import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np

import megengine as mge
import megengine.functional as F


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, mge.Tensor)
            ), "MaskData only supports list, numpy arrays, and mge tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, mge.Tensor)
        ), "MaskData only supports list, numpy arrays, and mge tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: mge.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, mge.Tensor):
                self._stats[k] = v[mge.tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().numpy()]
            elif isinstance(v, list) and keep.dtype == np.dtype("bool"):
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, mge.Tensor):
                self._stats[k] = F.concat([self._stats[k], v], axis=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, mge.Tensor):
                self._stats[k] = v.detach().numpy()


def is_box_near_crop_edge(
    boxes: mge.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> mge.Tensor:
    def isclose(
        a: mge.Tensor,
        b: mge.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan=None,
    ) -> mge.Tensor:
        assert equal_nan is None
        return F.abs(a - b) <= atol + rtol * F.abs(b)

    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    crop_box_mge = mge.tensor(crop_box, dtype="float32", device=boxes.device)
    orig_box_mge = mge.tensor(orig_box, dtype="float32", device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).astype("float32")
    near_crop_edge = isclose(boxes, crop_box_mge[None, :], atol=atol, rtol=0)
    near_image_edge = isclose(boxes, orig_box_mge[None, :], atol=atol, rtol=0)
    near_crop_edge = F.logical_and(near_crop_edge, ~near_image_edge)

    def mge_any(x, axis):
        return F.sum(x, axis=axis) > 0

    return mge_any(near_crop_edge, axis=1)


def box_xyxy_to_xywh(box_xyxy: mge.Tensor) -> mge.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle_mge(tensor: mge.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = F.flatten(tensor.transpose(0, 2, 1), 1)

    # Compute change indices
    # there is a bug when indexing bool tensor with shape (1, n)
    tensor_int32 = tensor.astype("int32")
    diff = tensor_int32[:, 1:].astype("bool") ^ tensor_int32[:, :-1].astype("bool")

    def nonzero2d(x):
        _, raw_idx = F.cond_take(x != 0, x)
        idx = F.zeros((raw_idx.shape[0], 2), dtype="int32")
        idx[:, 0] = raw_idx // x.shape[1]
        idx[:, 1] = raw_idx % x.shape[1]
        return idx

    change_indices = nonzero2d(diff)

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = F.concat(
            [
                mge.Tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                mge.Tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


def calculate_stability_score(
    masks: mge.Tensor, mask_threshold: float, threshold_offset: float
) -> mge.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1)
        .astype("int32")
        .sum(-1)
        .astype("int32")
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1)
        .astype("int32")
        .sum(-1)
        .astype("int32")
    )
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer ** i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: mge.Tensor, crop_box: List[int]) -> mge.Tensor:
    x0, y0, _, _ = crop_box
    offset = mge.Tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: mge.Tensor, crop_box: List[int]) -> mge.Tensor:
    x0, y0, _, _ = crop_box
    offset = mge.Tensor([[x0, y0]], device=points.device)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(
    masks: mge.Tensor, crop_box: List[int], orig_h: int, orig_w: int
) -> mge.Tensor:
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return F.pad(masks, pad, value=0)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def batched_mask_to_box(masks: mge.Tensor) -> mge.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    if np.prod(masks.shape) == 0:
        return F.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = F.flatten(masks, 0, -3)
    else:
        masks = F.expand_dims(masks, 0)

    # Get top and bottom edges
    in_height = F.max(masks, axis=-1)
    in_height_coords = in_height * F.arange(h, device=in_height.device)[None, :]
    bottom_edges = F.max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges = F.min(in_height_coords, axis=-1)

    # Get left and right edges
    in_width = F.max(masks, axis=-2)
    in_width_coords = in_width * F.arange(w, device=in_width.device)[None, :]
    right_edges = F.max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges = F.min(in_width_coords, axis=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = F.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * F.expand_dims((~empty_filter), -1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out
