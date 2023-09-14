from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import megengine as mge
import megengine.data.transform as T
import megengine.functional as F


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched mge tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )

        def to_pil_image(npimg, mode=None):
            assert isinstance(npimg, np.ndarray)
            if npimg.ndim not in {2, 3}:
                raise ValueError(
                    f"pic should be 2/3 dimensional. Got {npimg.ndim} dimensions."
                )
            elif npimg.ndim == 2:
                npimg = np.expand_dims(npimg, 2)
            if npimg.shape[-1] > 4:
                raise ValueError(
                    f"pic should not have > 4 channels. Got {npimg.shape[-1]} channels."
                )

            if npimg.shape[2] == 1:
                expected_mode = None
                npimg = npimg[:, :, 0]
                if npimg.dtype == np.uint8:
                    expected_mode = "L"
                elif npimg.dtype == np.int16:
                    expected_mode = "I;16"
                elif npimg.dtype == np.int32:
                    expected_mode = "I"
                elif npimg.dtype == np.float32:
                    expected_mode = "F"
                if mode is not None and mode != expected_mode:
                    raise ValueError(
                        f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}"
                    )
                mode = expected_mode
            elif npimg.shape[2] == 2:
                permitted_2_channel_modes = ["LA"]
                if mode is not None and mode not in permitted_2_channel_modes:
                    raise ValueError(
                        f"Only modes {permitted_2_channel_modes} are supported for 2D inputs"
                    )
                if mode is None and npimg.dtype == np.uint8:
                    mode = "LA"
            elif npimg.shape[2] == 4:
                permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
                if mode is not None and mode not in permitted_4_channel_modes:
                    raise ValueError(
                        f"Only modes {permitted_4_channel_modes} are supported for 4D inputs"
                    )

                if mode is None and npimg.dtype == np.uint8:
                    mode = "RGBA"
            else:
                permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
                if mode is not None and mode not in permitted_3_channel_modes:
                    raise ValueError(
                        f"Only modes {permitted_3_channel_modes} are supported for 3D inputs"
                    )
                if mode is None and npimg.dtype == np.uint8:
                    mode = "RGB"

            if mode is None:
                raise TypeError(f"Input type {npimg.dtype} is not supported")

            return Image.fromarray(npimg, mode=mode)

        def _compute_resized_output_size(
            image_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
        ) -> List[int]:
            if len(size) == 1:
                h, w = image_size
                short, long = (w, h) if w <= h else (h, w)
                requested_new_short = size if isinstance(size, int) else size[0]

                new_short, new_long = (
                    requested_new_short,
                    int(requested_new_short * long / short),
                )

                if max_size is not None:
                    if max_size <= requested_new_short:
                        raise ValueError(
                            f"max_size = {max_size} must be strictly greater than the requested "
                            f"size for the smaller edge size = {size}"
                        )
                    if new_long > max_size:
                        new_short, new_long = (
                            int(max_size * new_short / new_long),
                            max_size,
                        )

                new_w, new_h = (
                    (new_short, new_long) if w <= h else (new_long, new_short)
                )
            else:  # specified both h and w
                new_w, new_h = size[1], size[0]
            return [new_h, new_w]

        def resize_use_pil(
            img,
            size: List[int],
            interpolation="BILINEAR",
            max_size: Optional[int] = None,
            antialias: Optional[Union[str, bool]] = "warn",
        ):
            if isinstance(size, (list, tuple)):
                if len(size) not in [1, 2]:
                    raise ValueError(
                        f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
                    )
                if max_size is not None and len(size) != 1:
                    raise ValueError(
                        "max_size should only be passed if size specifies the length of the smaller edge, "
                        "i.e. size should be an int or a sequence of length 1"
                    )

            if isinstance(size, int):
                size = [size]
            output_size = _compute_resized_output_size(img.size, size, max_size)

            if img.size == output_size:
                return img

            pil_modes_mapping = {
                "BILINEAR": 2,
            }

            pil_interpolation = pil_modes_mapping[interpolation]
            return img.resize(tuple(output_size[::-1]), pil_interpolation)

        return np.array(resize_use_pil(to_pil_image(image), target_size))

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_mge(self, image: mge.Tensor) -> mge.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_mge(
        self, coords: mge.Tensor, original_size: Tuple[int, ...]
    ) -> mge.Tensor:
        """
        Expects a mge tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to("float32")
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_mge(
        self, boxes: mge.Tensor, original_size: Tuple[int, ...]
    ) -> mge.Tensor:
        """
        Expects a mge tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_mge(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
