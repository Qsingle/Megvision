from typing import Any, Optional, Tuple, Type

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .common import LayerNorm2d


class PromptEncoder(M.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[M.Module] = M.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            M.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = M.Sequential(*point_embeddings)
        self.not_a_point_embed = M.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = M.Sequential(
            M.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            M.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            M.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = M.Embedding(1, embed_dim)

    def get_dense_pe(self) -> mge.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          mge.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return F.expand_dims(self.pe_layer(self.image_embedding_size), 0)

    def _embed_points(
        self, points: mge.Tensor, labels: mge.Tensor, pad: bool,
    ) -> mge.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = F.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -F.ones((labels.shape[0], 1), device=labels.device)
            points = F.concat([points, padding_point], axis=1)
            labels = F.concat([labels, padding_label], axis=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: mge.Tensor) -> mge.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: mge.Tensor) -> mge.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[mge.Tensor, mge.Tensor]],
        boxes: Optional[mge.Tensor],
        masks: Optional[mge.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> mge.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[mge.Tensor, mge.Tensor]],
        boxes: Optional[mge.Tensor],
        masks: Optional[mge.Tensor],
    ) -> Tuple[mge.Tensor, mge.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(mge.Tensor, mge.Tensor) or none): point coordinates
            and labels to embed.
          boxes (mge.Tensor or none): boxes to embed
          masks (mge.Tensor or none): masks to embed

        Returns:
          mge.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          mge.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = F.zeros((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = F.concat([sparse_embeddings, point_embeddings], axis=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = F.concat([sparse_embeddings, box_embeddings], axis=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = F.broadcast_to(
                self.no_mask_embed.weight.reshape(1, -1, 1, 1),
                (bs, None, self.image_embedding_size[0], self.image_embedding_size[1]),
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(M.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.positional_encoding_gaussian_matrix = mge.Tensor(
            scale * np.random.randn(2, num_pos_feats)
        )

    def _pe_encoding(self, coords: mge.Tensor) -> mge.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return F.concat([F.sin(coords), F.cos(coords)], axis=-1)

    def forward(self, size: Tuple[int, int]) -> mge.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = F.ones((h, w), device=device, dtype="float32")
        y_embed = F.cumsum(grid, axis=0) - 0.5
        x_embed = F.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(F.stack([x_embed, y_embed], axis=-1))
        return pe.transpose(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: mge.Tensor, image_size: Tuple[int, int]
    ) -> mge.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.to(coords_input.device)
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.astype("float32"))  # B x N x C
