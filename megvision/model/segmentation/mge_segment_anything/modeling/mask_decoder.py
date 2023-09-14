from typing import List, Tuple, Type

import megengine as mge
import megengine.functional as F
import megengine.module as M

from .common import LayerNorm2d

try:
    from megengine.functional import repeat_interleave
except:
    from ..utils.function import repeat_interleave


class MaskDecoder(M.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: M.Module,
        num_multimask_outputs: int = 3,
        activation: Type[M.Module] = M.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = M.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = M.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = M.Sequential(
            M.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            M.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ]

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: mge.Tensor,
        image_pe: mge.Tensor,
        sparse_prompt_embeddings: mge.Tensor,
        dense_prompt_embeddings: mge.Tensor,
        multimask_output: bool,
    ) -> Tuple[mge.Tensor, mge.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (mge.Tensor): the embeddings from the image encoder
          image_pe (mge.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (mge.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (mge.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          mge.Tensor: batched predicted masks
          mge.Tensor: batched predictions of mask quality
        """

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: mge.Tensor,
        image_pe: mge.Tensor,
        sparse_prompt_embeddings: mge.Tensor,
        dense_prompt_embeddings: mge.Tensor,
    ) -> Tuple[mge.Tensor, mge.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = F.concat(
            [self.iou_token.weight, self.mask_tokens.weight], axis=0
        )
        output_tokens = F.broadcast_to(
            F.expand_dims(output_tokens, 0),
            (sparse_prompt_embeddings.shape[0], None, None),
        )
        tokens = F.concat((output_tokens, sparse_prompt_embeddings), axis=1)

        # Expand per-image data in batch direction to be per-mask
        src = repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(0, 2, 1).reshape(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[mge.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = F.stack(hyper_in_list, axis=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.reshape(b, c, h * w)).reshape(
            b, -1, h, w
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(M.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            M.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
