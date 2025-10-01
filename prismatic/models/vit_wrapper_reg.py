"""Implementation of additional modules for the VLA's vision transformer."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from .film_vit_wrapper import FiLMedVisionTransformerBlock, NullVisionTransformerBlockWrapper


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """Utility function for monkey-patching functions."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class FiLMedVisionTransformerRegister(VisionTransformer):
    """
    Wrapper for timm.models.vision_transformer.VisionTransformer that overrides functions to enable infusing language
    embeddings into visual embeddings via FiLM.
    """

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        vision_aggr: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ):
        """
        Copy of timm.models.vision_transformer.VisionTransformer._intermediate_layers() with modifications
        to take in language embeddings as additional input.
        """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        vision_aggr_batch = vision_aggr.expand(x.shape[0], -1, -1)
        x = torch.cat([
            x, vision_aggr_batch.to(x.dtype).to(x.device)
        ], dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, language_embeddings)  # Modified to receive language_embeddings
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        vision_aggr: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Copy of timm.models.vision_transformer.VisionTransformer.get_intermediate_layers() with modifications
        to allow language embeddings as additional input.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(
            x, language_embeddings=language_embeddings, vision_aggr=vision_aggr, n=n
        )
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]
        outputs = [out[:, self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]:] for out in outputs]  # treat register as visual embedding

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class FiLMedPrismaticVisionBackboneAggregator(nn.Module):
    """
    Wrapper for OpenVLA's vision backbone that implements feature-wise linear modulation (FiLM).

    Wraps the Vision Transformers in the vision backbone to enable language conditioning through FiLM.
    Supports processing 1-3 images using dual vision backbones (SigLIP + DINOv2).
    """

    def __init__(
        self,
        vision_backbone,
        llm_dim: int = 4096,  # 4096 for Llama-2 7B
        num_vision_aggr=32
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (PrismaticVisionBackbone): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim
        self.num_vision_aggr = num_vision_aggr

        # Wrap vision transformers
        self.vision_aggr_featurizer = nn.Parameter(
            torch.randn(1, num_vision_aggr, self.vision_backbone.featurizer.embed_dim)
        )
        self._wrap_vit(
            self.vision_backbone.featurizer,
            self.vision_aggr_featurizer
        )  # SigLIP

        if self.vision_backbone.use_fused_vision_backbone:
            self.vision_aggr_fused_featurizer = nn.Parameter(
                torch.randn(1, num_vision_aggr, self.vision_backbone.fused_featurizer.embed_dim)
            )
            self._wrap_vit(
                self.vision_backbone.fused_featurizer,
                self.vision_aggr_fused_featurizer
            )  # DINOv2

    def _wrap_vit(self, vit, vision_aggr) -> None:
        """
        Creates wrapper around an individual vision transformer to allow for infusion of language inputs.

        Args:
            vit (VisionTransformer): Original vision transformer.
        """
        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.num_features, llm_dim=self.llm_dim)
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = FiLMedVisionTransformerRegister
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, vision_aggr=vision_aggr, n={len(vit.blocks) - 2}))

    # def get_num_patches(self) -> int:
    #     """Returns the number of vision patches output by the vision backbone."""
    #     return self.vision_backbone.get_num_patches()

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        return self.num_vision_aggr

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, seq_len, llm_dim).
        """
        # For FiLM: Average the language embeddings of the task description
        average_language_embedding = language_embeddings.mean(dim=1)

        if self.get_num_images_in_input() == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values, average_language_embedding)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img, average_language_embedding)
            patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

            return (patches, patches_fused)  # Tuple[Tensor]

        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.get_num_images_in_input(), dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.vision_backbone.featurizer(img_regular, average_language_embedding)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

                all_patches.append((patches, patches_fused))

            return all_patches  # List[Tuple[Tensor]]


# ===================================================
class VisionTransformerRegister(VisionTransformer):
    def _intermediate_layers(
        self,
        x: torch.Tensor,
        vision_aggr: torch.Tensor,
        n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        vision_aggr_batch = vision_aggr.expand(x.shape[0], -1, -1)
        x = torch.cat([
            x, vision_aggr_batch.to(x.dtype).to(x.device)
        ], dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x, None)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        vision_aggr: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(
            x, vision_aggr=vision_aggr, n=n
        )
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]
        outputs = [out[:, self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]:] for out in outputs]  # treat register as visual embedding

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class PrismaticVisionBackboneAggregator(FiLMedPrismaticVisionBackboneAggregator):
    def _wrap_vit(self, vit, vision_aggr) -> None:
        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                NullVisionTransformerBlockWrapper(block=block)
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = VisionTransformerRegister
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, vision_aggr=vision_aggr, n={len(vit.blocks) - 2}))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.get_num_images_in_input() == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.vision_backbone.featurizer(img)
            patches_fused = self.vision_backbone.fused_featurizer(img_fused)

            return (patches, patches_fused)  # Tuple[Tensor]

        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.get_num_images_in_input(), dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.vision_backbone.featurizer(img_regular)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused)

                all_patches.append((patches, patches_fused))

            return all_patches  # List[Tuple[Tensor]]
