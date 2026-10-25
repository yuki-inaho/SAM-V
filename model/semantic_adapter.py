"""Learned class-query semantic adapter for the native SAM-V head.

This module adds an automatic, prompt-free semantic route on top of the native
SAM-V components.  It does *not* replace or re-implement the native encoders or
the native fusion/decoder modules:

* ``embedding_fusion_mlp`` and ``cross_attention_fusion`` are the actual
  :class:`model.sam_vggt_model.PerPixelMLP` / ``CrossAttentionFusion`` modules.
* ``sam.mask_decoder`` is the native ``segment_anything`` ``MaskDecoder`` (with
  its native ``TwoWayTransformer``).
* ``sam.prompt_encoder`` is only used for its native dense positional encoding;
  the original point/box/mask prompt route on ``SamVGGT.forward`` is untouched.

The primary API consumes *cached* native encoder outputs, so decoder training
from a feature cache never needs both frozen encoders resident on the GPU::

    sam_features   [B, N, 256, 64, 64]   native SAM ViT-H image features
    vggt_features  [B, N, 2048, 64, 64]  native VGGT-1B joint patch features
    camera_tokens  [B, N, 2048]          native VGGT camera tokens

plus a measured depth map and validity mask aligned to the native RGB
preprocessing.  Depth is a real inference input (RGB-D); missing depth is
represented by ``validity == 0`` and never by a forced background class.

Outputs are per-view logits ``[B, N, K, H, W]`` plus the native-grid
``[B, N, K, 64, 64]`` logits, or target-only ``[B, K, H, W]`` when a view is
selected.  ``K`` is a constructor argument and is not tied to a dataset.

No ground-truth points, masks, boxes, crops, labels or target masks are accepted
by the inference API.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from model.sam_vggt_model import CrossAttentionFusion, PerPixelMLP

__all__ = [
    "DepthAlignment",
    "DepthValidityAdapter",
    "SemanticClassQuery",
    "SemanticAdapterOutput",
    "SamVSemanticAdapter",
    "NATIVE_STAGE2_PREFIXES",
    "align_depth_to_grid",
    "invert_view_logits",
    "build_native_semantic_adapter",
    "competition_probabilities",
    "predict_views",
    "predict_target",
    "freeze_encoder_parameters_",
    "offload_frozen_encoders_",
]

#: State-dict prefixes that the released SAM-V stage-2 head actually contains.
NATIVE_STAGE2_PREFIXES: Tuple[str, ...] = (
    "embedding_fusion_mlp.",
    "cross_attention_fusion.",
    "sam.mask_decoder.",
)

_SAM_IMAGE_SIZE = 1024
_SAM_GRID = 64
_SAM_PATCH = _SAM_IMAGE_SIZE // _SAM_GRID
_SAM_DIM = 256
_VGGT_DIM = 2048
_DECODER_UPSAMPLE = 4


# ---------------------------------------------------------------------------
# Geometry: depth/validity alignment and logit inversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthAlignment:
    """Exact aspect-preserving alignment of an original frame to the SAM grid.

    The native SAM preprocessing used by :class:`SamVGGT` resizes the longest
    side to ``canvas_size`` (rounding half up, exactly like
    ``ResizeLongestSide.get_preprocess_shape``) and pads the right/bottom to a
    square canvas.  This dataclass records that plan so depth and validity can
    be resampled with the *same* geometry instead of being stretched to the
    square canvas.

    ``decoder_canvas_size`` is the width/height of one view in the mask
    decoder's low-resolution output (4x the 64x64 embedding grid).
    """

    original_h: int
    original_w: int
    resized_h: int
    resized_w: int
    pad_top: int = 0
    pad_left: int = 0
    canvas_size: int = _SAM_IMAGE_SIZE
    grid_h: int = _SAM_GRID
    grid_w: int = _SAM_GRID

    def __post_init__(self) -> None:
        for name in ("original_h", "original_w", "resized_h", "resized_w", "canvas_size"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.resized_h > self.canvas_size or self.resized_w > self.canvas_size:
            raise ValueError("resized size must fit inside the square canvas")
        if self.pad_top < 0 or self.pad_left < 0:
            raise ValueError("pads must be non-negative")
        if self.resized_h + self.pad_top > self.canvas_size:
            raise ValueError("resized_h + pad_top exceeds the canvas")
        if self.resized_w + self.pad_left > self.canvas_size:
            raise ValueError("resized_w + pad_left exceeds the canvas")
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError("grid size must be positive")

    # -- construction -----------------------------------------------------
    @classmethod
    def from_size(
        cls,
        height: int,
        width: int,
        canvas_size: int = _SAM_IMAGE_SIZE,
        grid_size: Union[int, Tuple[int, int]] = _SAM_GRID,
    ) -> "DepthAlignment":
        """Build the native SAM resize/pad plan for an original ``H, W``."""
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        scale = canvas_size * 1.0 / max(height, width)
        # Same rounding as segment_anything.utils.transforms.ResizeLongestSide.
        resized_h = int(height * scale + 0.5)
        resized_w = int(width * scale + 0.5)
        if isinstance(grid_size, int):
            grid_h = grid_w = int(grid_size)
        else:
            grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
        return cls(
            original_h=height,
            original_w=width,
            resized_h=resized_h,
            resized_w=resized_w,
            pad_top=0,
            pad_left=0,
            canvas_size=int(canvas_size),
            grid_h=grid_h,
            grid_w=grid_w,
        )

    # -- derived values ---------------------------------------------------
    @property
    def decoder_canvas_size(self) -> int:
        """Width/height of one view in the low-resolution decoder output."""
        return int(self.grid_w * _DECODER_UPSAMPLE)

    @property
    def decoder_valid_h(self) -> float:
        return self.resized_h * self.decoder_canvas_size / self.canvas_size

    @property
    def decoder_valid_w(self) -> float:
        return self.resized_w * self.decoder_canvas_size / self.canvas_size

    def matches(self, other: "DepthAlignment") -> bool:
        return (
            self.original_h == other.original_h
            and self.original_w == other.original_w
            and self.resized_h == other.resized_h
            and self.resized_w == other.resized_w
            and self.pad_top == other.pad_top
            and self.pad_left == other.pad_left
            and self.canvas_size == other.canvas_size
            and self.grid_h == other.grid_h
            and self.grid_w == other.grid_w
        )

    def validate_against_size(self, height: int, width: int) -> None:
        """Reject stretched or otherwise non-native alignment plans."""
        expected = DepthAlignment.from_size(
            height, width, self.canvas_size, (self.grid_h, self.grid_w)
        )
        if not self.matches(expected):
            raise ValueError(
                "depth/validity alignment does not match the native aspect-preserving "
                f"RGB resize/pad plan: got {(self.resized_h, self.resized_w, self.pad_top, self.pad_left)}, "
                f"expected {(expected.resized_h, expected.resized_w, expected.pad_top, expected.pad_left)}"
            )


def _as_bn_depth(
    tensor: torch.Tensor, name: str, batch: int, views: int
) -> torch.Tensor:
    if tensor.dim() == 5 and tensor.shape[2] == 1:
        tensor = tensor[:, :, 0]
    if tensor.dim() != 4:
        raise ValueError(f"{name} must have shape [B,N,H,W] or [B,N,1,H,W]")
    if tensor.shape[0] != batch or tensor.shape[1] != views:
        raise ValueError(
            f"{name} batch/view dimensions {tuple(tensor.shape[:2])} do not match "
            f"features {(batch, views)}"
        )
    return tensor


def align_depth_to_grid(
    measured_depth: torch.Tensor,
    validity: torch.Tensor,
    alignment: DepthAlignment,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample depth/validity onto the 64x64 native grid.

    Depth is clamped to ``[0, 3]`` metres and divided by 3.  Only valid pixels
    contribute; the resampling is a validity-weighted area average so padding
    and missing pixels cannot bleed invalid *zero depth* into valid cells.

    Returns ``(depth_grid, valid_grid)`` with shapes ``[B,N,64,64]``.
    """
    if measured_depth.shape != validity.shape:
        raise ValueError(
            f"measured_depth {tuple(measured_depth.shape)} and validity "
            f"{tuple(validity.shape)} must have identical shapes"
        )
    alignment.validate_against_size(measured_depth.shape[-2], measured_depth.shape[-1])
    valid = (validity > 0).to(dtype=measured_depth.dtype)
    depth = torch.clamp(measured_depth, min=0.0, max=3.0) / 3.0
    depth = depth * valid

    b, n, _, _ = depth.shape
    flat_depth = depth.reshape(b * n, 1, *depth.shape[-2:])
    flat_valid = valid.reshape(b * n, 1, *valid.shape[-2:])

    if (alignment.resized_h, alignment.resized_w) != (depth.shape[-2], depth.shape[-1]):
        flat_depth = F.interpolate(
            flat_depth,
            size=(alignment.resized_h, alignment.resized_w),
            mode="bilinear",
            align_corners=False,
        )
        flat_valid = F.interpolate(
            flat_valid,
            size=(alignment.resized_h, alignment.resized_w),
            mode="bilinear",
            align_corners=False,
        )

    pad_h = alignment.canvas_size - alignment.resized_h
    pad_w = alignment.canvas_size - alignment.resized_w
    # SAM pads right/bottom; the left/top pads are recorded for completeness.
    if alignment.pad_top or alignment.pad_left:
        pad_top, pad_left = alignment.pad_top, alignment.pad_left
        pad_bottom = alignment.canvas_size - alignment.resized_h - pad_top
        pad_right = alignment.canvas_size - alignment.resized_w - pad_left
    else:
        pad_top = pad_left = 0
        pad_bottom, pad_right = pad_h, pad_w
    flat_depth = F.pad(flat_depth, (pad_left, pad_right, pad_top, pad_bottom))
    flat_valid = F.pad(flat_valid, (pad_left, pad_right, pad_top, pad_bottom))

    weight = F.adaptive_avg_pool2d(flat_valid, (alignment.grid_h, alignment.grid_w))
    weighted = F.adaptive_avg_pool2d(flat_depth, (alignment.grid_h, alignment.grid_w))
    present = weight > eps
    depth_grid = torch.where(present, weighted / weight.clamp_min(eps), torch.zeros_like(weighted))
    valid_grid = present.to(dtype=measured_depth.dtype)
    depth_grid = depth_grid.reshape(b, n, alignment.grid_h, alignment.grid_w)
    valid_grid = valid_grid.reshape(b, n, alignment.grid_h, alignment.grid_w)
    return depth_grid, valid_grid


def _view_affine_grid(
    alignment: DepthAlignment,
    batch: int,
    out_h: int,
    out_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine grid mapping an original frame onto one wide-canvas view crop.

    The crop covers the valid (non-padded) part of the view's decoder canvas.
    ``align_corners=False`` pixel-centre arithmetic is used throughout, so a
    landmark at original pixel centre ``x + 0.5`` lands on output pixel ``x``
    with no half-pixel shift.
    """
    in_h = float(alignment.decoder_canvas_size)
    in_w = float(alignment.decoder_canvas_size)
    valid_h = float(alignment.decoder_valid_h)
    valid_w = float(alignment.decoder_valid_w)
    x0 = alignment.pad_left * in_w / alignment.canvas_size
    y0 = alignment.pad_top * in_h / alignment.canvas_size
    if valid_h <= 0 or valid_w <= 0:
        raise ValueError("empty valid crop in decoder canvas")

    # grid_sample normalized coordinates: input_norm = a * output_norm + b.
    ax = valid_w / in_w
    ay = valid_h / in_h
    bx = ax + 2.0 * x0 / in_w - 1.0
    by = ay + 2.0 * y0 / in_h - 1.0
    theta = torch.tensor(
        [[[ax, 0.0, bx], [0.0, ay, by]]], device=device, dtype=dtype
    ).expand(batch, -1, -1)
    return F.affine_grid(theta, size=(batch, 1, out_h, out_w), align_corners=False)


def invert_view_logits(
    view_logits: torch.Tensor,
    alignments: Union[DepthAlignment, Sequence[DepthAlignment]],
) -> torch.Tensor:
    """Invert per-view decoder logits from the square canvas to original H/W.

    Args:
        view_logits: ``[B, N, K, C, C]`` low-resolution decoder logits, where
            ``C == grid * 4`` (256 for the native 64x64 grid).
        alignments: one :class:`DepthAlignment` for every view, or a single
            plan reused for all views.

    Returns:
        ``[B, N, K, H, W]`` logits in the original frame geometry.  The valid
        crop of each square view canvas is resampled without stretching the
        padded region and without an alignment shift.
    """
    if view_logits.dim() != 5:
        raise ValueError("view_logits must be [B,N,K,C,C]")
    b, n, k, c_h, c_w = view_logits.shape
    if isinstance(alignments, DepthAlignment):
        plans = [alignments] * n
    else:
        plans = list(alignments)
        if len(plans) != n:
            raise ValueError(f"expected {n} alignments, got {len(plans)}")

    outputs: List[torch.Tensor] = []
    for index, plan in enumerate(plans):
        if c_h != plan.decoder_canvas_size or c_w != plan.decoder_canvas_size:
            raise ValueError(
                f"view {index} decoder canvas {(c_h, c_w)} does not match "
                f"alignment decoder canvas {plan.decoder_canvas_size}"
            )
        grid = _view_affine_grid(
            plan,
            batch=b,
            out_h=plan.original_h,
            out_w=plan.original_w,
            device=view_logits.device,
            dtype=torch.float32,
        )
        sampled = F.grid_sample(
            view_logits[:, index].float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        outputs.append(sampled.to(dtype=view_logits.dtype))
    return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Trainable adapter pieces
# ---------------------------------------------------------------------------


class DepthValidityAdapter(nn.Module):
    """Small trainable measured-depth + validity adapter.

    ``2 -> hidden -> 256`` convolutional adapter, as used for RGB-D inference.
    Input channels are ``(clamp(depth, 0, 3) / 3) * valid`` and ``valid``.
    The output is gated by grid validity, so an entirely missing view produces
    exactly zero depth features: missing depth cannot force any class.
    """

    def __init__(self, hidden_dim: int = 64, out_dim: int = _SAM_DIM) -> None:
        super().__init__()
        if hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("hidden_dim and out_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2, self.hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1, bias=True),
        )

    def forward(
        self,
        measured_depth: torch.Tensor,
        validity: torch.Tensor,
        alignment: Union[DepthAlignment, Sequence[DepthAlignment]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, h, w = measured_depth.shape
        if isinstance(alignment, DepthAlignment):
            plans = [alignment] * n
        else:
            plans = list(alignment)
            if len(plans) != n:
                raise ValueError(f"expected {n} alignments, got {len(plans)}")
        if measured_depth.shape != validity.shape:
            raise ValueError("measured_depth and validity must have identical shapes")
        param_device = next(self.parameters()).device
        measured_depth = measured_depth.to(device=param_device)
        validity = validity.to(device=param_device)
        if plans[0].grid_h != plans[0].grid_w:
            raise ValueError("depth adapter currently requires a square native grid")
        grid_h, grid_w = plans[0].grid_h, plans[0].grid_w
        for plan in plans:
            if (plan.grid_h, plan.grid_w) != (grid_h, grid_w):
                raise ValueError("all views must share one native grid size")

        depths = []
        valids = []
        for index, plan in enumerate(plans):
            d_i, v_i = align_depth_to_grid(
                measured_depth[:, index : index + 1],
                validity[:, index : index + 1],
                plan,
            )
            depths.append(d_i)
            valids.append(v_i)
        depth_grid = torch.cat(depths, dim=1).reshape(b * n, 1, grid_h, grid_w)
        valid_grid = torch.cat(valids, dim=1).reshape(b * n, 1, grid_h, grid_w)

        dtype = next(self.parameters()).dtype
        features = self.net(torch.cat([depth_grid, valid_grid], dim=1).to(dtype=dtype))
        features = features * valid_grid.to(dtype=dtype)
        return features.reshape(b, n, self.out_dim, grid_h, grid_w), valid_grid.reshape(
            b, n, grid_h, grid_w
        )


class SemanticClassQuery(nn.Module):
    """Learnable class queries and their dense/decoder calibration."""

    def __init__(self, num_classes: int, dim: int = _SAM_DIM, init_std: float = 0.02) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.query = nn.Parameter(torch.randn(self.num_classes, self.dim) * init_std)
        # Dense bias added inside the native mask decoder, and a final per-class
        # logit calibration. Both are new parameters (not in the stage-2 head).
        self.class_dense = nn.Parameter(torch.zeros(self.num_classes, self.dim))
        self.class_bias = nn.Parameter(torch.zeros(self.num_classes))


class _SamDecoderBundle(nn.Module):
    """Native SAM decoder pieces, with stage-2-compatible key names."""

    def __init__(self, mask_decoder: MaskDecoder, prompt_encoder: PromptEncoder) -> None:
        super().__init__()
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder


ModuleLike = Any


class SemanticAdapterOutput:
    """Result of :meth:`SamVSemanticAdapter.forward`.

    Attributes:
        grid_logits: ``[B,N,K,64,64]`` logits on the native embedding grid.
        logits: ``[B,N,K,H,W]`` logits in original frame geometry, or ``None``
            when original geometry could not be derived.
        target_logits: ``[B,K,H,W]`` logits for the selected ``output_view``,
            or ``None``.
        iou_predictions: ``[B,K]`` native mask-decoder quality predictions.
        class_tokens: ``[B,K,256]`` class queries after native cross-attention.
        depth_features: ``[B,N,256,64,64]`` depth contribution (before scale).
        valid_grid: ``[B,N,64,64]`` validity after native alignment.
        alignment: the :class:`DepthAlignment` or sequence used.
    """

    __slots__ = (
        "grid_logits",
        "logits",
        "target_logits",
        "iou_predictions",
        "class_tokens",
        "depth_features",
        "valid_grid",
        "alignment",
    )

    def __init__(
        self,
        grid_logits: torch.Tensor,
        logits: Optional[torch.Tensor],
        target_logits: Optional[torch.Tensor],
        iou_predictions: torch.Tensor,
        class_tokens: torch.Tensor,
        depth_features: Optional[torch.Tensor],
        valid_grid: Optional[torch.Tensor],
        alignment: Union[DepthAlignment, Sequence[DepthAlignment]],
    ) -> None:
        self.grid_logits = grid_logits
        self.logits = logits
        self.target_logits = target_logits
        self.iou_predictions = iou_predictions
        self.class_tokens = class_tokens
        self.depth_features = depth_features
        self.valid_grid = valid_grid
        self.alignment = alignment

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


def _native_prompt_encoder(
    image_size: int = _SAM_IMAGE_SIZE, patch_size: int = _SAM_PATCH, prompt_dim: int = _SAM_DIM
) -> PromptEncoder:
    return PromptEncoder(
        embed_dim=prompt_dim,
        image_embedding_size=(image_size // patch_size, image_size // patch_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )


def _native_mask_decoder(prompt_dim: int = _SAM_DIM) -> MaskDecoder:
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )


class SamVSemanticAdapter(nn.Module):
    """Prompt-free semantic head built from the native SAM-V modules.

    The constructor takes *instances* of the native modules so callers can pass
    the modules of a real :class:`SamVGGT` (use :meth:`from_sam_vggt`) or build
    the native architecture directly (use :func:`build_native_semantic_adapter`).
    No substitute backbone is created here.
    """

    def __init__(
        self,
        embedding_fusion_mlp: PerPixelMLP,
        cross_attention_fusion: CrossAttentionFusion,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
        num_classes: int,
        depth_hidden_dim: int = 64,
        depth_scale_init: float = 1.0,
        class_query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if not isinstance(embedding_fusion_mlp, PerPixelMLP):
            raise TypeError("embedding_fusion_mlp must be the native PerPixelMLP")
        if not isinstance(cross_attention_fusion, CrossAttentionFusion):
            raise TypeError("cross_attention_fusion must be the native CrossAttentionFusion")
        if not isinstance(mask_decoder, MaskDecoder):
            raise TypeError("mask_decoder must be the native segment_anything MaskDecoder")
        if not isinstance(prompt_encoder, PromptEncoder):
            raise TypeError("prompt_encoder must be the native segment_anything PromptEncoder")

        self.num_classes = int(num_classes)
        self.embedding_fusion_mlp = embedding_fusion_mlp
        self.cross_attention_fusion = cross_attention_fusion
        self.sam = _SamDecoderBundle(mask_decoder, prompt_encoder)
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad_(False)
        self.sam.prompt_encoder.eval()
        self.prompt_encoder_provenance: Dict[str, Any] = {
            "source": "unloaded_random",
            "native_pretrained": False,
            "keys_loaded": 0,
            "strict": False,
        }
        self.semantic_class_query = SemanticClassQuery(
            self.num_classes, dim=_SAM_DIM, init_std=class_query_init_std
        )
        self.depth_adapter = DepthValidityAdapter(hidden_dim=depth_hidden_dim, out_dim=_SAM_DIM)
        self.depth_scale = nn.Parameter(torch.tensor(float(depth_scale_init)))
        self._native_model_ref: Optional[weakref.ReferenceType] = None

    # -- construction -----------------------------------------------------
    @classmethod
    def from_sam_vggt(
        cls,
        sam_vggt: ModuleLike,
        num_classes: int,
        depth_hidden_dim: int = 64,
        depth_scale_init: float = 1.0,
        class_query_init_std: float = 0.02,
        freeze_encoders: bool = True,
    ) -> "SamVSemanticAdapter":
        """Build around the real modules of an existing :class:`SamVGGT`.

        The native ``embedding_fusion_mlp``, ``cross_attention_fusion`` and
        ``sam.mask_decoder`` are shared by identity, not copied.  The full
        SAM ViT-H and VGGT-1B encoders remain owned by ``sam_vggt`` and are
        frozen here; they are not required for cached-feature decoder training.
        """
        required = ("embedding_fusion_mlp", "cross_attention_fusion", "sam")
        for name in required:
            if not hasattr(sam_vggt, name):
                raise TypeError(f"sam_vggt is missing native attribute {name!r}")
        if not hasattr(sam_vggt.sam, "mask_decoder") or not hasattr(sam_vggt.sam, "prompt_encoder"):
            raise TypeError("sam_vggt.sam must expose native mask_decoder and prompt_encoder")

        adapter = cls(
            embedding_fusion_mlp=sam_vggt.embedding_fusion_mlp,
            cross_attention_fusion=sam_vggt.cross_attention_fusion,
            mask_decoder=sam_vggt.sam.mask_decoder,
            prompt_encoder=sam_vggt.sam.prompt_encoder,
            num_classes=num_classes,
            depth_hidden_dim=depth_hidden_dim,
            depth_scale_init=depth_scale_init,
            class_query_init_std=class_query_init_std,
        )
        adapter._native_model_ref = None
        try:
            adapter._native_model_ref = weakref.ref(sam_vggt)
        except TypeError:
            # Lightweight stubs (e.g. SimpleNamespace in unit tests) are not
            # weak-referenceable; the modules are still shared by identity.
            pass
        adapter.prompt_encoder_provenance = {
            "source": "from_sam_vggt",
            "native_pretrained": True,
            "keys_loaded": len(
                [k for k in adapter.state_dict() if k.startswith("sam.prompt_encoder.")]
            ),
            "strict": True,
        }
        if freeze_encoders:
            freeze_encoder_parameters_(sam_vggt)
        return adapter

    @property
    def native_model(self) -> Optional[ModuleLike]:
        return None if self._native_model_ref is None else self._native_model_ref()

    # -- stage-2 checkpoint ----------------------------------------------
    def load_native_stage2(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> Dict[str, Any]:
        """Strictly load the released stage-2 native tensors.

        Only the native prefixes (fusion MLP, cross-attention fusion, native
        mask decoder) are matched.  New adapter parameters and the native,
        non-stage-2 prompt-encoder buffers are reported separately; they are
        never silently treated as stage-2 tensors.
        """
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            cleaned[str(key).replace("module.", "", 1) if str(key).startswith("module.") else str(key)] = value

        native_keys = {key for key in cleaned if key.startswith(NATIVE_STAGE2_PREFIXES)}
        expected_native = {
            key for key in self.state_dict() if key.startswith(NATIVE_STAGE2_PREFIXES)
        }
        missing_native = sorted(expected_native - native_keys)
        unexpected_native = sorted(native_keys - expected_native)
        if strict and (missing_native or unexpected_native):
            raise RuntimeError(
                "native stage-2 state dict does not match the adapter exactly: "
                f"missing={missing_native[:8]} unexpected={unexpected_native[:8]}"
            )

        native_state = {key: cleaned[key] for key in native_keys if key in expected_native}
        self.load_state_dict(native_state, strict=False)

        all_keys = set(self.state_dict())
        absent = sorted(all_keys - native_keys)
        new_trainable = sorted(
            key
            for key in absent
            if key.startswith(
                (
                    "semantic_class_query.",
                    "depth_adapter.",
                    "depth_scale",
                )
            )
        )
        non_stage2_native = sorted(key for key in absent if key.startswith("sam.prompt_encoder."))
        return {
            "strict_native": not (missing_native or unexpected_native),
            "native_loaded": len(native_keys),
            "native_expected": len(expected_native),
            "missing_native": missing_native,
            "unexpected_native": unexpected_native,
            "non_native_checkpoint_keys": sorted(set(cleaned) - native_keys),
            "new_trainable_keys": new_trainable,
            "new_trainable_parameter_count": int(
                sum(self.state_dict()[key].numel() for key in new_trainable)
            ),
            "non_stage2_native_keys": non_stage2_native,
            "prompt_encoder_provenance": dict(self.prompt_encoder_provenance),
        }

    def load_native_stage2_file(
        self, path: str, map_location: Union[str, torch.device] = "cpu", strict: bool = True
    ) -> Dict[str, Any]:
        raw = torch.load(path, map_location=map_location, weights_only=True)
        return self.load_native_stage2(raw, strict=strict)

    # -- native SAM prompt encoder provenance ----------------------------
    def load_sam_prompt_encoder(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        source: str = "sam_checkpoint",
    ) -> Dict[str, Any]:
        """Strictly load the native SAM prompt encoder from a SAM state dict.

        The standalone cached-feature route must not silently run with a random
        native positional encoding.  This loads the original ``prompt_encoder.*``
        tensors (strict keys and shapes) into the frozen native prompt encoder
        used only for dense positional encoding.
        """
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = str(key)
            if key.startswith("sam.prompt_encoder."):
                mapped[key] = value
            elif key.startswith("prompt_encoder."):
                mapped["sam." + key] = value
        expected = {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith("sam.prompt_encoder.")
        }
        missing = sorted(set(expected) - set(mapped))
        unexpected = sorted(set(mapped) - set(expected))
        shape_mismatch = sorted(
            key
            for key in set(expected) & set(mapped)
            if tuple(expected[key].shape) != tuple(mapped[key].shape)
        )
        if strict and (missing or unexpected or shape_mismatch):
            raise RuntimeError(
                "SAM prompt-encoder state does not match the native encoder: "
                f"missing={missing[:8]} unexpected={unexpected[:8]} "
                f"shape_mismatch={shape_mismatch[:8]}"
            )
        if shape_mismatch:
            raise RuntimeError(f"shape mismatch for prompt encoder keys: {shape_mismatch[:8]}")
        local = {
            key[len("sam.prompt_encoder.") :]: value
            for key, value in mapped.items()
            if key in expected
        }
        self.sam.prompt_encoder.load_state_dict(local, strict=strict)
        bitwise = True
        for key, value in mapped.items():
            if key not in expected:
                continue
            actual = self.sam.prompt_encoder.state_dict()[key[len("sam.prompt_encoder.") :]]
            reference = value.to(device=actual.device, dtype=actual.dtype)
            if not torch.equal(actual, reference):
                bitwise = False
                break
        self.prompt_encoder_provenance = {
            "source": str(source),
            "native_pretrained": True,
            "keys_loaded": len(local),
            "strict": bool(strict and not (missing or unexpected or shape_mismatch)),
            "bitwise_equal": bool(bitwise),
        }
        return {
            **self.prompt_encoder_provenance,
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
        }

    def load_sam_prompt_encoder_file(
        self,
        path: str,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        source: str = "sam_checkpoint",
    ) -> Dict[str, Any]:
        raw = torch.load(path, map_location=map_location, weights_only=True)
        return self.load_sam_prompt_encoder(raw, strict=strict, source=source)

    # -- inference --------------------------------------------------------
    def forward(
        self,
        sam_features: torch.Tensor,
        vggt_features: torch.Tensor,
        camera_tokens: torch.Tensor,
        measured_depth: torch.Tensor,
        validity: torch.Tensor,
        alignments: Optional[Union[DepthAlignment, Sequence[DepthAlignment]]] = None,
        output_view: Optional[int] = None,
        include_depth: bool = True,
        return_grid: bool = True,
    ) -> SemanticAdapterOutput:
        """Run automatic semantic segmentation from cached native features.

        Args:
            sam_features: native SAM image features ``[B,N,256,64,64]``.
            vggt_features: native VGGT joint patch features ``[B,N,2048,64,64]``.
            camera_tokens: native VGGT camera tokens ``[B,N,2048]``.
            measured_depth: metric depth ``[B,N,H,W]`` (metres); ``0`` is only
                meaningful together with ``validity == 0``.
            validity: depth validity mask ``[B,N,H,W]``; ``0`` means missing.
            alignments: optional native :class:`DepthAlignment` plan (or one per
                view).  It is validated against the depth shape and rejected if
                it does not match the aspect-preserving native resize/pad.
            output_view: optional view index for target-only ``[B,K,H,W]``
                logits.  This is a view selector, not a target mask.
            include_depth: set ``False`` only for an ablation/control that must
                ignore the depth adapter.  Normal RGB-D inference keeps it on.
            return_grid: include the native 64x64 logits in the output.

        Returns:
            :class:`SemanticAdapterOutput`.
        """
        b, n, c_sam, gh, gw = sam_features.shape
        if c_sam != _SAM_DIM or (gh, gw) != (_SAM_GRID, _SAM_GRID):
            raise ValueError(
                f"sam_features must be [B,N,256,64,64], got {tuple(sam_features.shape)}"
            )
        if vggt_features.shape[:2] != (b, n) or vggt_features.shape[2] != _VGGT_DIM:
            raise ValueError(
                f"vggt_features must be [B,N,2048,64,64], got {tuple(vggt_features.shape)}"
            )
        if vggt_features.shape[-2:] != (_SAM_GRID, _SAM_GRID):
            raise ValueError("vggt_features spatial grid must be 64x64")
        if camera_tokens.shape[:2] != (b, n) or camera_tokens.shape[-1] != _VGGT_DIM:
            raise ValueError(
                f"camera_tokens must be [B,N,2048], got {tuple(camera_tokens.shape)}"
            )
        measured_depth = _as_bn_depth(measured_depth, "measured_depth", b, n)
        validity = _as_bn_depth(validity, "validity", b, n)
        if measured_depth.shape != validity.shape:
            raise ValueError("measured_depth and validity must have identical shapes")
        h, w = measured_depth.shape[-2:]
        if output_view is not None and not (0 <= int(output_view) < n):
            raise ValueError(f"output_view must be in [0,{n}), got {output_view}")

        if alignments is None:
            plans: Union[DepthAlignment, List[DepthAlignment]] = DepthAlignment.from_size(h, w)
        elif isinstance(alignments, DepthAlignment):
            alignments.validate_against_size(h, w)
            plans = alignments
        else:
            plan_list = list(alignments)
            if len(plan_list) != n:
                raise ValueError(f"expected {n} alignments, got {len(plan_list)}")
            for plan in plan_list:
                plan.validate_against_size(h, w)
            plans = plan_list

        dtype = sam_features.dtype
        device = sam_features.device

        # 1) Native SAM-V embedding fusion (actual trained PerPixelMLP).
        fusion_input = torch.cat([sam_features, vggt_features], dim=2).reshape(
            b * n, _SAM_DIM + _VGGT_DIM, gh, gw
        )
        fused = self.embedding_fusion_mlp(fusion_input).reshape(
            b, n, _SAM_DIM, gh, gw
        )  # [B,N,256,64,64]

        # 2) Measured depth + validity adapter into the native fused embedding.
        depth_features: Optional[torch.Tensor] = None
        valid_grid: Optional[torch.Tensor] = None
        if include_depth:
            depth_features, valid_grid = self.depth_adapter(measured_depth, validity, plans)
            depth_features = depth_features.to(dtype=dtype)
            fused = fused + self.depth_scale.to(dtype=dtype) * depth_features

        # 3) Class queries attend to native camera tokens and pooled VGGT
        #    features. No point/mask/crop or ground-truth geometry is involved.
        kv = torch.cat(
            [camera_tokens, vggt_features.mean(dim=(-2, -1))], dim=1
        ).to(dtype=dtype)
        queries = self.semantic_class_query.query.to(dtype=dtype).unsqueeze(0).expand(b, -1, -1)
        class_tokens, _ = self.cross_attention_fusion(queries, kv)  # [B,K,256]

        # 4) Native mask decoder on the wide multi-view canvas.
        wide = fused.permute(0, 2, 3, 1, 4).reshape(
            b, _SAM_DIM, _SAM_GRID, _SAM_GRID * n
        )
        dense_pe = self.sam.prompt_encoder.get_dense_pe().to(dtype=dtype, device=device)
        dense_pe_wide = dense_pe.repeat(b, 1, 1, n)
        class_dense = self.semantic_class_query.class_dense.to(dtype=dtype)
        class_bias = self.semantic_class_query.class_bias.to(dtype=dtype)
        low_res: List[torch.Tensor] = []
        iou: List[torch.Tensor] = []
        # The native decoder's batch semantics are per-image (it repeats the
        # image embedding by the prompt batch), matching the original SamVGGT
        # per-sample loop. Iterate over batch and class, sharing the decoder.
        for b_idx in range(b):
            wide_b = wide[b_idx : b_idx + 1]
            pe_b = dense_pe_wide[b_idx : b_idx + 1]
            for k in range(self.num_classes):
                dense_prompt = class_dense[k].view(1, _SAM_DIM, 1, 1).expand(
                    1, _SAM_DIM, _SAM_GRID, _SAM_GRID * n
                )
                masks, iou_pred = self.sam.mask_decoder(
                    image_embeddings=wide_b,
                    image_pe=pe_b,
                    sparse_prompt_embeddings=class_tokens[b_idx : b_idx + 1, k : k + 1],
                    dense_prompt_embeddings=dense_prompt,
                    multimask_output=False,
                )
                low_res.append(masks[0, 0])
                iou.append(iou_pred[0, 0])
        c = _SAM_GRID * _DECODER_UPSAMPLE
        low_res_canvas = torch.stack(low_res, dim=0).view(
            b, self.num_classes, c, n * c
        )  # [B,K,256,256*N]
        iou_predictions = torch.stack(iou, dim=0).view(b, self.num_classes)

        # 5) Split the wide canvas per view and invert the native crop/pad.
        view_canvas = low_res_canvas.view(b, self.num_classes, c, n, c).permute(
            0, 3, 1, 2, 4
        )  # [B,N,K,C,C]
        logits = invert_view_logits(view_canvas, plans)
        logits = logits + class_bias.view(1, 1, self.num_classes, 1, 1)

        # Native-grid logits: per-view decoder canvas downsampled to 64x64.
        grid_logits = F.interpolate(
            view_canvas.reshape(b * n * self.num_classes, 1, c, c).float(),
            size=(gh, gw),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, n, self.num_classes, gh, gw)
        grid_logits = grid_logits.to(dtype=dtype)
        grid_logits = grid_logits + class_bias.view(1, 1, self.num_classes, 1, 1)
        if not return_grid:
            grid_logits = None  # type: ignore[assignment]

        target_logits = None
        if output_view is not None:
            target_logits = logits[:, int(output_view)]

        return SemanticAdapterOutput(
            grid_logits=grid_logits,
            logits=logits,
            target_logits=target_logits,
            iou_predictions=iou_predictions,
            class_tokens=class_tokens,
            depth_features=depth_features,
            valid_grid=valid_grid,
            alignment=plans,
        )

    # -- controls / helpers ----------------------------------------------
    def trainable_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Named trainable groups, including the retained native groups."""
        encoder_ids = set()
        native = self.native_model
        if native is not None:
            for module in (getattr(native.sam, "image_encoder", None), getattr(native, "vggt", None)):
                if isinstance(module, nn.Module):
                    encoder_ids.update(id(p) for p in module.parameters())
        groups: Dict[str, List[nn.Parameter]] = {
            "embedding_fusion_mlp": [],
            "cross_attention_fusion": [],
            "mask_decoder": [],
            "class_query": [],
            "depth_adapter": [],
            "calibration": [],
            "frozen_encoder": [],
        }
        for name, param in self.named_parameters():
            if id(param) in encoder_ids:
                groups["frozen_encoder"].append(param)
            elif name.startswith("embedding_fusion_mlp."):
                groups["embedding_fusion_mlp"].append(param)
            elif name.startswith("cross_attention_fusion."):
                groups["cross_attention_fusion"].append(param)
            elif name.startswith("sam.mask_decoder."):
                groups["mask_decoder"].append(param)
            elif name.startswith("semantic_class_query.query"):
                groups["class_query"].append(param)
            elif name.startswith("semantic_class_query."):
                groups["calibration"].append(param)
            elif name.startswith("depth_adapter."):
                groups["depth_adapter"].append(param)
            elif name in ("depth_scale",):
                groups["calibration"].append(param)
            elif name.startswith("sam.prompt_encoder."):
                groups["frozen_encoder"].append(param)
        return groups


# ---------------------------------------------------------------------------
# Native construction helpers and frozen-encoder controls
# ---------------------------------------------------------------------------


def build_native_semantic_adapter(
    num_classes: int,
    depth_hidden_dim: int = 64,
    embed_fusion_hidden: int = 768,
    embed_fusion_depth: int = 2,
    embed_fusion_dropout: float = 0.0,
    image_size: int = _SAM_IMAGE_SIZE,
    patch_size: int = _SAM_PATCH,
    prompt_dim: int = _SAM_DIM,
    depth_scale_init: float = 1.0,
    prompt_encoder_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    sam_checkpoint: Optional[str] = None,
    allow_random_prompt_encoder: bool = False,
) -> SamVSemanticAdapter:
    """Build the native semantic architecture without the frozen encoders.

    This uses the real native classes (``PerPixelMLP``, ``CrossAttentionFusion``,
    ``PromptEncoder``, ``MaskDecoder``, ``TwoWayTransformer``) with the native
    SAM ViT-H head dimensions.  It is the standalone cached-feature API; the
    encoders are supplied separately as their cached outputs.

    The native prompt encoder is used for dense positional encoding.  Its
    weights are not in the stage-2 head, so the standalone route must load them
    explicitly with ``prompt_encoder_state_dict`` or ``sam_checkpoint``
    (original SAM checkpoint, ``weights_only=True``).  Passing
    ``allow_random_prompt_encoder=True`` is only for synthetic architecture
    tests and is recorded as non-pretrained provenance; it is never a silent
    default.
    """
    embedding_fusion_mlp = PerPixelMLP(
        c_in=_SAM_DIM + _VGGT_DIM,
        c_hidden=embed_fusion_hidden,
        c_out=_SAM_DIM,
        depth=embed_fusion_depth,
        dropout=embed_fusion_dropout,
    )
    cross_attention_fusion = CrossAttentionFusion(
        d_clip=_SAM_DIM,
        d_spatial_encoder=_VGGT_DIM,
        d_attn=_SAM_DIM,
        num_heads=8,
    )
    adapter = SamVSemanticAdapter(
        embedding_fusion_mlp=embedding_fusion_mlp,
        cross_attention_fusion=cross_attention_fusion,
        mask_decoder=_native_mask_decoder(prompt_dim),
        prompt_encoder=_native_prompt_encoder(image_size, patch_size, prompt_dim),
        num_classes=num_classes,
        depth_hidden_dim=depth_hidden_dim,
        depth_scale_init=depth_scale_init,
    )
    if prompt_encoder_state_dict is not None:
        adapter.load_sam_prompt_encoder(prompt_encoder_state_dict, strict=True)
    elif sam_checkpoint is not None:
        adapter.load_sam_prompt_encoder_file(sam_checkpoint, strict=True)
    elif allow_random_prompt_encoder:
        adapter.prompt_encoder_provenance = {
            "source": "synthetic_random",
            "native_pretrained": False,
            "keys_loaded": 0,
            "strict": False,
        }
    else:
        raise ValueError(
            "the standalone cached-feature route requires the original SAM "
            "prompt-encoder state (prompt_encoder_state_dict or sam_checkpoint), "
            "or use SamVSemanticAdapter.from_sam_vggt with a SAM-loaded model; "
            "pass allow_random_prompt_encoder=True only for synthetic tests"
        )
    return adapter


def freeze_encoder_parameters_(model: ModuleLike) -> Dict[str, int]:
    """Freeze the SAM ViT-H and VGGT-1B encoders in place.

    Only encoder parameters are frozen; the native fusion MLP, cross-attention
    fusion and mask decoder stay trainable.
    """
    report: Dict[str, int] = {}
    sam = getattr(model, "sam", None)
    image_encoder = getattr(sam, "image_encoder", None)
    if isinstance(image_encoder, nn.Module):
        count = 0
        for param in image_encoder.parameters():
            param.requires_grad_(False)
            count += param.numel()
        image_encoder.eval()
        report["sam_image_encoder_parameters"] = count
    vggt = getattr(model, "vggt", None)
    if isinstance(vggt, nn.Module):
        count = 0
        for param in vggt.parameters():
            param.requires_grad_(False)
            count += param.numel()
        vggt.eval()
        report["vggt_parameters"] = count
    return report


def offload_frozen_encoders_(model: ModuleLike, device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Move frozen encoders off the training device for cached decoder training.

    The fusion/decoder head is left where it is.  This is what makes cached
    decoder training independent of both encoders' residency.
    """
    frozen = freeze_encoder_parameters_(model)
    moved: Dict[str, str] = {}
    sam = getattr(model, "sam", None)
    image_encoder = getattr(sam, "image_encoder", None)
    if isinstance(image_encoder, nn.Module):
        image_encoder.to(device)
        moved["sam_image_encoder"] = str(device)
    vggt = getattr(model, "vggt", None)
    if isinstance(vggt, nn.Module):
        vggt.to(device)
        moved["vggt"] = str(device)
    return {"frozen_parameters": frozen, "moved": moved}


def competition_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the class axis for per-view or target-only logits."""
    if logits.dim() == 5:
        return torch.softmax(logits, dim=2)
    if logits.dim() == 4:
        return torch.softmax(logits, dim=1)
    raise ValueError("expected [B,N,K,H,W] or [B,K,H,W] logits")


def predict_views(output: SemanticAdapterOutput) -> torch.Tensor:
    """Per-view class indices ``[B,N,H,W]`` from competition logits."""
    if output.logits is None:
        raise ValueError("logits are not available")
    return output.logits.argmax(dim=2)


def predict_target(output: SemanticAdapterOutput) -> torch.Tensor:
    """Target-only class indices ``[B,H,W]`` from competition logits."""
    if output.target_logits is None:
        raise ValueError("target_logits require output_view")
    return output.target_logits.argmax(dim=1)
