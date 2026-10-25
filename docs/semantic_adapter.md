# Native class-query semantic adapter

This document describes the generic, prompt-free semantic head added in
`model/semantic_adapter.py`, with cache helpers in `model/feature_cache.py`.

The adapter is an *additive* route. It does not change the original
`SamVGGT.forward` point/box/mask prompt path, and it does not replace the native
encoders, fusion modules, or mask decoder.

## Retained native modules

`SamVSemanticAdapter` is built from the real native modules:

| role | module |
|---|---|
| SAM/VGGT embedding fusion | `model.sam_vggt_model.PerPixelMLP` (`embedding_fusion_mlp`) |
| class-query cross-attention | `model.sam_vggt_model.CrossAttentionFusion` |
| native SAM decoder | `segment_anything.modeling.MaskDecoder` with `TwoWayTransformer` |
| native dense positional encoding | `segment_anything.modeling.PromptEncoder` |

`SamVSemanticAdapter.from_sam_vggt(sam_vggt, num_classes=K)` shares those
instances with an existing `SamVGGT` by identity and freezes the SAM ViT-H and
VGGT-1B encoders. `build_native_semantic_adapter(K)` builds the same native
head architecture for the cached-feature API, without instantiating the
encoders.

The native prompt encoder contributes the dense positional encoding but is not
part of the stage-2 head. The standalone route therefore requires the original
SAM prompt-encoder state, either as `prompt_encoder_state_dict` or as a
`sam_checkpoint` loaded with `weights_only=True`; it raises rather than silently
using a random positional encoding. `allow_random_prompt_encoder=True` exists
only for synthetic architecture tests and is recorded as
`prompt_encoder_provenance.source == "synthetic_random"`. The `from_sam_vggt`
route records `source == "from_sam_vggt"`, because `SamVGGT` loads the original
SAM checkpoint in its constructor.

No unrelated ConvNet, ResNet, or substitute mask decoder is used.

## Primary cached-feature API

```python
from model.semantic_adapter import build_native_semantic_adapter

adapter = build_native_semantic_adapter(
    num_classes=K,
    sam_checkpoint="sam_vit_h_4b8939.pth",  # original SAM checkpoint
)
out = adapter(
    sam_features=sam,          # [B, N, 256, 64, 64] native SAM ViT-H features
    vggt_features=vggt,        # [B, N, 2048, 64, 64] native VGGT-1B features
    camera_tokens=camera,      # [B, N, 2048]        native VGGT camera tokens
    measured_depth=depth,      # [B, N, H, W]        metric depth (metres)
    validity=validity,         # [B, N, H, W]        0 = missing
    output_view=0,             # optional; target-only logits
)
out.grid_logits   # [B, N, K, 64, 64]
out.logits        # [B, N, K, H, W]
out.target_logits # [B, K, H, W] when output_view is set
out.iou_predictions  # [B, K]
```

The native SAM preprocessing uses a 1024 canvas and the native VGGT
preprocessing uses 896; both produce the same 64x64 feature grid. The cached
features must come from the actual native encoders — `encode_sam_batched`,
`encode_vggt_batched` in `SamVGGT` — not from a different backbone.

The forward path is:

1. `embedding_fusion_mlp(concat(sam, vggt))` -> native fused embeddings.
2. The measured depth/validity adapter adds a 256-channel feature to those
   embeddings.
3. `K` learned class queries attend to the native camera tokens and pooled VGGT
   features through the actual `cross_attention_fusion`.
4. The native `sam.mask_decoder` is called once per class with that class query
   as the sparse prompt (a shared decoder, not a new decoder).
5. Logits are inverted from the wide decoder canvas to each original view.

No ground-truth points, masks, boxes, crops or labels are inputs to inference.
`output_view` selects a view, not a target mask. Ground truth is only for the
loss/evaluation outside this module.

## RGB-D depth semantics

Depth is mandatory at inference. The depth adapter is a small trainable
`2 -> hidden -> 256` convolution:

```text
input channels = (clamp(depth, 0, 3) / 3) * validity, validity
```

* `validity == 0` means missing. Missing is never interpreted as measured
  zero depth and never forces a background class. An entirely invalid view
  produces exactly zero depth features.
* `depth == 0, validity == 1` is a legitimate measured zero and is distinct
  from missing.
* Depth and validity are resampled with the *same* aspect-preserving resize/pad
  as the native RGB preprocessing. A plan that would stretch the frame is
  rejected by `DepthAlignment.validate_against_size`.
* Resampling is a validity-weighted area average, so the zero padding and
  missing pixels do not bleed into valid cells.

## Geometry: inverse to original H/W

`DepthAlignment.from_size(H, W)` mirrors
`ResizeLongestSide.get_preprocess_shape`: longest side to 1024, round half up,
pad right/bottom. Each view's decoder output is a `256 x 256` square canvas.

`invert_view_logits` crops the valid (non-padded) rectangle and resamples it to
the original `H, W` using an `align_corners=False` pixel-centre affine grid.
This preserves rectangular inputs (for example 768x480) without stretching the
pad and without a half-pixel alignment shift. Only the valid crop is used; the
padded square region is not stretched into the output.

The class axis mapping is documented and fixed: channel `k` of
`grid_logits`/`logits` corresponds to class index `k`; `target_logits` keeps the
same order. Use `competition_probabilities` (softmax over the class axis) and
`predict_views`/`predict_target` for the competition output.

## Checkpoint loading and new parameters

`load_native_stage2(...)` strictly matches the three native stage-2 prefixes:

```text
embedding_fusion_mlp.*
cross_attention_fusion.*
sam.mask_decoder.*
```

It raises if the native key set is not exact. New trainable parameters are
reported separately and are never treated as released stage-2 tensors:

```text
semantic_class_query.query
semantic_class_query.class_dense
semantic_class_query.class_bias
depth_adapter.*
depth_scale
```

The native frozen prompt encoder's positional buffer is reported separately as
non-stage-2 native state. The original point/box/mask prompt route remains
available on `SamVGGT.forward`.

## Feature cache and frozen encoders

`model/feature_cache.py` defines an explicit cache key:

```text
stage (n1 | n2), ordered view ids, RGB SHA-256s, encoder source SHA-256,
encoder weights SHA-256, preprocessing signature, dtype, N, SAM/VGGT sizes
```

N1 entries contain exactly one independently encoded view. N2 entries contain
the joint ordered views. An N1 digest can never equal an N2 digest, and an N2
feature is never reused as an N1 feature (that would be a context drop).

For decoder training from cache:

```python
from model.semantic_adapter import offload_frozen_encoders_

offload_frozen_encoders_(sam_vggt, device="cpu")
```

This freezes and moves both encoders off the training device; the cached
feature adapter and its native decoder stay trainable. Depth and ground truth
are not part of the encoder cache key.

## Scope and limitations

* The unit tests use small synthetic feature grids on CPU. They verify shapes,
  gradient flow, optimiser updates, depth semantics, geometry, and cache
  identity. They are **not** GPU performance, full-training, or accuracy
  evidence.
* GPU feasibility (native 1024/896 encoders, memory, decoder backward) must be
  measured separately on a granted device.
* The depth adapter is aligned to the native SAM resize/pad, which is the
  preprocessing that produces the 64x64 SAM feature grid. The native fusion
  already assumes the SAM and VGGT grids are spatially compatible; this module
  does not change that assumption.
* The prompt-free route is a new semantic head. It does not replace the
  released promptable instance route.
