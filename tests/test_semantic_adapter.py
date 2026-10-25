"""CPU unit tests for the native class-query semantic adapter.

These are synthetic feature-grid tests only. They prove shapes, gradients and
geometry invariants on CPU; they are not GPU performance, training, or accuracy
evidence.
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from model.feature_cache import (
    FeatureCacheStore,
    NativeFeaturePayload,
    make_feature_cache_key,
)
from model.semantic_adapter import (
    DepthAlignment,
    SamVSemanticAdapter,
    align_depth_to_grid,
    build_native_semantic_adapter,
    competition_probabilities,
    freeze_encoder_parameters_,
    invert_view_logits,
    offload_frozen_encoders_,
)
from model.sam_vggt_model import CrossAttentionFusion, PerPixelMLP, SamVGGT


def _synthetic_adapter(num_classes=2):
    # Synthetic tests explicitly opt out of the original-SAM prompt-encoder
    # provenance requirement; they make no native-pretrained claim.
    return build_native_semantic_adapter(
        num_classes=num_classes,
        depth_hidden_dim=8,
        allow_random_prompt_encoder=True,
    )


def _features(batch=1, views=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sam = torch.randn(batch, views, 256, 64, 64, generator=generator)
    vggt = torch.randn(batch, views, 2048, 64, 64, generator=generator)
    camera = torch.randn(batch, views, 2048, generator=generator)
    return sam, vggt, camera


def _depth(batch=1, views=1, height=96, width=128, value=0.5, valid=1.0):
    depth = torch.full((batch, views, height, width), float(value))
    validity = torch.full((batch, views, height, width), float(valid))
    return depth, validity


def _run(adapter, batch=1, views=1, height=96, width=128, **kwargs):
    sam, vggt, camera = _features(batch, views)
    depth, validity = _depth(batch, views, height, width)
    return adapter(
        sam_features=sam,
        vggt_features=vggt,
        camera_tokens=camera,
        measured_depth=depth,
        validity=validity,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Native architecture and checkpoint
# ---------------------------------------------------------------------------


def test_native_module_types_retained():
    adapter = _synthetic_adapter()
    assert isinstance(adapter.embedding_fusion_mlp, PerPixelMLP)
    assert isinstance(adapter.cross_attention_fusion, CrossAttentionFusion)
    assert isinstance(adapter.sam.mask_decoder, MaskDecoder)
    assert isinstance(adapter.sam.prompt_encoder, PromptEncoder)
    assert isinstance(adapter.sam.mask_decoder.transformer, TwoWayTransformer)
    # The native prompt encoder is frozen positional infrastructure.
    assert all(not p.requires_grad for p in adapter.sam.prompt_encoder.parameters())


def test_stage2_checkpoint_strict_load():
    path = os.environ.get("SAMV_STAGE2_CHECKPOINT")
    if not path:
        pytest.skip("SAMV_STAGE2_CHECKPOINT not set")
    adapter = _synthetic_adapter()
    report = adapter.load_native_stage2_file(path, strict=True)
    assert report["strict_native"] is True
    assert report["missing_native"] == []
    assert report["unexpected_native"] == []
    assert report["native_loaded"] == report["native_expected"] > 0
    assert report["new_trainable_keys"]
    assert all(
        key.startswith(("semantic_class_query.", "depth_adapter.", "depth_scale"))
        for key in report["new_trainable_keys"]
    )
    assert all(key.startswith("sam.prompt_encoder.") for key in report["non_stage2_native_keys"])


def test_standalone_requires_prompt_encoder_provenance():
    with pytest.raises(ValueError):
        build_native_semantic_adapter(num_classes=2, depth_hidden_dim=8)
    adapter = _synthetic_adapter()
    assert adapter.prompt_encoder_provenance["source"] == "synthetic_random"
    assert adapter.prompt_encoder_provenance["native_pretrained"] is False


def test_sam_prompt_encoder_bitwise_loaded():
    sam_path = os.environ.get("SAMV_SAM_CHECKPOINT")
    if not sam_path:
        pytest.skip("SAMV_SAM_CHECKPOINT not set")
    adapter = build_native_semantic_adapter(
        num_classes=2, depth_hidden_dim=8, sam_checkpoint=sam_path
    )
    provenance = adapter.prompt_encoder_provenance
    assert provenance["source"] == "sam_checkpoint"
    assert provenance["native_pretrained"] is True
    assert provenance["keys_loaded"] == 17
    assert provenance["bitwise_equal"] is True
    assert provenance["strict"] is True

    raw = torch.load(sam_path, map_location="cpu", weights_only=True)
    key = "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    assert torch.equal(
        adapter.sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix, raw[key]
    )

    stage2_path = os.environ.get("SAMV_STAGE2_CHECKPOINT")
    if stage2_path:
        report = adapter.load_native_stage2_file(stage2_path, strict=True)
        assert report["prompt_encoder_provenance"]["source"] == "sam_checkpoint"
        assert report["prompt_encoder_provenance"]["native_pretrained"] is True
        assert report["non_stage2_native_keys"]
        assert all(
            key.startswith("sam.prompt_encoder.") for key in report["non_stage2_native_keys"]
        )


def test_batch_two_uses_shared_prompt_encoder_pe():
    adapter = _synthetic_adapter(num_classes=1)
    adapter.eval()
    with torch.no_grad():
        out = _run(adapter, batch=2, views=1, height=96, width=128)
    assert out.logits.shape == (2, 1, 1, 96, 128)
    assert out.grid_logits.shape == (2, 1, 1, 64, 64)
    assert torch.isfinite(out.logits).all()


def test_from_sam_vggt_shares_modules_and_freezes_encoders():
    native = _synthetic_adapter()
    fake = SimpleNamespace(
        embedding_fusion_mlp=native.embedding_fusion_mlp,
        cross_attention_fusion=native.cross_attention_fusion,
        sam=SimpleNamespace(
            mask_decoder=native.sam.mask_decoder,
            prompt_encoder=native.sam.prompt_encoder,
            image_encoder=torch.nn.Linear(4, 4),
        ),
        vggt=torch.nn.Linear(4, 4),
    )
    adapter = SamVSemanticAdapter.from_sam_vggt(fake, num_classes=3, depth_hidden_dim=8)
    assert adapter.embedding_fusion_mlp is native.embedding_fusion_mlp
    assert adapter.cross_attention_fusion is native.cross_attention_fusion
    assert adapter.sam.mask_decoder is native.sam.mask_decoder
    assert adapter.num_classes == 3
    assert all(not p.requires_grad for p in fake.sam.image_encoder.parameters())
    assert all(not p.requires_grad for p in fake.vggt.parameters())
    # Native fusion/decoder remain trainable.
    assert all(p.requires_grad for p in fake.embedding_fusion_mlp.parameters())
    assert all(p.requires_grad for p in fake.sam.mask_decoder.parameters())


# ---------------------------------------------------------------------------
# Forward contract, depth semantics, geometry
# ---------------------------------------------------------------------------


def test_forward_output_contract_and_depth_semantics():
    adapter = _synthetic_adapter()
    adapter.eval()
    sam, vggt, camera = _features(1, 1)
    depth_a = torch.full((1, 1, 96, 128), 0.4)
    depth_b = torch.full((1, 1, 96, 128), 1.8)
    valid = torch.ones_like(depth_a)
    with torch.no_grad():
        out_a = adapter(sam, vggt, camera, depth_a, valid)
        out_b = adapter(sam, vggt, camera, depth_b, valid)
        out_missing = adapter(
            sam, vggt, camera, torch.zeros_like(depth_a), torch.zeros_like(valid)
        )
        out_no_depth = adapter(sam, vggt, camera, depth_a, valid, include_depth=False)
        out_measured_zero = adapter(
            sam, vggt, camera, torch.zeros_like(depth_a), valid
        )
        out_target = adapter(sam, vggt, camera, depth_a, valid, output_view=0)

    assert out_a.grid_logits.shape == (1, 1, 2, 64, 64)
    assert out_a.logits.shape == (1, 1, 2, 96, 128)
    assert out_target.target_logits.shape == (1, 2, 96, 128)
    assert torch.isfinite(out_a.logits).all()
    assert torch.isfinite(out_a.grid_logits).all()
    probs = competition_probabilities(out_a.logits)
    assert torch.allclose(probs.sum(dim=2), torch.ones_like(probs[:, :, 0]))
    assert torch.allclose(out_target.target_logits, out_target.logits[:, 0])

    # Measured depth changes the logits when the RGB features are fixed.
    assert not torch.allclose(out_a.logits, out_b.logits)
    # Missing depth contributes exactly nothing; it cannot force a class.
    assert torch.allclose(out_missing.logits, out_no_depth.logits)
    # Measured 0 m with validity=1 is not the same as missing depth.
    assert not torch.allclose(out_missing.logits, out_measured_zero.logits)


def test_per_view_output_for_multiple_views():
    adapter = _synthetic_adapter()
    adapter.eval()
    with torch.no_grad():
        out = _run(adapter, batch=1, views=2, height=120, width=160)
    assert out.logits.shape == (1, 2, 2, 120, 160)
    assert out.grid_logits.shape == (1, 2, 2, 64, 64)
    assert out.iou_predictions.shape == (1, 2)


def test_rectangular_crop_pad_inverse_has_no_align_shift():
    # Native rectangular preprocessing: longest side -> 1024, pad right/bottom.
    plan = DepthAlignment.from_size(480, 768)
    assert (plan.resized_h, plan.resized_w) == (640, 1024)
    assert plan.decoder_canvas_size == 256

    landmarks = [(240.0, 384.0), (40.0, 700.0)]
    views = []
    for y, x in landmarks:
        # Continuous 256-canvas coordinate of the original pixel centre.
        cy = (y + 0.5) * plan.decoder_valid_h / plan.original_h
        cx = (x + 0.5) * plan.decoder_valid_w / plan.original_w
        yy = torch.arange(256, dtype=torch.float32).view(1, 1, 256, 1) + 0.5
        xx = torch.arange(256, dtype=torch.float32).view(1, 1, 1, 256) + 0.5
        blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.0**2))
        views.append(blob)
    canvas = torch.stack(views, dim=0).view(2, 1, 1, 256, 256)

    restored = invert_view_logits(canvas, plan)
    assert restored.shape == (2, 1, 1, 480, 768)
    for index, (y, x) in enumerate(landmarks):
        flat = int(restored[index, 0, 0].argmax())
        got_y, got_x = divmod(flat, 768)
        assert abs(got_y - y) <= 1, (got_y, y)
        assert abs(got_x - x) <= 1, (got_x, x)


def test_depth_alignment_rejects_stretch_and_tracks_missing():
    plan = DepthAlignment.from_size(96, 128)
    bad = DepthAlignment(
        original_h=96,
        original_w=128,
        resized_h=96,
        resized_w=128,
    )
    with pytest.raises(ValueError):
        bad.validate_against_size(96, 128)

    depth = torch.full((1, 1, 96, 128), 2.0)
    valid = torch.zeros_like(depth)
    depth_grid, valid_grid = align_depth_to_grid(depth, valid, plan)
    assert depth_grid.shape == valid_grid.shape == (1, 1, 64, 64)
    assert torch.count_nonzero(valid_grid) == 0
    assert torch.count_nonzero(depth_grid) == 0

    valid[:] = 1
    depth_grid, valid_grid = align_depth_to_grid(depth, valid, plan)
    assert valid_grid.min() == 0  # padded region
    assert depth_grid.max() <= 2.0 / 3.0 + 1e-5


# ---------------------------------------------------------------------------
# Gradients, optimiser update, encoder freeze
# ---------------------------------------------------------------------------


def test_native_groups_gradients_and_optimizer_step():
    adapter = _synthetic_adapter()
    sam, vggt, camera = _features(1, 1)
    depth, validity = _depth(1, 1, 96, 128)
    out = adapter(sam, vggt, camera, depth, validity)
    loss = out.logits.mean() + out.grid_logits.mean() + out.iou_predictions.mean()
    loss.backward()

    groups = adapter.trainable_parameter_groups()
    for name in (
        "embedding_fusion_mlp",
        "cross_attention_fusion",
        "mask_decoder",
        "depth_adapter",
        "class_query",
        "calibration",
    ):
        params = groups[name]
        assert params, name
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in params
        ), f"no finite non-zero gradient for {name}"

    trainable = [p for name, params in groups.items() if name != "frozen_encoder" for p in params]
    before = {id(p): p.detach().clone() for p in trainable}
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    optimizer.step()
    changed = {name: False for name in groups if name != "frozen_encoder"}
    for name, params in groups.items():
        if name == "frozen_encoder":
            continue
        for p in params:
            if not torch.allclose(p.detach(), before[id(p)]):
                changed[name] = True
    assert all(changed.values()), changed


def test_frozen_encoder_offload_keeps_head_trainable():
    fake = SimpleNamespace(
        sam=SimpleNamespace(image_encoder=torch.nn.Linear(4, 4)),
        vggt=torch.nn.Linear(4, 4),
        embedding_fusion_mlp=torch.nn.Linear(4, 4),
    )
    report = offload_frozen_encoders_(fake, device="cpu")
    assert report["frozen_parameters"]["sam_image_encoder_parameters"] == 20
    assert all(not p.requires_grad for p in fake.sam.image_encoder.parameters())
    assert all(not p.requires_grad for p in fake.vggt.parameters())
    assert all(p.requires_grad for p in fake.embedding_fusion_mlp.parameters())
    assert fake.sam.image_encoder.weight.device.type == "cpu"


# ---------------------------------------------------------------------------
# API constraints and original prompt route
# ---------------------------------------------------------------------------


def test_inference_api_is_gt_free_and_prompt_route_unchanged():
    signature = inspect.signature(SamVSemanticAdapter.forward)
    banned = {
        "gt",
        "label",
        "labels",
        "mask",
        "masks",
        "point",
        "points",
        "box",
        "boxes",
        "crop",
        "crops",
        "target_mask",
    }
    for name in signature.parameters:
        tokens = set(name.lower().replace(".", "_").split("_"))
        assert not (tokens & banned), name

    prompt_signature = inspect.signature(SamVGGT.forward)
    for name in (
        "point_coords_list",
        "point_labels_list",
        "point_frame_indices_list",
        "multimask_output",
    ):
        assert name in prompt_signature.parameters


# ---------------------------------------------------------------------------
# Feature-cache identity
# ---------------------------------------------------------------------------


def _key_kwargs(stage="n2"):
    return dict(
        stage=stage,
        view_ids=["view-a", "view-b"],
        rgb_sha256=["a" * 64, "b" * 64],
        encoder_source_sha256="c" * 64,
        weights_sha256="d" * 64,
        preprocess="sam1024-longest-pad-rb/vggt896-center-pad",
        dtype="torch.bfloat16",
    )


def test_n1_cache_key_is_independent_of_n2():
    n2 = make_feature_cache_key(**_key_kwargs("n2"))
    n1 = make_feature_cache_key(
        **{
            **_key_kwargs("n1"),
            "view_ids": ["view-a"],
            "rgb_sha256": ["a" * 64],
            "n_views": 1,
        }
    )
    assert n1.digest != n2.digest
    with pytest.raises(ValueError):
        make_feature_cache_key(
            **{
                **_key_kwargs("n1"),
                "n_views": 2,
            }
        )
    reordered = make_feature_cache_key(
        **{
            **_key_kwargs("n2"),
            "view_ids": ["view-b", "view-a"],
            "rgb_sha256": ["b" * 64, "a" * 64],
        }
    )
    assert reordered.digest != n2.digest


def test_feature_cache_round_trip(tmp_path):
    key = make_feature_cache_key(
        **{
            **_key_kwargs("n2"),
            "view_ids": ["view-a"],
            "rgb_sha256": ["a" * 64],
            "n_views": 1,
        }
    )
    payload = NativeFeaturePayload(
        key=key,
        sam_features=torch.zeros(1, 1, 256, 64, 64),
        vggt_features=torch.zeros(1, 1, 2048, 64, 64),
        camera_tokens=torch.zeros(1, 1, 2048),
        metadata={"note": "synthetic"},
    )
    store = FeatureCacheStore(tmp_path)
    path = store.save(payload)
    loaded = store.load(key)
    assert path.exists()
    assert torch.equal(loaded.sam_features, payload.sam_features)
    assert loaded.metadata["note"] == "synthetic"
