"""Native feature-cache keys and frozen-encoder offload helpers.

The cache is intentionally explicit about what produced a feature tensor:
ordered view identities, RGB content hashes, encoder source hash, encoder weight
hash, preprocessing/dtype, and the number of views.  N1 and N2 are distinct
cache entries: an N2 feature is never a valid N1 feature (that would be a
context drop), and an N1 feature is never padded into an N2 entry.

Only encoder outputs are cached.  Depth and ground truth are *not* encoder
inputs and are not part of the encoder cache key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "FeatureCacheKey",
    "FeatureCacheStore",
    "make_feature_cache_key",
    "NativeFeaturePayload",
    "save_native_features",
    "load_native_features",
]


def _sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class FeatureCacheKey:
    """Immutable identity of one native encoder feature extraction.

    ``stage`` is ``"n1"`` (one independently encoded view) or ``"n2"`` (joint
    multi-view context).  The stage and view count are part of the digest.
    """

    stage: str
    view_ids: Tuple[str, ...]
    rgb_sha256: Tuple[str, ...]
    encoder_source_sha256: str
    weights_sha256: str
    preprocess: str
    dtype: str
    n_views: int
    sam_image_size: int = 1024
    vggt_image_size: int = 896
    extra: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.stage not in ("n1", "n2"):
            raise ValueError("stage must be 'n1' or 'n2'")
        if len(self.view_ids) != len(self.rgb_sha256):
            raise ValueError("view_ids and rgb_sha256 must have the same length")
        if self.n_views <= 0:
            raise ValueError("n_views must be positive")
        if len(self.view_ids) != self.n_views:
            raise ValueError("view_ids length must equal n_views")
        if self.stage == "n1" and self.n_views != 1:
            raise ValueError("an N1 cache entry must contain exactly one independent view")
        if self.stage == "n2" and self.n_views < 1:
            raise ValueError("an N2 cache entry needs at least one view")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "view_ids": list(self.view_ids),
            "rgb_sha256": list(self.rgb_sha256),
            "encoder_source_sha256": self.encoder_source_sha256,
            "weights_sha256": self.weights_sha256,
            "preprocess": self.preprocess,
            "dtype": self.dtype,
            "n_views": self.n_views,
            "sam_image_size": self.sam_image_size,
            "vggt_image_size": self.vggt_image_size,
            "extra": [list(pair) for pair in self.extra],
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def filename(self) -> str:
        return f"{self.stage}-{self.digest}.pt"


def make_feature_cache_key(
    *,
    stage: str,
    view_ids: Sequence[str],
    rgb_sha256: Sequence[str],
    encoder_source_sha256: str,
    weights_sha256: str,
    preprocess: str,
    dtype: str,
    n_views: Optional[int] = None,
    sam_image_size: int = 1024,
    vggt_image_size: int = 896,
    extra: Optional[Iterable[Tuple[str, str]]] = None,
) -> FeatureCacheKey:
    """Create and validate an encoder-cache key.

    N1 keys are single-view by construction, so a feature extracted with joint
    N2 context can never be looked up as an independent N1 feature.
    """
    ids = tuple(str(item) for item in view_ids)
    hashes = tuple(str(item) for item in rgb_sha256)
    if n_views is None:
        n_views = len(ids)
    return FeatureCacheKey(
        stage=str(stage),
        view_ids=ids,
        rgb_sha256=hashes,
        encoder_source_sha256=str(encoder_source_sha256),
        weights_sha256=str(weights_sha256),
        preprocess=str(preprocess),
        dtype=str(dtype),
        n_views=int(n_views),
        sam_image_size=int(sam_image_size),
        vggt_image_size=int(vggt_image_size),
        extra=tuple(extra or ()),
    )


@dataclass
class NativeFeaturePayload:
    key: FeatureCacheKey
    sam_features: torch.Tensor
    vggt_features: torch.Tensor
    camera_tokens: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = self.key.n_views
        if self.sam_features.dim() != 5 or self.sam_features.shape[1] != n:
            raise ValueError("sam_features must be [B,N,256,64,64] with N matching the key")
        if tuple(self.sam_features.shape[2:]) != (256, 64, 64):
            raise ValueError("sam_features must be [B,N,256,64,64]")
        if self.vggt_features.dim() != 5 or self.vggt_features.shape[1] != n:
            raise ValueError("vggt_features must be [B,N,2048,64,64] with N matching the key")
        if tuple(self.vggt_features.shape[2:]) != (2048, 64, 64):
            raise ValueError("vggt_features must be [B,N,2048,64,64]")
        if tuple(self.camera_tokens.shape[:2])[1] != n or self.camera_tokens.shape[-1] != 2048:
            raise ValueError("camera_tokens must be [B,N,2048]")


class FeatureCacheStore:
    """Small filesystem store for native encoder features."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def path_for(self, key: FeatureCacheKey) -> Path:
        return self.root / key.stage / key.filename()

    def save(self, payload: NativeFeaturePayload) -> Path:
        payload.validate()
        path = self.path_for(payload.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "digest": payload.key.digest,
                "key": json.dumps(payload.key.to_dict(), sort_keys=True),
                "sam_features": payload.sam_features,
                "vggt_features": payload.vggt_features,
                "camera_tokens": payload.camera_tokens,
                "metadata": payload.metadata,
            },
            path,
        )
        return path

    def load(self, key: FeatureCacheKey) -> NativeFeaturePayload:
        path = self.path_for(key)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw.get("digest") != key.digest:
            raise ValueError("feature cache digest mismatch")
        return NativeFeaturePayload(
            key=key,
            sam_features=raw["sam_features"],
            vggt_features=raw["vggt_features"],
            camera_tokens=raw["camera_tokens"],
            metadata=dict(raw.get("metadata", {})),
        )


def save_native_features(path: Union[str, Path], payload: NativeFeaturePayload) -> Path:
    path = Path(path)
    payload.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "digest": payload.key.digest,
            "key": json.dumps(payload.key.to_dict(), sort_keys=True),
            "sam_features": payload.sam_features,
            "vggt_features": payload.vggt_features,
            "camera_tokens": payload.camera_tokens,
            "metadata": payload.metadata,
        },
        path,
    )
    return path


def load_native_features(
    path: Union[str, Path], key: FeatureCacheKey
) -> NativeFeaturePayload:
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if raw.get("digest") != key.digest:
        raise ValueError("feature cache digest mismatch")
    return NativeFeaturePayload(
        key=key,
        sam_features=raw["sam_features"],
        vggt_features=raw["vggt_features"],
        camera_tokens=raw["camera_tokens"],
        metadata=dict(raw.get("metadata", {})),
    )
