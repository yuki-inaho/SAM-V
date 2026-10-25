# Generic uv environment (SAM-V native)

This repository can be installed with [uv](https://docs.astral.sh/uv/) instead
of the legacy pip requirements. The environment is CPU-importable out of the box;
a successful `uv sync` and import is **not** proof that the CUDA kernels run on a
particular GPU.

## Requirements

- Python 3.10 (`.python-version`)
- Linux with an NVIDIA driver suitable for CUDA 12.8 for GPU execution
- The native packages come from the pinned submodules, not PyPI:

```bash
git submodule update --init submodules/sam-hq submodules/vggt
```

## Install

```bash
uv sync
```

This resolves the locked environment in `uv.lock`:

| package | pin | why |
|---|---|---|
| `torch` | `2.8.0+cu128` | CUDA 12.8 build with Blackwell (sm_120) kernels; the historical 2.3.1/cu121 pin has none |
| `torchvision` | `0.23.0+cu128` | matches torch 2.8.0 |
| `timm` | `1.0.22` | 1.0.23 breaks the patched sam-hq fork at import (`RotaryEmbedding` moved out of `timm.layers.attention_pool2d` while the compatibility shim still imports it) |

`torch` and `torchvision` are fetched from the explicit
`https://download.pytorch.org/whl/cu128` index declared in `pyproject.toml`.
The remaining dependencies are ordinary PyPI ranges. `numpy` is held below 2,
matching the VGGT submodule.

## Run

`PYTHONPATH` must include the repo root and both submodules (the VGGT submodule
ships its own `training` package that would otherwise shadow this one):

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
uv run python tools/<script>.py
```

## Validate the CPU import only

```bash
uv run python -c "import torch, torchvision, timm, segment_anything, vggt; print(torch.__version__)"
```

This confirms import compatibility and CPU runtime only. Validate the actual
CUDA/sm_120 path separately, on a granted GPU, with an explicit bounded probe.

## Base weights

Weights are large and are not downloaded by `uv sync`. The native model expects:

- SAM ViT-H: the original `sam_vit_h_4b8939.pth` from the official
  segment-anything release. Converted transformers or HQ-SAM checkpoints are not
  drop-in substitutes for the native `segment_anything` state dict.
- VGGT-1B: `model.pt` from `facebook/VGGT-1B`, loaded by the VGGT submodule.
- SAM-V stage 2: `sam_v_stage2.pth` from `Frank-Gong123/SAM-V`; it contains only
  the three trained module prefixes and is loaded together with the two frozen
  base checkpoints above.

Keep weights outside the repository and pass their paths explicitly.
