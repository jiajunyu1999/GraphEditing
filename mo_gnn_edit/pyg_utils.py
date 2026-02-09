from __future__ import annotations


def require_torch_geometric() -> None:
    try:
        import torch_geometric  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: torch-geometric. Install it along with PyTorch for your CUDA/CPU setup."
        ) from exc

