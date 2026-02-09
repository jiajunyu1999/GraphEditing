from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root_dir: Path

    @staticmethod
    def load(path: str | Path) -> "Config":
        path = Path(path)
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: PyYAML. Install it (e.g. `pip install pyyaml`)."
            ) from exc

        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a dict, got: {type(raw)}")
        return Config(raw=raw, root_dir=path.parent.resolve())

    def resolve_path(self, maybe_path: str) -> Path:
        p = Path(maybe_path)
        return p if p.is_absolute() else (self.root_dir / p)

