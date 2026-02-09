from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionalGroupVocab:
    smiles_to_id: dict[str, int]
    id_to_smiles: dict[int, str]

    @staticmethod
    def load(path: str | Path) -> "FunctionalGroupVocab":
        path = Path(path)
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Functional group vocab must be a dict: {fg_smiles: id}")
        smiles_to_id: dict[str, int] = {}
        id_to_smiles: dict[int, str] = {}
        for smiles, idx in raw.items():
            if not isinstance(smiles, str) or not isinstance(idx, int):
                raise ValueError("Functional group vocab keys must be str and values must be int")
            smiles_to_id[smiles] = idx
            id_to_smiles[idx] = smiles
        if not smiles_to_id:
            raise ValueError(f"Empty functional group vocab: {path}")
        return FunctionalGroupVocab(smiles_to_id=smiles_to_id, id_to_smiles=id_to_smiles)

    @property
    def size(self) -> int:
        return max(self.id_to_smiles) + 1

