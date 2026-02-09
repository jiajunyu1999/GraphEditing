from __future__ import annotations

from typing import Callable
from .drd2_scorer import get_score

def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: RDKit. Install it (e.g. `conda install -c conda-forge rdkit`)."
        ) from exc


def build_property_fns(property_names: list[str]) -> dict[str, Callable[["Chem.Mol"], float]]:
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors

    fns: dict[str, Callable[[Chem.Mol], float]] = {
        "qed": lambda m: float(QED.qed(m)),
        "logp": lambda m: float(Descriptors.MolLogP(m)),
        "tpsa": lambda m: float(rdMolDescriptors.CalcTPSA(m)),
        "hba": lambda m: float(Lipinski.NumHAcceptors(m)),
        "hbd": lambda m: float(Lipinski.NumHDonors(m)),
        "drd2": lambda m: float(get_score(m))
    }

    missing = [p for p in property_names if p not in fns]
    if missing:
        raise KeyError(
            f"Unknown property names: {missing}. "
            "Add them to `mo_gnn_edit/properties.py` (build_property_fns)."
        )

    return {p: fns[p] for p in property_names}


def calc_properties(mol: "Chem.Mol", property_fns: dict[str, Callable[["Chem.Mol"], float]]) -> dict[str, float]:
    return {name: fn(mol) for name, fn in property_fns.items()}

