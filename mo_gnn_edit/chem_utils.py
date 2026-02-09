from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: RDKit. Install it (e.g. `conda install -c conda-forge rdkit`)."
        ) from exc


@dataclass(frozen=True)
class MolWithSmiles:
    mol: "Chem.Mol"
    smiles: str


def mol_from_smiles(smiles: str) -> "Chem.Mol":
    _require_rdkit()
    from rdkit import Chem

    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return mol

    # Fallback: some edited aromatic systems can't be kekulized (RDKit returns None when sanitizing),
    # but are still useful to keep as an aromatic representation. Parse without sanitization then
    # sanitize with a no-kekulize fallback.
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    try:
        Chem.SanitizeMol(mol)
    except Chem.KekulizeException:
        ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        Chem.SanitizeMol(mol, sanitizeOps=ops)
    except Exception:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def smiles_without_atom_maps(mol: "Chem.Mol") -> str:
    _require_rdkit()
    from rdkit import Chem

    mol2 = Chem.Mol(mol)
    for atom in mol2.GetAtoms():
        atom.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)
    frags = smi.split(".")
    frags.sort()
    return ".".join(frags)


def ensure_atom_maps(mol: "Chem.Mol", start_map: int = 1) -> "Chem.Mol":
    _require_rdkit()
    from rdkit import Chem

    mol2 = Chem.Mol(mol)
    used = {a.GetAtomMapNum() for a in mol2.GetAtoms() if a.GetAtomMapNum() > 0}
    next_map = max(used) + 1 if used else start_map
    for atom in mol2.GetAtoms():
        if atom.GetAtomMapNum() <= 0:
            while next_map in used:
                next_map += 1
            atom.SetAtomMapNum(next_map)
            used.add(next_map)
            next_map += 1
    return mol2


def max_atom_map(mol: "Chem.Mol") -> int:
    return max((a.GetAtomMapNum() for a in mol.GetAtoms()), default=0)


def mol_to_tagged_smiles(mol: "Chem.Mol") -> str:
    _require_rdkit()
    from rdkit import Chem

    mol2 = ensure_atom_maps(mol)
    smi = Chem.MolToSmiles(mol2, isomericSmiles=True, canonical=True)
    frags = smi.split(".")
    frags.sort()
    return ".".join(frags)


def normalize_bracket_atom_hydrogens(mol: "Chem.Mol") -> "Chem.Mol":
    """
    RDKit encodes bracket atoms like `[CH2:8]` by storing H as *explicit H count* on the atom,
    which is not automatically updated when we add/remove bonds via RWMol edits.

    For stable editing, convert most explicit-H counts back to implicit-H form.
    """
    _require_rdkit()
    from rdkit import Chem

    mol2 = Chem.Mol(mol)
    changed = False
    for atom in mol2.GetAtoms():
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetAtomMapNum() <= 0:
            continue
        # Keep aromatic [nH] as-is (pyrrolic N); dropping H changes chemistry.
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0:
            continue
        # Be conservative for charged species.
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(0)
            changed = True
        if atom.GetNoImplicit():
            atom.SetNoImplicit(False)
            changed = True

    if changed:
        mol2.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(mol2)
        except Chem.KekulizeException:
            ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            Chem.SanitizeMol(mol2, sanitizeOps=ops)
    return mol2


def atom_idx_by_map(mol: "Chem.Mol", atom_map: int) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == atom_map:
            return atom.GetIdx()
    raise KeyError(f"Atom map {atom_map} not found in molecule")


def pick_anchor_atom_maps(mol: "Chem.Mol") -> list[int]:
    return [a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def is_terminal_atom(atom: "Chem.Atom") -> bool:
    return atom.GetDegree() == 1 and not atom.IsInRing()


def terminal_neighbors(mol: "Chem.Mol", anchor_idx: int) -> list[int]:
    _require_rdkit()
    from rdkit import Chem

    anchor = mol.GetAtomWithIdx(anchor_idx)
    out: list[int] = []
    for nbr in anchor.GetNeighbors():
        if not is_terminal_atom(nbr):
            continue
        if nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(anchor.GetIdx(), nbr.GetIdx())
        if bond is None:
            continue
        # Very conservative "removable substituent" definition:
        # SINGLE, non-aromatic, non-ring bond to a terminal heavy atom.
        if bond.GetIsAromatic():
            continue
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        out.append(nbr.GetIdx())
    return out
