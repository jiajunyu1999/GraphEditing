from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from .chem_utils import atom_idx_by_map, ensure_atom_maps, max_atom_map, mol_from_smiles, terminal_neighbors

Op = Literal["add", "remove", "replace"]


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: RDKit. Install it (e.g. `conda install -c conda-forge rdkit`)."
        ) from exc


def _sanitize_with_fallback(mol: "Chem.Mol") -> "Chem.Mol":
    from rdkit import Chem

    mol_full = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol_full)
        return mol_full
    except Chem.KekulizeException:
        # Some aromatic systems cannot be kekulized after edits, but we can still keep a
        # valid aromatic representation for SMILES + many descriptors.
        mol_partial = Chem.Mol(mol)
        ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        Chem.SanitizeMol(mol_partial, sanitizeOps=ops)
        return mol_partial


@dataclass(frozen=True)
class Edit:
    step: int
    anchor_atom_map: int
    op: Op
    fg_smiles: str | None = None
    fg_id: int | None = None
    removed_atom_map: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "anchor_atom_map": self.anchor_atom_map,
            "op": self.op,
            "fg_smiles": self.fg_smiles,
            "fg_id": self.fg_id,
            "removed_atom_map": self.removed_atom_map,
        }


def _find_dummy_atom_idx(frag: "Chem.Mol") -> int:
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return atom.GetIdx()
    raise ValueError("Functional group must contain a '*' dummy atom for attachment")


@lru_cache(maxsize=1024)
def _cached_fg_mol(fg_smiles: str) -> "Chem.Mol":
    """
    Cache parsed functional-group molecules. Returned Mol must NOT be mutated.
    """
    _require_rdkit()
    from rdkit import Chem

    frag = mol_from_smiles(fg_smiles)
    # Keep a sanitized, immutable copy in the cache.
    frag = Chem.Mol(frag)
    Chem.SanitizeMol(frag, catchErrors=True)
    return frag


def attach_functional_group(mol: "Chem.Mol", anchor_idx: int, fg_smiles: str) -> "Chem.Mol":
    _require_rdkit()
    from rdkit import Chem

    mol = ensure_atom_maps(mol)
    frag = Chem.Mol(_cached_fg_mol(fg_smiles))
    frag = ensure_atom_maps(frag, start_map=max_atom_map(mol) + 1)

    dummy_idx = _find_dummy_atom_idx(frag)
    dummy_atom = frag.GetAtomWithIdx(dummy_idx)
    nbrs = list(dummy_atom.GetNeighbors())
    if len(nbrs) != 1:
        raise ValueError("Dummy atom must have exactly 1 neighbor in functional group")
    attach_idx = nbrs[0].GetIdx()
    bond = frag.GetBondBetweenAtoms(dummy_idx, attach_idx)
    if bond is None:
        raise RuntimeError("Internal error: dummy-attach bond missing")

    combined = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combined)

    offset = mol.GetNumAtoms()
    rw.AddBond(anchor_idx, offset + attach_idx, bond.GetBondType())

    # Remove dummy atom (highest index first to keep indices stable).
    rw.RemoveAtom(offset + dummy_idx)
    out = rw.GetMol()
    out = _sanitize_with_fallback(out)
    return ensure_atom_maps(out)


def remove_terminal_substituent(
    mol: "Chem.Mol",
    anchor_atom_map: int,
    *,
    removed_atom_map: int | None = None,
) -> tuple["Chem.Mol", int]:
    _require_rdkit()
    from rdkit import Chem

    mol = ensure_atom_maps(mol)
    anchor_idx = atom_idx_by_map(mol, anchor_atom_map)
    candidates = terminal_neighbors(mol, anchor_idx)
    if not candidates:
        raise ValueError("No removable terminal substituent found")

    if removed_atom_map is not None:
        idx_by_map = {mol.GetAtomWithIdx(i).GetAtomMapNum(): i for i in candidates}
        if removed_atom_map not in idx_by_map:
            raise ValueError(
                f"removed_atom_map={removed_atom_map} is not a removable terminal neighbor "
                f"of anchor_atom_map={anchor_atom_map}"
            )
        remove_idx = idx_by_map[removed_atom_map]
    else:
        # Heuristic: remove the smallest terminal atom (by atomic number).
        candidates.sort(key=lambda idx: mol.GetAtomWithIdx(idx).GetAtomicNum())
        remove_idx = candidates[0]
    removed_map = mol.GetAtomWithIdx(remove_idx).GetAtomMapNum()

    rw = Chem.RWMol(mol)
    rw.RemoveAtom(remove_idx)
    out = rw.GetMol()
    out = _sanitize_with_fallback(out)
    return ensure_atom_maps(out), removed_map


def replace_terminal_substituent(
    mol: "Chem.Mol", anchor_atom_map: int, fg_smiles: str
) -> tuple["Chem.Mol", int]:
    mol2, removed_map = remove_terminal_substituent(mol, anchor_atom_map)
    anchor_idx2 = atom_idx_by_map(mol2, anchor_atom_map)
    mol3 = attach_functional_group(mol2, anchor_idx2, fg_smiles)
    return mol3, removed_map


def _scaffold_atom_indices(mol: "Chem.Mol") -> set[int]:
    _require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return set()
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        return set()
    return set(int(i) for i in match)


def _non_scaffold_branch_atoms(
    mol: "Chem.Mol", start_idx: int, scaffold_idx: set[int]
) -> set[int]:
    """
    Return the connected non-scaffold atoms reachable from start_idx without crossing scaffold atoms.
    """
    if start_idx in scaffold_idx:
        return set()
    seen: set[int] = set()
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in seen or idx in scaffold_idx:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in seen or nidx in scaffold_idx:
                continue
            stack.append(nidx)
    return seen


def remove_non_scaffold_branch(
    mol: "Chem.Mol",
    anchor_atom_map: int,
    removed_atom_map: int,
) -> tuple["Chem.Mol", int]:
    """
    Remove a non-scaffold atom and its non-scaffold branch, anchored on a scaffold atom.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = ensure_atom_maps(mol)
    anchor_idx = atom_idx_by_map(mol, anchor_atom_map)
    removed_idx = atom_idx_by_map(mol, removed_atom_map)
    scaffold_idx = _scaffold_atom_indices(mol)
    if anchor_idx not in scaffold_idx:
        raise ValueError("Anchor atom is not in the Murcko scaffold")
    if removed_idx in scaffold_idx:
        raise ValueError("Removed atom is in the Murcko scaffold")
    if not mol.GetBondBetweenAtoms(anchor_idx, removed_idx):
        raise ValueError("Removed atom must be a direct neighbor of anchor atom")

    to_remove = _non_scaffold_branch_atoms(mol, removed_idx, scaffold_idx)
    if not to_remove:
        raise ValueError("No removable non-scaffold branch found")

    rw = Chem.RWMol(mol)
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(int(idx))
    out = rw.GetMol()
    out = _sanitize_with_fallback(out)
    return ensure_atom_maps(out), removed_atom_map


def replace_non_scaffold_branch(
    mol: "Chem.Mol",
    anchor_atom_map: int,
    removed_atom_map: int,
    fg_smiles: str,
) -> tuple["Chem.Mol", int]:
    mol2, removed_map = remove_non_scaffold_branch(mol, anchor_atom_map, removed_atom_map)
    anchor_idx2 = atom_idx_by_map(mol2, anchor_atom_map)
    mol3 = attach_functional_group(mol2, anchor_idx2, fg_smiles)
    return mol3, removed_map
