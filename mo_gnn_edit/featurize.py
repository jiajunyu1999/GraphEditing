from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .chem_utils import ensure_atom_maps, mol_from_smiles, normalize_bracket_atom_hydrogens

if TYPE_CHECKING:  # pragma: no cover
    from rdkit import Chem


def _rdkit_feature_factory():
    """
    Optional RDKit ChemicalFeatures factory (Donor/Acceptor/Hydrophobe/LumpedHydrophobe).
    Returns None if unavailable.
    """
    try:
        from functools import lru_cache

        @lru_cache(maxsize=1)
        def _build():
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures

            import os

            fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            return ChemicalFeatures.BuildFeatureFactory(fdef)

        return _build()
    except Exception:
        return None


@dataclass(frozen=True)
class FeaturizedMol:
    z: torch.Tensor  # [num_nodes] atomic numbers (long)
    x: torch.Tensor  # [num_nodes, num_scalar_feats] float features
    edge_index: torch.Tensor  # [2, num_edges]
    edge_type: torch.Tensor  # [num_edges] long bond type index
    atom_map: torch.Tensor  # [num_nodes] long


def _bond_type_id(bond: "Chem.Bond") -> int:
    from rdkit import Chem

    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return 0
    if bt == Chem.BondType.DOUBLE:
        return 1
    if bt == Chem.BondType.TRIPLE:
        return 2
    if bt == Chem.BondType.AROMATIC:
        return 3
    return 4


def _smarts_patterns():
    try:
        from functools import lru_cache

        @lru_cache(maxsize=1)
        def _build():
            from rdkit import Chem

            return {
                "amide": Chem.MolFromSmarts("C(=O)N"),
                "urea": Chem.MolFromSmarts("NC(=O)N"),
                "sulfonamide": Chem.MolFromSmarts("S(=O)(=O)N"),
            }

        return _build()
    except Exception:
        return {}


def featurize_tagged_smiles(smiles_tagged: str) -> FeaturizedMol:
    from rdkit import Chem
    from rdkit import RDLogger
    import time 
    t=time.time()
    try:  # pragma: no cover
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    mol = mol_from_smiles(smiles_tagged)
    mol = ensure_atom_maps(mol)
    mol = normalize_bracket_atom_hydrogens(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    num_nodes = mol.GetNumAtoms()
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    atom_map = torch.tensor([a.GetAtomMapNum() for a in mol.GetAtoms()], dtype=torch.long)
    pt = Chem.GetPeriodicTable()

    patterns = _smarts_patterns()
    amide_atoms: set[int] = set()
    urea_atoms: set[int] = set()
    sulfonamide_atoms: set[int] = set()
    if patterns:
        amide = patterns.get("amide")
        urea = patterns.get("urea")
        sulfonamide = patterns.get("sulfonamide")
        if amide is not None:
            for match in mol.GetSubstructMatches(amide):
                amide_atoms.update(int(i) for i in match)
        if urea is not None:
            for match in mol.GetSubstructMatches(urea):
                urea_atoms.update(int(i) for i in match)
        if sulfonamide is not None:
            for match in mol.GetSubstructMatches(sulfonamide):
                sulfonamide_atoms.update(int(i) for i in match)

    def _hyb_onehot(a: "Chem.Atom") -> list[float]:
        hyb = a.GetHybridization()
        out = [0.0] * 6
        if hyb == Chem.rdchem.HybridizationType.SP:
            out[0] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP2:
            out[1] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3:
            out[2] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3D:
            out[3] = 1.0
        elif hyb == Chem.rdchem.HybridizationType.SP3D2:
            out[4] = 1.0
        else:
            out[5] = 1.0
        return out

    
    
    feat_types = ("Donor", "Acceptor", "Hydrophobe", "LumpedHydrophobe")
    extra = torch.zeros((num_nodes, len(feat_types)), dtype=torch.float32)
    factory = None
    if factory is not None:
        try:
            for f in factory.GetFeaturesForMol(mol):
                fam = f.GetFamily()
                if fam not in feat_types:
                    continue
                j = feat_types.index(fam)
                for atom_id in f.GetAtomIds():
                    if 0 <= int(atom_id) < num_nodes:
                        extra[int(atom_id), j] = 1.0
        except Exception:
            pass
    
    scalar_feats: list[list[float]] = []
    for i, a in enumerate(mol.GetAtoms()):
        nbrs = list(a.GetNeighbors())
        nbr_type_counts = [0.0] * 6  # C, N, O, S, halogen, other
        nbr_arom_count = 0.0
        nbr_bond_counts = [0.0, 0.0, 0.0, 0.0]  # single, double, triple, aromatic
        for n in nbrs:
            anum = int(n.GetAtomicNum())
            if anum == 6:
                nbr_type_counts[0] += 1.0
            elif anum == 7:
                nbr_type_counts[1] += 1.0
            elif anum == 8:
                nbr_type_counts[2] += 1.0
            elif anum == 16:
                nbr_type_counts[3] += 1.0
            elif anum in (9, 17, 35, 53):
                nbr_type_counts[4] += 1.0
            else:
                nbr_type_counts[5] += 1.0
            if n.GetIsAromatic():
                nbr_arom_count += 1.0
            b = mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx())
            if b is not None:
                bt = b.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    nbr_bond_counts[0] += 1.0
                elif bt == Chem.BondType.DOUBLE:
                    nbr_bond_counts[1] += 1.0
                elif bt == Chem.BondType.TRIPLE:
                    nbr_bond_counts[2] += 1.0
                elif bt == Chem.BondType.AROMATIC:
                    nbr_bond_counts[3] += 1.0
        max_valence = float(pt.GetDefaultValence(int(a.GetAtomicNum())))
        total_valence = float(a.GetTotalValence())
        free_valence = max(0.0, max_valence - total_valence)
        cip = a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else ""
        cip_r = 1.0 if cip == "R" else 0.0
        cip_s = 1.0 if cip == "S" else 0.0
        scalar_feats.append(
            [
                float(a.GetDegree()),
                float(a.GetTotalDegree()),
                total_valence,
                max_valence,
                free_valence,
                float(a.GetImplicitValence()),
                float(a.GetFormalCharge()),
                float(a.GetNumRadicalElectrons()),
                float(a.GetTotalNumHs()),
                float(a.GetNumImplicitHs()),
                float(a.GetNumExplicitHs()),
                1.0 if a.GetIsAromatic() else 0.0,
                1.0 if a.IsInRing() else 0.0,
                *_hyb_onehot(a),
                cip_r,
                cip_s,
                1.0 if a.HasProp("_ChiralityPossible") else 0.0,
                1.0 if i in amide_atoms else 0.0,
                1.0 if i in urea_atoms else 0.0,
                1.0 if i in sulfonamide_atoms else 0.0,
                *nbr_type_counts,
                nbr_arom_count,
                *nbr_bond_counts,
                *extra[i].tolist(),
            ]
        )
    
    x = torch.tensor(scalar_feats, dtype=torch.float32)
    
    src = []
    dst = []
    et = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        t = _bond_type_id(b)
        src.extend([i, j])
        dst.extend([j, i])
        et.extend([t, t])
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(et, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    # Final sanity: ensure RDKit keeps atom map numbers.
    if atom_map.min().item() <= 0:
        raise ValueError("featurize_tagged_smiles expects atom-mapped (tagged) SMILES")

    return FeaturizedMol(z=z, x=x, edge_index=edge_index, edge_type=edge_type, atom_map=atom_map)
