"""Quantifies the porosity/stiffness trade-off of a gyroid scaffold and
sweeps the lattice parameter to pick a candidate that respects both a
tissue-engineering porosity target (bone ingrowth needs pore space) and a
minimum mechanical-stiffness constraint (the scaffold must bear load).

Relative stiffness is estimated with the classic Gibson-Ashby cellular-
solids scaling law for open-cell foams/lattices:

    E_scaffold / E_solid ~= C * (rho_scaffold / rho_solid) ** n

with C ~= 1 and n ~= 2 for bending-dominated open-cell lattices (gyroid TPMS
behaves close to this regime at low-to-moderate relative density). This is
a standard first-order estimate used in the scaffold-design literature
before committing to full finite-element verification -- it is NOT a
replacement for FE/compression testing, and the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scaffold_gyroid import build_scaffold, mask_to_mesh


@dataclass
class ScaffoldMetrics:
    period_vox: float
    gyroid_thresh: float
    porosity: float              # fraction of defect volume that is empty (pore) space
    relative_density: float      # 1 - porosity
    rel_stiffness_gibson_ashby: float  # E_scaffold / E_solid_bone estimate
    surface_area_mm2: float
    volume_mm3: float
    sa_to_vol: float             # mm^-1, higher = more surface for cell attachment
    n_struts_watertight: bool


def evaluate_scaffold(bone_mask, defect_mask, voxel_mm, period_vox, gyroh_thresh,
                       gibson_ashby_n: float = 2.0, gibson_ashby_c: float = 1.0) -> ScaffoldMetrics:
    sc = build_scaffold(bone_mask, defect_mask, period_vox=period_vox, gyroid_thresh=gyroh_thresh)

    defect_vox = defect_mask.sum()
    lattice_vox = sc.lattice_mask.sum()
    porosity = 1.0 - (lattice_vox / defect_vox if defect_vox else 0.0)
    rel_density = 1.0 - porosity
    rel_stiffness = gibson_ashby_c * (max(rel_density, 1e-6) ** gibson_ashby_n)

    mesh = mask_to_mesh(sc.solid_mask, voxel_mm=voxel_mm)
    return ScaffoldMetrics(
        period_vox=period_vox, gyroid_thresh=gyroh_thresh,
        porosity=float(porosity), relative_density=float(rel_density),
        rel_stiffness_gibson_ashby=float(rel_stiffness),
        surface_area_mm2=float(mesh.area), volume_mm3=float(mesh.volume),
        sa_to_vol=float(mesh.area / mesh.volume) if mesh.volume else float("nan"),
        n_struts_watertight=bool(mesh.is_watertight),
    )


def sweep(bone_mask, defect_mask, voxel_mm,
          periods=(6, 8, 10, 12, 14), thresh=0.25,
          min_rel_stiffness=0.08, min_porosity=0.5):
    """Sweep gyroid period (pore size) at fixed wall threshold and report
    the trade-off curve plus the 'optimal' pick: the most porous design
    that still clears the minimum stiffness constraint."""
    rows = [evaluate_scaffold(bone_mask, defect_mask, voxel_mm, p, thresh) for p in periods]
    feasible = [r for r in rows if r.rel_stiffness_gibson_ashby >= min_rel_stiffness
                and r.porosity >= min_porosity]
    pick = max(feasible, key=lambda r: r.porosity) if feasible else \
        max(rows, key=lambda r: r.rel_stiffness_gibson_ashby)
    return rows, pick


if __name__ == "__main__":
    from synth_ct import generate_sample
    s = generate_sample(size=96, seed=0)
    rows, pick = sweep(s.bone_mask, s.defect_mask, s.voxel_mm)
    print(f"{'period':>7} {'porosity':>9} {'rel.density':>11} {'E/E_solid':>10} {'SA:V(/mm)':>10} {'watertight':>10}")
    for r in rows:
        flag = " <= picked" if r is pick else ""
        print(f"{r.period_vox:7.1f} {r.porosity:9.3f} {r.relative_density:11.3f} "
              f"{r.rel_stiffness_gibson_ashby:10.4f} {r.sa_to_vol:10.3f} {str(r.n_struts_watertight):>10}{flag}")
