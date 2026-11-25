"""Builds a 3D-printable scaffold for a segmented bone defect.

Approach:
  1. Take the CNN-predicted bone mask + the (known, for this synthetic demo)
     defect region that needs filling.
  2. Fill the defect with a TPMS gyroid lattice -- a classic scaffold
     geometry in bone-tissue engineering because it is smooth, self-
     supporting for FDM/resin printing, and its pore size/porosity is tuned
     by a single period parameter, which is exactly the "biomechanical
     optimization" knob exposed in biomechanics.py.
  3. Keep the native cortical bone shell as solid material around the
     lattice infill so the scaffold key properly into the host bone.
  4. Extract an iso-surface with marching cubes and export an STL mesh
     that is immediately ready for slicing/printing.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from skimage.measure import marching_cubes
import trimesh


def gyroid_field(shape, period_vox: float):
    """Evaluate the gyroid implicit function on a voxel grid.
    g(x,y,z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
    period_vox controls how many voxels one lattice unit cell spans
    (smaller => finer lattice => more struts => lower porosity).
    """
    d, h, w = shape
    k = 2 * np.pi / period_vox
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    g = (np.sin(k * xx) * np.cos(k * yy)
         + np.sin(k * yy) * np.cos(k * zz)
         + np.sin(k * zz) * np.cos(k * xx))
    return g


@dataclass
class ScaffoldResult:
    solid_mask: np.ndarray      # final printable solid voxel mask (shell + lattice infill)
    lattice_mask: np.ndarray    # just the lattice infill voxels (inside the defect)
    shell_mask: np.ndarray      # native bone shell voxels kept as-is
    gyroid_thresh: float
    period_vox: float


def build_scaffold(bone_mask: np.ndarray, defect_mask: np.ndarray,
                    period_vox: float = 9.0, gyroid_thresh: float = 0.25) -> ScaffoldResult:
    """period_vox: lattice unit-cell size in voxels (smaller -> denser/stiffer).
    gyroid_thresh: |g| <= thresh keeps material (thin-walled TPMS shell);
    larger thresh -> thicker struts -> lower porosity, higher stiffness.
    """
    g = gyroid_field(bone_mask.shape, period_vox)
    lattice = defect_mask & (np.abs(g) <= gyroid_thresh)
    solid = bone_mask | lattice
    return ScaffoldResult(solid_mask=solid, lattice_mask=lattice,
                           shell_mask=bone_mask.copy(),
                           gyroid_thresh=gyroid_thresh, period_vox=period_vox)


def mask_to_mesh(mask: np.ndarray, voxel_mm: float = 0.4, step_size: int = 1):
    """Marching-cubes surface extraction -> trimesh.Trimesh, scaled to mm."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    verts, faces, normals, _ = marching_cubes(padded, level=0.5, step_size=step_size)
    verts = (verts - 1) * voxel_mm  # undo padding offset, scale to mm
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair()
            mesh = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
            mesh.fix_normals()
        except Exception:
            trimesh.repair.fill_holes(mesh)
            mesh.fix_normals()
    return mesh


def export_stl(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)
    return path


if __name__ == "__main__":
    import os
    from synth_ct import generate_sample

    os.makedirs("outputs", exist_ok=True)
    s = generate_sample(size=96, seed=0)
    sc = build_scaffold(s.bone_mask, s.defect_mask, period_vox=9.0, gyroid_thresh=0.25)
    mesh = mask_to_mesh(sc.solid_mask, voxel_mm=s.voxel_mm)
    export_stl(mesh, "outputs/scaffold_demo.stl")
    print("mesh verts/faces:", len(mesh.vertices), len(mesh.faces))
    print("watertight:", mesh.is_watertight, "volume_mm3:", mesh.volume)
