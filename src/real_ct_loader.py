"""Loads a real patient CT + a real TotalSegmentator bone mask (e.g. a
femur) and carves a synthetic defect into it, so the rest of the pipeline
(CNN segmentation, scaffold generation, biomechanics) can be exercised on
real anatomy without touching any of that existing code.

Data used here: subject s1405 from the public "Small subset of
TotalSegmentator Dataset" (Zenodo, Wasserthal et al., CC BY 4.0,
https://zenodo.org/records/10047263). Only the CT volume and the
`femur_left` mask were kept (data/s1405/).

Why a real mask + a synthetic defect: TotalSegmentator's masks label
healthy, intact anatomy -- there is no "bone with a reconstruction defect"
ground truth in any public dataset of this kind. Carving a synthetic void
into a *real* bone mask gives real cortical/trabecular CT texture and
real bone geometry, while still letting us validate the scaffold-filling
step against a known, intentionally-removed region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import nibabel as nib


@dataclass
class RealSample:
    volume: np.ndarray
    bone_mask: np.ndarray       # real bone mask, defect carved out
    defect_mask: np.ndarray     # the carved-out void to be scaffolded
    full_bone_mask: np.ndarray  # original real mask, defect included (for reference/eval)
    voxel_mm: float
    meta: dict = field(default_factory=dict)


def _bbox(mask: np.ndarray, margin_vox: int):
    idx = np.argwhere(mask)
    mn = np.maximum(idx.min(axis=0) - margin_vox, 0)
    mx = np.minimum(idx.max(axis=0) + margin_vox + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(mn, mx))


def load_real_case(ct_path: str, mask_path: str, margin_mm: float = 12.0,
                    add_defect: bool = True, defect_frac: float = 0.18,
                    seed: int = 0, subject_id: str = "", bone_name: str = "") -> RealSample:
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    ct = ct_img.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata() > 0

    zooms = ct_img.header.get_zooms()[:3]
    voxel_mm = float(np.mean(zooms))  # TotalSegmentator volumes are isotropic (1.5mm)
    margin_vox = max(1, int(round(margin_mm / voxel_mm)))

    sl = _bbox(mask, margin_vox)
    vol_crop = ct[sl]
    mask_crop = mask[sl]

    full_bone_mask = mask_crop.copy()
    defect_mask = np.zeros_like(mask_crop)

    if add_defect:
        rng = np.random.default_rng(seed)
        idx = np.argwhere(mask_crop)
        # place the defect center somewhere in the middle third of the shaft
        # (avoid the joint ends, which are the widest/most irregular part)
        z_lo, z_hi = np.quantile(idx[:, 2], [0.35, 0.65])
        candidates = idx[(idx[:, 2] >= z_lo) & (idx[:, 2] <= z_hi)]
        center = candidates[rng.integers(0, len(candidates))].astype(float)

        extent = np.array(mask_crop.shape)
        radius_vox = defect_frac * extent.max()

        zz, yy, xx = np.meshgrid(np.arange(mask_crop.shape[0]),
                                  np.arange(mask_crop.shape[1]),
                                  np.arange(mask_crop.shape[2]), indexing="ij")
        dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
        defect_mask = (dist <= radius_vox) & mask_crop
        mask_crop = mask_crop & ~defect_mask
        # zero out the removed bone's CT signal to mimic a resection void
        vol_crop = vol_crop.copy()
        vol_crop[defect_mask] = rng.normal(20, 10, size=defect_mask.sum()).astype(np.float32)

    return RealSample(
        volume=vol_crop.astype(np.float32),
        bone_mask=mask_crop.astype(bool),
        defect_mask=defect_mask.astype(bool),
        full_bone_mask=full_bone_mask.astype(bool),
        voxel_mm=voxel_mm,
        meta={"subject": subject_id, "bone": bone_name,
              "source": "TotalSegmentator small subset (Zenodo 10.5281/zenodo.10047263)",
              "crop_shape": vol_crop.shape},
    )


if __name__ == "__main__":
    s = load_real_case("data/s1405/ct.nii.gz", "data/s1405/femur_left.nii.gz",
                        subject_id="s1405", bone_name="femur_left")
    print("crop shape:", s.volume.shape, "voxel_mm:", s.voxel_mm)
    print("bone voxels:", s.bone_mask.sum(), "defect voxels:", s.defect_mask.sum())
    print("HU range:", s.volume.min(), s.volume.max())
