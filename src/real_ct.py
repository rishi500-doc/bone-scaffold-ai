"""Loads a REAL CT volume + a real bone segmentation mask (from the public
TotalSegmentator dataset, https://doi.org/10.5281/zenodo.10047263) and
packages it into the exact same `SynthSample`-shaped object that
`synth_ct.generate_sample()` produces -- so every downstream module
(unet3d, scaffold_gyroid, biomechanics, pipeline, the Streamlit app) runs
completely unchanged on real data vs. synthetic data.

The dataset ships healthy-anatomy masks only (no injury/defect ground
truth), so the one synthetic step kept here is carving a spherical defect
into the real bone mask -- this is an honest, explainable design choice:
*real bone geometry + real CT noise/texture, synthetic injury* for scaffold
-design demonstration purposes. Swap in a real defect-annotated dataset
(e.g. a fracture/resection cohort) to remove this step entirely; nothing
else would need to change.

Usage:
    python3 src/real_ct.py          # sanity-check load + report sizes
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from synth_ct import SynthSample

CT_PATH = "data/s1405/ct.nii.gz"
MASK_PATH = "data/s1405/femur_left.nii.gz"


def _crop_to_bbox(volume, mask, pad_vox=10):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - pad_vox, 0)
    hi = np.minimum(idx.max(axis=0) + pad_vox + 1, np.array(mask.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    return volume[sl], mask[sl]


def load_real_sample(ct_path=CT_PATH, mask_path=MASK_PATH,
                      out_size: int = 64, seed: int | None = 0,
                      with_defect: bool = True) -> SynthSample:
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)
    volume = ct_img.get_fdata().astype(np.float32)
    mask = mask_img.get_fdata().astype(bool)

    native_voxel_mm = float(np.mean(ct_img.header.get_zooms()[:3]))

    vol_crop, mask_crop = _crop_to_bbox(volume, mask, pad_vox=8)

    # Resample the cropped bounding box onto an out_size^3 cube so it feeds
    # the same TinyUNet3D input resolution the model was trained at.
    zoom_factors = [out_size / s for s in vol_crop.shape]
    vol_rs = ndi.zoom(vol_crop, zoom_factors, order=1)
    mask_rs = ndi.zoom(mask_crop.astype(np.float32), zoom_factors, order=1) > 0.5
    vol_rs = vol_rs[:out_size, :out_size, :out_size]
    mask_rs = mask_rs[:out_size, :out_size, :out_size]
    if vol_rs.shape != (out_size, out_size, out_size):
        pad = [(0, out_size - s) for s in vol_rs.shape]
        vol_rs = np.pad(vol_rs, pad, mode="edge")
        mask_rs = np.pad(mask_rs, [(0, out_size - s) for s in mask_rs.shape], mode="constant")

    # effective voxel size after resampling the cropped box onto out_size^3
    eff_voxel_mm = native_voxel_mm * np.mean(
        [c / out_size for c in vol_crop.shape]
    )

    defect_mask = np.zeros_like(mask_rs)
    bone_mask = mask_rs.copy()
    if with_defect:
        rng = np.random.default_rng(seed)
        bone_idx = np.argwhere(mask_rs)
        center_vox = bone_idx[rng.integers(0, len(bone_idx))]
        zz, yy, xx = np.meshgrid(*[np.arange(out_size)] * 3, indexing="ij")
        defect_r = rng.uniform(0.12, 0.18) * out_size
        dist = np.sqrt((zz - center_vox[0]) ** 2 + (yy - center_vox[1]) ** 2 +
                        (xx - center_vox[2]) ** 2)
        defect = (dist <= defect_r) & mask_rs
        defect_mask = defect
        bone_mask = mask_rs & ~defect
        vol_rs = vol_rs.copy()
        vol_rs[defect] = np.percentile(vol_rs[~mask_rs], 50)  # soft-tissue-like HU

    return SynthSample(volume=vol_rs.astype(np.float32), bone_mask=bone_mask.astype(bool),
                        defect_mask=defect_mask.astype(bool), voxel_mm=float(eff_voxel_mm))


if __name__ == "__main__":
    s = load_real_sample()
    print("volume", s.volume.shape, "HU range", s.volume.min(), s.volume.max())
    print("bone voxels", s.bone_mask.sum(), "defect voxels", s.defect_mask.sum(),
          "voxel_mm", s.voxel_mm)
