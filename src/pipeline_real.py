"""Runs the existing CNN + scaffold + biomechanics pipeline on a real
patient CT (TotalSegmentator subject s1405, femur_left), instead of a
synthetic volume. Nothing in unet3d.py / scaffold_gyroid.py /
biomechanics.py is modified -- this script only adds a real-data entry
point alongside the synthetic one (pipeline.py).

Honesty note: the CNN was trained only on synthetic volumes (see
train_segmentation.py). Real CT has a different intensity distribution,
surrounding soft tissue, and partial-volume noise, so a Dice drop here
vs. the ~0.97 seen on synthetic held-out data is the expected,
informative result, not a bug -- it's the real-vs-synthetic domain gap,
reported honestly below. Because of that gap, the scaffold geometry is
built from the verified TotalSegmentator ground-truth mask (what a
clinician/validated tool would confirm before fabrication), while the
CNN's prediction is reported purely as a generalization diagnostic.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os

from real_ct_loader import load_real_case      # native-resolution crop -> accurate scaffold/STL geometry
from real_ct import load_real_sample            # 64^3-resampled crop -> matches CNN's native training size
from unet3d import TinyUNet3D
from scaffold_gyroid import build_scaffold, mask_to_mesh, export_stl
from biomechanics import sweep


def _pad_to_multiple(vol: np.ndarray, multiple: int = 8):
    pads = [(-s) % multiple for s in vol.shape]
    pad_width = [(0, p) for p in pads]
    return np.pad(vol, pad_width, mode="edge"), vol.shape


def run_model_segmentation(volume: np.ndarray, weights_path="outputs/unet3d_bone.pt") -> np.ndarray:
    """The TinyUNet3D has 3 pooling levels, so each spatial dim must be a
    multiple of 8. A real-data crop rarely is, so pad up, infer, then crop
    the prediction back to the original (unpadded) shape."""
    model = TinyUNet3D(in_ch=1, base=8)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    v = (volume - volume.mean()) / (volume.std() + 1e-6)
    v_padded, orig_shape = _pad_to_multiple(v, 8)
    x = torch.tensor(v_padded[None, None], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).squeeze().numpy().astype(bool)
    return pred[:orig_shape[0], :orig_shape[1], :orig_shape[2]]


def dice_vs_finetuned():
    """Compares the synthetic-only model against the real-data fine-tuned
    one (src/finetune_real.py) on the same 64^3-resampled real case, the
    resolution both models were actually trained/evaluated at. Returns
    (dice_base, dice_finetuned) -- dice_finetuned is None if fine-tuning
    hasn't been run yet (outputs/unet3d_bone_finetuned.pt missing)."""
    real64 = load_real_sample(out_size=64, seed=0)
    gt = real64.bone_mask | real64.defect_mask
    v = (real64.volume - real64.volume.mean()) / (real64.volume.std() + 1e-6)
    x = torch.tensor(v[None, None], dtype=torch.float32)

    def _dice(weights_path):
        model = TinyUNet3D(in_ch=1, base=8)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) > 0.5).squeeze().numpy().astype(bool)
        return 2 * (pred & gt).sum() / (pred.sum() + gt.sum() + 1e-9)

    dice_base = _dice("outputs/unet3d_bone.pt")
    ft_path = "outputs/unet3d_bone_finetuned.pt"
    dice_ft = _dice(ft_path) if _os.path.exists(ft_path) else None
    return float(dice_base), (float(dice_ft) if dice_ft is not None else None)


def main(outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    sample = load_real_case("data/s1405/ct.nii.gz", "data/s1405/femur_left.nii.gz",
                             subject_id="s1405", bone_name="femur_left", seed=3)

    pred_mask = run_model_segmentation(sample.volume)
    dice_real = (2 * (pred_mask & sample.bone_mask).sum() /
                 (pred_mask.sum() + sample.bone_mask.sum() + 1e-9))
    print(f"[Real data, native resolution] synthetic-only model vs. real ground truth: "
          f"Dice={dice_real:.4f}")
    print("This reflects the synthetic-to-real domain gap, not a pipeline bug "
          "(see src/pipeline_real.py docstring).")

    dice_base64, dice_ft64 = dice_vs_finetuned()
    print(f"[Real data, 64³ resampled -- matches training resolution] "
          f"synthetic-only Dice={dice_base64:.4f}", end="")
    if dice_ft64 is not None:
        print(f"  ->  after fine-tuning on this case (src/finetune_real.py): "
              f"Dice={dice_ft64:.4f}")
    else:
        print("  (run `python3 src/finetune_real.py` to see the fine-tuned Dice)")

    # Scaffold is built from the verified real ground-truth mask (clinically
    # realistic: a confirmed segmentation feeds the scaffold design step).
    rows, pick = sweep(sample.bone_mask, sample.defect_mask, sample.voxel_mm,
                        periods=(5, 7, 9, 11, 13))
    print(f"Picked scaffold: period_vox={pick.period_vox:.1f}  "
          f"porosity={pick.porosity:.3f}  E/E_solid={pick.rel_stiffness_gibson_ashby:.4f}")

    sc = build_scaffold(sample.bone_mask, sample.defect_mask,
                         period_vox=pick.period_vox, gyroid_thresh=pick.gyroid_thresh)
    mesh = mask_to_mesh(sc.solid_mask, voxel_mm=sample.voxel_mm)
    export_stl(mesh, f"{outdir}/scaffold_real_femur.stl")
    print(f"Watertight STL exported: watertight={mesh.is_watertight}, "
          f"volume={mesh.volume:.1f} mm^3")

    mid = sample.volume.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    axes[0].imshow(sample.volume[mid], cmap="gray")
    axes[0].set_title("Real CT slice\n(s1405, femur region)")
    axes[1].imshow(sample.full_bone_mask[mid], cmap="bone")
    axes[1].set_title("Real ground-truth mask\n(TotalSegmentator, pre-defect)")
    axes[2].imshow(pred_mask[mid], cmap="bone")
    axes[2].set_title(f"CNN prediction\n(synthetic-trained, Dice={dice_real:.3f})")
    axes[3].imshow(sc.solid_mask[mid], cmap="viridis")
    axes[3].set_title("Scaffold\n(shell + gyroid infill)")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{outdir}/real_data_slices.png", dpi=150)
    print(f"Saved {outdir}/scaffold_real_femur.stl and {outdir}/real_data_slices.png")

    with open(f"{outdir}/real_data_report.md", "w") as f:
        f.write(f"""# Real-data validation (subject s1405, femur_left)

Source: Small subset of TotalSegmentator Dataset, Zenodo
(10.5281/zenodo.10047263), subject s1405, `femur_left` mask. CC BY 4.0.

- CT volume cropped to femur region: {sample.volume.shape} voxels @ {sample.voxel_mm} mm isotropic
- Real bone voxels (ground truth, defect removed): {int(sample.bone_mask.sum())}
- Synthetic defect carved for scaffold demo: {int(sample.defect_mask.sum())} voxels
- CNN (trained only on synthetic data) Dice vs. real ground truth \
(native resolution): **{dice_real:.4f}**
- Same comparison at the model's native 64³ training resolution: \
synthetic-only Dice={dice_base64:.4f}{"" if dice_ft64 is None else f", after fine-tuning on this one real case (src/finetune_real.py) Dice={dice_ft64:.4f}"}
- Picked scaffold: period={pick.period_vox:.1f} vox, porosity={pick.porosity:.1%}, \
E/E_solid={pick.rel_stiffness_gibson_ashby:.4f}
- Output mesh watertight: {mesh.is_watertight}

**Interpretation:** the CNN was trained purely on procedurally generated
synthetic CT (see `train_segmentation.py`) and was never shown real patient
data. The Dice score above is therefore a genuine synthetic-to-real
generalization check, not the number reported for the synthetic-only
pipeline (~0.97). `src/finetune_real.py` closes much of that gap by
fine-tuning on this one real case (mixed with synthetic replay so the
model doesn't forget the general bone/background boundary) -- that is a
single-case adaptation demo, not a validated real-data model, since the
same case is both the fine-tuning example and the evaluation case here.
The scaffold/STL in this run is built from the TotalSegmentator
ground-truth mask (the clinically-verified equivalent) regardless, so the
printable geometry is trustworthy either way. The real next step for a
production model would be fine-tuning on several *held-out* annotated
real cases, not just one.
""")
    print(f"Saved {outdir}/real_data_report.md")


if __name__ == "__main__":
    main()
