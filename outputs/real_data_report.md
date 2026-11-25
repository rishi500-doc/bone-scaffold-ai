# Real-data validation (subject s1405, femur_left)

Source: Small subset of TotalSegmentator Dataset, Zenodo
(10.5281/zenodo.10047263), subject s1405, `femur_left` mask. CC BY 4.0.

- CT volume cropped to femur region: (78, 59, 74) voxels @ 1.5 mm isotropic
- Real bone voxels (ground truth, defect removed): 45470
- Synthetic defect carved for scaffold demo: 7407 voxels
- CNN (trained only on synthetic data) Dice vs. real ground truth (native resolution): **0.5675**
- Same comparison at the model's native 64³ training resolution: synthetic-only Dice=0.5999, after fine-tuning on this one real case (src/finetune_real.py) Dice=0.8974
- Picked scaffold: period=9.0 vox, porosity=83.4%, E/E_solid=0.0275
- Output mesh watertight: True

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
