# AI-Assisted Scaffold Design for Bone Reconstruction

A compact, end-to-end pipeline that goes from a CT-like volume of a bone
with a focal defect to a **3D-printable, porosity-tuned lattice scaffold**,
with the geometry and biomechanical trade-offs quantified along the way.

Built to mirror the core technical ask of AICTE-RPS-type scaffold-design
roles: *AI-based reconstruction of bone geometry from medical images +
biomechanically reasoned 3D-printable scaffold generation.*

```
CT volume --> 3D CNN segmentation --> gyroid-lattice scaffold design
   (synthetic)   (PyTorch U-Net)        --> porosity/stiffness sweep
                                         --> watertight STL export
```

## Why synthetic CT data

Real CT of non-paired bones with traumatic/resection defects isn't freely
redistributable, and acquiring it wasn't feasible in the project's time
box. `src/synth_ct.py` procedurally generates paired (CT-intensity-volume,
ground-truth-mask) samples: a curved, tapering long-bone shaft with a dense
cortical shell, a textured trabecular core, realistic scanner noise, and a
randomly placed defect void. This is a standard way to bootstrap and sanity
-check a segmentation pipeline before fine-tuning on real, access
-controlled clinical CT (the same code would just point at a `DataLoader`
over real NIfTI volumes instead).

## Pipeline

1. **Segmentation** (`src/unet3d.py`, `src/train_segmentation.py`) — a small
   3D U-Net (encoder/decoder with skip connections) trained from scratch on
   on-the-fly synthetic volumes with a combined BCE + soft-Dice loss.
   Reaches **~0.97 Dice** against ground truth on held-out synthetic cases
   after 12 epochs (~2 min on CPU).
2. **Scaffold generation** (`src/scaffold_gyroid.py`) — the segmented
   defect is filled with a **TPMS gyroid lattice** (a standard bone-
   tissue-engineering scaffold geometry: smooth, self-supporting for
   FDM/resin printing, porosity tunable by one period parameter), fused
   with the native cortical shell so the scaffold keys into the host bone.
   The result is surfaced with marching cubes and repaired
   (`pymeshfix`) into a **watertight STL** ready to slice.
3. **Biomechanical trade-off** (`src/biomechanics.py`) — sweeps the gyroid
   period to compute porosity, relative density, surface-area:volume ratio
   (cell-attachment proxy), and a **Gibson-Ashby relative-stiffness
   estimate** (`E_scaffold/E_solid ≈ (ρ/ρ₀)²`), then picks the most porous
   design that still clears a minimum-stiffness constraint. This is a
   first-order design estimate, not a substitute for FE/compression
   validation — the README and the app both say so explicitly.
4. **Interactive demo** (`app/streamlit_app.py`) — live sliders for lattice
   period/wall-thickness and porosity/stiffness constraints, CT-slice and
   3D mesh viewers (Plotly), the porosity-vs-stiffness curve, and an STL
   download button.

## Run it

```bash
pip install -r requirements.txt

# 1. train the segmentation CNN (~2 min on CPU)
python3 src/train_segmentation.py

# 2. run the full pipeline on one synthetic case -> outputs/*.stl, *.png
python3 src/pipeline.py --seed 7

# 3. interactive app
streamlit run app/streamlit_app.py
```

## Results (seed=7, 64³ volume)

| Metric | Value |
|---|---|
| Segmentation Dice (held-out) | 0.966 (training), 0.972 (pipeline case) |
| Picked scaffold porosity | 73.5% |
| Picked scaffold rel. stiffness (E/E₀) | 0.070 |
| STL mesh | watertight, ready to slice |

See `outputs/pipeline_slices.png` (CT slice → predicted mask → scaffold
slice) and `outputs/porosity_stiffness_tradeoff.png` (design sweep).

## Real-data validation

The pipeline is also run on **real patient data**: subject `s1405` from the
public [Small subset of TotalSegmentator Dataset](https://zenodo.org/records/10047263)
(Zenodo 10.5281/zenodo.10047263, Wasserthal et al., CC BY 4.0), using the
real CT volume and the `femur_left` ground-truth mask (`data/s1405/`). A
synthetic defect is carved into the real mask (no public dataset labels
actual reconstruction defects), so the scaffold-filling step still has a
known target to validate against.

```bash
python3 src/pipeline_real.py           # evaluate + build the scaffold from real data
python3 src/finetune_real.py           # (optional) adapt the model to this real case
```

| Metric | Value |
|---|---|
| CNN (synthetic-trained) Dice vs. real ground truth | **0.56–0.60** |
| Dice after fine-tuning on this one real case | **0.897** |
| Picked scaffold porosity | 83.4% |
| Picked scaffold rel. stiffness (E/E₀) | 0.028 |
| Output mesh | watertight STL (`outputs/scaffold_real_femur.stl`) |

**This Dice drop from ~0.97 (synthetic) to ~0.6 (real) is the expected,
honestly-reported synthetic-to-real domain gap** — the CNN was never shown
real CT during training. `src/finetune_real.py` then adapts the model to
this one real case (mixed with synthetic replay so it doesn't forget the
general bone/background boundary), recovering to **0.897 Dice** — a
single-case adaptation demo, not a validated real-data model, since the
same case is both the fine-tuning example and the evaluation case. The
scaffold geometry itself is built from the **verified TotalSegmentator
ground-truth mask** regardless (the equivalent of a clinician/validated
tool confirming a segmentation before fabrication), so the exported STL is
trustworthy either way. The real next step for a production model would be
fine-tuning on several *held-out* annotated real cases, not just one.

See `outputs/real_data_slices.png` (real CT → ground truth → CNN
prediction → scaffold) and `outputs/real_data_report.md`.

The Streamlit app also has a **Data source** toggle (sidebar) to switch
live between the synthetic demo and this real femur case.

## Honest limitations

- Trained entirely on **synthetic** CT; validated on one real case (above) —
  Dice is markedly lower on real data, as expected, and the model would need
  fine-tuning on more annotated clinical scans before being relied on for
  segmentation on real CT.
- The stiffness number is a **Gibson-Ashby scaling estimate**, useful for
  rapid design iteration, not a finite-element or physical compression-test
  result.
- Single bone-defect topology (spherical void in a long-bone shaft); real
  anatomical reconstruction cases (per the job posting: non-paired bones,
  upper-limb structures) have more complex geometry.

## Tech stack

Python, PyTorch (3D CNN), scikit-image (marching cubes), trimesh +
pymeshfix (mesh repair/STL export), NumPy/SciPy (volumetric geometry,
TPMS fields), Streamlit + Plotly (interactive 3D viewer).


