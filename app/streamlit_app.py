"""Interactive demo for the AI-assisted bone-scaffold design pipeline.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch

from synth_ct import generate_sample
from real_ct_loader import load_real_case
from unet3d import TinyUNet3D
from scaffold_gyroid import build_scaffold, mask_to_mesh
from biomechanics import evaluate_scaffold

st.set_page_config(page_title="AI Bone Scaffold Designer", layout="wide")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
WEIGHTS = os.path.join(OUT, "unet3d_bone.pt")

st.title("AI-Assisted Scaffold Design for Bone Reconstruction")
st.caption(
    "CT segmentation (3D CNN) → gyroid-lattice scaffold generation → "
    "porosity/stiffness trade-off → 3D-printable STL. Demonstrated on "
    "procedurally generated synthetic CT volumes of a long-bone shaft with a "
    "focal defect."
)

REAL_CT = os.path.join(OUT, "..", "data", "s1405", "ct.nii.gz")
REAL_MASK = os.path.join(OUT, "..", "data", "s1405", "femur_left.nii.gz")
HAS_REAL_DATA = os.path.exists(REAL_CT) and os.path.exists(REAL_MASK)

with st.sidebar:
    st.header("Case")
    data_source = "Synthetic"
    if HAS_REAL_DATA:
        data_source = st.radio(
            "Data source", ["Synthetic", "Real CT (s1405, femur)"],
            help="Real case: subject s1405, femur_left, from the public "
                 "TotalSegmentator small subset (Zenodo 10.5281/zenodo.10047263). "
                 "A synthetic defect is carved in for the scaffold demo.")
    if data_source == "Synthetic":
        seed = st.number_input("Synthetic patient seed", 0, 10_000, 7)
        size = st.select_slider("Volume resolution (voxels/side)", [48, 64, 80], value=64)
    else:
        seed = st.number_input("Defect placement seed", 0, 10_000, 3)
        size = None
    st.header("Scaffold design")
    period = st.slider("Gyroid period (voxels) — smaller = finer lattice", 4.0, 16.0, 8.0, 0.5)
    thresh = st.slider("Gyroid wall thickness (|g| ≤ t)", 0.10, 0.45, 0.25, 0.01)
    st.header("Design constraints")
    min_porosity = st.slider("Min. porosity for bone ingrowth", 0.3, 0.9, 0.5, 0.05)
    min_stiffness = st.slider("Min. relative stiffness E/E_solid", 0.0, 0.3, 0.08, 0.01)

if not os.path.exists(WEIGHTS):
    st.error("No trained model found at outputs/unet3d_bone.pt. Run "
             "`python3 src/train_segmentation.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    m = TinyUNet3D(in_ch=1, base=8)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    return m


@st.cache_data
def make_synthetic_case(seed, size):
    return generate_sample(size=size, seed=seed, with_defect=True)


@st.cache_data
def make_real_case(seed):
    return load_real_case(REAL_CT, REAL_MASK, subject_id="s1405",
                           bone_name="femur_left", seed=seed)


def _pad_to_multiple(vol, multiple=8):
    pads = [(-s) % multiple for s in vol.shape]
    return np.pad(vol, [(0, p) for p in pads], mode="edge"), vol.shape


def segment(volume):
    model = load_model()
    v = (volume - volume.mean()) / (volume.std() + 1e-6)
    v_padded, orig_shape = _pad_to_multiple(v, 8)
    x = torch.tensor(v_padded[None, None], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).squeeze().numpy().astype(bool)
    return pred[:orig_shape[0], :orig_shape[1], :orig_shape[2]]


is_real = data_source != "Synthetic"
sample = make_real_case(seed) if is_real else make_synthetic_case(seed, size)
pred_mask = segment(sample.volume)
dice = 2 * (pred_mask & sample.bone_mask).sum() / (pred_mask.sum() + sample.bone_mask.sum())

if is_real:
    st.info(
        "Real-data mode: the CNN was trained only on synthetic volumes, so this "
        "Dice score is a genuine synthetic-to-real generalization check (expect it "
        "lower than the ~0.97 seen on synthetic cases) — not a bug. The scaffold "
        "below is still built from the verified TotalSegmentator ground-truth mask, "
        "so the exported geometry is trustworthy regardless.",
        icon="ℹ️",
    )

col1, col2, col3 = st.columns(3)
col1.metric("CNN segmentation Dice", f"{dice:.3f}")

# Real mode: build the scaffold from the verified TotalSegmentator mask (what a
# clinician/validated tool would confirm before fabrication), not the CNN's
# own prediction, since the CNN here is only evaluated for generalization.
scaffold_mask = sample.bone_mask if is_real else pred_mask
sc = build_scaffold(scaffold_mask, sample.defect_mask, period_vox=period, gyroid_thresh=thresh)
metrics = evaluate_scaffold(scaffold_mask, sample.defect_mask, sample.voxel_mm, period, thresh)

col2.metric("Porosity", f"{metrics.porosity*100:.1f}%")
col3.metric("Relative stiffness (E/Eₐ)", f"{metrics.rel_stiffness_gibson_ashby:.3f}")

ok_porosity = metrics.porosity >= min_porosity
ok_stiffness = metrics.rel_stiffness_gibson_ashby >= min_stiffness
if ok_porosity and ok_stiffness:
    st.success("This design satisfies both the porosity and stiffness constraints.")
else:
    msgs = []
    if not ok_porosity:
        msgs.append("porosity below the bone-ingrowth target")
    if not ok_stiffness:
        msgs.append("relative stiffness below the load-bearing target")
    st.warning("Constraint(s) not met: " + ", ".join(msgs) +
               ". Adjust the gyroid period / wall thickness sliders.")

tab1, tab2, tab3 = st.tabs(["CT & segmentation", "3D scaffold preview", "Porosity vs. stiffness sweep"])

with tab1:
    mid = sample.volume.shape[0] // 2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(sample.volume[mid], cmap="gray")
    axes[0].set_title("Real CT slice" if is_real else "Synthetic CT slice")
    axes[1].imshow(pred_mask[mid], cmap="bone")
    axes[1].set_title("CNN-predicted bone mask")
    axes[2].imshow(sc.solid_mask[mid], cmap="viridis")
    axes[2].set_title("Scaffold slice")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    with st.spinner("Extracting 3D surface mesh..."):
        mesh = mask_to_mesh(sc.solid_mask, voxel_mm=sample.voxel_mm)
    v, f = mesh.vertices, mesh.faces
    fig3d = go.Figure(data=[go.Mesh3d(
        x=v[:, 2], y=v[:, 1], z=v[:, 0],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="goldenrod", opacity=1.0, flatshading=True,
    )])
    fig3d.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=520)
    st.plotly_chart(fig3d, width='stretch')
    st.caption(f"Watertight: {mesh.is_watertight} · Volume: {mesh.volume:.1f} mm³ · "
               f"Surface area: {mesh.area:.1f} mm²")

    stl_path = os.path.join(OUT, "scaffold_app_export.stl")
    mesh.export(stl_path)
    with open(stl_path, "rb") as fh:
        st.download_button("Download scaffold STL", fh, file_name="bone_scaffold.stl", mime="model/stl")

with tab3:
    periods = np.linspace(5, 15, 9)
    rows = [evaluate_scaffold(scaffold_mask, sample.defect_mask, sample.voxel_mm, p, thresh) for p in periods]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[r.period_vox for r in rows], y=[r.porosity for r in rows],
                               mode="lines+markers", name="Porosity"))
    fig2.add_trace(go.Scatter(x=[r.period_vox for r in rows],
                               y=[r.rel_stiffness_gibson_ashby for r in rows],
                               mode="lines+markers", name="Relative stiffness (E/Eₐ)", yaxis="y2"))
    fig2.add_vline(x=period, line_dash="dash", line_color="black")
    fig2.update_layout(
        xaxis_title="Gyroid period (voxels)",
        yaxis=dict(title="Porosity"),
        yaxis2=dict(title="Relative stiffness", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        height=450,
    )
    st.plotly_chart(fig2, width='stretch')
    st.caption("Dashed line marks the currently selected gyroid period. Relative stiffness is "
               "a Gibson-Ashby first-order estimate (E/Eₐ ≈ (ρ/ρₐ)²), not a "
               "finite-element result.")
