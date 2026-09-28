"""
app.py

Streamlit demo: upload an MRI slice, get a predicted CDR severity stage
with a confidence bar chart and a Grad-CAM heatmap.

Run with:  streamlit run app.py

Research demo only -- not a diagnostic tool.
"""

import os

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

import config
from AppModel import AlzhiNet
from gradcam_cli import apply_gradcam, build_transform, make_overlay

st.set_page_config(page_title="AlzhiNet MRI Staging", layout="centered")
st.title("AlzhiNet: MRI-based CDR staging")
st.caption("Research demo only. Not a medical device and not for clinical use.")


@st.cache_resource
def load_model():
    model = AlzhiNet(num_classes=len(config.CLASS_ORDER))
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


if not os.path.exists(config.BEST_MODEL_PATH):
    st.warning(
        f"No trained model found at `{config.BEST_MODEL_PATH}`. "
        "Run `python train.py` first, then reload this page."
    )
    st.stop()

uploaded = st.file_uploader("Upload an MRI slice (.jpg / .png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0)
    model = load_model()

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred_idx = int(torch.argmax(probs))

    st.subheader(f"Predicted stage: {config.CLASS_ORDER[pred_idx]}")
    st.write(f"Confidence: {probs[pred_idx].item() * 100:.1f}%")
    st.bar_chart({name: float(p) for name, p in zip(config.CLASS_ORDER, probs)})

    col1, col2 = st.columns(2)
    col1.image(image, caption="Input", use_container_width=True)
    cam = apply_gradcam(model, tensor.clone().requires_grad_(True), pred_idx)
    col2.image(make_overlay(image, cam), caption="Grad-CAM (model attention)", use_container_width=True)
