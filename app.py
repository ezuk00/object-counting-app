import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.write("Morphological processing으로 물체의 수를 세어줍니다.")

# ---------------- Sidebar ----------------
st.sidebar.header("Parameters")
polarity = st.sidebar.radio(
    "Object vs background",
    ["Darker than background", "Brighter than background", "Either (differs from background)"],
)
bg_pct = st.sidebar.slider("Background percentile (50 = median)", 0, 100, 50)
use_tophat = st.sidebar.checkbox("Remove uneven background (top-hat)", False)
th_size = st.sidebar.slider("Top-hat size (> object size)", 11, 151, 51, step=2)
th_adj = st.sidebar.slider("Threshold adjust (Otsu ±)", -60, 60, 0)
ksize = st.sidebar.slider("Median filter size (odd)", 1, 9, 3, step=2)
open_k = st.sidebar.slider("Opening size", 1, 15, 3, step=2)
close_k = st.sidebar.slider("Closing size (1 = off)", 1, 15, 1, step=2)
hole_max = st.sidebar.slider("Max hole size to fill (px)", 0, 3000, 50)
erode_n = st.sidebar.slider("Erosion iterations", 0, 15, 2)
min_area = st.sidebar.slider("Min object area (px, after erosion)", 0, 3000, 40)
mode = st.sidebar.radio("Counting", ["Connected components", "Connected + area split"])
split_k = st.sidebar.slider("Typical object area multiplier", 0.5, 2.0, 1.0, step=0.1)
skip_border = st.sidebar.checkbox("Ignore objects touching image border", True)

uploaded = st.file_uploader("이미지 파일 선택", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("이미지를 업로드해주세요.")
    st.stop()


def otsu(x):
    hist, _ = np.histogram(x.ravel(), bins=256, range=(0, 256))
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sb = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(sb))


# ---------------- 1. load ----------------
rgb = Image.open(uploaded).convert("RGB")
if rgb.width > 800:
    rgb = rgb.resize((800, int(rgb.height * 800 / rgb.width)))
gray = np.array(rgb.convert("L")).astype(np.float32)
f = nd.median_filter(gray, size=ksize) if ksize > 1 else gray

# ---------------- 2. signal (object strength) ----------------
if use_tophat:
    s = (th_size, th_size)
    dark_s = nd.grey_closing(f, size=s) - f
    bright_s = f - nd.grey_opening(f, size=s)
else:
    bg = np.percentile(f, bg_pct)
    dark_s = bg - f
    bright_s = f - bg

if polarity.startswith("Darker"):
    signal = dark_s
elif polarity.startswith("Brighter"):
    signal = bright_s
else:
    signal = np.maximum(dark_s, bright_s)
signal = np.clip(signal, 0, 255)

# ---------------- 3. mask ----------------
th = otsu(signal) + th_adj
mask = signal > th
mask = nd.binary_opening(mask, structure=np.ones((open_k, open_k)))
if close_k > 1:
    mask = nd.binary_closing(mask, structure=np.ones((close_k, close_k)))

holes = nd.binary_fill_holes(mask) & ~mask
hl, _ = nd.label(holes)
hs = np.bincount(hl.ravel())
small = np.isin(hl, np.where(hs < hole_max)[0]) & (hl > 0)
mask = mask | small

mask_e = mask
if erode_n > 0:
    mask_e = nd.binary_erosion(mask, structure=np.ones((3, 3)), iterations=erode_n)

# ---------------- 4. counting ----------------
labels_e, n = nd.label(mask_e)
areas = np.bincount(labels_e.ravel(), minlength=n + 1)[1:]
keep = np.where(areas >= min_area)[0] + 1
if skip_border:
    border = np.unique(np.concatenate(
        [labels_e[0, :], labels_e[-1, :], labels_e[:, 0], labels_e[:, -1]]))
    keep = np.array([k for k in keep if k not in border], dtype=int)
if len(keep) == 0:
    st.warning("검출된 object가 없습니다. 파라미터를 조절해보세요.")
    st.stop()

a_keep = areas[keep - 1]
ref = float(np.median(a_keep)) * split_k
if mode.startswith("Connected +"):
    per = np.maximum(1, np.round(a_keep / ref).astype(int))
else:
    per = np.ones(len(keep), dtype=int)
count = int(per.sum())

# grow eroded components back to full object size (for outlines)
lab_keep = np.where(np.isin(labels_e, keep), labels_e, 0)
dist, idx = nd.distance_transform_edt(lab_keep == 0, return_indices=True)
grown = lab_keep[idx[0], idx[1]]
full = np.where(mask & (dist <= erode_n + 2), grown, 0)
outline = (nd.maximum_filter(full, size=3) != nd.minimum_filter(full, size=3)) & (full > 0)

cents = nd.center_of_mass(mask_e, labels_e, keep)

# ---------------- display ----------------
def show_img(img, title, cmap="gray"):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def show_result():
    ov = np.stack([gray / 255.0] * 3, axis=-1)
    ov[outline] = [1, 0, 0]
    fig, ax = plt.subplots()
    ax.imshow(ov)
    fs = max(4, 9 - len(keep) // 25)
    for i, ((cy, cx), k) in enumerate(zip(cents, per), start=1):
        txt = f"{i}" if k == 1 else f"{i}×{k}"
        ax.text(cx, cy, txt, color="yellow", fontsize=fs, ha="center", va="center",
                fontweight="bold")
    ax.set_title(f"Detected objects: {count}")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


st.metric("Object count", count)
st.caption(f"Otsu threshold + adjust = {th} | median object area = {int(np.median(a_keep))} px")

c1, c2 = st.columns(2)
with c1:
    show_img(gray, "Input (grayscale)")
with c2:
    show_img(signal, "Object signal")

c3, c4 = st.columns(2)
with c3:
    show_img(mask_e, "Binary mask (after erosion)")
with c4:
    show_result()

with st.expander("Pixel count per object"):
    st.dataframe(pd.DataFrame({"id": range(1, len(keep) + 1), "pixels": a_keep, "count": per}))
