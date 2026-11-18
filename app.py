import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.write("이미지를 업로드하면 thresholding과 morphology로 물체의 수를 세어줍니다.")

# ---------------- Sidebar ----------------
st.sidebar.header("Parameters")
smooth = st.sidebar.slider("Distance smoothing (sigma)", 0.0, 8.0, 3.0, step=0.5)
ksize = st.sidebar.slider("Median filter size (odd)", 3, 15, 5, step=2)
sat_th = st.sidebar.slider("Saturation threshold", 5, 100, 35)
val_th = st.sidebar.slider("Darkness threshold (V)", 10, 250, 80)
hole_max = st.sidebar.slider("Max hole size to fill (px)", 0, 3000, 20)
open_k = st.sidebar.slider("Opening size", 1, 15, 3, step=2)
erode_n = st.sidebar.slider("Erosion iterations", 0, 15, 3)
method = st.sidebar.radio(
    "Counting method",
    ["Distance transform (touching objects)", "Connected components"],
)
min_dist = st.sidebar.slider("Min distance value (px)", 1, 30, 4)
sup = st.sidebar.slider("Suppression factor", 0.3, 2.0, 0.8, step=0.1)
min_area = st.sidebar.slider("Min object size (pixels, connected comp.)", 0, 2000, 100)

uploaded = st.file_uploader("이미지 파일 선택", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("이미지를 업로드해주세요.")
    st.stop()

# ---------------- 1. load ----------------
rgb = Image.open(uploaded).convert("RGB")
if rgb.width > 1000:
    rgb = rgb.resize((1000, int(rgb.height * 1000 / rgb.width)))
f = np.array(rgb.convert("L"))
hsv = np.array(rgb.convert("HSV"))
S = hsv[..., 1].astype(int)
V = hsv[..., 2].astype(int)

# ---------------- 2. denoise ----------------
f_denoised = signal.medfilt2d(f, ksize)
bg = int(np.percentile(f_denoised, 95))

# ---------------- 3. mask ----------------
mask = (S > sat_th) | (V < val_th)
mask = nd.binary_opening(mask, structure=np.ones((open_k, open_k)))

# fill only SMALL holes (inside objects), not gaps between objects
holes = nd.binary_fill_holes(mask) & ~mask
hl, hn = nd.label(holes)
hs = np.bincount(hl.ravel())
small = np.isin(hl, np.where(hs < hole_max)[0]) & (hl > 0)
mask = mask | small

# erosion: separate touching objects
mask_c = mask.copy()
if erode_n > 0:
    mask_c = nd.binary_erosion(mask, structure=np.ones((3, 3)), iterations=erode_n)

# ---------------- 4. counting ----------------
centers = []
keep = np.array([], dtype=int)
sizes = np.array([])


if method.startswith("Distance"):
    dist = nd.distance_transform_edt(mask_c)
    dist_s = nd.gaussian_filter(dist, smooth)
    cand = (dist_s == nd.maximum_filter(dist_s, size=5)) & (dist_s > min_dist)
    ys, xs = np.nonzero(cand)
    order = np.argsort(-dist_s[ys, xs])
    accepted = []
    for i in order:
        y, x = ys[i], xs[i]
        ok = True
        for (ay, ax, ar) in accepted:
            if (y - ay) ** 2 + (x - ax) ** 2 < (sup * ar) ** 2:
                ok = False
                break
        if ok:
            accepted.append((y, x, dist_s[y, x]))
    centers = [(a[0], a[1]) for a in accepted]
    count = len(centers)
else:
    labels, nums = nd.label(mask_c)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.where(sizes >= min_area)[0] + 1
    count = len(keep)
    if count > 0:
        labels_f = np.where(np.isin(labels, keep), labels, 0)
        centers = nd.center_of_mass(mask_c, labels_f, keep)


# ---------------- display ----------------
def show(img, title, cmap="gray", points=None):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)
    if points is not None and len(points) > 0:
        py, px = zip(*points)
        ax.scatter(px, py, c="red", s=12)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


st.metric("Object count", count)
st.caption(f"background level: {bg}")

c1, c2 = st.columns(2)
with c1:
    show(f, "Input (grayscale)")
with c2:
    show(f_denoised, "Denoised (median filter)")

c3, c4 = st.columns(2)
with c3:
    show(mask_c, "Binary mask (after erosion)")
with c4:
    show(f, f"Detected objects: {count}", points=centers)

if method.startswith("Connected") and count > 0:
    st.subheader("Pixel count per object")
    df = pd.DataFrame({"label": keep, "pixels": sizes[keep - 1]})
    st.dataframe(df, use_container_width=True)
