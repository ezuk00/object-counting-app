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

st.sidebar.header("Parameters")
ksize = st.sidebar.slider("Median filter size (odd)", 3, 15, 5, step=2)
th_val = st.sidebar.slider("Threshold (background - f)", 1, 100, 25)
dark_obj = st.sidebar.radio("Object color", ["Darker than background", "Brighter than background"])
method = st.sidebar.radio("Counting method", ["Distance transform (touching objects)", "Connected components"])
peak_size = st.sidebar.slider("Peak window size (≈ object diameter)", 5, 61, 21, step=2)
min_dist = st.sidebar.slider("Min distance value (px)", 1, 30, 5)
min_area = st.sidebar.slider("Min object size (pixels)", 0, 2000, 100)

uploaded = st.file_uploader("이미지 파일 선택", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("이미지를 업로드해주세요.")
    st.stop()

# 1. grayscale
I = Image.open(uploaded).convert("L")
if I.width > 1000:
    I = I.resize((1000, int(I.height * 1000 / I.width)))
f = np.array(I)

# 2. noise 제거
f_denoised = signal.medfilt2d(f, ksize)

# 3. thresholding
if dark_obj == "Darker than background":
    bg = int(np.percentile(f_denoised, 95))   # 가장 밝은 영역 = 배경
    mask = (bg - f_denoised.astype(int)) > th_val
else:
    bg = int(np.percentile(f_denoised, 5))    # 가장 어두운 영역 = 배경
    mask = (f_denoised.astype(int) - bg) > th_val

mask = nd.binary_fill_holes(mask)
mask = nd.binary_opening(mask, structure=np.ones((3, 3)))
mask = nd.binary_closing(mask, structure=np.ones((3, 3)))

# 4. counting
if method.startswith("Distance"):
    dist = nd.distance_transform_edt(mask)
    dist_s = nd.gaussian_filter(dist, 2)
    peaks = (dist_s == nd.maximum_filter(dist_s, size=peak_size)) & (dist_s > min_dist)
    labels, nums = nd.label(peaks)
    centers = nd.center_of_mass(peaks, labels, range(1, nums + 1))
    count = nums
    labels_f = labels
else:
    labels, nums = nd.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.where(sizes >= min_area)[0] + 1
    labels_f = np.where(np.isin(labels, keep), labels, 0)
    count = len(keep)
    centers = nd.center_of_mass(mask, labels_f, keep) if count > 0 else []


def show(img, title, cmap="gray", points=None):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)
    if points is not None and len(points) > 0:
        ys, xs = zip(*points)
        ax.scatter(xs, ys, c="red", s=12)
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
    show(mask, "Binary mask")
with c4:
    show(f, f"Detected objects: {count}", points=centers)

if method.startswith("Connected") and count > 0:
    st.subheader("Pixel count per object")
    df = pd.DataFrame({"label": keep, "pixels": sizes[keep - 1]})
    st.dataframe(df, use_container_width=True)
