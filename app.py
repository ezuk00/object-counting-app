import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.write("이미지를 업로드하면 thresholding과 connected component 분석으로 물체의 수를 세어줍니다.")

st.sidebar.header("Parameters")
ksize = st.sidebar.slider("Median filter size (odd)", 3, 15, 7, step=2)
th_val = st.sidebar.slider("Threshold |f - background|", 1, 100, 20)
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
bg = int(np.median(f_denoised))
mask = np.abs(f_denoised.astype(int) - bg) > th_val
mask = nd.binary_fill_holes(mask)
mask = nd.binary_opening(mask, iterations=1)
mask = nd.binary_closing(mask, iterations=2)

# 4. connected component
labels, nums = nd.label(mask)
sizes = np.bincount(labels.ravel())[1:]
keep = np.where(sizes >= min_area)[0] + 1
if len(keep) == 0:
    st.warning("검출된 object가 없습니다. threshold 또는 min size를 조절해보세요.")
    st.stop()

labels_f = np.where(np.isin(labels, keep), labels, 0)
count = len(keep)

# 5. largest / smallest
largest = keep[np.argmax(sizes[keep - 1])]
smallest = keep[np.argmin(sizes[keep - 1])]


def show(img, title, cmap="gray"):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=cmap)
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
    show(labels_f, "Connected components", cmap="Spectral_r")

st.subheader("Largest / Smallest component")
c5, c6 = st.columns(2)
with c5:
    show(labels_f == largest, f"Largest: label {largest}, {sizes[largest-1]} px")
with c6:
    show(labels_f == smallest, f"Smallest: label {smallest}, {sizes[smallest-1]} px")

st.subheader("Pixel count per object")
df = pd.DataFrame({"label": keep, "pixels": sizes[keep - 1]})
st.dataframe(df, use_container_width=True)
