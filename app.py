
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import streamlit as st
from PIL import Image

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("Object Counting")
st.write(
    "이미지를 업로드하면 thresholding, morphology, "
    "distance transform 및 watershed로 물체의 수를 세어줍니다."
)

# ---------------- Sidebar ----------------
st.sidebar.header("Parameters")

ksize = st.sidebar.slider(
    "Median filter size (odd)", 3, 15, 5, step=2
)

sat_th = st.sidebar.slider(
    "Saturation threshold", 5, 100, 25
)

val_th = st.sidebar.slider(
    "Darkness threshold (V)", 100, 255, 235
)

open_k = st.sidebar.slider(
    "Opening size", 1, 9, 3, step=2
)

min_dist = st.sidebar.slider(
    "Minimum distance between objects", 3, 40, 15
)

min_area = st.sidebar.slider(
    "Minimum object size (pixels)", 10, 3000, 100
)

# ---------------- Upload ----------------
uploaded = st.file_uploader(
    "이미지 파일 선택",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("이미지를 업로드해주세요.")
    st.stop()

# ---------------- 1. Load image ----------------
rgb = Image.open(uploaded).convert("RGB")

if rgb.width > 1200:
    rgb = rgb.resize(
        (1200, int(rgb.height * 1200 / rgb.width))
    )

rgb_array = np.array(rgb)

gray = np.array(rgb.convert("L"))

hsv = np.array(rgb.convert("HSV"))

S = hsv[..., 1].astype(int)
V = hsv[..., 2].astype(int)

# ---------------- 2. Denoising ----------------
denoised = nd.median_filter(gray, size=ksize)

# ---------------- 3. Thresholding ----------------
# Detect colored and dark objects on a bright background

mask = (S > sat_th) | (V < val_th)

# Remove small noise
mask = nd.binary_opening(
    mask,
    structure=np.ones((open_k, open_k))
)

# Connect small gaps
mask = nd.binary_closing(
    mask,
    structure=np.ones((3, 3))
)

# Fill holes inside objects
mask = nd.binary_fill_holes(mask)

# ---------------- 4. Distance Transform ----------------
distance = nd.distance_transform_edt(mask)

# Smooth distance map
distance_smooth = nd.gaussian_filter(distance, sigma=1)

# ---------------- 5. Find object centers ----------------
coordinates = peak_local_max(
    distance_smooth,
    min_distance=min_dist,
    labels=mask.astype(np.uint8),
    exclude_border=False
)

# Create markers
markers = np.zeros(mask.shape, dtype=np.int32)

for i, (y, x) in enumerate(coordinates, start=1):
    markers[y, x] = i

# ---------------- 6. Watershed Segmentation ----------------
if len(coordinates) > 0:

    labels = watershed(
        -distance_smooth,
        markers,
        mask=mask
    )

else:
    labels, _ = nd.label(mask)

# ---------------- 7. Filter small objects ----------------
sizes = np.bincount(labels.ravel())

valid_labels = np.where(sizes >= min_area)[0]

valid_labels = valid_labels[valid_labels != 0]

final_mask = np.isin(labels, valid_labels)

count = len(valid_labels)

# Calculate centers
centers = []

for label_id in valid_labels:
    center = nd.center_of_mass(
        final_mask,
        labels,
        label_id
    )
    centers.append(center)

# ---------------- 8. Display ----------------
def show(img, title, cmap="gray", points=None):

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(img, cmap=cmap)

    if points is not None and len(points) > 0:
        py, px = zip(*points)

        ax.scatter(
            px, py,
            c="red",
            s=15,
            marker="o"
        )

    ax.set_title(title)
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)


# ---------------- Result ----------------
st.metric("Object count", count)

st.caption(
    f"Detected {count} individual objects using Watershed."
)

c1, c2 = st.columns(2)

with c1:
    show(gray, "Input (grayscale)")

with c2:
    show(denoised, "Denoised (median filter)")

c3, c4 = st.columns(2)

with c3:
    show(mask, "Binary mask")

with c4:
    show(
        final_mask,
        f"Detected objects: {count}",
        points=centers
    )

# ---------------- 9. Object information ----------------
if count > 0:

    st.subheader("Pixel count per object")

    object_sizes = []

    for label_id in valid_labels:
        object_sizes.append({
            "Object ID": int(label_id),
            "Pixel area": int(sizes[label_id])
        })

    df = pd.DataFrame(object_sizes)

    st.dataframe(df, use_container_width=True)
