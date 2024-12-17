import tempfile
from typing import Protocol

import cv2
import numpy as np
import streamlit as st
import pillow_heif
from PIL import Image


def resize(image, width):
    height = int((width / image.shape[1]) * image.shape[0])
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def simplify(image, num_bins, white_point):
    # Normalize pixel values to the range 0-1
    normalized_image = image / 255.0

    # Create bins and assign each pixel to a bin
    bin_edges = np.linspace(0, white_point, num_bins)
    binned_image = np.digitize(normalized_image, bin_edges) - 1

    # Map bins back to grayscale values (0 to 255)
    scale_values = np.linspace(0, 255, num_bins, endpoint=True)
    simplified_image = scale_values[binned_image].astype(np.uint8)

    return simplified_image


def overlay_grid(image, rows, cols, color=(100, 255, 0), thickness=2):
    grid_image = image.copy()
    height, width = grid_image.shape[:2]

    # Draw horizontal lines
    for i in range(1, rows):
        y = i * height // rows
        cv2.line(grid_image, (0, y), (width, y), color, thickness)

    # Draw vertical lines
    for j in range(1, cols):
        x = j * width // cols
        cv2.line(grid_image, (x, 0), (x, height), color, thickness)

    return grid_image


class ImageLoader(Protocol):
    def load(self, file_path: str) -> np.ndarray:
        ...


class HEICImageLoader:
    def load(self, file_path: str) -> np.ndarray:
        heif_image = pillow_heif.read_heif(file_path)
        pil_image = Image.frombytes(
            heif_image.mode, heif_image.size, heif_image.data
        )
        return np.array(pil_image)


class JPGImageLoader:
    def load(self, file_path: str) -> np.ndarray:
        return np.flip(cv2.imread(file_path), axis=-1)


def get_image_loader(file_path: str) -> ImageLoader:
    image_loaders = {
        ".heic": HEICImageLoader,
        ".jpg": JPGImageLoader,
        ".jpeg": JPGImageLoader,
        ".png": JPGImageLoader,
    }
    for extension, image_loader_class in image_loaders.items():
        if file_path.lower().endswith(extension):
            return image_loader_class()
    raise ValueError("Unsupported file format")


def load_uploaded_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        image_loader = get_image_loader(uploaded_file.name)
        image = image_loader.load(temp_file.name)
    return image


def render_ui():
    st.title("Art Assist")
    st.info(
        "This app provides tools to simplify and analyze images. It's meant to help artists "
        "better understand the values, colors, and compositions of their reference images.\n\n"
        "The app is currently optimized for computer, not mobile, use.\n\n"
        "Upload an image and adjust the settings to see the simplified version.\n\n"
    )
    st.warning(
        "This app is a work in progress. If you encounter any issues or have suggestions, "
        "please let me know by creating an issue on the [GitHub repository]"
        "(https://github.com/carolinebinley/art-assist)."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "heic"])
    if uploaded_file is None:
        st.info("Please upload an image.")
        st.stop()

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        n_bins = st.slider("Bins", min_value=2, max_value=10, value=5)

    with column_2:
        white_point = st.slider("White point", min_value=0.0, max_value=1.0, value=0.75)

    with column_3:
        black_and_white = st.checkbox("Black-and-white", value=False)
        grid_overlay = st.checkbox("Overlay Grid", value=False)
        side_by_side = st.checkbox("Side-by-side", value=True)
        output_width = 350 if side_by_side else 700

    original_image = load_uploaded_image(uploaded_file)
    original_image = simplified_image = resize(original_image, output_width)

    if black_and_white:
        simplified_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    simplified_image = simplify(simplified_image, n_bins, white_point)

    if grid_overlay:
        rows = cols = st.slider("Grid Rows", min_value=1, max_value=20, value=10)
        grid_color = st.color_picker("Grid Color", value="#8C8C8C")
        grid_color = tuple(int(grid_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        simplified_image = overlay_grid(simplified_image, rows, cols, grid_color)

    st.image(
        [simplified_image, original_image],
        caption=["Simplified", "Original"],
    )


if __name__ == "__main__":
    render_ui()
