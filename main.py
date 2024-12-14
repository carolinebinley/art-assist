import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image
import pillow_heif


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


def render_ui():
    st.title("Image Simplifier")

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
        side_by_side = st.checkbox("Side-by-side", value=True)
        output_width = 350 if side_by_side else 700

    grid_overlay = st.checkbox("Overlay Grid", value=False)
    if grid_overlay:
        rows = cols = st.slider("Grid Rows", min_value=1, max_value=20, value=10)
        grid_color = st.color_picker("Grid Color", value="#8C8C8C")
        grid_color = tuple(int(grid_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

        if uploaded_file.name.lower().endswith(".heic"):
            heif_image = pillow_heif.read_heif(temp_file.name)
            pil_image = Image.frombytes(
                heif_image.mode, heif_image.size, heif_image.data
            )
            original_image = np.array(pil_image)
        else:
            original_image = np.flip(cv2.imread(temp_file.name), axis=-1)

    resized_image = simplified_image = resize(original_image, output_width)

    if black_and_white:
        simplified_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    simplified_image = simplify(simplified_image, n_bins, white_point)

    if grid_overlay:
        simplified_image = overlay_grid(simplified_image, rows, cols, grid_color)

    st.image(
        [simplified_image, resized_image],
        caption=["Simplified", "Original"],
    )


if __name__ == "__main__":
    render_ui()
