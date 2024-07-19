import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from numba import jit
import io
import time

st.set_page_config(layout="wide")

st.title("Drawing a Dot and Zooming on the Mandelbrot Set")

# Function to generate the Mandelbrot set
@jit(nopython=True, fastmath=True)
def mandelbrot(h, w, x_min, x_max, y_min, y_max, max_iter):
    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    C = x + y[:, None] * 1j
    Z = np.zeros(C.shape, dtype=np.complex128)
    div_time = np.zeros(C.shape, dtype=np.int32)

    for i in range(max_iter):
        for j in range(h):
            for k in range(w):
                if div_time[j, k] == 0:
                    Z[j, k] = Z[j, k] * Z[j, k] + C[j, k]
                    if (Z[j, k].real * Z[j, k].real + Z[j, k].imag * Z[j, k].imag) > 4.0:
                        div_time[j, k] = i

    return div_time

# Function to generate and return the Mandelbrot image as an in-memory file
def generate_mandelbrot_image(x_min, x_max, y_min, y_max, width, height, max_iter):
    mandelbrot_image = mandelbrot(height, width, x_min, x_max, y_min, y_max, max_iter)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(mandelbrot_image, cmap='inferno', extent=(x_min, x_max, y_min, y_max), interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf

# Initial bounds of the Mandelbrot set
if "x_min" not in st.session_state:
    st.session_state.x_min = -2.5
if "x_max" not in st.session_state:
    st.session_state.x_max = 1.0
if "y_min" not in st.session_state:
    st.session_state.y_min = -1.25
if "y_max" not in st.session_state:
    st.session_state.y_max = 1.25
if "resolution" not in st.session_state:
    st.session_state.resolution = (1600, 900)  # Rectangular resolution
if "zoom_factor" not in st.session_state:
    st.session_state.zoom_factor = 1  # Ensure this is an int
if "total_zoom" not in st.session_state:
    st.session_state.total_zoom = 1.0

# App state to store the last drawn dot information
if "last_point" not in st.session_state:
    st.session_state.last_point = None
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False
if "zoom_pending" not in st.session_state:
    st.session_state.zoom_pending = False
if "image_quality" not in st.session_state:
    st.session_state.image_quality = 'very_low'
if "last_click_time" not in st.session_state:
    st.session_state.last_click_time = time.time()

# Generate initial Mandelbrot image
if "mandelbrot_bg" not in st.session_state:
    buf = generate_mandelbrot_image(st.session_state.x_min, st.session_state.x_max,
                                    st.session_state.y_min, st.session_state.y_max,
                                    st.session_state.resolution[0], st.session_state.resolution[1],
                                    50)
    st.session_state.mandelbrot_bg = buf

# Display the Mandelbrot image
try:
    mandelbrot_image = Image.open(st.session_state.mandelbrot_bg)
    st.image(mandelbrot_image, caption="Generated Mandelbrot Set", use_column_width=True)
except Exception as e:
    st.write(f"Error loading Mandelbrot image: {e}")

# Add slider for setting the zoom factor
zoom_factor = st.slider('Zoom Factor', min_value=-1000, max_value=1000, value=st.session_state.zoom_factor, step=1)
st.session_state.zoom_factor = zoom_factor

# Display total zoom level
st.write(f"Total Zoom: {st.session_state.total_zoom:.2f}")

# Setup drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Semi-transparent orange fill color
    stroke_width=10,
    stroke_color="#000000",
    background_image=mandelbrot_image if 'mandelbrot_image' in locals() else None,
    height=900,  # Height of canvas
    width=1600,  # Width of canvas
    drawing_mode="point",  # Drawing points mode
    key="canvas",
    initial_drawing=None if not st.session_state.clear_canvas else {"objects": []}
)

# Reset clear canvas flag
st.session_state.clear_canvas = False

# Check if a dot was drawn
if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0 and not st.session_state.zoom_pending:
    # Get dot location
    dot = canvas_result.json_data["objects"][-1]
    x, y = dot["left"], dot["top"]

    # Display dot coordinates
    st.write(f"Dot drawn at position: ({x}, {y})")

    # Save the last dot to session state
    st.session_state.last_point = (x, y)
    st.session_state.last_click_time = time.time()

    # Rescale coordinates to the Mandelbrot set range
    x_range = st.session_state.x_max - st.session_state.x_min
    y_range = st.session_state.y_max - st.session_state.y_min
    new_x = st.session_state.x_min + (x / 1600) * x_range
    new_y = st.session_state.y_min + (y / 900) * y_range

    # Display rescaled dot coordinates
    st.write(f"Rescaled dot position: ({new_x}, {new_y})")

    # Calculate zoom factor
    if st.session_state.zoom_factor > 0:
        zoom_factor = 1 / st.session_state.zoom_factor
    else:
        zoom_factor = abs(st.session_state.zoom_factor)

    # Set new bounds for zoom
    x_range_zoomed = x_range * zoom_factor
    y_range_zoomed = y_range * zoom_factor
    st.session_state.x_min = new_x - x_range_zoomed / 2
    st.session_state.x_max = new_x + x_range_zoomed / 2
    st.session_state.y_min = new_y - y_range_zoomed / 2
    st.session_state.y_max = new_y + y_range_zoomed / 2

    # Update total zoom level
    st.session_state.total_zoom *= zoom_factor

    # Set clear canvas and zoom flags
    st.session_state.clear_canvas = True
    st.session_state.zoom_pending = True
    st.session_state.image_quality = 'very_low'

    # Re-render empty canvas
    st.experimental_rerun()

# Check zoom flag
if st.session_state.zoom_pending:
    # Generate new Mandelbrot image in low quality
    buf = generate_mandelbrot_image(st.session_state.x_min, st.session_state.x_max,
                                    st.session_state.y_min, st.session_state.y_max,
                                    st.session_state.resolution[0], st.session_state.resolution[1],
                                    50)
    st.session_state.mandelbrot_bg = buf
    st.session_state.image_quality = 'very_low'
    st.session_state.zoom_pending = False

    # Re-render empty canvas
    st.experimental_rerun()

# Improve image quality after 10 seconds if no new clicks
if st.session_state.image_quality == 'very_low':
    placeholder = st.empty()
    while time.time() - st.session_state.last_click_time < 10:
        placeholder.text(f"Waiting for {10 - int(time.time() - st.session_state.last_click_time)} seconds to improve image quality...")
        time.sleep(1)
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            st.session_state.last_click_time = time.time()
            break
    else:
        # Generate new Mandelbrot image in high quality
        buf = generate_mandelbrot_image(st.session_state.x_min, st.session_state.x_max,
                                        st.session_state.y_min, st.session_state.y_max,
                                        st.session_state.resolution[0], st.session_state.resolution[1],
                                        500)
        st.session_state.mandelbrot_bg = buf
        st.session_state.image_quality = 'high'

        # Re-render empty canvas
        st.experimental_rerun()

# Display last drawn dot
if st.session_state.last_point:
    st.write(f"Last dot was at position: {st.session_state.last_point}")
