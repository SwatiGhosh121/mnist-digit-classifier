
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import center_of_mass, shift


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 17px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Draw a digit from 0 to 9 and let an Artificial Neural Network recognize it.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mnist_ann.keras")


model = load_model()


# =====================================================
# 5. SESSION STATE FOR CLEAR CANVAS
# =====================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# =====================================================
# 6. MAIN LAYOUT
# =====================================================

draw_col, info_col = st.columns([1.2, 1])


# ---------------- DRAWING SECTION ----------------

with draw_col:

    st.subheader("Draw Your Digit")

    st.write(
        "Use your mouse or trackpad to draw a single digit."
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        return_image_data=True
    )

    predict_col, clear_col = st.columns(2)

    with predict_col:
        predict_button = st.button(
            "Predict Digit",
            type="primary",
            use_container_width=True
        )

    with clear_col:
        clear_button = st.button(
            "Clear Canvas",
            use_container_width=True
        )

    if clear_button:
        st.session_state.canvas_key += 1
        st.rerun()


# ---------------- INFORMATION SECTION ----------------

with info_col:

    st.subheader("How to Use")

    st.markdown("""
    1. Draw one digit between **0 and 9**.
    2. Click **Predict Digit**.
    3. View the predicted digit and confidence.
    4. Explore the probability chart.
    5. Click **Clear Canvas** to try another digit.
    """)

    st.info(
        "Tip: Draw a large, clear digit near the center "
        "of the canvas for better recognition."
    )

    st.subheader("About the Model")

    st.markdown("""
    - **Model:** Artificial Neural Network (ANN)
    - **Dataset:** MNIST
    - **Input:** 28 × 28 grayscale image
    - **Output:** 10 possible digits (0–9)
    """)


# =====================================================
# 7. IMAGE PREPROCESSING AND PREDICTION
# =====================================================

if predict_button:

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        # Convert canvas output into grayscale
        image = PILImage.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        image = image.convert("L")

        image_array = np.array(image)

        # Find pixels belonging to the drawn digit
        coords = np.argwhere(image_array > 30)

        if len(coords) == 0:

            st.warning("Please draw a digit first.")

        else:

            # -------------------------------------------------
            # STEP 1: CROP THE DIGIT
            # -------------------------------------------------

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            cropped = image_array[
                y_min:y_max + 1,
                x_min:x_max + 1
            ]

            digit = PILImage.fromarray(cropped)

            # -------------------------------------------------
            # STEP 2: RESIZE TO FIT 20 × 20
            # -------------------------------------------------

            digit.thumbnail((20, 20))

            # -------------------------------------------------
            # STEP 3: PLACE INSIDE 28 × 28 IMAGE
            # -------------------------------------------------

            final_image = PILImage.new(
                "L",
                (28, 28),
                0
            )

            x_offset = (28 - digit.width) // 2
            y_offset = (28 - digit.height) // 2

            final_image.paste(
                digit,
                (x_offset, y_offset)
            )

            # -------------------------------------------------
            # STEP 4: CENTRE OF MASS ALIGNMENT
            # -------------------------------------------------

            image_array = np.array(
                final_image
            ).astype("float32")

            cy, cx = center_of_mass(image_array)

            target_y = 13.5
            target_x = 13.5

            shift_y = target_y - cy
            shift_x = target_x - cx

            aligned_image = shift(
                image_array,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=0
            )

            # -------------------------------------------------
            # STEP 5: NORMALIZATION
            # -------------------------------------------------

            processed_image = aligned_image / 255.0

            processed_image = np.expand_dims(
                processed_image,
                axis=0
            )

            # -------------------------------------------------
            # STEP 6: MODEL PREDICTION
            # -------------------------------------------------

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            predicted_digit = int(
                np.argmax(predictions[0])
            )

            confidence = float(
                np.max(predictions[0]) * 100
            )


            # =================================================
            # 8. DISPLAY PREDICTION RESULTS
            # =================================================

            st.divider()

            st.header("Prediction Results")

            result_col, confidence_col = st.columns(2)

            with result_col:

                st.success(
                    f"Predicted Digit: {predicted_digit}"
                )

            with confidence_col:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )


            # =================================================
            # 9. PROBABILITY CHART
            # =================================================

            st.subheader("Probability for Each Digit")

            probability_data = {
                str(i): float(
                    predictions[0][i] * 100
                )
                for i in range(10)
            }

            st.bar_chart(probability_data)


            # =================================================
            # 10. PROCESSED IMAGE PREVIEW
            # =================================================

            st.subheader("Processed Image")

            st.write(
                "This is the aligned 28 × 28 image passed to the ANN."
            )

            image_col, explanation_col = st.columns([1, 2])

            with image_col:

                st.image(
                    aligned_image,
                    width=180,
                    clamp=True
                )

            with explanation_col:

                st.markdown("""
                **What happens behind the scenes?**

                - The drawn digit is converted to grayscale.
                - The digit is cropped from the canvas.
                - It is resized to fit within 20 × 20 pixels.
                - It is placed inside a 28 × 28 image.
                - Centre-of-mass alignment adjusts its position.
                - Pixel values are normalized between 0 and 1.
                - The ANN predicts the most likely digit.
                """)


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Built using Python, TensorFlow, Streamlit and the MNIST dataset."
)
