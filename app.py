
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MNIST | Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mnist_model.pkl.gz"
SAMPLE_DIR = BASE_DIR 


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
    color: #1d2939;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #142b45, #27658c);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 22px;
    color: white;
}

.hero p {
    color: #e5eef7;
    font-size: 17px;
    margin-bottom: 0;
}

.intro {
    background: white;
    padding: 17px 20px;
    border: 1px solid #e4eaf1;
    border-radius: 12px;
    color: #526174;
    margin-bottom: 25px;
    line-height: 1.6;
}

div.stButton > button {
    background-color: #245f86;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #173f5d;
    color: white;
}

.result-card {
    background: #eaf5ee;
    border: 1px solid #cce5d5;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.result-label {
    color: #456b55;
    font-size: 14px;
    font-weight: 600;
}

.result-digit {
    color: #17633b;
    font-size: 65px;
    font-weight: 800;
    line-height: 1.3;
}

.note {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found. Keep best_mnist_model.pkl beside app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

if getattr(model, "n_features_in_", None) != 784:
    st.error("The loaded model is not compatible with MNIST input.")
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🔢 Handwritten Digit Recognition</h1>
    <p>Recognising handwritten digits from 0 to 9 using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro">
    Select a handwritten digit, view its image and test the trained classifier.
</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. LOAD BUILT-IN TEST IMAGES
# =====================================================

if not SAMPLE_DIR.exists():
    st.error("The sample_images folder is missing.")
    st.stop()

samples = {}

for digit in range(10):
    image_paths = sorted(
        SAMPLE_DIR.glob(f"digit_{digit}_*.png")
    )

    if image_paths:
        samples[digit] = image_paths[0]

if not samples:
    st.error("No MNIST sample images were found.")
    st.stop()


# =====================================================
# 6. SELECT A DIGIT
# =====================================================

st.subheader("Test the classifier")

st.caption("Choose a digit to view its handwritten test image.")

selected_digit = st.selectbox(
    "Select a digit",
    options=list(samples.keys()),
    format_func=lambda x: f"Digit {x}"
)


# =====================================================
# 7. DISPLAY IMAGE
# =====================================================

selected_path = samples[selected_digit]

sample_image = Image.open(
    selected_path
).convert("L").resize((28, 28))

pixels = np.asarray(
    sample_image,
    dtype=np.float32
) / 255.0

model_input = pixels.reshape(1, 784)

st.divider()

image_col, info_col = st.columns([1, 2], gap="large")

with image_col:
    st.markdown("### Handwritten digit")

    st.image(
        sample_image.resize((196, 196)),
        caption=f"Actual digit: {selected_digit}",
        width=196
    )

with info_col:
    st.markdown("### Image information")

    st.write("Dataset: MNIST test dataset")
    st.write("Image dimensions: 28 × 28 pixels")
    st.write("Image type: Grayscale")
    st.write("Input features: 784 pixel values")

    st.markdown("""
    <p class="note">
    Pixel values are normalised between 0 and 1
    before being passed to the trained classifier.
    </p>
    """, unsafe_allow_html=True)


# =====================================================
# 8. PREDICTION
# =====================================================

st.markdown("### Run prediction")

predict_clicked = st.button(
    "Predict Digit",
    type="primary"
)

if predict_clicked:

    predicted_digit = int(
        model.predict(model_input)[0]
    )

    st.divider()
    st.subheader("Prediction result")

    result_col, details_col = st.columns([1, 2], gap="large")

    with result_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED DIGIT</div>
            <div class="result-digit">{predicted_digit}</div>
        </div>
        """, unsafe_allow_html=True)

    with details_col:
        st.metric("Actual digit", str(selected_digit))
        st.metric("Predicted digit", str(predicted_digit))

        if predicted_digit == selected_digit:
            st.success("Correct classification! The prediction matches the actual digit.")
        else:
            st.warning("Misclassification: the prediction does not match the actual digit.")

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(model_input)[0]

        probability_df = pd.DataFrame({
            "Digit": [str(value) for value in model.classes_],
            "Estimated probability": probabilities
        })

        st.markdown("### Prediction probability distribution")

        st.bar_chart(
            probability_df.set_index("Digit"),
            horizontal=True
        )

        st.caption(
            "These are model-estimated probabilities, not a guarantee of correctness."
        )


# =====================================================
# 9. PROJECT INFORMATION
# =====================================================

st.divider()

with st.expander("About this project"):

    st.markdown("""
    **Project objective**

    To recognise handwritten digits from 0 to 9 using machine learning.

    **Dataset**

    MNIST contains 70,000 labelled grayscale images.
    Each image has dimensions of 28 × 28 pixels.

    **Algorithms evaluated**

    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest Classifier

    **Selected model**

    Random Forest achieved the highest test accuracy
    among the three evaluated models.

    **Evaluation methodology**

    The models were trained using the first 10,000
    images from the official 60,000-image training split
    and evaluated on the official 10,000-image test split.

    **Limitation**

    This demonstration uses MNIST test images and is
    designed for single-digit grayscale images.
    Predictions on ordinary photographs or substantially
    different handwriting may be unreliable.
    """)


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.markdown(
    "<p style='text-align:center; color:#7b8794; font-size:13px;'>"
    "MNIST Handwritten Digit Recognition | Machine Learning Project"
    "</p>",
    unsafe_allow_html=True
)
