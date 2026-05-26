import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import pandas as pd
import os
import gdown
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_damage_cnn.h5")

@st.cache_resource
def load_labels():
    with open("label_mapping.pkl", "rb") as f:
        return pickle.load(f)
    
    
import os
import gdown

MODEL_FILE = "road_damage_cnn.h5"
FILE_ID = "1VUHYjEqRFsTLqlFCtCY1w0EGmFnW5dcO"

if not os.path.exists(MODEL_FILE):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_FILE,
        quiet=False
    )

model = load_model()
class_names = load_labels()

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_damage(img):

    img = img.resize((224, 224))

    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    return predicted_class, confidence, prediction


# --------------------------------------------------
# Severity Logic
# --------------------------------------------------
def get_severity(prediction_class, confidence):

    prediction_class = prediction_class.lower()

    # POTHOLE
    if prediction_class == "pothole":

        if confidence >= 85:
            return "High"

        elif confidence >= 60:
            return "Medium"

        else:
            return "Low"

    # CRACK
    elif prediction_class == "crack":

        if confidence >= 70:
            return "Medium"

        else:
            return "Low"

    # MANHOLE
    elif prediction_class == "manhole":

        return "Low"

    return "Unknown"

# --------------------------------------------------
# Recommendation Logic
# --------------------------------------------------
def get_recommendation(prediction_class):

    if prediction_class.lower() == "pothole":
        return (
            "Immediate maintenance recommended.",
            "High-risk road condition detected."
        )

    elif prediction_class.lower() == "crack":
        return (
            "Schedule repair work soon.",
            "Road damage may worsen over time."
        )

    elif prediction_class.lower() == "manhole":
        return (
            "Routine inspection recommended.",
            "Monitor infrastructure condition."
        )

    return (
        "No recommendation available.",
        ""
    )


# ==================================================
# SECTION 1 — HEADER
# ==================================================
st.title("🛣️ AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")

# ==================================================
# SECTION 2 — ABOUT PROJECT
# ==================================================
st.header("📘 About the Project")

st.write("""
Road monitoring is essential for ensuring public safety,
reducing vehicle damage, and improving transportation infrastructure.

Convolutional Neural Networks (CNNs) are powerful deep learning models
used in computer vision for automatically identifying patterns in images.

This system helps detect road damages such as potholes, cracks,
and manholes to support smart city infrastructure management.
""")

st.write("""
### Industry Applications

- Smart City Monitoring
- Road Maintenance Planning
- Highway Inspection
- Transportation Safety
- Municipal Infrastructure Management
""")

st.markdown("---")

# ==================================================
# SECTION 3 — IMAGE UPLOAD
# ==================================================
st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# SECTION 4 — IMAGE PREVIEW
# ==================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.header("🖼 Uploaded Image")

    st.image(
        image,
        caption="Road Surface Image",
        use_container_width=True
    )
# ==================================================
# SECTION 5 — PREDICTION AREA
# ==================================================
predicted_class, confidence, probs = predict_damage(image)

severity = get_severity(
    predicted_class,
    confidence
)

st.header("🔍 Prediction Area")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Damage Type",
        value=predicted_class.upper()
    )

with col2:
    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        label="Severity",
        value=severity
    )

# Risk Indicator

if severity == "High":
    st.error("🚨 High-Risk Road Condition Detected")

elif severity == "Medium":
    st.warning("⚠ Moderate Road Damage Detected")

else:
    st.success("✅ Low-Risk Infrastructure Condition")


# ==================================================
# SECTION 6 — VISUALIZATION AREA
# ==================================================
st.header("📊 Visualization Area")

chart_df = pd.DataFrame({
    "Class": class_names,
    "Confidence (%)": probs * 100
})

st.subheader("Class Confidence Scores")

import plotly.express as px

fig = px.bar(
    chart_df,
    x="Class",
    y="Confidence (%)",
    title="Probability Distribution Across Classes",
    text="Confidence (%)"
)

fig.update_traces(
    texttemplate='%{text:.2f}',
    textposition='outside'
)

fig.update_layout(
    yaxis_title="Confidence (%)",
    xaxis_title="Damage Class"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    chart_df,
    use_container_width=True
)


# ==================================================
# SECTION 7 — RECOMMENDATIONS
# ==================================================
st.header("🚧 Recommendations")

if predicted_class.lower() == "pothole":

    st.error("""
🚨 Immediate maintenance recommended.

• High accident risk

• Potential vehicle damage

• Repair required urgently
""")

    st.warning("""
Safety Warning:

Drivers should reduce speed and avoid sudden maneuvers.
""")

elif predicted_class.lower() == "crack":

    st.warning("""
⚠ Maintenance recommended.

• Crack propagation is possible

• Damage may worsen over time

• Schedule repair work soon
""")

    st.info("""
Safety Warning:

Continuous monitoring is advised to prevent deterioration.
""")

elif predicted_class.lower() == "manhole":

    st.info("""
ℹ Routine inspection recommended.

• Verify manhole alignment

• Inspect surrounding pavement

• Prevent future damage
""")

    st.success("""
Safety Status:

Currently low-risk but periodic inspection is recommended.
""")


# ==================================================
# PROJECT SUMMARY
# ==================================================
st.markdown("---")

st.header("📋 Project Information")

st.write("""
**Model:** Custom CNN

**Dataset:** Road Damage Dataset (2009 Images)

**Classes:** Crack, Manhole, Pothole

**Framework:** TensorFlow + Streamlit

**Application:** Smart City Infrastructure Monitoring
""")