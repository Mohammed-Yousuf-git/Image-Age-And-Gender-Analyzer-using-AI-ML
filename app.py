import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
from deepface import DeepFace
import matplotlib.pyplot as plt
import tempfile
import atexit

# ========== Custom Topic-Related CSS ========== #
st.markdown(
    """
    <style>
    body, .main {
        background-color: #f7fafc;
    }
    .app-card {
        background: linear-gradient(120deg, #213a5c 70%, #5780c2 100%);
        color: #fff;
        padding: 22px 18px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(30,40,50,0.10);
    }
    .section-title {
        font-size: 1.18rem;
        font-weight: 600;
        color: #e2eaf7;
        margin-bottom: 8px;
        letter-spacing: 0.2px;
    }
    .info-section {
        background: #e5ecfa;
        border-left: 6px solid #326fff;
        color: #234;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
    h1, h2, h3 {
        color: #23365e;
        text-align: left;
    }
    .stButton>button {
        background: linear-gradient(80deg, #4f8cff 80%, #00bfff 100%);
        color: #fff;
        font-weight: bold;
        border: none;
        border-radius: 7px;
        padding: 9px 22px;
        font-size: 1rem;
        margin-top: 7px;
        transition: 0.19s;
    }
    .stButton>button:hover {
        background: linear-gradient(80deg, #457acc 70%, #27b1ea 100%);
    }
    .emoji {
        font-size: 2rem;
        margin-right: 10px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== ICONS ==========
age_icon = "👶 🧑 🧓"
gender_icon = "👨‍🦰 👩‍🦰"
ai_icon = "🤖"
cam_icon = "🔎"

# ========== Load Model ==========
model_path = "/Users/usufahmed/Desktop/gender_app/faces/models/age_classifier.keras"
age_model = None
IMG_SIZE = (224, 224)

try:
    age_model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"⚠️ Error loading age model: {e}")

# ========== Age Prediction ==========
def predict_age_group(image_path):
    if age_model is None:
        return "Age model could not be loaded.", None
    try:
        img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = age_model.predict(img_array)
        class_names = ['MIDDLE', 'OLD', 'YOUNG']
        return class_names[np.argmax(pred)], pred[0]
    except Exception as e:
        return f"Error predicting age: {e}", None

# ========== Gender Prediction ==========
def predict_gender(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['gender'], enforce_detection=False)
        gender_probs = {}
        if isinstance(result, dict) and 'gender' in result:
            gender_probs = result['gender']
        elif isinstance(result, list) and len(result) and 'gender' in result[0]:
            gender_probs = result[0]['gender']
        return gender_probs
    except Exception as e:
        st.error(f"DeepFace error: {e}")
        return {}

# ========== Main Title & Context ==========
st.markdown(f"<h1>{ai_icon} Real-time Age & Gender Analyzer Using AI/ML</h1>", unsafe_allow_html=True)
st.markdown(
    f"""<div class="info-section">
    <b>{cam_icon} Upload a face image to predict age group and gender, powered by deep learning.
    </b>
    <br>See how AI can interpret faces!<br>
    <span style="font-size:1.05rem;color:#456fc9;">Image quality and visibility affect results.</span>
    </div>
    """, unsafe_allow_html=True
)

# ========== Sidebar: Upload + Stats ==========
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload face image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    image_path = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded_file.getbuffer())
            image_path = temp.name
            st.image(image_path, caption="Preview", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.subheader("Dataset Stats")
    try:
        csv_path = "/Users/usufahmed/Desktop/gender_app/faces/train.csv"
        image_dir = "/Users/usufahmed/Desktop/gender_app/faces/Train"
        if os.path.exists(csv_path) and os.path.isdir(image_dir):
            df = pd.read_csv(csv_path)
            df['ID'] = df['ID'].str.lower().str.replace(r'\.jpg|\.jpeg|\.png', '', regex=True).str.strip()
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            base_to_full = {os.path.splitext(f)[0].lower(): f for f in image_files}
            df['filename'] = df['ID'].map(base_to_full)
            df = df.dropna(subset=['filename'])
            st.write(f"**Valid Images:** {len(df)}")
            class_counts = df['Class'].value_counts().to_dict()
            st.write("**Class Distribution:**")
            st.json(class_counts)
        else:
            st.info("Dataset not found.")
    except Exception as e:
        st.error(f"Error loading dataset stats: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.subheader("Feedback")
    feedback = st.text_area("What do you think of the app?")
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Main Output ==========
if image_path and age_model is not None:
    # --- Age Prediction ---
    st.markdown(f'<div class="app-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{age_icon} Age Prediction</div>', unsafe_allow_html=True)
    age_result, age_probs = predict_age_group(image_path)
    if isinstance(age_result, str) and 'Error' in age_result:
        st.error(age_result)
    else:
        st.markdown(
            f"<span class='emoji'>🧑‍🔬</span> <b>Predicted Age Group:</b> <span style='font-size:1.18em;color:#ffe580'>{age_result}</span>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots()
        ax.bar(['MIDDLE', 'OLD', 'YOUNG'], age_probs, color=['#4f8cff', '#f67279', '#79e686'])
        ax.set_title("Probability by Age Group")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Gender Prediction ---
    st.markdown(f'<div class="app-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{gender_icon} Gender Prediction</div>', unsafe_allow_html=True)
    gender_result = predict_gender(image_path)
    if isinstance(gender_result, dict) and gender_result:
        fig, ax = plt.subplots()
        labels = list(gender_result.keys())
        values = list(gender_result.values())
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#28c3f0', '#fd6fc3'])
        ax.set_title("Gender Classification Probabilities")
        st.pyplot(fig)
        pred_gender = max(gender_result, key=gender_result.get)
        st.markdown(
            f"<span class='emoji'>{'👨' if pred_gender.lower() == 'man' else '👩'}</span> <b>Predicted:</b> "
            f"<span style='font-size:1.12em;color:#4f8cff'>{pred_gender}</span>",
            unsafe_allow_html=True)
    else:
        st.error("Gender prediction failed. Please try another clear, frontal face image.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Cleanup Temporary File ==========
def cleanup():
    try:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    except Exception:
        pass

atexit.register(cleanup)
