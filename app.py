import streamlit as st
import os
import json
import joblib
import numpy as np

# === Constants ===
PROJECT_DIR = os.path.dirname(__file__)
USER_DATA_FILE = os.path.join(PROJECT_DIR, 'users.json')
MODEL_FILE = os.path.join(PROJECT_DIR, 'svm_model.joblib')
SCALER_FILE = os.path.join(PROJECT_DIR, 'scaler.joblib')
PERFORMANCE_FILE = os.path.join(PROJECT_DIR, 'model_performance.json')
DATASET_FILE_PATH = os.path.join(PROJECT_DIR, 'parkinsons.csv')

FEATURES_FOR_MODEL = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Shimmer", "Shimmer:DDA", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

# === Load model, scaler, metrics ===
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    model_loaded = True
except:
    model, scaler, model_loaded = None, None, False

try:
    with open(PERFORMANCE_FILE, "r") as f:
        model_performance = json.load(f)
except:
    model_performance = {}

# === Ensure users.json exists ===
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)

# === Streamlit App ===
st.set_page_config(page_title="Parkinson's Detection App", layout="wide")

# Session state for login
if "user" not in st.session_state:
    st.session_state.user = None

# Sidebar Menu (dynamic)
if st.session_state.user:
    menu = st.sidebar.radio("📌 Navigation", 
        ["Home", "Predict Parkinson's", "Abstract", 
         "Algorithm & Example", "Dataset Info", "Help", "Logout"])
else:
    menu = st.sidebar.radio("📌 Navigation", 
        ["Home", "Abstract", "Algorithm & Example", 
         "Dataset Info", "Help", "Login / Sign Up"])

# === Pages ===
if menu == "Home":
    st.title("🧠 Parkinson's Detection App")
    st.write("Welcome to the Parkinson’s Disease detection system using voice features and SVM.")
    if os.path.exists("home.png"):
        st.image("home.png", use_column_width=True)

elif menu == "Login / Sign Up":
    st.subheader("🔑 Login / Sign Up")

    choice = st.radio("Select Action", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    with open(USER_DATA_FILE, "r") as f:
        users = json.load(f)

    if choice == "Login":
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.user = username
                st.success(f"✅ Welcome {username}")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")
    else:
        if st.button("Sign Up"):
            if username in users:
                st.error("⚠️ Username already exists")
            else:
                users[username] = password
                with open(USER_DATA_FILE, "w") as f:
                    json.dump(users, f, indent=4)
                st.success("🎉 Account created, please login.")

elif menu == "Predict Parkinson's":
    if not model_loaded:
        st.error("❌ Model or scaler not found. Run training first.")
    else:
        st.subheader("🔬 Enter Voice Measurements")

        input_data = {}
        for feature in FEATURES_FOR_MODEL:
            input_data[feature] = st.number_input(
                f"{feature}", 
                value=0.0, 
                format="%.6f"   # allows typing decimals
            )

        if st.button("Get Prediction"):
            values = [input_data[feat] for feat in FEATURES_FOR_MODEL]
            scaled = scaler.transform([values])
            pred = model.predict(scaled)[0]

            if pred == 1:
                st.error("🩺 Parkinson's Disease Detected")
            else:
                st.success("✅ No Disease Detected")

            # Show metrics
            st.subheader("📊 Model Performance")
            st.write(f"**Accuracy**: {model_performance.get('accuracy', 0):.2f}")
            st.write(f"**Precision**: {model_performance.get('precision', 0):.2f}")
            st.write(f"**Recall**: {model_performance.get('recall', 0):.2f}")
            st.write(f"**F1 Score**: {model_performance.get('f1_score', 0):.2f}")

elif menu == "Abstract":
    st.subheader("📄 Abstract")
    st.write("""
    Parkinson’s Disease (PD) is a progressive neurological disorder that primarily affects movement control.
    This project focuses on the development of a machine learning-based system to detect PD using
    biomedical voice measurements. We employ a Support Vector Machine (SVM) classifier trained on a dataset of vocal
    features such as jitter, shimmer, and harmonics-to-noise ratio.
    """)

elif menu == "Algorithm & Example":
    st.subheader("⚙️ Algorithm & Example")
    st.write("""
    - Algorithm: Support Vector Machine (SVM)  
    - Data preprocessing: features scaled  
    - Model trained on voice dataset  
    - Predicts PD vs Healthy  
    """)

elif menu == "Dataset Info":
    st.subheader("📊 Dataset Info")
    st.write("Dataset: Parkinson’s Disease dataset from UCI ML Repository.")
    if os.path.exists(DATASET_FILE_PATH):
        st.download_button("📥 Download Dataset", open(DATASET_FILE_PATH, "rb"), "parkinsons.csv")

elif menu == "Help":
    st.subheader("🆘 Help")
    st.write("""
    1. Login/Sign up from sidebar  
    2. Go to 'Predict Parkinson's' to enter features  
    3. Get prediction + model performance  
    4. If model not found, run training first  
    """)

elif menu == "Logout":
    st.session_state.user = None
    st.success("🔓 Logged out successfully")
    st.experimental_rerun()

