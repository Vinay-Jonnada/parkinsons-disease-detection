import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Parkinson's Disease Detection", layout="centered")

# App title
st.title("🧠 Parkinson's Disease Detection")
st.write("This app predicts whether a person has Parkinson's disease based on input features.")

# Collect user input
# (Replace with the actual features your model expects)
fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.3f")
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.3f")
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.3f")
Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.6f")
Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.6f")
RAP = st.number_input("MDVP:RAP", min_value=0.0, format="%.6f")
PPQ = st.number_input("MDVP:PPQ", min_value=0.0, format="%.6f")
Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.6f")
Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.6f")

# Add all required features here (matching your training dataset)

# Prediction button
if st.button("Predict"):
    features = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, Shimmer, Shimmer_dB]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts: **Parkinson's Disease Detected**")
    else:
        st.success("✅ The model predicts: **No Parkinson's Disease**")
