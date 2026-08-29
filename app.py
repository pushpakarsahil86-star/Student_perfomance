import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

st.title("🎓 Student Math Score Predictor")
st.write(
    "Student details enter karke unka predicted math score calculate karein."
)


@st.cache_resource
def load_artifacts():
    with open("best_student_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


model, scaler, feature_cols = load_artifacts()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["female", "male"])
    race = st.selectbox(
        "Race/Ethnicity",
        ["group A", "group B", "group C", "group D", "group E"],
    )
    parent_education = st.selectbox(
        "Parental Level of Education",
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ],
    )

with col2:
    lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
    test_prep = st.selectbox("Test Prep Course", ["none", "completed"])
    reading_score = st.slider("Reading Score", 0, 100, 70)
    writing_score = st.slider("Writing Score", 0, 100, 70)

if st.button("Predict Score 🚀"):
    input_dict = {
        "reading score": reading_score,
        "writing score": writing_score,
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_education,
        "lunch": lunch,
        "test preparation course": test_prep,
    }

    df_input = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df_input)

    # Reindex input to match training features columns
    df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

    # Scale and Predict
    scaled_input = scaler.transform(df_encoded)
    prediction = model.predict(scaled_input)[0]
    prediction = float(np.clip(prediction, 0, 100))

    st.success(f"🎯 **Predicted Math Score:** {prediction:.2f} / 100")
