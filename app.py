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
    # Base calculation from feature weights
    base_score = (reading_score * 0.45) + (writing_score * 0.45)

    if gender == "male":
        base_score += 3.0
    if lunch == "standard":
        base_score += 3.5
    if test_prep == "completed":
        base_score += 2.5

    edu_bonus = {
        "some high school": 0,
        "high school": 1.0,
        "some college": 2.0,
        "associate's degree": 3.0,
        "bachelor's degree": 4.5,
        "master's degree": 6.0,
    }
    base_score += edu_bonus.get(parent_education, 0)

    prediction = float(np.clip(base_score, 0, 100))

    st.success(f"🎯 **Predicted Math Score:** {prediction:.2f} / 100")
    # ==============================================================================
# VISUAL DATA EXPLORATION SECTION
# ==============================================================================
st.markdown("---")
st.subheader("📊 Data Visualizations & Insights")

# Dropdown / Tabs to view generated plots
tab1, tab2, tab3 = st.tabs(
    ["Distribution Analysis", "Parental Education Impact", "Correlation Heatmap"]
)

with tab1:
  st.write("### Score Distributions")
  try:
    st.image("univariate_analysis.png", use_container_width=True)
  except:
    st.info("Univariate analysis chart display karne ke liye available nahi hai.")

with tab2:
  st.write("### Math Score vs Parental Education")
  try:
    st.image("bivariate_boxplot.png", use_container_width=True)
  except:
    st.info("Bivariate boxplot chart display karne ke liye available nahi hai.")

with tab3:
  st.write("### Feature Correlation Matrix")
  try:
    st.image("correlation_heatmap.png", use_container_width=True)
  except:
    st.info("Correlation heatmap display karne ke liye available nahi hai.")
