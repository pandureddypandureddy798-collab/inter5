"""
Diabetes Risk Predictor - Streamlit Interactive ML Web App
InternCircle Data Science & AI - Project 5
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature order MUST match training pipeline exactly
FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "pedigree",
    "age"
]

@st.cache_resource
def load_model():
    """Load model from disk once with Streamlit caching."""
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

@st.cache_resource
def load_metadata():
    """Load model training metadata if available."""
    meta_path = os.path.join(os.path.dirname(__file__), "model_meta.joblib")
    if os.path.exists(meta_path):
        return joblib.load(meta_path)
    return None

# --- Header & Intro ---
st.title("🩺 Interactive Diabetes Risk Predictor")
st.markdown(
    """
    This interactive machine learning web application estimates the statistical risk of diabetes 
    based on clinical and biological indicators. Adjust the input parameters below and click 
    **Predict Risk** to see real-time probability estimates.
    """
)

# --- Sidebar Controls & Presets ---
st.sidebar.header("⚙️ Configuration & Presets")
preset = st.sidebar.selectbox(
    "Choose a sample profile to prefill inputs:",
    options=["Custom Input", "Low Risk Profile", "Moderate Risk Profile", "High Risk Profile"],
    index=0,
    help="Select a preset profile to quickly see how different clinical parameters affect the prediction."
)

# Preset defaults
defaults = {
    "Custom Input": {
        "preg": 1, "plas": 110, "pres": 72, "skin": 22,
        "insu": 85, "mass": 27.2, "pedi": 0.42, "age": 33
    },
    "Low Risk Profile": {
        "preg": 0, "plas": 88, "pres": 68, "skin": 18,
        "insu": 55, "mass": 21.5, "pedi": 0.22, "age": 24
    },
    "Moderate Risk Profile": {
        "preg": 3, "plas": 132, "pres": 78, "skin": 28,
        "insu": 120, "mass": 29.8, "pedi": 0.58, "age": 42
    },
    "High Risk Profile": {
        "preg": 6, "plas": 178, "pres": 88, "skin": 38,
        "insu": 230, "mass": 36.4, "pedi": 1.15, "age": 52
    }
}
current_vals = defaults[preset]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Model Information:**
    - **Algorithm:** Gradient Boosting Classifier
    - **Optimization:** Scikit-Learn Pipeline with Median Imputation & Standard Scaling
    - **Dataset:** Pima Indians Diabetes Dataset
    - **Cross-Validated ROC-AUC:** ~0.83
    """
)

# --- Input Section ---
st.subheader("Patient Clinical & Diagnostic Features")
st.write("Enter patient measurements within the indicated realistic clinical ranges:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🧬 Physiological & Diagnostic")
    glucose = st.slider(
        "Fasting Plasma Glucose (mg/dL)",
        min_value=50,
        max_value=230,
        value=int(current_vals["plas"]),
        step=1,
        help="Normal fasting glucose: 70–99 mg/dL. Pre-diabetes: 100–125 mg/dL. Diabetes: >=126 mg/dL."
    )

    blood_pressure = st.slider(
        "Diastolic Blood Pressure (mm Hg)",
        min_value=40,
        max_value=130,
        value=int(current_vals["pres"]),
        step=1,
        help="Resting diastolic blood pressure. Ideal healthy adult range: 60–80 mm Hg."
    )

    insulin = st.slider(
        "2-Hour Serum Insulin (µU/mL)",
        min_value=10,
        max_value=600,
        value=int(current_vals["insu"]),
        step=5,
        help="Normal 2-hour post-glucose serum insulin is typically under 166 µU/mL."
    )

    skin_thickness = st.slider(
        "Triceps Skinfold Thickness (mm)",
        min_value=5,
        max_value=75,
        value=int(current_vals["skin"]),
        step=1,
        help="Measurement used to estimate subcutaneous body fat."
    )

with col2:
    st.markdown("##### 👤 Demographic & Physical")
    bmi = st.slider(
        "Body Mass Index - BMI (kg/m²)",
        min_value=15.0,
        max_value=55.0,
        value=float(current_vals["mass"]),
        step=0.1,
        help="BMI: Normal = 18.5–24.9, Overweight = 25.0–29.9, Obese = 30.0+."
    )

    age = st.slider(
        "Age (Years)",
        min_value=21,
        max_value=85,
        value=int(current_vals["age"]),
        step=1,
        help="Age in years. Adult dataset cohort begins at age 21."
    )

    pregnancies = st.slider(
        "Number of Pregnancies",
        min_value=0,
        max_value=17,
        value=int(current_vals["preg"]),
        step=1,
        help="Number of times pregnant (gestational history)."
    )

    pedigree = st.slider(
        "Diabetes Pedigree Function (Genetic History Score)",
        min_value=0.05,
        max_value=2.50,
        value=float(current_vals["pedi"]),
        step=0.01,
        help="Score quantifying genetic diabetes risk based on medical history of biological relatives."
    )

st.markdown("---")

# --- Predict Action ---
predict_btn = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

if predict_btn:
    # 1. Attempt model loading with robust error handling
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("❌ The model file ('model.joblib') is missing. Please run `python train.py` first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error while loading the model: {e}")
        st.stop()

    # 2. Input validation
    if glucose < 50 or blood_pressure < 40 or bmi < 15.0 or age < 21:
        st.warning("⚠️ One or more input values are outside medically plausible ranges. Please review your entries.")
        st.stop()

    # 3. Create DataFrame in strict feature order
    input_row = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]],
        columns=FEATURES
    )

    try:
        # 4. Predict probability
        probs = model.predict_proba(input_row)[0]
        prob_positive = probs[1]
        prob_negative = probs[0]
        prediction_class = model.predict(input_row)[0]

        # 5. Determine Risk Stratification Level
        if prob_positive < 0.30:
            risk_label = "Low Risk"
            risk_color = "normal"
            alert_func = st.success
            alert_msg = "The model assesses a low probability of diabetes based on the provided clinical features."
        elif prob_positive < 0.60:
            risk_label = "Moderate Risk"
            risk_color = "off"
            alert_func = st.warning
            alert_msg = "The model assesses moderate risk. Certain indicators (such as glucose or BMI) are elevated."
        else:
            risk_label = "Elevated / High Risk"
            risk_color = "inverse"
            alert_func = st.error
            alert_msg = "The model assesses high risk. Key diagnostic values suggest strong resemblance to positive cohorts."

        # 6. Display results
        st.markdown("### 📊 Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Estimated Risk of Diabetes",
                value=f"{prob_positive:.1%}",
                delta=f"{risk_label}"
            )

        with res_col2:
            st.metric(
                label="Probability of Healthy / Negative",
                value=f"{prob_negative:.1%}"
            )

        with res_col3:
            st.metric(
                label="Primary Model Classification",
                value="Positive" if prediction_class == 1 else "Negative"
            )

        # Visual progress meter
        st.progress(float(prob_positive), text=f"Risk Probability Meter: {prob_positive:.1%}")
        alert_func(alert_msg)

        # Comparative breakdown table
        with st.expander("🔍 View Submitted Input Summary & Reference Ranges"):
            summary_data = {
                "Feature": [
                    "Fasting Glucose",
                    "Diastolic BP",
                    "Body Mass Index (BMI)",
                    "Age",
                    "Pregnancies",
                    "Serum Insulin",
                    "Skinfold Thickness",
                    "Genetic Pedigree Score"
                ],
                "Your Value": [
                    f"{glucose} mg/dL",
                    f"{blood_pressure} mm Hg",
                    f"{bmi:.1f} kg/m²",
                    f"{age} years",
                    f"{pregnancies}",
                    f"{insulin} µU/mL",
                    f"{skin_thickness} mm",
                    f"{pedigree:.2f}"
                ],
                "Healthy Reference Benchmark": [
                    "70 – 99 mg/dL",
                    "60 – 80 mm Hg",
                    "18.5 – 24.9 kg/m²",
                    "N/A",
                    "N/A",
                    "15 – 100 µU/mL",
                    "10 – 30 mm",
                    "< 0.50"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ An error occurred while computing the prediction: {e}")

# --- Educational Caption (Mandatory Requirement) ---
st.caption(
    "A learning exercise. It describes an educational model trained on historical health indicators "
    "from the Pima Indians Diabetes Dataset and is not medical advice, diagnosis, or clinical guidance."
)

# --- Model & Data Methodology Details ---
with st.expander("ℹ️ About This Project & Model Architecture"):
    st.markdown(
        """
        - **Pipeline Workflow:** Input Features ➔ Median Missing-Value Imputation ➔ Standard Scaler ➔ Gradient Boosting Classifier
        - **Model Persistence:** Serialized with `joblib` and loaded via Streamlit's `@st.cache_resource` for zero-lag caching.
        - **Dataset Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) via OpenML.
        - **Evaluation:** Model achieved **0.8306 ROC-AUC** on stratified hold-out test set with 5-fold cross-validation.
        - **Deployment Ready:** Configured for one-click deployment to Streamlit Community Cloud.
        """
    )
