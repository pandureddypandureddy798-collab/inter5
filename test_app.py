"""
Automated Unit and Integration Tests for Streamlit ML Web App
Validates model integrity, prediction accuracy, pipeline robustness, and Streamlit App rendering.
"""

import os
import pytest
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

MODEL_FILE = "model.joblib"
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

def test_model_file_exists():
    """Verify that model.joblib exists in the project root."""
    assert os.path.exists(MODEL_FILE), f"Missing '{MODEL_FILE}'. Run train.py first."

def test_model_loading_and_type():
    """Verify that model loads cleanly and has expected pipeline stages."""
    model = joblib.load(MODEL_FILE)
    assert hasattr(model, "predict"), "Loaded model has no predict method"
    assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba method"
    assert hasattr(model, "named_steps"), "Loaded model is not a scikit-learn Pipeline"
    assert "imputer" in model.named_steps
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps

def test_prediction_output_format_and_probabilities():
    """Verify predict_proba returns normalized probabilities for standard input."""
    model = joblib.load(MODEL_FILE)
    sample_input = pd.DataFrame([{
        "pregnancies": 2,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 28.5,
        "pedigree": 0.45,
        "age": 35
    }], columns=FEATURES)

    proba = model.predict_proba(sample_input)
    assert proba.shape == (1, 2), f"Expected shape (1, 2), got {proba.shape}"
    assert np.isclose(np.sum(proba[0]), 1.0, atol=1e-5), "Probabilities must sum to 1"
    assert 0.0 <= proba[0][0] <= 1.0, "Probability out of [0, 1] bounds"
    assert 0.0 <= proba[0][1] <= 1.0, "Probability out of [0, 1] bounds"

    pred = model.predict(sample_input)
    assert pred[0] in [0, 1], f"Prediction class must be 0 or 1, got {pred[0]}"

def test_handling_missing_biological_values():
    """Verify pipeline handles NaN values through median imputation without crashing."""
    model = joblib.load(MODEL_FILE)
    nan_input = pd.DataFrame([{
        "pregnancies": 0,
        "glucose": np.nan,  # Missing glucose
        "blood_pressure": np.nan,  # Missing BP
        "skin_thickness": 20,
        "insulin": np.nan,
        "bmi": 25.0,
        "pedigree": 0.35,
        "age": 28
    }], columns=FEATURES)

    # Should not raise any ValueError
    proba = model.predict_proba(nan_input)
    assert proba.shape == (1, 2)
    assert not np.isnan(proba).any(), "Probabilities should not contain NaN"

def test_risk_directionality():
    """Verify high-risk profile produces higher predicted risk than low-risk profile."""
    model = joblib.load(MODEL_FILE)

    low_risk = pd.DataFrame([{
        "pregnancies": 0,
        "glucose": 80,
        "blood_pressure": 65,
        "skin_thickness": 18,
        "insulin": 45,
        "bmi": 20.5,
        "pedigree": 0.15,
        "age": 22
    }], columns=FEATURES)

    high_risk = pd.DataFrame([{
        "pregnancies": 7,
        "glucose": 185,
        "blood_pressure": 92,
        "skin_thickness": 42,
        "insulin": 260,
        "bmi": 38.5,
        "pedigree": 1.25,
        "age": 55
    }], columns=FEATURES)

    prob_low = model.predict_proba(low_risk)[0][1]
    prob_high = model.predict_proba(high_risk)[0][1]

    assert prob_high > prob_low, f"High risk prob ({prob_high:.3f}) should exceed low risk ({prob_low:.3f})"

def test_streamlit_app_loads_and_runs():
    """Simulate Streamlit app run using AppTest to ensure no runtime errors."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run(timeout=20)
    assert not at.exception, f"App raised an exception on load: {at.exception}"
    assert len(at.title) >= 1, "App missing title"
    assert "Diabetes" in at.title[0].value

def test_streamlit_app_prediction_button_interaction():
    """Simulate user clicking Predict button in the Streamlit app."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run(timeout=20)

    # Find the predict button and click it
    assert len(at.button) >= 1, "App has no button"
    predict_button = at.button[0]
    predict_button.click().run(timeout=20)

    assert not at.exception, f"App raised exception upon clicking predict: {at.exception}"
    # Verify metric cards are rendered
    assert len(at.metric) >= 1, "Metric card not displayed after clicking Predict"
    assert "Estimated Risk" in at.metric[0].label
