"""
Train and export production-grade Diabetes Risk Prediction Model.
Dataset: Pima Indians Diabetes Dataset (National Institute of Diabetes and Digestive and Kidney Diseases).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

def train_and_save_model():
    print("1. Fetching dataset...")
    data = fetch_openml('diabetes', version=1, as_frame=True)
    df = data.frame.copy()

    # Column mapping & feature names
    feature_names = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "pedigree",
        "age"
    ]
    df.columns = feature_names + ["target"]

    # Target: tested_positive -> 1, tested_negative -> 0
    y = (df["target"] == "tested_positive").astype(int)
    X = df[feature_names].copy()

    # Biological zeroes that are actually missing values
    zero_cols = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    for col in zero_cols:
        X[col] = X[col].replace(0, np.nan)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    # Candidate models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    best_name = None
    best_pipeline = None
    best_score = 0.0
    best_metrics = {}

    for name, clf in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
        print(f"[{name}] CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | Test Acc: {acc:.4f} | Test ROC-AUC: {auc:.4f} | Test F1: {f1:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_name = name
            best_pipeline = pipeline
            best_metrics = {
                "model_name": name,
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4),
                "f1_score": round(f1, 4),
                "features": feature_names,
                "n_train": len(X_train),
                "n_test": len(X_test)
            }

    print(f"\nBest Model selected: {best_name} with Test ROC-AUC: {best_score:.4f}")

    # Retrain best pipeline on entire dataset to maximize production utility
    full_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", best_pipeline.named_steps["classifier"])
    ])
    full_pipeline.fit(X, y)

    # Save model artifact
    model_path = "model.joblib"
    joblib.dump(full_pipeline, model_path)
    print(f"Model successfully saved to '{model_path}'.")

    # Save metadata dictionary
    meta_path = "model_meta.joblib"
    joblib.dump(best_metrics, meta_path)
    print(f"Model metadata saved to '{meta_path}'.")

    # Also save a sample CSV of raw data for user reference/inspection in the app
    sample_df = df.head(50)
    sample_df.to_csv("sample_data.csv", index=False)
    print("Saved 'sample_data.csv'.")

    return full_pipeline, best_metrics

if __name__ == "__main__":
    train_and_save_model()
