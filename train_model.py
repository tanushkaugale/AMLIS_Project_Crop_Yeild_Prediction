"""
train_model.py
==============
Crop Yield Prediction using Weather and Soil Data
Training and Evaluation Pipeline for Linear Regression and Random Forest Regressor.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data"
MODELS_DIR = "models"
RANDOM_STATE = 42

def load_and_preprocess_data():
    """
    Load raw historical datasets and merge them into a unified feature set.
    """
    print("[1/5] Loading historical datasets...")
    crop_path = os.path.join(DATA_DIR, "crop_yield.csv")
    soil_path = os.path.join(DATA_DIR, "state_soil_data.csv")
    weather_path = os.path.join(DATA_DIR, "state_weather_data_1997_2020.csv")

    if not (os.path.exists(crop_path) and os.path.exists(soil_path) and os.path.exists(weather_path)):
        raise FileNotFoundError(
            f"Missing required CSV files in '{DATA_DIR}/'. Expected: crop_yield.csv, state_soil_data.csv, state_weather_data_1997_2020.csv"
        )

    df_yield = pd.read_csv(crop_path)
    df_soil = pd.read_csv(soil_path)
    df_weather = pd.read_csv(weather_path)

    # Standardize string keys
    df_yield["state"] = df_yield["state"].astype(str).str.strip()
    df_yield["crop"] = df_yield["crop"].astype(str).str.strip()
    df_soil["state"] = df_soil["state"].astype(str).str.strip()
    df_weather["state"] = df_weather["state"].astype(str).str.strip()

    # Merge on state and year
    print("[2/5] Merging soil, weather, and yield records...")
    merged = df_yield.merge(df_soil, on="state", how="inner")
    merged = merged.merge(df_weather, on=["state", "year"], how="inner")

    # Document target variable & units:
    # Most crops in Indian agricultural statistics report Yield as Tonnes per Hectare (Production in Tonnes / Area in Hectares).
    # 'Coconut' in Indian national agricultural reports is recorded in 'Thousand Nuts per Hectare', producing values up to 21,000.
    # To maintain consistent yield units (Tonnes/Hectare) throughout training, evaluation, and prediction,
    # Coconut records are isolated from the Tonnes/Hectare regression benchmark.
    merged_clean = merged[merged["crop"] != "Coconut"].copy()

    merged_clean = merged_clean.rename(columns={
        "total_rainfall_mm": "rainfall",
        "avg_temp_c": "temperature",
        "avg_humidity_percent": "humidity",
        "crop": "crop_type"
    })

    required_features = ["rainfall", "temperature", "humidity", "pH", "N", "P", "K", "crop_type"]
    target = "yield"

    # Drop any null values if present
    data_clean = merged_clean.dropna(subset=required_features + [target]).copy()

    # Save cleaned merged dataset for easy inspection & Streamlit EDA
    cleaned_dataset_path = os.path.join(DATA_DIR, "processed_crop_yield_data.csv")
    data_clean[required_features + [target, "state", "year"]].to_csv(cleaned_dataset_path, index=False)
    print(f"Cleaned dataset saved to: {cleaned_dataset_path} ({len(data_clean)} records)")

    X = data_clean[required_features]
    y = data_clean[target]

    return X, y, data_clean

def train_and_evaluate():
    """
    Split data, train both models, evaluate on held-out test data, and persist artifacts.
    """
    X, y, df = load_and_preprocess_data()

    print("[3/5] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Define preprocessing pipeline
    # Numerical features are standardized; categorical crop types are One-Hot Encoded.
    num_features = ["rainfall", "temperature", "humidity", "pH", "N", "P", "K"]
    cat_features = ["crop_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="drop"
    )

    print("[4/5] Training Linear Regression and Random Forest Regressor...")
    # 1. Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    # 2. Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    print("[5/5] Evaluating on held-out test set...")
    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    def compute_metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }

    lr_metrics = compute_metrics(y_test, lr_preds)
    rf_metrics = compute_metrics(y_test, rf_preds)

    print("\n" + "="*50)
    print("EVALUATION RESULTS (Held-Out Test Set, 20%):")
    print("="*50)
    print(f"Linear Regression:       MAE={lr_metrics['MAE']}, MSE={lr_metrics['MSE']}, RMSE={lr_metrics['RMSE']}, R²={lr_metrics['R2']}")
    print(f"Random Forest Regressor: MAE={rf_metrics['MAE']}, MSE={rf_metrics['MSE']}, RMSE={rf_metrics['RMSE']}, R²={rf_metrics['R2']}")
    print("="*50)

    # Model comparison conclusion
    better_model_name = "Random Forest Regressor" if rf_metrics["R2"] > lr_metrics["R2"] else "Linear Regression"
    print(f"Selected Champion Model: {better_model_name} (Higher R², Lower RMSE/MAE)")

    # Save test sample predictions for Streamlit actual-vs-predicted visualization
    sample_size = min(300, len(y_test))
    test_eval_df = X_test.iloc[:sample_size].copy()
    test_eval_df["Actual_Yield"] = y_test.iloc[:sample_size].values
    test_eval_df["Predicted_LR"] = lr_preds[:sample_size]
    test_eval_df["Predicted_RF"] = rf_preds[:sample_size]

    os.makedirs(MODELS_DIR, exist_ok=True)
    test_eval_path = os.path.join(MODELS_DIR, "test_predictions_sample.csv")
    test_eval_df.to_csv(test_eval_path, index=False)

    # Save artifacts
    joblib.dump(lr_pipeline, os.path.join(MODELS_DIR, "linear_regression_pipeline.joblib"))
    joblib.dump(rf_pipeline, os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"))

    # Save metadata and crop list
    crop_list = sorted(df["crop_type"].unique().tolist())
    metadata = {
        "dataset_name": "Indian Agricultural Crop Yield with Soil & Meteorological Data (1997-2020)",
        "source": "Kaggle (anshumish/crop-yield-data-with-soil-and-weather-dataset) & Ministry of Agriculture & Farmers Welfare, India",
        "target_variable": "historical crop yield",
        "yield_units": "Tonnes per Hectare (t/ha)",
        "features": {
            "rainfall": "Total Annual Rainfall (mm)",
            "temperature": "Average Temperature (°C)",
            "humidity": "Average Relative Humidity (%)",
            "pH": "Soil pH Level",
            "N": "Nitrogen Ratio in Soil (kg/ha)",
            "P": "Phosphorus Ratio in Soil (kg/ha)",
            "K": "Potassium Ratio in Soil (kg/ha)",
            "crop_type": "Crop Category Name"
        },
        "crops": crop_list,
        "evaluation_metrics": {
            "Linear_Regression": lr_metrics,
            "Random_Forest": rf_metrics
        },
        "better_model": better_model_name,
        "test_sample_count": len(y_test),
        "train_sample_count": len(X_train),
        "random_state": RANDOM_STATE
    }

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nAll models, evaluation metadata, and test artifacts saved to '{MODELS_DIR}/'.")
    return metadata

if __name__ == "__main__":
    train_and_evaluate()
