"""
test_pipeline.py
================
Unit & Integration Tests for Crop Yield Prediction Machine Learning Pipeline,
Agronomic Suggestion Engine, Crop Recommendation, and FastMCP Server Tools.
Does NOT require an LLM API key to run.
"""

import os
import sys
import json
import warnings
import joblib
import pandas as pd
import numpy as np

# Suppress sklearn unpickle warnings during tests
warnings.filterwarnings("ignore")

# Ensure base directory is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import farm_logic
import chatbot


def test_artifacts_exist():
    """Verify all model artifacts and metadata files are present."""
    assert os.path.exists("models/linear_regression_pipeline.joblib"), "LR pipeline missing!"
    assert os.path.exists("models/random_forest_pipeline.joblib"), "RF pipeline missing!"
    assert os.path.exists("models/model_metadata.json"), "Model metadata JSON missing!"
    assert os.path.exists("models/test_predictions_sample.csv"), "Test predictions sample CSV missing!"
    print("[PASS] All artifact files exist.")


def test_metadata_contents():
    """Verify required metadata fields and metric keys."""
    with open("models/model_metadata.json", "r") as f:
        meta = json.load(f)
    
    assert "evaluation_metrics" in meta
    assert "Linear_Regression" in meta["evaluation_metrics"]
    assert "Random_Forest" in meta["evaluation_metrics"]
    for model_key in ["Linear_Regression", "Random_Forest"]:
        for metric_name in ["MAE", "MSE", "RMSE", "R2"]:
            assert metric_name in meta["evaluation_metrics"][model_key]
            assert isinstance(meta["evaluation_metrics"][model_key][metric_name], (int, float))
    
    assert meta["yield_units"] == "Tonnes per Hectare (t/ha)"
    assert len(meta["crops"]) > 50
    print(f"[PASS] Metadata contains valid metrics for both models (Random Forest R²: {meta['evaluation_metrics']['Random_Forest']['R2']}).")


def test_model_inference():
    """Verify that both pipelines successfully predict on user inputs without shape or encoding mismatch."""
    lr_pipe = joblib.load("models/linear_regression_pipeline.joblib")
    rf_pipe = joblib.load("models/random_forest_pipeline.joblib")

    test_input = pd.DataFrame([{
        "rainfall": 1200.0,
        "temperature": 27.0,
        "humidity": 70.0,
        "pH": 6.5,
        "N": 80.0,
        "P": 40.0,
        "K": 30.0,
        "crop_type": "Rice"
    }])

    pred_lr = lr_pipe.predict(test_input)
    pred_rf = rf_pipe.predict(test_input)

    assert len(pred_lr) == 1 and not np.isnan(pred_lr[0])
    assert len(pred_rf) == 1 and not np.isnan(pred_rf[0])
    assert pred_rf[0] > 0, "Predicted yield should be positive for realistic inputs"
    print(f"[PASS] Inference test passed! Rice Yield -> LR: {pred_lr[0]:.2f} t/ha, RF: {pred_rf[0]:.2f} t/ha")


def test_unseen_category_handling():
    """Verify that OneHotEncoder handles unknown crop types gracefully without throwing exceptions."""
    rf_pipe = joblib.load("models/random_forest_pipeline.joblib")
    novel_input = pd.DataFrame([{
        "rainfall": 900.0,
        "temperature": 25.0,
        "humidity": 65.0,
        "pH": 6.8,
        "N": 70.0,
        "P": 35.0,
        "K": 25.0,
        "crop_type": "UnknownNovelCrop"
    }])
    pred_novel = rf_pipe.predict(novel_input)
    assert len(pred_novel) == 1 and not np.isnan(pred_novel[0])
    print(f"[PASS] Novel categorical handling passed! Output: {pred_novel[0]:.2f} t/ha")


def test_recommend_crops():
    """Verify crop recommendation returns top N crops ranked descending by predicted yield."""
    top5 = farm_logic.recommend_crops(
        rainfall=1150.0,
        temperature=26.5,
        humidity=72.0,
        pH=6.5,
        N=78.0,
        P=35.0,
        K=28.0,
        top_n=5
    )
    assert isinstance(top5, list), "Expected list of recommendations"
    assert len(top5) == 5, f"Expected 5 crops, got {len(top5)}"
    
    # Verify ranking order (descending)
    for i in range(len(top5) - 1):
        assert top5[i]["predicted_yield"] >= top5[i+1]["predicted_yield"], "Recommendations should be sorted descending"
        assert "crop" in top5[i]
        assert top5[i]["units"] == "t/ha"
        assert top5[i]["predicted_yield"] >= 0.0

    print(f"[PASS] Crop Recommendation test passed! Top 3: {[c['crop'] for c in top5[:3]]}")


def test_suggest_improvements():
    """Verify that yield improvement testing evaluates soil factors and ranks gains."""
    sugg = farm_logic.suggest_improvements(
        crop="Rice",
        rainfall=1150.0,
        temperature=26.5,
        humidity=72.0,
        pH=5.0,  # Acidic, should recommend lime
        N=40.0,  # Deficient N, should recommend N boost
        P=20.0,  # Deficient P
        K=20.0
    )
    assert isinstance(sugg, dict)
    assert "crop" in sugg and sugg["crop"] == "Rice"
    assert "baseline_yield" in sugg
    assert "suggestions" in sugg
    assert "soil_advisory" in sugg
    assert "weather_tips" in sugg
    assert "disclaimer" in sugg
    assert len(sugg["weather_tips"]) > 0

    # If suggestions found, verify sorted descending by yield gain
    suggestions = sugg["suggestions"]
    for i in range(len(suggestions) - 1):
        assert suggestions[i]["yield_gain"] >= suggestions[i+1]["yield_gain"]
    print(f"[PASS] Yield improvement suggestion test passed! Generated {len(suggestions)} ranked suggestions.")


def test_soil_advisory():
    """Verify crop-specific and generic nutrient advisory rules."""
    # Test 1: Low values for Rice
    adv_low = farm_logic.get_soil_advisory("Rice", pH=4.8, N=35.0, P=15.0, K=15.0)
    assert "acidic" in adv_low["statuses"]["pH"]
    assert adv_low["statuses"]["N"] == "low"
    assert adv_low["statuses"]["P"] == "low"
    assert adv_low["statuses"]["K"] == "low"
    assert len(adv_low["notes"]) == 4

    # Test 2: Optimal values for Rice
    adv_opt = farm_logic.get_soil_advisory("Rice", pH=6.5, N=95.0, P=40.0, K=40.0)
    assert adv_opt["statuses"]["pH"] == "balanced"
    assert adv_opt["statuses"]["N"] == "balanced"
    assert adv_opt["statuses"]["P"] == "balanced"
    assert adv_opt["statuses"]["K"] == "balanced"

    # Test 3: Generic crop fallback
    adv_gen = farm_logic.get_soil_advisory("Other Cereals", pH=6.5, N=80.0, P=40.0, K=30.0)
    assert adv_gen["statuses"]["pH"] == "balanced"
    print("[PASS] Soil nutrient advisory rules test passed!")


def test_what_if_comparison():
    """Verify what-if comparison between current and improved soil."""
    wi = farm_logic.compare_what_if(
        crop="Wheat",
        rainfall=900.0,
        temperature=24.0,
        humidity=65.0,
        cur_pH=5.5,
        cur_N=50.0,
        cur_P=25.0,
        cur_K=20.0,
        imp_pH=6.5,
        imp_N=120.0,
        imp_P=45.0,
        imp_K=40.0
    )
    assert "current_yield" in wi
    assert "improved_yield" in wi
    assert "difference" in wi
    assert wi["difference"] == round(wi["improved_yield"] - wi["current_yield"], 2)
    print(f"[PASS] What-If comparison test passed! Delta: {wi['difference']:+.2f} t/ha")


def test_chatbot_unconfigured_fallback():
    """Ensure chatbot returns a clean setup guide without raising an unhandled exception when unconfigured."""
    os.environ["TEST_UNCONFIGURED"] = "1"
    try:
        resp = chatbot.ask_farmer_assistant("Hello, how are you?")
        assert isinstance(resp, str)
        assert "Kisan Mitra" in resp or "Farmer Assistant" in resp
        print("[PASS] Chatbot unconfigured fallback test passed!")
    finally:
        os.environ.pop("TEST_UNCONFIGURED", None)


if __name__ == "__main__":
    print("Running automated pipeline verification tests...")
    test_artifacts_exist()
    test_metadata_contents()
    test_model_inference()
    test_unseen_category_handling()
    test_recommend_crops()
    test_suggest_improvements()
    test_soil_advisory()
    test_what_if_comparison()
    test_chatbot_unconfigured_fallback()
    print("\n=============================================")
    print("All 9 automated verification tests PASSED successfully!")
    print("=============================================")
