"""
farm_logic.py
=============
Shared Agronomic & Machine Learning Inference Logic for Crop Yield Prediction.
Reused across Streamlit UI (app.py) and Model Context Protocol Server (mcp_server.py).
Adheres strictly to the pre-trained Random Forest and Linear Regression pipelines.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Base paths (resolves correctly whether running from root or child directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
LR_FILE = os.path.join(MODELS_DIR, "linear_regression_pipeline.joblib")
RF_FILE = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
TEST_PRED_FILE = os.path.join(MODELS_DIR, "test_predictions_sample.csv")

# ---------------------------------------------------------------------------
# Singleton Cache for Model Artifacts
# ---------------------------------------------------------------------------
_METADATA: Optional[Dict[str, Any]] = None
_LR_MODEL = None
_RF_MODEL = None
_TEST_PREDS: Optional[pd.DataFrame] = None


def load_artifacts() -> Tuple[Dict[str, Any], Any, Any, Optional[pd.DataFrame]]:
    """
    Load pre-trained models and metadata JSON. Cached in memory.
    """
    global _METADATA, _LR_MODEL, _RF_MODEL, _TEST_PREDS
    if _METADATA is None:
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Model metadata not found at: {METADATA_FILE}")
        with open(METADATA_FILE, "r") as f:
            _METADATA = json.load(f)
        _LR_MODEL = joblib.load(LR_FILE)
        _RF_MODEL = joblib.load(RF_FILE)
        if os.path.exists(TEST_PRED_FILE):
            _TEST_PREDS = pd.read_csv(TEST_PRED_FILE)
    return _METADATA, _LR_MODEL, _RF_MODEL, _TEST_PREDS


# ---------------------------------------------------------------------------
# Crop-Specific Agronomic Benchmark Ranges (N, P, K in kg/ha, Soil pH)
# Sourced from Indian Council of Agricultural Research (ICAR) guidelines.
# ---------------------------------------------------------------------------
CROP_AGRONOMIC_TARGETS: Dict[str, Dict[str, Any]] = {
    "Rice": {
        "N": (80.0, 120.0),
        "P": (30.0, 50.0),
        "K": (30.0, 50.0),
        "pH": (5.5, 7.0),
        "desc": "Paddy requires moderate-to-high nitrogen in split doses with balanced phosphate for root anchorage in puddled soil."
    },
    "Wheat": {
        "N": (100.0, 150.0),
        "P": (40.0, 60.0),
        "K": (30.0, 50.0),
        "pH": (6.0, 7.5),
        "desc": "Rabi wheat needs solid early nitrogen and basal phosphate for tillering and crown root initiation."
    },
    "Maize": {
        "N": (100.0, 140.0),
        "P": (40.0, 60.0),
        "K": (30.0, 50.0),
        "pH": (5.8, 7.2),
        "desc": "Maize is a heavy feeder that responds well to nitrogen top-dressing at knee-high and tasseling stages."
    },
    "Cotton(lint)": {
        "N": (90.0, 130.0),
        "P": (40.0, 60.0),
        "K": (40.0, 60.0),
        "pH": (6.0, 7.5),
        "desc": "Cotton demands balanced potassium to support boll weight and prevent premature leaf shedding."
    },
    "Sugarcane": {
        "N": (120.0, 180.0),
        "P": (50.0, 75.0),
        "K": (50.0, 80.0),
        "pH": (6.0, 7.8),
        "desc": "Sugarcane has very high biomass output, necessitating high nitrogen and potassium for cane sucrose accumulation."
    },
    "Groundnut": {
        "N": (20.0, 40.0),
        "P": (40.0, 60.0),
        "K": (30.0, 50.0),
        "pH": (6.0, 7.0),
        "desc": "Leguminous groundnut fixes biological nitrogen; emphasis is on phosphorus and calcium/gypsum for pod development."
    },
    "Gram": {
        "N": (20.0, 35.0),
        "P": (35.0, 55.0),
        "K": (20.0, 40.0),
        "pH": (6.0, 7.5),
        "desc": "Chickpea (Gram) requires minimal starter nitrogen but benefits from seed rhizobium inoculation and phosphorus."
    },
    "Potato": {
        "N": (120.0, 160.0),
        "P": (60.0, 90.0),
        "K": (80.0, 120.0),
        "pH": (5.2, 6.5),
        "desc": "Potato requires abundant potassium for tuber bulking and slightly acidic soil to deter potato scab."
    }
}

DEFAULT_AGRONOMIC_TARGET = {
    "N": (60.0, 120.0),
    "P": (25.0, 60.0),
    "K": (20.0, 50.0),
    "pH": (5.8, 7.2),
    "desc": "Standard balanced agronomic guidelines for Indian cropping systems."
}

MANDATORY_DISCLAIMER = (
    "Educational guidance and machine learning estimate based on historical regional datasets (1997-2020). "
    "Not a chemical fertilizer prescription. Always verify with physical soil testing from an accredited laboratory "
    "and consult your local Krishi Vigyan Kendra (KVK) or State Agricultural Extension Officer."
)


def get_crop_target(crop: str) -> Dict[str, Any]:
    """Retrieve ideal nutrient and pH ranges for a crop, or return general defaults."""
    return CROP_AGRONOMIC_TARGETS.get(crop, DEFAULT_AGRONOMIC_TARGET)


def list_supported_crops() -> List[str]:
    """Return the list of all 54 supported crop varieties."""
    meta, _, _, _ = load_artifacts()
    return meta.get("crops", [])


# ---------------------------------------------------------------------------
# 1. Yield Prediction
# ---------------------------------------------------------------------------
def predict_yield(
    crop: str,
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float
) -> Dict[str, Any]:
    """
    Predict crop yield using the trained Random Forest and Linear Regression pipelines.
    Returns predicted yield in t/ha with +/- MAE confidence margin.
    """
    meta, lr_pipe, rf_pipe, _ = load_artifacts()
    rf_mae = meta["evaluation_metrics"]["Random_Forest"]["MAE"]
    lr_mae = meta["evaluation_metrics"]["Linear_Regression"]["MAE"]

    input_df = pd.DataFrame([{
        "rainfall": float(rainfall),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "pH": float(pH),
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "crop_type": str(crop)
    }])

    pred_rf = float(rf_pipe.predict(input_df)[0])
    pred_lr = float(lr_pipe.predict(input_df)[0])

    pred_rf_clamped = max(0.0, pred_rf)
    pred_lr_clamped = max(0.0, pred_lr)

    return {
        "crop": crop,
        "predicted_yield": round(pred_rf_clamped, 2),
        "predicted_yield_rf": round(pred_rf_clamped, 2),
        "predicted_yield_lr": round(pred_lr_clamped, 2),
        "mae": round(rf_mae, 2),
        "yield_range_min": max(0.0, round(pred_rf_clamped - rf_mae, 2)),
        "yield_range_max": round(pred_rf_clamped + rf_mae, 2),
        "units": "Tonnes per Hectare (t/ha)",
        "inputs": {
            "crop": crop,
            "rainfall": rainfall,
            "temperature": temperature,
            "humidity": humidity,
            "pH": pH,
            "N": N,
            "P": P,
            "K": K
        },
        "disclaimer": MANDATORY_DISCLAIMER
    }


# ---------------------------------------------------------------------------
# 2. Crop Recommendation
# ---------------------------------------------------------------------------
def recommend_crops(
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Simulate predicted yields across all supported crops for the given weather & soil.
    Returns the top N crops ranked by expected yield in descending order.
    """
    meta, _, rf_pipe, _ = load_artifacts()
    all_crops = meta.get("crops", [])

    rows = [{
        "rainfall": float(rainfall),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "pH": float(pH),
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "crop_type": crop
    } for crop in all_crops]

    eval_df = pd.DataFrame(rows)
    preds = rf_pipe.predict(eval_df)

    scored = []
    for crop, pred in zip(all_crops, preds):
        scored.append({
            "crop": crop,
            "predicted_yield": round(max(0.0, float(pred)), 2),
            "units": "t/ha"
        })

    # Sort descending by predicted yield
    scored.sort(key=lambda x: x["predicted_yield"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# 3. Rule-Based Crop-Specific Soil & Fertilizer Advisory
# ---------------------------------------------------------------------------
def get_soil_advisory(
    crop: str,
    pH: float,
    N: float,
    P: float,
    K: float
) -> Dict[str, Any]:
    """
    Evaluates soil N, P, K, and pH against crop-specific ideal agronomic ranges.
    Returns categorized statuses (low, balanced, high) with actionable amendment notes.
    """
    target = get_crop_target(crop)
    notes = []
    status_summary = {}

    # Soil pH Check
    ph_min, ph_max = target["pH"]
    if pH < ph_min:
        status_summary["pH"] = "low (acidic)"
        notes.append(
            f"Soil pH ({pH:.1f}) is acidic for {crop} (ideal: {ph_min:.1f}–{ph_max:.1f}). "
            "Nutrient availability like phosphorus is restricted. Consider applying agricultural lime (calcium carbonate) "
            "or dolomite at 200–500 kg/ha to gradually neutralize soil acidity."
        )
    elif pH > ph_max:
        status_summary["pH"] = "high (alkaline)"
        notes.append(
            f"Soil pH ({pH:.1f}) is alkaline for {crop} (ideal: {ph_min:.1f}–{ph_max:.1f}). "
            "Iron, Zinc, and Manganese availability may be reduced. Incorporate farmyard manure (FYM) or gypsum to buffer pH."
        )
    else:
        status_summary["pH"] = "balanced"
        notes.append(
            f"Soil pH ({pH:.1f}) is in the optimal range ({ph_min:.1f}–{ph_max:.1f}) for {crop}."
        )

    # Nitrogen (N) Check
    n_min, n_max = target["N"]
    if N < n_min:
        status_summary["N"] = "low"
        notes.append(
            f"Nitrogen ({N:.0f} kg/ha) is below recommended levels for {crop} ({n_min:.0f}–{n_max:.0f} kg/ha). "
            "Leaf yellowing or stunted tillering may occur. Consider applying Urea in split doses: basal at sowing and top-dressing at vegetative stage."
        )
    elif N > n_max:
        status_summary["N"] = "high"
        notes.append(
            f"Nitrogen ({N:.0f} kg/ha) exceeds optimal requirements for {crop} ({n_min:.0f}–{n_max:.0f} kg/ha). "
            "Excessive nitrogen can lead to vegetative overgrowth, crop lodging, and increased insect vulnerability. Avoid further N fertilizer."
        )
    else:
        status_summary["N"] = "balanced"
        notes.append(f"Nitrogen ({N:.0f} kg/ha) is well-balanced for {crop} ({n_min:.0f}–{n_max:.0f} kg/ha).")

    # Phosphorus (P) Check
    p_min, p_max = target["P"]
    if P < p_min:
        status_summary["P"] = "low"
        notes.append(
            f"Phosphorus ({P:.0f} kg/ha) is deficient for {crop} ({p_min:.0f}–{p_max:.0f} kg/ha). "
            "Crucial for early root elongation and flowering. Apply Single Super Phosphate (SSP) or DAP as a basal fertilizer during seedbed preparation."
        )
    elif P > p_max:
        status_summary["P"] = "high"
        notes.append(
            f"Phosphorus ({P:.0f} kg/ha) is abundant for {crop} ({p_min:.0f}–{p_max:.0f} kg/ha). Maintenance doses are sufficient."
        )
    else:
        status_summary["P"] = "balanced"
        notes.append(f"Phosphorus ({P:.0f} kg/ha) is optimal for {crop} root architecture and grain filling.")

    # Potassium (K) Check
    k_min, k_max = target["K"]
    if K < k_min:
        status_summary["K"] = "low"
        notes.append(
            f"Potassium ({K:.0f} kg/ha) is low for {crop} ({k_min:.0f}–{k_max:.0f} kg/ha). "
            "Potassium regulates water transpiration and disease resilience. Apply Muriate of Potash (MOP) before flowering or tuber filling."
        )
    elif K > k_max:
        status_summary["K"] = "high"
        notes.append(
            f"Potassium ({K:.0f} kg/ha) is high for {crop} ({k_min:.0f}–{k_max:.0f} kg/ha). No supplementary potash required."
        )
    else:
        status_summary["K"] = "balanced"
        notes.append(f"Potassium ({K:.0f} kg/ha) is sufficient for drought resistance and crop quality.")

    return {
        "crop": crop,
        "crop_description": target["desc"],
        "target_ranges": {
            "N": f"{n_min:.0f}–{n_max:.0f} kg/ha",
            "P": f"{p_min:.0f}–{p_max:.0f} kg/ha",
            "K": f"{k_min:.0f}–{k_max:.0f} kg/ha",
            "pH": f"{ph_min:.1f}–{ph_max:.1f}"
        },
        "statuses": status_summary,
        "notes": notes,
        "disclaimer": MANDATORY_DISCLAIMER
    }


# ---------------------------------------------------------------------------
# 4. Weather-Based Agronomic Tips
# ---------------------------------------------------------------------------
def get_weather_tips(
    rainfall: float,
    temperature: float,
    humidity: float
) -> List[str]:
    """
    Practical weather management tips based on current environmental inputs.
    """
    tips = []
    # Rainfall tips
    if rainfall < 600.0:
        tips.append("💧 **Low Rainfall Advisory:** Precipitation is below 600 mm. Implement mulching with crop residue to reduce evaporation, and arrange supplemental drip or sprinkler irrigation.")
    elif rainfall > 2200.0:
        tips.append("🌊 **Heavy Rainfall Advisory:** High annual precipitation (>2200 mm) detected. Ensure broadbed furrow or raised bed drainage channels to avoid root waterlogging and aeration loss.")
    else:
        tips.append("🌧️ **Adequate Rainfall:** Regional rainfall is within normal rainfed agriculture thresholds.")

    # Temperature tips
    if temperature > 34.0:
        tips.append("☀️ **Heat Stress Warning:** Temperatures exceed 34°C. Irrigate during early morning or late dusk to alleviate thermal shock and prevent floral shedding.")
    elif temperature < 16.0:
        tips.append("❄️ **Cool Weather Notice:** Mean temperature is below 16°C. Germination and seedling establishment may be slow; avoid cold water pooling.")
    else:
        tips.append("🌡️ **Favorable Temperature:** Ambient thermal conditions are conducive to healthy vegetative metabolism.")

    # Humidity tips
    if humidity > 80.0:
        tips.append("🌫️ **High Humidity Risk:** Relative humidity exceeds 80%. Humid microclimates accelerate foliar fungal and bacterial blights. Regularly inspect field borders for early pest/disease symptoms.")
    elif humidity < 40.0:
        tips.append("🌵 **Dry Air Notice:** Low humidity (<40%) elevates crop evapotranspiration demand. Monitor soil moisture closely.")
    else:
        tips.append("🍃 **Normal Humidity:** Atmospheric moisture level is moderate.")

    return tips


# ---------------------------------------------------------------------------
# 5. Yield Improvement Suggestion Engine (Optimization Simulation)
# ---------------------------------------------------------------------------
def suggest_improvements(
    crop: str,
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float
) -> Dict[str, Any]:
    """
    Systematically evaluates potential yield gains from realistic adjustments
    to farmer-controllable soil parameters (pH, N, P, K) while keeping immutable
    weather variables constant.

    Ranks suggestions by predicted yield gain (impact) and integrates crop-specific
    soil advisories, weather management tips, and academic disclaimers.
    """
    meta, _, rf_pipe, _ = load_artifacts()
    target = get_crop_target(crop)

    # 1. Baseline prediction
    base_pred_dict = predict_yield(crop, rainfall, temperature, humidity, pH, N, P, K)
    base_yield = base_pred_dict["predicted_yield"]

    # 2. Explore single parameter adjustments
    test_scenarios = []

    # Safe boundaries
    bounds = {
        "pH": (4.5, 8.5),
        "N": (10.0, 200.0),
        "P": (5.0, 120.0),
        "K": (5.0, 120.0)
    }

    # Ideal midpoints from target
    ideal_mid = {
        "pH": (target["pH"][0] + target["pH"][1]) / 2.0,
        "N": (target["N"][0] + target["N"][1]) / 2.0,
        "P": (target["P"][0] + target["P"][1]) / 2.0,
        "K": (target["K"][0] + target["K"][1]) / 2.0
    }

    current_soil = {"pH": pH, "N": N, "P": P, "K": K}

    # Test shifting each soil parameter toward ideal or small incremental adjustments
    param_labels = {
        "N": "Nitrogen (N)",
        "P": "Phosphorus (P)",
        "K": "Potassium (K)",
        "pH": "Soil pH"
    }
    units_map = {"N": "kg/ha", "P": "kg/ha", "K": "kg/ha", "pH": ""}

    for param in ["N", "P", "K", "pH"]:
        cur_v = current_soil[param]
        mid_v = ideal_mid[param]

        # Generate candidate adjustments
        candidates = []
        # Candidate 1: shift toward ideal midpoint
        if abs(cur_v - mid_v) > (0.2 if param == "pH" else 10.0):
            step = (mid_v - cur_v) * 0.5
            candidates.append(cur_v + step)
            candidates.append(mid_v)

        # Candidate 2: small incremental steps
        delta_step = 0.4 if param == "pH" else 20.0
        candidates.append(cur_v + delta_step)
        candidates.append(cur_v - delta_step)

        # Evaluate unique candidates within safe bounds
        tested = set()
        for cand in candidates:
            cand = float(np.clip(cand, bounds[param][0], bounds[param][1]))
            # Round candidate
            cand = round(cand, 1 if param == "pH" else 0)
            if cand == cur_v or cand in tested:
                continue
            tested.add(cand)

            row = {
                "rainfall": rainfall,
                "temperature": temperature,
                "humidity": humidity,
                "pH": cand if param == "pH" else pH,
                "N": cand if param == "N" else N,
                "P": cand if param == "P" else P,
                "K": cand if param == "K" else K,
                "crop_type": crop
            }
            pred_cand = max(0.0, float(rf_pipe.predict(pd.DataFrame([row]))[0]))
            gain = pred_cand - base_yield

            if gain > 0.02:  # Meaningful positive gain (> 0.02 t/ha)
                direction = "Raising" if cand > cur_v else "Lowering"
                unit_str = f" {units_map[param]}" if units_map[param] else ""
                cur_str = f"{cur_v:.1f}" if param == "pH" else f"{cur_v:.0f}"
                cand_str = f"{cand:.1f}" if param == "pH" else f"{cand:.0f}"

                action_text = f"{direction} {param_labels[param]} from {cur_str}{unit_str} to {cand_str}{unit_str}"
                explanation = f"Increases estimated yield by about {gain:.2f} t/ha (+{(gain/base_yield*100):.1f}%)." if base_yield > 0 else f"Increases estimated yield by {gain:.2f} t/ha."

                # Agronomic amendment advice
                if param == "N":
                    advice = "Apply split-dose nitrogenous fertilizer (Urea / Ammonium Sulphate) at active tillering." if cand > cur_v else "Reduce nitrogen to avoid lodging and pathogen susceptibility."
                elif param == "P":
                    advice = "Incorporate basal Single Super Phosphate (SSP) or DAP during land preparation." if cand > cur_v else "Avoid excess phosphorus build-up."
                elif param == "K":
                    advice = "Apply Muriate of Potash (MOP) to enhance grain filling and stress resilience." if cand > cur_v else "Maintain standard potash application."
                else:
                    advice = "Apply agricultural lime (calcium carbonate) to reduce acidity." if cand > cur_v else "Incorporate gypsum or organic compost to buffer alkalinity."

                test_scenarios.append({
                    "parameter": param,
                    "label": param_labels[param],
                    "current_value": cur_v,
                    "suggested_value": cand,
                    "yield_gain": round(gain, 2),
                    "new_yield": round(pred_cand, 2),
                    "action": action_text,
                    "explanation": explanation,
                    "amendment_advice": advice
                })

    # Sort scenarios descending by yield gain
    test_scenarios.sort(key=lambda x: x["yield_gain"], reverse=True)

    # Keep the top distinct recommendations (avoid duplicate params)
    seen_params = set()
    best_single_suggestions = []
    for s in test_scenarios:
        if s["parameter"] not in seen_params:
            seen_params.add(s["parameter"])
            best_single_suggestions.append(s)

    # 3. Test Full Target Balanced Soil Package
    package_row = {
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "pH": ideal_mid["pH"],
        "N": ideal_mid["N"],
        "P": ideal_mid["P"],
        "K": ideal_mid["K"],
        "crop_type": crop
    }
    package_pred = max(0.0, float(rf_pipe.predict(pd.DataFrame([package_row]))[0]))
    package_gain = package_pred - base_yield

    package_suggestion = None
    if package_gain > 0.04:
        package_suggestion = {
            "title": f"Combined Balanced Soil Package for {crop}",
            "description": (
                f"Adjusting all soil nutrients toward optimal benchmarks "
                f"(pH: {ideal_mid['pH']:.1f}, N: {ideal_mid['N']:.0f} kg/ha, "
                f"P: {ideal_mid['P']:.0f} kg/ha, K: {ideal_mid['K']:.0f} kg/ha)"
            ),
            "current_yield": round(base_yield, 2),
            "improved_yield": round(package_pred, 2),
            "yield_gain": round(package_gain, 2),
            "improved_inputs": {
                "pH": round(ideal_mid["pH"], 1),
                "N": round(ideal_mid["N"], 0),
                "P": round(ideal_mid["P"], 0),
                "K": round(ideal_mid["K"], 0)
            }
        }

    soil_adv = get_soil_advisory(crop, pH, N, P, K)
    weather_tips = get_weather_tips(rainfall, temperature, humidity)

    return {
        "crop": crop,
        "baseline_yield": round(base_yield, 2),
        "yield_units": "t/ha",
        "suggestions": best_single_suggestions,
        "package_suggestion": package_suggestion,
        "soil_advisory": soil_adv,
        "weather_tips": weather_tips,
        "disclaimer": MANDATORY_DISCLAIMER
    }


# ---------------------------------------------------------------------------
# 6. What-If Comparison
# ---------------------------------------------------------------------------
def compare_what_if(
    crop: str,
    rainfall: float,
    temperature: float,
    humidity: float,
    cur_pH: float,
    cur_N: float,
    cur_P: float,
    cur_K: float,
    imp_pH: float,
    imp_N: float,
    imp_P: float,
    imp_K: float
) -> Dict[str, Any]:
    """
    Compare current vs improved soil parameters side-by-side.
    """
    cur_res = predict_yield(crop, rainfall, temperature, humidity, cur_pH, cur_N, cur_P, cur_K)
    imp_res = predict_yield(crop, rainfall, temperature, humidity, imp_pH, imp_N, imp_P, imp_K)

    diff = imp_res["predicted_yield"] - cur_res["predicted_yield"]
    pct_diff = (diff / cur_res["predicted_yield"] * 100) if cur_res["predicted_yield"] > 0 else 0.0

    return {
        "crop": crop,
        "current_yield": cur_res["predicted_yield"],
        "improved_yield": imp_res["predicted_yield"],
        "difference": round(diff, 2),
        "percent_change": round(pct_diff, 1),
        "units": "t/ha",
        "current_inputs": {"pH": cur_pH, "N": cur_N, "P": cur_P, "K": cur_K},
        "improved_inputs": {"pH": imp_pH, "N": imp_N, "P": imp_P, "K": imp_K}
    }
