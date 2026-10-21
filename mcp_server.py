"""
mcp_server.py
=============
Model Context Protocol (MCP) Server for Crop Yield Prediction & Soil Advisory.
Built using the official Python MCP SDK (FastMCP).
Exposes agricultural inference tools for LLM agent integration via stdio transport.
"""

import os
import sys
import json
import warnings

# Suppress sklearn unpickle warnings for clean stdio protocol communication
warnings.filterwarnings("ignore")

# Ensure local imports work regardless of current working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_logic
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP(
    name="CropYieldAssistantServer",
    instructions="Agricultural expert server offering crop yield estimation, crop recommendations, and soil nutrient guidance for Indian agriculture."
)


@mcp.tool()
def predict_yield(
    crop: str,
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float
) -> str:
    """
    Predict expected crop yield in Tonnes per Hectare (t/ha) using machine learning.
    
    Parameters:
        crop: Target crop name (e.g. 'Rice', 'Wheat', 'Maize', 'Cotton(lint)', 'Sugarcane').
        rainfall: Annual rainfall in millimeters (mm) (range: 100 - 3500 mm).
        temperature: Mean temperature in degrees Celsius (°C) (range: 10 - 45 °C).
        humidity: Mean relative humidity percentage (%) (range: 20 - 100 %).
        pH: Soil pH level (range: 3.5 - 9.5).
        N: Available soil Nitrogen in kg/ha (range: 10 - 200 kg/ha).
        P: Available soil Phosphorus in kg/ha (range: 5 - 120 kg/ha).
        K: Available soil Potassium in kg/ha (range: 5 - 120 kg/ha).
    
    Returns:
        JSON string containing predicted yield in t/ha, evaluated MAE margin, and confidence interval.
    """
    result = farm_logic.predict_yield(
        crop=crop,
        rainfall=rainfall,
        temperature=temperature,
        humidity=humidity,
        pH=pH,
        N=N,
        P=P,
        K=K
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def recommend_crops(
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float,
    top_n: int = 5
) -> str:
    """
    Recommend the top crops with the highest predicted yields for the given weather and soil conditions.
    
    Parameters:
        rainfall: Annual rainfall in millimeters (mm).
        temperature: Mean temperature in degrees Celsius (°C).
        humidity: Mean relative humidity percentage (%).
        pH: Soil pH level.
        N: Available soil Nitrogen in kg/ha.
        P: Available soil Phosphorus in kg/ha.
        K: Available soil Potassium in kg/ha.
        top_n: Number of top crops to return (default: 5).
    
    Returns:
        JSON string listing the top recommended crops ranked by estimated productivity (t/ha).
    """
    recommendations = farm_logic.recommend_crops(
        rainfall=rainfall,
        temperature=temperature,
        humidity=humidity,
        pH=pH,
        N=N,
        P=P,
        K=K,
        top_n=top_n
    )
    return json.dumps(recommendations, indent=2)


@mcp.tool()
def suggest_improvements(
    crop: str,
    rainfall: float,
    temperature: float,
    humidity: float,
    pH: float,
    N: float,
    P: float,
    K: float
) -> str:
    """
    Evaluate potential yield gains by testing realistic adjustments to soil factors (pH, N, P, K).
    Weather parameters cannot be changed. Returns ranked actionable suggestions sorted by impact,
    along with weather risk management tips and nutrient advisories.
    
    Parameters:
        crop: Target crop name.
        rainfall: Annual rainfall in millimeters (mm).
        temperature: Mean temperature in degrees Celsius (°C).
        humidity: Mean relative humidity percentage (%).
        pH: Soil pH level.
        N: Available soil Nitrogen in kg/ha.
        P: Available soil Phosphorus in kg/ha.
        K: Available soil Potassium in kg/ha.
    
    Returns:
        JSON string containing ranked suggestions, expected yield gain (t/ha), and advisory tips.
    """
    suggestions = farm_logic.suggest_improvements(
        crop=crop,
        rainfall=rainfall,
        temperature=temperature,
        humidity=humidity,
        pH=pH,
        N=N,
        P=P,
        K=K
    )
    return json.dumps(suggestions, indent=2)


@mcp.tool()
def get_soil_advisory(
    crop: str,
    pH: float,
    N: float,
    P: float,
    K: float
) -> str:
    """
    Evaluate whether soil pH, Nitrogen, Phosphorus, and Potassium are low, balanced, or high
    against crop-specific agronomic benchmarks. Provides educational fertilizer/amendment advice.
    
    Parameters:
        crop: Target crop name.
        pH: Soil pH level.
        N: Available soil Nitrogen in kg/ha.
        P: Available soil Phosphorus in kg/ha.
        K: Available soil Potassium in kg/ha.
    
    Returns:
        JSON string detailing nutrient status (low/balanced/high) and educational guidance.
    """
    advisory = farm_logic.get_soil_advisory(
        crop=crop,
        pH=pH,
        N=N,
        P=P,
        K=K
    )
    return json.dumps(advisory, indent=2)


@mcp.tool()
def list_supported_crops() -> str:
    """
    List all supported crop varieties available in the machine learning model.
    
    Returns:
        JSON string containing an array of 54 crop names.
    """
    crops = farm_logic.list_supported_crops()
    return json.dumps({"supported_crops": crops, "total_count": len(crops)}, indent=2)


if __name__ == "__main__":
    # Run the MCP server over standard I/O (stdio) transport
    mcp.run(transport="stdio")
