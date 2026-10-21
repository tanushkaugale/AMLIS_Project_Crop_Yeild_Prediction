# Crop Yield Prediction using Weather and Soil Data

An applied Machine Learning and Model Context Protocol (MCP) project developed for a 5th-semester Information Technology capstone viva examination.

---

## 📌 Project Overview
Predicts agricultural crop yield based on environmental parameters (annual rainfall, temperature, relative humidity) and topsoil fertility indicators (soil pH, Nitrogen, Phosphorus, Potassium). 

The system provides:
1. **Accurate Yield Forecasting:** Powered by a pre-trained **Random Forest Regressor** pipeline benchmarked against a **Linear Regression** baseline on held-out test data ($n = 3,904$).
2. **Smart Yield Improvement Engine:** Simulates realistic adjustments to farmer-controllable soil parameters (pH, N, P, K) to identify the highest predicted yield gains ($\Delta \text{Yield}$).
3. **Crop Recommendation:** Evaluates all 54 supported Indian crop varieties to recommend the top 5 highest-yielding crops for any given soil and weather profile.
4. **"Kisan Mitra" Farmer Assistant (MCP Chatbot):** An AI agricultural assistant built on Anthropic's **Model Context Protocol (FastMCP)** and powered by modern LLMs (Google Gemini / OpenAI / Groq) that speaks English, Hindi (हिन्दी), and Marathi (मराठी).
5. **Evaluator Dashboard:** A dedicated academic section featuring dynamic metric comparisons, actual-vs-predicted scatter charts, dynamic sensitivity simulation, and dataset provenance.

---

## 🛠️ Technology Stack
- **Language & Runtime:** Python 3.12 / 3.13 (Windows virtual environment)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning Pipelines:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `RandomForestRegressor`, `LinearRegression`)
- **Model Serialization:** Joblib
- **Agent Architecture:** Model Context Protocol Python SDK (`mcp` / `FastMCP`)
- **LLM Integration:** `google-genai` (Google Gemini) & `openai` SDKs, `python-dotenv`
- **Frontend Dashboard:** Streamlit
- **Visualizations:** Plotly Express & Plotly Graph Objects

---

## 📂 Project Directory Structure
```text
Yield-Predict/
│
├── farm_logic.py                           # Shared agronomic, ML inference, and optimization engine
├── mcp_server.py                           # FastMCP Server exposing 5 agricultural tools via stdio
├── chatbot.py                              # MCP client & LLM bridge with function calling & fallback
├── app.py                                  # 4-page modern Streamlit dashboard
├── train_model.py                          # Data cleaning, split, training & evaluation pipeline
├── test_pipeline.py                       # Automated unit & integration verification test suite
├── requirements.txt                        # Project dependencies
├── .env.example                            # Configuration template for LLM API keys
├── .gitignore                              # Protects secrets (.env), venv/, and cache files
├── README.md                               # Project documentation & viva examination guide
│
├── data/
│   ├── crop_yield.csv                      # Raw historical crop records (1997-2020)
│   ├── state_soil_data.csv                 # Raw soil nutrient profiles (N, P, K, pH)
│   ├── state_weather_data_1997_2020.csv    # Raw meteorological records
│   ├── processed_crop_yield_data.csv       # Cleaned unified dataset (19,517 rows)
│   └── README.md                           # Dataset provenance & acquisition instructions
│
└── models/
    ├── random_forest_pipeline.joblib       # Pre-trained Random Forest pipeline (Champion)
    ├── linear_regression_pipeline.joblib   # Pre-trained Linear Regression pipeline (Baseline)
    ├── model_metadata.json                 # Evaluated test metrics, feature names, crop list
    └── test_predictions_sample.csv        # Held-out actual vs predicted records
```

---

## 🌾 Core Features & Pages

### 1. 🌾 Yield Prediction & Soil Guidance (Farmer-Facing)
- Input 8 agricultural variables: Crop, Annual Rainfall (mm), Temperature (°C), Humidity (%), Soil pH, Nitrogen (N), Phosphorus (P), Potassium (K).
- Displays Random Forest predicted yield as primary metric in **Tonnes per Hectare (t/ha)** with a realistic $\pm \text{MAE}$ confidence interval.
- **Smart Yield Improvement Checklist:** Tests candidate soil nutrient adjustments and ranks suggestions by yield gain (e.g. *"Raising N from 78 to 100 kg/ha increases estimated yield by +0.22 t/ha"*).
- **What-If Soil Simulator:** Side-by-side comparison between current soil inputs and target improved values.
- **Weather Mitigation Tips:** Advice for rainfall extremes (irrigation/drainage), heat stress, and high humidity pathogen risks.
- **One-Click Download:** Exports prediction report as a formatted text file.

### 2. 🌱 Crop Recommendation Engine (Farmer-Facing)
- Simulates yield across all 54 Indian crop categories for the user's active climate and soil conditions.
- Displays the top 5 highest-yielding crops in an interactive Plotly horizontal bar chart and summary table.

### 3. 🤖 Kisan Mitra — Farmer Assistant (Chatbot with MCP)
- Interactive chatbot with conversational memory (`st.session_state`).
- Powered by the **Model Context Protocol (MCP)**: The LLM interacts with `mcp_server.py` to retrieve verified numbers.
- **Zero Hallucination Guarantee:** The model never fabricates yield numbers; all figures come from tool execution.
- **Multilingual Support:** Converses naturally in **English**, **Hindi (हिन्दी)**, and **Marathi (मराठी)**.
- **Safe & Ethical:** Refuses to dispense hazardous pesticide/fungicide chemical dosages; always recommends local Krishi Vigyan Kendra (KVK) officers.
- **Graceful Degradation:** If no API key is configured, the rest of the app works completely, and the chatbot displays an informative setup guide.

### 4. 📖 About Model (for evaluators)
- **Rigorous Model Comparison:** Random Forest vs Linear Regression across 4 standard regression metrics:
  - **MAE (Mean Absolute Error):** Random Forest reduces error by **56.3%** (`1.2291 t/ha` vs `2.8152 t/ha`).
  - **RMSE (Root Mean Squared Error):** Random Forest reduces error by **23.5%** (`4.9314 t/ha` vs `6.4491 t/ha`).
  - **R² Score:** Random Forest explains **84.6% of variance** (`0.8460`) vs Linear Regression's `0.7367`.
- **Actual vs. Predicted Scatter Plot:** Visualizes predictions against ideal $45^\circ$ line on 3,904 held-out test points.
- **Dynamic Sensitivity Simulation:** Sweeps a chosen variable while holding all other parameters at the **user's active inputs**.
- **Dataset Provenance & Documented Limitations:** Sourced from Ministry of Agriculture & Farmers Welfare, India Meteorological Department (IMD), and Kaggle.

---

## 🚀 Installation & Running Guide (Windows PowerShell)

### 1. Open Windows PowerShell and navigate to the project directory:
```powershell
cd "c:\Users\admin\Desktop\Academy(2026)\Yield-Predict\Yield-Predict"
```

### 2. Activate Virtual Environment:
```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies:
```powershell
pip install -r requirements.txt
```

### 4. Configure Chatbot API Key (Optional):
Copy the example environment template:
```powershell
copy .env.example .env
```
Open `.env` in any text editor and insert your key:
```ini
LLM_API_KEY=your_gemini_or_openai_api_key_here
LLM_MODEL=gemini-2.5-flash
LLM_PROVIDER=gemini
```
*(If omitted, all prediction, recommendation, and evaluator features still function 100%.)*

### 5. Run Automated Tests:
```powershell
python test_pipeline.py
```
*Expected: 9 automated verification tests pass successfully.*

### 6. Run FastMCP Agricultural Server (Standalone check):
```powershell
python mcp_server.py
```

### 7. Launch the Streamlit Dashboard:
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧠 Model Context Protocol (MCP) Architecture

This project implements Anthropic's **Model Context Protocol (MCP)** to decouple AI model reasoning from backend machine learning inference tools:

```text
┌──────────────────────────┐
│   Streamlit Web Client   │
│   (app.py / Chatbot UI)  │
└────────────┬─────────────┘
             │ (User Message)
             ▼
┌──────────────────────────┐
│        chatbot.py        │  ◄── Reads .env (LLM_API_KEY, LLM_MODEL)
│   (LLM + MCP Client)     │
└────────────┬─────────────┘
             │ (JSON-RPC Tool Calls / stdio)
             ▼
┌──────────────────────────┐
│      mcp_server.py       │  Exposes 5 Standard Agricultural Tools:
│    (FastMCP Server)      │  1. predict_yield
└────────────┬─────────────┘  2. recommend_crops
             │                3. suggest_improvements
             ▼                4. get_soil_advisory
┌──────────────────────────┐  5. list_supported_crops
│      farm_logic.py       │
│  (Inference & Optimizer) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Trained ML Artifacts   │
│ (Random Forest & LR OHE) │
└──────────────────────────┘
```

---

## 🎓 Capstone Viva Examination Q&A

### Q1: Why did you compare Linear Regression and Random Forest Regressor?
**Answer:** Linear Regression serves as our transparent, parametric baseline. However, agricultural biological response curves are inherently non-linear (e.g. crop yield follows a bell-curve where extreme rainfall causes waterlogging and high heat causes flower drop). Random Forest Regressor, as a non-parametric ensemble of 100 decision trees, captures non-linear plateaus and complex multi-feature interactions (such as soil pH controlling Nitrogen and Phosphorus availability) without requiring manual interaction terms. On our strictly isolated 20% test partition, Random Forest achieved an $R^2$ of **0.8460** and MAE of **1.23 t/ha**, compared to Linear Regression's $R^2$ of **0.7367** and MAE of **2.82 t/ha**.

### Q2: How does your pipeline prevent data leakage?
**Answer:** Data leakage is strictly prevented by fitting all preprocessing transformers exclusively on the 80% training split (`X_train`). Numerical standardization (`StandardScaler`) and categorical one-hot encoding (`OneHotEncoder`) are encapsulated inside a Scikit-Learn `Pipeline` and `ColumnTransformer`. The held-out 20% test split (`X_test`) was only transformed during evaluation and never exposed to scaler fit parameters.

### Q3: Why is Coconut excluded from the continuous yield benchmark?
**Answer:** In official agricultural statistics compiled by the Ministry of Agriculture & Farmers Welfare (Government of India), Coconut yield is reported in *thousands of nuts per hectare* (producing raw metrics upwards of 21,000), whereas all other crops are quantified in *Tonnes per Hectare (t/ha)*. To maintain rigorous unit consistency across our regression targets, Coconut records were isolated.

### Q4: How does the Yield Improvement Suggestion algorithm work?
**Answer:** The algorithm recognizes that weather variables (rainfall, temperature, humidity) are immutable exogenous factors that a farmer cannot alter. However, topsoil chemical properties (pH, N, P, K) are controllable through agricultural conditioning and fertilization. The `suggest_improvements` engine takes the farmer's current inputs, benchmarks them against crop-specific agronomic ranges (ICAR guidelines), and evaluates candidate adjustments through the pre-trained Random Forest pipeline. It calculates $\Delta \text{Yield} = Y_{\text{candidate}} - Y_{\text{current}}$, filters for positive gains ($> 0.02\text{ t/ha}$), and ranks actionable suggestions in descending order of yield impact.

### Q5: What is Model Context Protocol (MCP) and why is it used in this project?
**Answer:** The Model Context Protocol (MCP) is an open architectural standard that allows Large Language Models (LLMs) to securely communicate with external execution environments and data tools via standardized JSON-RPC protocols. In our project, `mcp_server.py` runs a FastMCP server exposing 5 agricultural tools (`predict_yield`, `recommend_crops`, `suggest_improvements`, `get_soil_advisory`, `list_supported_crops`). This ensures the chatbot cannot hallucinate or invent crop yield numbers: the LLM must call the MCP tools to receive verified predictions computed by our Scikit-Learn pipelines.

### Q6: What are the primary limitations of this prototype?
**Answer:** 
1. **Regional Macro-Data:** Historical meteorological and soil indicators are compiled at state/district annual levels. Field-level precision soil tests may exhibit microclimate and topographical variations.
2. **Exogenous Variables:** Unrecorded variables like seed genetics (hybrid vs indigenous), specific micro-irrigation schedules, sudden pest attacks, and unseasonal hail storms are outside the tabular feature set.
3. **Academic Scope:** The advisory engine is an educational heuristic demonstration and must always be paired with certified physical soil testing and guidance from local Krishi Vigyan Kendras (KVK).
