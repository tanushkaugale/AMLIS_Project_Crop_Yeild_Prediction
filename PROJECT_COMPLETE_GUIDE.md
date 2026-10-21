# CROP YIELD PREDICTION USING WEATHER AND SOIL DATA
## Complete Project Guide & Documentation (5th-Semester IT Capstone Project)

---

## 1. PROJECT IN 1 MINUTE (Viva Speaking Script)

> **Speak this script when the examiner says: "Introduce your project in one minute."**

"Good morning, teachers. My project is titled **'Crop Yield Prediction using Weather and Soil Data'**. 

* **The Problem:** In India, agriculture depends heavily on unpredictable weather and varying soil quality. Farmers often invest in seeds and fertilizers without knowing how much crop they will actually harvest, leading to financial loss and debt.
* **Who Uses It:** Farmers, agricultural extension officers, and agriculture students.
* **The Inputs:** 8 parameters — 3 climate factors (Annual Rainfall, Temperature, Humidity), 4 soil factors (Soil pH, Nitrogen, Phosphorus, Potassium), and 1 Crop Variety (out of 54 Indian crops).
* **The Outputs:** 
  1. A machine learning yield prediction in **Tonnes per Hectare (t/ha)** with a realistic margin of error ($\pm 1.23\text{ t/ha}$).
  2. Smart yield improvement suggestions showing which soil nutrients to adjust for maximum harvest gain.
  3. A top-5 crop recommendation list for the farmer's soil and climate.
  4. An AI chatbot named **'Kisan Mitra'** built with the **Model Context Protocol (FastMCP)** that answers farming questions in English, Hindi, and Marathi without making up numbers.

We benchmarked Linear Regression against Random Forest Regressor on 19,517 historical records. Random Forest won with an **$R^2$ score of 0.8460**, explaining 84.6% of yield variance."

---

## 2. PROBLEM AND OBJECTIVES

### Why Crop Yield Prediction Matters in India
* Over 55% of India's population relies on agriculture for their livelihood.
* Most smallholder farmers practice rainfed agriculture and lack access to expensive soil testing labs or scientific forecasting tools.
* Over-application of fertilizers (especially Nitrogen/Urea) damages soil acidity and groundwater, while under-application causes crop starvation.
* Having an accurate, free, and easy-to-use digital tool helps farmers plan sowing seasons, select profitable crops, and balance fertilizers responsibly.

### Project Objectives
1. **Accurate Yield Forecasting:** Train and compare regression pipelines on historical Indian crop, soil, and meteorological records (1997–2020) without data leakage.
2. **Actionable Farmer Guidance:** Separate immutable climate factors (which a farmer cannot change) from controllable soil factors (pH, N, P, K), testing adjustments to find the highest yield gain ($\Delta \text{Yield}$).
3. **Crop Recommendation:** Allow a farmer to enter their local soil and weather and instantly see the 5 best crops to plant.
4. **Model Context Protocol (MCP) AI Assistant:** Build an AI chatbot using FastMCP tools so that large language models retrieve verified ML predictions instead of hallucinating numbers.
5. **Academic Rigor for Evaluation:** Provide a dedicated evaluator section comparing metrics (MAE, RMSE, MSE, $R^2$), actual-vs-predicted scatter plots, and dynamic sensitivity charts.

---

## 3. FOLDER TREE

```text
Yield-Predict/
│
├── farm_logic.py                 # Core backend: predictions, crop recommendations, soil suggestions, and what-if logic
├── mcp_server.py                 # FastMCP server exposing 5 tools over standard I/O (stdio transport)
├── chatbot.py                    # LLM chat client with MCP tool calling, multi-language support, and error retry
├── app.py                        # Streamlit web application with 4 pages (Farmer UI + Evaluator Section)
├── train_model.py                # Offline pipeline: merges CSVs, cleans data, fits scalers/encoders, trains RF & LR models
├── test_pipeline.py              # Automated test suite with 9 unit/integration tests (runs without API keys)
├── requirements.txt              # List of Python dependencies with version bounds
├── .env.example                  # Template configuration showing how to set LLM_API_KEY and LLM_MODEL
├── .env                          # Secret file holding real API keys (gitignored to prevent leaks)
├── .gitignore                    # Tells Git to ignore .env, venv/, and Python cache folders
├── README.md                     # Project overview, installation steps, MCP architecture, and viva Q&A
│
├── data/                         # Historical datasets folder
│   ├── crop_yield.csv            # 19,689 raw crop production and yield observations across Indian states (1997–2020)
│   ├── state_soil_data.csv       # 30 state-level topsoil nutrient averages (Nitrogen, Phosphorus, Potassium, pH)
│   ├── state_weather_data_1997_2020.csv # 720 meteorological rows (annual rainfall, temperature, humidity for 30 states)
│   ├── processed_crop_yield_data.csv    # 19,517 cleaned, merged records created by train_model.py
│   └── README.md                 # Dataset provenance and Kaggle source documentation
│
└── models/                       # Serialized model artifacts folder
    ├── random_forest_pipeline.joblib     # Pre-trained Random Forest Regressor Scikit-Learn Pipeline (Champion)
    ├── linear_regression_pipeline.joblib # Pre-trained Linear Regression Scikit-Learn Pipeline (Baseline)
    ├── model_metadata.json               # JSON file storing test metrics, feature names, units, and 54 crop names
    └── test_predictions_sample.csv       # 300 held-out test points comparing actual vs predicted yield
```

---

## 4. TECH STACK TABLE

| Technology | Category | What It Is (1-Line Definition) | Why It Is Used in This Project | Which File Uses It |
| :--- | :--- | :--- | :--- | :--- |
| **Python 3.12 / 3.13** | Programming Language | High-level readable programming language for data science and AI. | Foundation of the whole project; supported by all ML libraries. | All `.py` files |
| **pandas** | Library | Fast tabular data manipulation library using DataFrames. | Merging CSVs, filtering columns, formatting inputs for model prediction. | `train_model.py`, `farm_logic.py`, `app.py`, `test_pipeline.py` |
| **NumPy** | Library | Numerical library for fast array math and linear algebra. | Calculating RMSE ($\sqrt{\text{MSE}}$) and generating numerical sweeps for sensitivity simulation. | `train_model.py`, `farm_logic.py`, `app.py`, `test_pipeline.py` |
| **scikit-learn** | Machine Learning | Machine learning library providing pipelines, scalers, and regressors. | Preprocessing (StandardScaler, OneHotEncoder) and model training (RandomForestRegressor, LinearRegression). | `train_model.py`, `farm_logic.py`, `test_pipeline.py` |
| **joblib** | Serialization | Lightweight file serializer optimized for Python objects with large NumPy arrays. | Saving and loading trained Scikit-Learn pipelines to/from disk (`.joblib`). | `train_model.py`, `farm_logic.py`, `test_pipeline.py` |
| **Streamlit** | Frontend UI | Python framework that turns data scripts into interactive web applications. | Powers the 4-page responsive farmer dashboard without writing HTML/JS. | `app.py` |
| **Plotly** | Visualization | Interactive charting library producing zoomable web graphs. | Draws the crop recommendation bar charts, metric comparison graphs, scatter plots, and sensitivity curves. | `app.py` |
| **mcp (FastMCP)** | Protocol / Agent | Anthropic's Model Context Protocol SDK connecting AI models to local tools. | Exposes 5 verified agricultural ML tools to the chatbot via stdio transport. | `mcp_server.py`, `chatbot.py` |
| **python-dotenv** | Configuration | Reads key-value pairs from a `.env` file into `os.environ`. | Safely loads `LLM_API_KEY` and `LLM_MODEL` without hardcoding secrets in code. | `chatbot.py` |
| **google-genai** | LLM Client SDK | Official Google SDK for communicating with Gemini models. | Connects chatbot to Gemini 2.5 Flash with native function calling. | `chatbot.py` |
| **openai** | LLM Client SDK | Industry-standard SDK for OpenAI and OpenAI-compatible endpoints (Groq). | Connects chatbot to Groq (`qwen/qwen3.8-27b`) or OpenAI (`gpt-4o-mini`). | `chatbot.py` |

---

## 5. FRONTEND (Streamlit Web Dashboard)

### How Streamlit Works
* **Execution Model:** Streamlit runs Python script `app.py` from top to bottom every time a user changes an input widget (slider, selectbox) or clicks a button.
* **State Preservation (`st.session_state`):** Because Streamlit reruns the whole script, normal local variables would reset to defaults. `st.session_state` acts like a dictionary that remembers values across reruns (such as the farmer's inputs, previous prediction results, and chat history).

### The 4 Dashboard Pages

#### Page 1: 🌾 Yield Prediction & Suggestions (Farmer-Facing)
* **Inputs:**
  * Selectbox: `Select Crop` (54 varieties, defaults to "Rice").
  * Slider: `Annual Rainfall (mm)` (100 to 3500 mm).
  * Slider: `Temperature (°C)` (10 to 45 °C).
  * Slider: `Relative Humidity (%)` (20 to 100 %).
  * Slider: `Soil pH` (3.5 to 9.5).
  * Slider: `Nitrogen (N) [kg/ha]` (10 to 200 kg/ha).
  * Slider: `Phosphorus (P) [kg/ha]` (5 to 120 kg/ha).
  * Slider: `Potassium (K) [kg/ha]` (5 to 120 kg/ha).
* **Buttons:**
  * `🚀 Predict Crop Yield`: Runs `farm_logic.predict_yield` and `farm_logic.suggest_improvements`, storing results in `st.session_state`.
  * `📥 Download Prediction Report (TXT)`: Downloads a formatted text summary of inputs, predictions, and suggestions.
* **Outputs & Visual Cards:**
  * **Big Yield Card:** Shows Random Forest predicted yield (e.g., `1.76 t/ha`), plus evaluated error range (`0.53 – 2.99 t/ha`).
  * **"What You Can Do" Checklist:** Cards sorted by yield impact (e.g., *Raising N from 78 to 100 kg/ha increases estimated yield by +0.22 t/ha*).
  * **Weather Management Tips:** Blue advice boxes for rain, heat, and fungal humidity risks.
  * **What-If Comparison:** Side-by-side comparison letting farmers adjust target soil values and instantly see the predicted difference ($\Delta \text{Yield}$).
  * **Academic & Agronomic Disclaimer:** Highlights that predictions are estimates, not chemical prescriptions.

#### Page 2: 🌱 Crop Recommendation (Farmer-Facing)
* **Inputs:** Uses the active soil and weather parameters from Page 1; slider to select number of top crops (3 to 10).
* **Outputs:**
  * Horizontal Plotly bar chart showing top crops ordered by predicted yield.
  * Summary table listing Crop Variety, Predicted Yield (t/ha), and Units.
  * Highlight banner declaring the single highest-yielding crop.

#### Page 3: 🤖 Farmer Assistant (Chatbot — Kisan Mitra)
* **Inputs:** `st.chat_input` text bar at the bottom; 3 quick-question shortcut buttons (Rice prediction, crop recommendation, Hindi wheat advice).
* **Outputs:** Chat message bubbles (`st.chat_message`) with farmer 👨‍🌾 and assistant 🌾 avatars.
* **Button:** `🗑️ Clear Conversation` resets `st.session_state.chat_history`.

#### Page 4: 📖 About Model (For Evaluators & Capstone Viva)
* **Inputs:** Radio button to filter Actual vs Predicted scatter plot ("Random Forest", "Linear Regression", or "Both Models"); Selectbox to sweep one parameter in Sensitivity Simulation.
* **Outputs:**
  * 4 Metric Cards: $R^2$ Score (`0.8460`), MAE (`1.23 t/ha`), RMSE (`4.93 t/ha`), Test Sample Count (`3,904`).
  * Metric Comparison Grouped Bar Chart comparing LR vs RF.
  * Agronomic & ML Rationale card explaining non-linear plateaus and pH-nutrient interactions.
  * Actual vs Predicted Scatter Plot with ideal $45^\circ$ reference line.
  * **Dynamic Sensitivity Simulation Chart:** Sweeps the chosen variable across realistic bounds while holding other features at the **user's active inputs**.
  * 3 Academic documentation tabs: Dataset Provenance, Windows Run Guide, and Limitations.

### Code Path: From Slider Click to Screen Number
1. Farmer adjusts the Nitrogen slider to `100.0`.
2. Streamlit records `st.session_state.N = 100.0` and triggers a script rerun.
3. Farmer clicks `🚀 Predict Crop Yield`.
4. `app.py` passes the 8 values to `farm_logic.predict_yield()`.
5. `farm_logic.py` formats a single-row pandas DataFrame matching the required columns.
6. The pre-trained Random Forest pipeline (`rf_pipeline.predict(df)`) scales numbers, encodes categorical crop, runs through 100 decision trees, and outputs `1.98`.
7. `app.py` reads the output dictionary, clamps to $\ge 0$, and renders the HTML card: `<div class="big-yield-number">1.98 t/ha</div>`.

---

## 6. BACKEND

The backend consists of clean Python modules with **zero code duplication** (DRY principle):

### Backend Files & Their Purpose
* **`train_model.py`:** Prepares raw data, fits pipelines on 80% training data, evaluates on 20% test data, and saves `.joblib` and `.json` artifacts into `models/`.
* **`farm_logic.py`:** Shared inference and optimization module. Imported by both `app.py` and `mcp_server.py`.
* **`mcp_server.py`:** FastMCP server exposing 5 tools for LLM integration.
* **`chatbot.py`:** LLM client connecting to MCP tools, handling multilingual translation, tool calling loops, and rate limit retries.

### Exact Functions Called

| Function Name | File | What It Does |
| :--- | :--- | :--- |
| `load_artifacts()` | `farm_logic.py` | Singleton loader that caches the metadata JSON and both joblib pipelines in memory. |
| `predict_yield(...)` | `farm_logic.py` | Runs Random Forest and Linear Regression on user inputs, returns predicted yield, MAE margin, and confidence range. |
| `recommend_crops(...)` | `farm_logic.py` | Runs RF inference on all 54 crops in a single batch, sorts descending, and returns top $N$ crops. |
| `get_soil_advisory(...)` | `farm_logic.py` | Compares pH, N, P, K against crop-specific agronomic benchmarks (ICAR ranges) and returns "low", "balanced", or "high" with fertilizer tips. |
| `get_weather_tips(...)` | `farm_logic.py` | Evaluates rainfall, temperature, and humidity thresholds and returns actionable risk management tips. |
| `suggest_improvements(...)` | `farm_logic.py` | Evaluates realistic candidate adjustments to soil factors (holding weather constant), computes $\Delta \text{Yield}$, and ranks suggestions by impact. |
| `compare_what_if(...)` | `farm_logic.py` | Takes current soil inputs and target improved inputs, runs both through RF, and calculates yield and percentage differences. |
| `list_supported_crops()` | `farm_logic.py` | Returns the list of all 54 crop varieties from metadata. |
| `ask_farmer_assistant(...)` | `chatbot.py` | Takes user chat message, connects to LLM, executes tool calls via FastMCP, and returns friendly advice in English/Hindi/Marathi. |
| `_create_completion_with_retry(...)` | `chatbot.py` | Helper wrapping LLM completions with exponential backoff on HTTP 429 rate limits. |

---

## 7. ARCHITECTURE AND DATA FLOW

### Text Diagram 1: Machine Learning & Prediction Pipeline

```text
[crop_yield.csv] + [state_soil_data.csv] + [state_weather_data.csv]
                           │
                           ▼ (1. Inner Join on state & year, remove Coconut)
              [processed_crop_yield_data.csv] (19,517 rows)
                           │
                           ▼ (2. train_test_split: 80% Train, 20% Held-Out Test)
             ┌─────────────┴─────────────┐
             ▼                           ▼
        X_train, y_train            X_test, y_test
        (15,613 samples)            (3,904 samples)
             │                           │
             ▼                           │
[ColumnTransformer & Preprocessing]      │
- StandardScaler on 7 numbers            │
- OneHotEncoder on crop_type             │
             │                           │
             ▼ (Fit exclusively on Train)│
     [Trained Pipelines]                 │
     - Linear Regression                 │
     - Random Forest (100 Trees)         │
             │                           │
             ▼ (Evaluate on Test) ◄──────┘
   [model_metadata.json] (MAE: 1.23 vs 2.82, R²: 0.846 vs 0.737)
             │
             ▼ (Serialized via Joblib)
    [models/*.joblib files]
             │
             ▼ (Loaded at startup into cache)
      [farm_logic.py] ◄─── (User enters 8 inputs) ─── [Streamlit app.py]
             │
             ▼
   [Predicted Yield (t/ha) + Soil Suggestions + Crop Recommendations]
```

#### Explanation of Diagram 1 Arrows:
1. **Raw CSVs $\rightarrow$ Processed CSV:** Three separate historical files are merged on state and year. Coconut is filtered out because its yield was recorded in thousands of nuts/ha rather than metric tonnes/ha.
2. **Processed CSV $\rightarrow$ Train/Test Split:** Data is split into 80% training ($n = 15,613$) and 20% testing ($n = 3,904$) using `random_state=42`.
3. **Train $\rightarrow$ Preprocessing:** Scalers and encoders are fitted **strictly on training data** to prevent data leakage.
4. **Pipelines $\rightarrow$ Evaluation:** Both pipelines predict on held-out test data. Random Forest outperforms Linear Regression across all 4 metrics.
5. **Artifacts $\rightarrow$ Streamlit:** Joblib loads pipelines into memory; `farm_logic.py` executes predictions instantly when the farmer clicks the predict button.

---

### Text Diagram 2: Chatbot & Model Context Protocol (MCP) Flow

```text
[Farmer types question in Hindi/English/Marathi]
                        │
                        ▼ (1. UI sends message + history)
                 [app.py (Page 3)]
                        │
                        ▼ (2. Calls ask_farmer_assistant)
                  [chatbot.py]
                        │
                        ▼ (3. Reads .env key & passes SYSTEM_PROMPT + TOOL_DEFINITIONS)
                [LLM API (Groq / Gemini)]
                        │
                        ▼ (4. LLM decides it needs real data; emits function call)
        Function Call: "predict_yield(Rice, rain=1150, ...)"
                        │
                        ▼ (5. chatbot.py routes call to FastMCP Server)
                 [mcp_server.py]
                        │
                        ▼ (6. Invokes shared logic)
                 [farm_logic.py]
                        │
                        ▼ (7. Runs pre-trained Scikit-Learn Random Forest pipeline)
             [random_forest_pipeline.joblib]
                        │
                        ▼ (8. Returns JSON: {"predicted_yield": 1.76, "mae": 1.23})
                 [mcp_server.py]
                        │
                        ▼ (9. Passes JSON tool output back to LLM)
                [LLM API (Groq / Gemini)]
                        │
                        ▼ (10. LLM formats warm, polite conversational reply with units)
                  [chatbot.py]
                        │
                        ▼ (11. Renders message bubble)
             [Streamlit Chatbot Screen]
```

#### Explanation of Diagram 2 Arrows:
1. **User $\rightarrow$ `app.py`:** Farmer asks: *"What is my rice yield with 1150mm rain?"*
2. **`app.py` $\rightarrow$ `chatbot.py`:** Chat history and prompt are passed to the controller.
3. **`chatbot.py` $\rightarrow$ LLM:** Prompt is sent with instructions to never invent numbers and always use tools.
4. **LLM $\rightarrow$ Tool Call:** LLM recognizes the need for calculation and outputs a structured tool request: `predict_yield(...)`.
5. **Tool Call $\rightarrow$ `mcp_server.py`:** Tool request is executed through the FastMCP server.
6. **`mcp_server.py` $\rightarrow$ `farm_logic.py`:** Shared inference engine runs the inputs through the trained Random Forest model.
7. **ML Pipeline $\rightarrow$ JSON:** Exact prediction (`1.76 t/ha`) is serialized to JSON.
8. **JSON $\rightarrow$ LLM:** LLM receives verified numerical facts from the model.
9. **LLM $\rightarrow$ Final Reply:** LLM crafts a polite farmer-friendly answer in the user's language, explaining units and disclaimers.

---

## 8. DATASET

### Primary Source & Provenance
* **Repository:** Kaggle Agricultural Benchmark (`anshumish/crop-yield-data-with-soil-and-weather-dataset`).
* **Underlying Government Source:** Directorate of Economics and Statistics, Ministry of Agriculture & Farmers Welfare (Government of India) paired with regional India Meteorological Department (IMD) historical archives.
* **Years Covered:** 1997 through 2020 (24 continuous agricultural seasons).

### Summary of Dataset Files

| File Name | Rows | Columns | Description of Columns | Source |
| :--- | :---: | :---: | :--- | :--- |
| `crop_yield.csv` | 19,689 | 9 | `crop`, `year`, `season`, `state`, `area`, `production`, `fertilizer`, `pesticide`, `yield` | Ministry of Agriculture |
| `state_soil_data.csv` | 30 | 5 | `state`, `N` (Nitrogen kg/ha), `P` (Phosphorus kg/ha), `K` (Potassium kg/ha), `pH` (soil reaction) | ICAR State Surveys |
| `state_weather_data_1997_2020.csv` | 720 | 5 | `state`, `year`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent` | India Meteorological Dept |
| `processed_crop_yield_data.csv` | 19,517 | 11 | `rainfall`, `temperature`, `humidity`, `pH`, `N`, `P`, `K`, `crop_type`, `yield`, `state`, `year` | Cleaned Merged Dataset |

### Target Variable & Units
* **Target Variable:** `yield` (Historical Crop Yield).
* **Target Units:** **Tonnes per Hectare (t/ha)** ($\text{Production in Tonnes} / \text{Cultivated Area in Hectares}$).

### Why Coconut Was Removed
In Indian national agricultural yearbooks, **Coconut production is quantified in *thousands of nuts per hectare*** rather than metric tonnes. This produced outlier numbers up to 21,000 in the raw dataset. To maintain strict unit integrity (**Tonnes per Hectare**) across all 54 crops and avoid corrupting regression error loss functions, Coconut records were explicitly isolated.

---

## 9. MACHINE LEARNING EXPLAINED SIMPLY (With Real-Life Analogies)

### 1. Regression (Continuous Prediction)
* **One-Line Meaning:** An ML task that predicts an actual continuous number (like crop yield in tonnes), rather than a yes/no label.
* **Real-Life Analogy:** Guessing someone's exact weight in kilograms based on their height, diet, and exercise, rather than just guessing "tall" or "short".

### 2. Train / Test Split (80% / 20%)
* **One-Line Meaning:** Dividing historical data into an 80% study book (training) and a 20% unseen final exam (testing).
* **Real-Life Analogy:** A student practices on 80 textbook questions during the semester, and the teacher tests them on 20 brand-new questions during the final exam to see if they truly learned the concepts.

### 3. Data Leakage
* **One-Line Meaning:** When information from the test dataset accidentally leaks into model training, creating falsely high performance that fails in the real world.
* **Real-Life Analogy:** A student who accidentally saw the final exam question paper before the exam. They get 100% on the test, but fail when asked to solve a problem in the real world.
* **How We Prevented It:** Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) were `fit()` **only on `X_train`**. `X_test` was only transformed.

### 4. StandardScaler (Feature Scaling)
* **One-Line Meaning:** Rescales numerical features so they have a mean of 0 and a standard deviation of 1.
* **Real-Life Analogy:** Converting rainfall (100–3500 mm) and soil pH (3.5–9.5) to a common grading scale so that rainfall doesn't dominate the math simply because its raw numbers are larger.

### 5. OneHotEncoder (Categorical Encoding)
* **OneHotEncoder:** Converts text categories (like "Rice", "Wheat") into binary 0-or-1 columns.
* **Real-Life Analogy:** Multiple-choice checkboxes on an application form where you tick `[X] Rice` and leave the other 53 crop boxes blank `[ ]`.

### 6. Scikit-Learn Pipeline
* **One-Line Meaning:** An automated assembly line that chains preprocessing and model estimation together.
* **Real-Life Analogy:** A factory conveyor belt where raw sugarcane is washed, chopped, crushed, and packaged into sugar in one unbroken sequence.

### 7. ColumnTransformer
* **One-Line Meaning:** Directs specific columns to specific preprocessing steps (numbers to scaler, text to one-hot encoder).
* **Real-Life Analogy:** A traffic police officer directing heavy trucks to the freight lane and passenger cars to the express lane.

### 8. Linear Regression (Baseline Model)
* **One-Line Meaning:** Fits a straight mathematical hyperplane through the data points assuming an additive relationship.
* **Real-Life Analogy:** Assuming that if 1 glass of water is good for a plant, 100 glasses of water will make it grow 100 times taller (which is false, because the plant drowns).

### 9. Random Forest Regressor (Champion Model)
* **One-Line Meaning:** An ensemble of 100 independent decision trees that vote together to produce a stable, non-linear prediction.
* **Real-Life Analogy:** Consulting 100 experienced farmers across the village. Each farmer gives an independent estimate based on soil and weather, and you take the average of all 100 answers to get the most reliable forecast.

### 10. Overfitting & `min_samples_leaf=5`
* **Overfitting:** When a model memorizes random noise in the training data rather than learning general rules.
* **`min_samples_leaf=5`:** Enforces that a decision tree cannot make a rule unless at least 5 similar historical records support it.
* **Real-Life Analogy:** Forbidding a doctor from declaring a new medical rule based on a single patient's fluke experience.

### Why Random Forest Beat Linear Regression
1. **Non-Linear Biological Curves:** Plants follow bell-shaped response curves. Extreme rainfall causes waterlogging; extreme heat causes flower abortion. Linear Regression forces a straight line, while Random Forest trees naturally split data into optimal plateaus.
2. **Multi-Way Nutrient Interactions:** Soil pH controls whether Nitrogen and Phosphorus dissolve in water for root absorption. Random Forest naturally splits on pH before testing NPK values, capturing chemistry interactions without manual math formulas.

---

## 10. RESULTS & EVALUATION METRICS

Evaluated on the exact same 20% held-out test set ($n = 3,904$ samples) from `models/model_metadata.json`:

| Metric | Linear Regression (Baseline) | Random Forest Regressor (Champion) | Improvement / Difference |
| :--- | :---: | :---: | :--- |
| **MAE (Mean Absolute Error)** | `2.8152 t/ha` | **`1.2291 t/ha`** | **56.3% lower error** |
| **MSE (Mean Squared Error)** | `41.5914` | **`24.3183`** | **41.5% reduction in squared error** |
| **RMSE (Root Mean Squared Error)** | `6.4491 t/ha` | **`4.9314 t/ha`** | **23.5% lower error** |
| **$R^2$ Score (Variance Explained)** | `0.7367` | **`0.8460`** | **+0.1093 (+14.8% more variance captured)** |

### Plain-Language Meaning of Each Metric
* **MAE (1.23 t/ha):** On average, the Random Forest model's prediction is off by only 1.23 tonnes per hectare from what actually grew historically.
* **RMSE (4.93 t/ha):** Similar to MAE, but heavily penalizes large rare misses. RF's lower RMSE shows it has far fewer catastrophic prediction mistakes than Linear Regression.
* **$R^2$ Score (0.8460):** The model explains **84.6% of all variation** in historical crop yields using just the 8 weather and soil inputs.

---

## 11. NEW FEATURES EXPLAINED

### 1. Crop Recommendation Engine
* **How It Works:** Takes the farmer's current 7 climate and soil values. In a single batch, it runs the trained Random Forest model across all 54 crop varieties. It sorts the predictions descending and outputs the top 5 crops.
* **Code Location:** `farm_logic.recommend_crops()` in `farm_logic.py`, displayed on Page 2 in `app.py`.

### 2. Yield Improvement Suggestions (Optimization Grid Search)
* **How It Works:** 
  1. Computes baseline yield $Y_{\text{base}}$ for the current inputs.
  2. Weather is locked (cannot be changed).
  3. Evaluates realistic candidate adjustments to soil pH ($\pm 0.4$), Nitrogen ($\pm 20$), Phosphorus ($\pm 15$), and Potassium ($\pm 15$) toward ICAR agronomic benchmarks.
  4. Runs each variation through Random Forest and calculates $\Delta \text{Yield} = Y_{\text{new}} - Y_{\text{base}}$.
  5. Ranks suggestions by highest yield gain first.
* **Code Location:** `farm_logic.suggest_improvements()` in `farm_logic.py`, displayed on Page 1 in `app.py`.

### 3. Crop-Specific Soil Advisory
* **How It Works:** Contains a dictionary of ideal nutrient and pH ranges for major Indian crops (Rice, Wheat, Maize, Cotton, Sugarcane, Groundnut, Gram, Potato) based on ICAR guidelines, plus a generic fallback. Labels each nutrient as "low", "balanced", or "high" and provides amendment advice (e.g. Urea for low N, Lime for acidic soil).
* **Code Location:** `farm_logic.get_soil_advisory()` in `farm_logic.py`.

### 4. Weather Risk Tips
* **How It Works:** Evaluates annual rainfall (<600mm indicates drought; >2200mm indicates waterlogging risk), temperature (>34°C heat stress), and humidity (>80% fungal disease microclimate) to provide preventive agricultural tips.
* **Code Location:** `farm_logic.get_weather_tips()` in `farm_logic.py`.

### 5. What-If Comparison
* **How It Works:** Lets the user manually adjust target soil values (pH, N, P, K) side-by-side against their current values and shows the live yield difference in t/ha and percentage.
* **Code Location:** `farm_logic.compare_what_if()` in `farm_logic.py`.

### 6. One-Click Report Download
* **How It Works:** Compiles inputs, predictions, confidence bounds, improvement suggestions, and disclaimers into a clean `.txt` file via Streamlit's `st.download_button`.
* **Code Location:** Inside Page 1 block in `app.py`.

---

## 12. MODEL CONTEXT PROTOCOL (MCP) CHATBOT EXPLAINED

### What is MCP?
**Model Context Protocol (MCP)** is an open industry protocol developed by Anthropic. It standardizes how AI applications and LLMs safely discover, call, and receive data from external tools, databases, and code execution servers.

### Why Use MCP in This Project?
Without MCP, an LLM would make up (hallucinate) believable-sounding but completely fictitious crop yield numbers. With MCP:
1. The LLM has **no permission** to invent yield numbers.
2. The LLM must request numerical calculations by calling tools on `mcp_server.py`.
3. Our verified Scikit-Learn pipelines compute the numbers and return them to the LLM.

### Tools Exposed by `mcp_server.py`
1. `predict_yield(crop, rainfall, temperature, humidity, pH, N, P, K)`
2. `recommend_crops(rainfall, temperature, humidity, pH, N, P, K, top_n=5)`
3. `suggest_improvements(crop, rainfall, temperature, humidity, pH, N, P, K)`
4. `get_soil_advisory(crop, pH, N, P, K)`
5. `list_supported_crops()`

### Example Conversation Step by Step
1. **Farmer:** *"Namaste! What will be my expected rice yield if rainfall is 1150mm, temp is 26.5C, humidity is 72%, pH is 6.5, N is 78, P is 35, and K is 28?"*
2. **`chatbot.py`:** Passes user query and tool schemas to LLM API (Groq/Gemini).
3. **LLM:** Decides to call tool: `predict_yield("Rice", 1150, 26.5, 72, 6.5, 78, 35, 28)`.
4. **FastMCP Server (`mcp_server.py`):** Runs the Random Forest pipeline and returns JSON:
   `{"crop": "Rice", "predicted_yield": 1.76, "mae": 1.23, "units": "t/ha"}`.
5. **LLM:** Reads the tool output and replies:
   *"Namaste! 🌾 Based on your soil and climate indicators, your expected Rice yield is approximately **1.76 Tonnes per Hectare (t/ha)** with a typical range of 0.53 to 2.99 t/ha. This is a model estimate based on historical data. Please confirm with your local Krishi Vigyan Kendra before making major purchases."*

### Security: Where the API Key is Stored
* The API key is stored in `.env` (`LLM_API_KEY=...`).
* `.env` is listed in `.gitignore` and is **never committed to GitHub or hardcoded in Python files**.
* `chatbot.py` loads the key into memory using `python-dotenv`.

---

## 13. HOW TO RUN ON WINDOWS

### Step-by-Step Commands

#### 1. Open Windows PowerShell and navigate to the project directory:
```powershell
cd "c:\Users\admin\Desktop\Academy(2026)\Yield-Predict\Yield-Predict"
```

#### 2. Activate the Virtual Environment:
```powershell
.\venv\Scripts\activate
```
*(You will see `(venv)` appear at the beginning of your command line.)*

#### 3. Install Dependencies:
```powershell
pip install -r requirements.txt
```

#### 4. Configure the `.env` File for Chatbot:
Copy the template file:
```powershell
copy .env.example .env
```
Open `.env` in Notepad and add your free Groq or Gemini key:
```ini
LLM_API_KEY=your_actual_key_here
LLM_MODEL=qwen/qwen3.8-27b
LLM_PROVIDER=groq
```

#### 5. Run the Automated Verification Suite:
```powershell
python test_pipeline.py
```
*(All 9 tests will run and print `[PASS]`.)*

#### 6. (Optional) Retrain Models:
```powershell
python train_model.py
```

#### 7. Launch the Streamlit Web Application:
```powershell
streamlit run app.py
```
Your browser will open at **`http://localhost:8501`**.

---

### Common Errors & Easy Fixes

| Error Encountered | Why It Happened | How to Fix It |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'joblib'` | You ran `python` from the global system instead of the virtual environment. | Run `.\venv\Scripts\python.exe` or activate venv with `.\venv\Scripts\activate`. |
| `Port 8501 is not available` | Another Streamlit instance is already running on port 8501. | Run with a different port: `streamlit run app.py --server.port 8502`. |
| `Kisan Mitra Setup Notice: The AI Chatbot requires an LLM API key` | `.env` was missing, named incorrectly (`.env.txt`), or not loaded. | Ensure `.env` is saved in the project folder with `LLM_API_KEY=...`. |
| `401 Unauthorized: Incorrect API key provided` | Key has a typo or wrong provider selected (e.g. sending OpenAI key to Gemini). | Verify key prefix (`gsk_` for Groq, `AIza` for Gemini, `sk-` for OpenAI) and set `LLM_PROVIDER` accordingly. |
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Windows terminal encoding (cp1252) cannot display crop emojis in stdout. | Already handled in `chatbot.py` via `sys.stdout.reconfigure(encoding="utf-8")`. |

---

## 14. 5-MINUTE DEMO SCRIPT FOR YOUR TEACHER

### Minute 1: Introduction & Title
* **Action:** Open browser at `http://localhost:8501`. Stay on Page 1.
* **Say:** *"Respected teachers, this is our project: 'Crop Yield Prediction using Weather and Soil Data'. Our system helps Indian farmers forecast harvest yields in tonnes per hectare using 8 meteorological and soil factors, and offers smart optimization suggestions."*

### Minute 2: Prediction & Confidence Margin
* **Action:** Leave crop as "Rice", adjust rainfall to 1150 mm, and click **`🚀 Predict Crop Yield`**.
* **Say:** *"When I click predict, the pre-trained Random Forest pipeline evaluates the inputs. We display the prediction prominently as 1.76 t/ha alongside a realistic confidence margin of $\pm 1.23\text{ t/ha}$ based on our held-out test MAE, avoiding false certainty."*

### Minute 3: Smart Yield Suggestions & What-If
* **Action:** Scroll down to "What You Can Do" and the "What-If Comparison".
* **Say:** *"Here is our core innovation: farmers cannot change rainfall, but they can condition their soil. Our algorithm simulates candidate soil adjustments holding weather constant. It recommends raising Nitrogen to gain +0.22 t/ha. In the what-if simulator, we can test custom soil values and see the yield difference in real-time."*

### Minute 4: Crop Recommendation & FastMCP Chatbot
* **Action:** Click Page 2 ("Crop Recommendation") to show the Plotly bar chart, then click Page 3 ("Farmer Assistant").
* **Say:** *"Page 2 simulates all 54 crops to recommend the top 5 varieties for this field. On Page 3, we built an AI Chatbot using the Model Context Protocol (FastMCP). When I ask a question, the LLM cannot fabricate numbers—it calls our FastMCP server tools and answers in English, Hindi, or Marathi."*

### Minute 5: Model Evaluation (Viva Proof)
* **Action:** Click Page 4 ("About Model").
* **Say:** *"Finally, for academic evaluation, Page 4 compares Linear Regression against Random Forest on 3,904 held-out test records. Random Forest achieved an $R^2$ of 0.8460 and 56.3% lower MAE than Linear Regression because decision trees effectively capture non-linear agricultural response curves and pH-nutrient interactions."*

---

## 15. LIMITATIONS AND FUTURE WORK

### Honest Limitations
1. **Regional Macro-Aggregations:** Soil indicators represent state averages (30 states), and weather data is annual regional averages. Field-level precision agriculture exhibits microclimate and topographical variations that macro-data cannot capture.
2. **Model Estimates vs. Causal Biology:** The yield improvement engine finds mathematical maxima in the regression function. It provides educational guidance, not guaranteed biological cause-and-effect.
3. **Unmodeled Exogenous Agronomic Factors:** Factors such as seed genetics (hybrid vs indigenous), pest infestations, micro-irrigation schedules, and sudden hailstorms are not recorded in the tabular dataset.
4. **Chatbot Requires Internet & API Key:** The MCP AI chatbot relies on cloud LLM APIs (Groq/Gemini), requiring internet connectivity.

### Future Work & Improvements
1. **IoT Sensor Integration:** Connect low-cost NPK and moisture sensors (e.g. Arduino / ESP32) to stream real-time field data into the Streamlit app.
2. **Hyperlocal Weather APIs:** Integrate free Open-Meteo or IMD live APIs to automatically pull forecast rainfall and temperature based on the farmer's GPS coordinates.
3. **Offline Small LLM:** Run a quantized local model (like Ollama with LLaMA 3.2 1B or Qwen 1.5B) locally on the laptop so the chatbot works without an internet connection.

---

## 16. WHAT IS INCOMPLETE OR NOT WORKING

1. **No Real-Time Field Sensors:** All 8 inputs are currently entered manually using sliders; there is no live hardware IoT sensor connection.
2. **Pesticide / Fungicide Dosage:** Intentionally omitted. The system deliberately refuses to prescribe chemical pesticide dosages for safety and ethical reasons, directing farmers to agricultural officers instead.
3. **Coconut Crop Modeling:** Coconut was filtered out of the continuous dataset because its production was officially recorded in thousands of nuts/ha rather than metric tonnes/ha. Coconut cannot be predicted in tonnes/ha.

---

## 17. GLOSSARY (Technical Terms Defined in 1 Line)

| Technical Term | 1-Line Meaning |
| :--- | :--- |
| **Regression** | An ML task predicting a continuous numerical value instead of categories. |
| **Random Forest** | An ensemble algorithm combining 100 decision trees to make accurate, non-linear predictions. |
| **Linear Regression** | A baseline algorithm fitting a straight additive line through data points. |
| **$R^2$ Score** | The proportion of variance in the target variable explained by the input features (1.0 is perfect). |
| **MAE (Mean Absolute Error)** | The average positive difference between predicted yield and actual harvest yield. |
| **RMSE (Root Mean Squared Error)** | The square root of squared errors; gives heavier penalties to large mistakes. |
| **Data Leakage** | When test information is accidentally shared with training data, inflating accuracy artificially. |
| **StandardScaler** | Transforms numbers so their mean is 0 and standard deviation is 1. |
| **OneHotEncoder** | Converts text categories into separate binary (0 or 1) columns. |
| **Scikit-Learn Pipeline** | A chain that automatically applies preprocessing and model inference in sequence. |
| **ColumnTransformer** | Directs different preprocessing pipelines to different subsets of columns. |
| **Joblib** | A Python library for efficiently saving and loading trained Scikit-Learn models to/from disk. |
| **Streamlit** | A Python framework that creates interactive web dashboards directly from Python code. |
| **Plotly** | A library that generates interactive, responsive charts and scatter plots. |
| **Model Context Protocol (MCP)** | An open protocol allowing AI models to execute external tools and fetch verified data safely. |
| **FastMCP** | The high-level Python SDK for defining and serving MCP tools quickly. |
| **LLM (Large Language Model)** | An AI system (like Groq/Gemini/GPT) trained on vast text to understand and generate natural language. |
| **`st.session_state`** | Streamlit's in-memory storage that preserves user inputs and variables across script reruns. |
| **Heuristic** | A practical, rule-of-thumb guideline based on experience rather than rigid mathematical certainty. |
| **Tonnes per Hectare (t/ha)** | The standard international agricultural unit of yield ($\text{production in tonnes} / \text{area in hectares}$). |
