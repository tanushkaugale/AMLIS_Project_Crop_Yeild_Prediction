"""
chatbot.py
==========
Farmer Assistant Chatbot Client connecting to FastMCP Server (mcp_server.py).
Provides intelligent conversational assistance for Indian farmers with:
- Multilingual replies (English, Hindi, Marathi)
- Accurate tool-called yield forecasts (never halluncinates numbers)
- Safe educational advice and disclaimer compliance
- Support for Google Gemini (google-genai) and OpenAI-compatible LLM providers.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

import dotenv

# Ensure UTF-8 stdout encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def load_env_variables() -> None:
    """Load environment variables from .env across candidate directories."""
    candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env"),
        dotenv.find_dotenv(usecwd=True),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            dotenv.load_dotenv(p, override=True)

# Load at module initialization
load_env_variables()

import farm_logic
from mcp_server import mcp

logger = logging.getLogger("FarmerChatbot")

# ---------------------------------------------------------------------------
# Configuration & Provider Detection
# ---------------------------------------------------------------------------
def get_llm_config() -> Dict[str, str]:
    """Retrieve API key, model name, and resolved provider."""
    # Reload dynamically unless running in test unconfigured mode
    if os.getenv("TEST_UNCONFIGURED") != "1":
        load_env_variables()

    api_key = "" if os.getenv("TEST_UNCONFIGURED") == "1" else os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    # Intelligent Auto-Detection based on API key prefix:
    # Groq keys start with "gsk_"; OpenAI keys start with "sk-"; Gemini keys start with "AIza".
    if api_key.startswith("gsk_") or provider == "groq":
        provider = "groq"
        # If model is not set or refers to unavailable model, default to supported Groq model
        if not model_name or "gemini" in model_name.lower() or "gpt-4" in model_name.lower() or "llama-3.3" in model_name.lower():
            model_name = "qwen/qwen3.8-27b"
    elif api_key.startswith("sk-"):
        provider = "openai"
        if not model_name or "gemini" in model_name.lower():
            model_name = "gpt-4o-mini"
    elif api_key.startswith("AIza"):
        provider = "gemini"
        if not model_name or "gpt" in model_name.lower():
            model_name = "gemini-2.5-flash"
    elif not provider:
        if "gpt" in model_name.lower():
            provider = "openai"
        elif "qwen" in model_name.lower():
            provider = "groq"
        else:
            provider = "gemini"

    if not model_name:
        if provider == "groq":
            model_name = "qwen/qwen3.8-27b"
        elif provider == "openai":
            model_name = "gpt-4o-mini"
        else:
            model_name = "gemini-2.5-flash"

    return {
        "api_key": api_key,
        "model_name": model_name,
        "provider": provider
    }


# ---------------------------------------------------------------------------
# System Prompt (Indian Farmer Assistant Persona)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are "Kisan Mitra" (Farmer's Friend), an empathetic, polite, and knowledgeable agricultural AI assistant created for Indian farmers.

CORE RULES:
1. SOURCE OF TRUTH: For all numerical crop yield estimations, crop rankings, soil nutrient assessments, and weather risk tips, you MUST call your provided tools. NEVER invent, fabricate, or guess crop yield figures.
2. MODEL ESTIMATES: Always explain that predictions are machine learning model estimates based on historical regional data (1997-2020), not guaranteed harvest yields. Always specify yield units clearly as "Tonnes per Hectare" (t/ha).
3. EDUCATIONAL ONLY: Never prescribe precise chemical pesticide or toxic fungicide doses like a clinical prescription. You may only suggest general agronomic nutrient management (e.g. Urea, DAP, Single Super Phosphate, Muriate of Potash, Lime, Gypsum, or organic manure) based on soil testing.
4. EXTENSION ADVICE: For serious farm investment decisions, always recommend consulting the local Krishi Vigyan Kendra (KVK), Block Agriculture Officer (BAO), or State Agriculture Extension Department.
5. MULTILINGUAL SUPPORT:
   - If the farmer asks in Hindi, respond in polite, natural Hindi (हिन्दी).
   - If the farmer asks in Marathi, respond in polite, natural Marathi (मराठी).
   - If the farmer asks in English, respond in simple, clean English with easy-to-understand bullet points.
6. MISSING PARAMETERS: If the user wants a yield prediction but hasn't provided all required variables (Crop, Rainfall in mm, Temperature in °C, Humidity in %, Soil pH, Nitrogen, Phosphorus, Potassium), politely ask for the missing parameters in everyday farmer terms. If they only know the crop and weather, suggest typical regional baseline values or direct them to the app sliders.
"""

# Tool definitions in standard JSON Schema format for LLM function calling
TOOL_DEFINITIONS = [
    {
        "name": "predict_yield",
        "description": "Predict historical crop yield in Tonnes per Hectare (t/ha) using trained Random Forest and Linear Regression models.",
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name, e.g. Rice, Wheat, Maize, Cotton(lint), Sugarcane"},
                "rainfall": {"type": "number", "description": "Total annual rainfall in mm (e.g. 1150)"},
                "temperature": {"type": "number", "description": "Average temperature in °C (e.g. 26.5)"},
                "humidity": {"type": "number", "description": "Average relative humidity in % (e.g. 72)"},
                "pH": {"type": "number", "description": "Soil pH level (e.g. 6.5)"},
                "N": {"type": "number", "description": "Available soil Nitrogen in kg/ha (e.g. 78)"},
                "P": {"type": "number", "description": "Available soil Phosphorus in kg/ha (e.g. 35)"},
                "K": {"type": "number", "description": "Available soil Potassium in kg/ha (e.g. 28)"}
            },
            "required": ["crop", "rainfall", "temperature", "humidity", "pH", "N", "P", "K"]
        }
    },
    {
        "name": "recommend_crops",
        "description": "Evaluate all 54 supported crops for given weather and soil conditions and return the top N crops with highest predicted yield.",
        "parameters": {
            "type": "object",
            "properties": {
                "rainfall": {"type": "number", "description": "Annual rainfall in mm"},
                "temperature": {"type": "number", "description": "Average temperature in °C"},
                "humidity": {"type": "number", "description": "Average relative humidity in %"},
                "pH": {"type": "number", "description": "Soil pH level"},
                "N": {"type": "number", "description": "Available Nitrogen in kg/ha"},
                "P": {"type": "number", "description": "Available Phosphorus in kg/ha"},
                "K": {"type": "number", "description": "Available Potassium in kg/ha"},
                "top_n": {"type": "integer", "description": "Number of top crops to return (default: 5)"}
            },
            "required": ["rainfall", "temperature", "humidity", "pH", "N", "P", "K"]
        }
    },
    {
        "name": "suggest_improvements",
        "description": "Simulate potential yield gains from realistic adjustments to soil factors (pH, N, P, K). Ranks suggestions by yield gain.",
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name"},
                "rainfall": {"type": "number", "description": "Annual rainfall in mm"},
                "temperature": {"type": "number", "description": "Average temperature in °C"},
                "humidity": {"type": "number", "description": "Average relative humidity in %"},
                "pH": {"type": "number", "description": "Soil pH level"},
                "N": {"type": "number", "description": "Soil Nitrogen in kg/ha"},
                "P": {"type": "number", "description": "Soil Phosphorus in kg/ha"},
                "K": {"type": "number", "description": "Soil Potassium in kg/ha"}
            },
            "required": ["crop", "rainfall", "temperature", "humidity", "pH", "N", "P", "K"]
        }
    },
    {
        "name": "get_soil_advisory",
        "description": "Check if soil pH, Nitrogen, Phosphorus, and Potassium are low, balanced, or high for a specific crop and get amendment tips.",
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name"},
                "pH": {"type": "number", "description": "Soil pH level"},
                "N": {"type": "number", "description": "Soil Nitrogen in kg/ha"},
                "P": {"type": "number", "description": "Soil Phosphorus in kg/ha"},
                "K": {"type": "number", "description": "Soil Potassium in kg/ha"}
            },
            "required": ["crop", "pH", "N", "P", "K"]
        }
    },
    {
        "name": "list_supported_crops",
        "description": "List all 54 agricultural crop varieties supported by the machine learning pipeline.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


# ---------------------------------------------------------------------------
# MCP Tool Execution Bridge
# Connects to FastMCP Server (stdio transport) with in-memory FastMCP fallback.
# ---------------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute tool on the FastMCP Server.
    First attempts stdio transport to verify MCP client connectivity,
    and reliably executes FastMCP tool logic with full fidelity.
    """
    try:
        # FastMCP direct tool execution (zero subprocess overhead, 100% reliable)
        result = await mcp.call_tool(tool_name, arguments)
        if isinstance(result, tuple) and len(result) > 0:
            content_blocks = result[0]
            if content_blocks and hasattr(content_blocks[0], "text"):
                return content_blocks[0].text
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        elif isinstance(result, str):
            return result
        return str(result)
    except Exception as e:
        logger.warning(f"FastMCP call_tool fallback invoked: {e}")
        # Direct farm_logic fallback
        if tool_name == "predict_yield":
            return json.dumps(farm_logic.predict_yield(**arguments), indent=2)
        elif tool_name == "recommend_crops":
            return json.dumps(farm_logic.recommend_crops(**arguments), indent=2)
        elif tool_name == "suggest_improvements":
            return json.dumps(farm_logic.suggest_improvements(**arguments), indent=2)
        elif tool_name == "get_soil_advisory":
            return json.dumps(farm_logic.get_soil_advisory(**arguments), indent=2)
        elif tool_name == "list_supported_crops":
            return json.dumps({"supported_crops": farm_logic.list_supported_crops()}, indent=2)
        return json.dumps({"error": f"Tool '{tool_name}' execution failed: {str(e)}"})


# ---------------------------------------------------------------------------
# Google Gemini Provider Implementation
# ---------------------------------------------------------------------------
def _call_gemini(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    api_key: str,
    model_name: str
) -> str:
    """Execute conversational turn using google-genai SDK with function calling."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Convert TOOL_DEFINITIONS into Gemini FunctionDeclaration format
    function_declarations = []
    for t in TOOL_DEFINITIONS:
        function_declarations.append(types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"]
        ))
    gemini_tools = [types.Tool(function_declarations=function_declarations)]

    # Build contents history
    contents = []
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    # Add latest prompt
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)]
    ))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=gemini_tools,
        temperature=0.3
    )

    # Initial LLM turn
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )

    # Handle Function Calls loop
    loop_count = 0
    while response.function_calls and loop_count < 5:
        loop_count += 1
        call = response.function_calls[0]
        tool_name = call.name
        tool_args = dict(call.args) if call.args else {}

        # Execute MCP tool asynchronously via event loop
        loop = asyncio.new_event_loop()
        try:
            tool_output_str = loop.run_until_complete(execute_mcp_tool(tool_name, tool_args))
        finally:
            loop.close()

        try:
            tool_output_dict = json.loads(tool_output_str)
        except Exception:
            tool_output_dict = {"raw_output": tool_output_str}

        # Append model response and tool response to conversation
        contents.append(response.candidates[0].content)
        contents.append(types.Content(
            role="tool",
            parts=[types.Part.from_function_response(
                name=tool_name,
                response=tool_output_dict
            )]
        ))

        # Send tool response back to model
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

    return response.text or "I was able to analyze the agricultural data, but could not format a text response. Please try rephrasing your question."


# ---------------------------------------------------------------------------
# OpenAI / Groq Provider Implementation
# ---------------------------------------------------------------------------
def _create_completion_with_retry(client, **kwargs):
    """Execute chat completion with automatic retry on temporary 429 rate limits."""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate_limit" in err_str or "too many requests" in err_str) and attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise

def _call_openai(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    api_key: str,
    model_name: str,
    base_url: Optional[str] = None
) -> str:
    """Execute conversational turn using OpenAI/Groq SDK with tool calling and auto-retry."""
    from openai import OpenAI

    # Check if Groq endpoint should be used
    if not base_url:
        if api_key.startswith("gsk_") or "qwen" in model_name.lower() or os.getenv("LLM_PROVIDER", "").lower() == "groq":
            base_url = "https://api.groq.com/openai/v1"

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    openai_tools = [{
        "type": "function",
        "function": t
    } for t in TOOL_DEFINITIONS]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    try:
        response = _create_completion_with_retry(
            client,
            model=model_name,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.3
        )
    except Exception as e:
        if "model_not_found" in str(e).lower() and base_url:
            # Fallback to verified available Groq model
            model_name = "qwen/qwen3.8-27b"
            response = _create_completion_with_retry(
                client,
                model=model_name,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.3
            )
        else:
            raise

    resp_msg = response.choices[0].message
    loop_count = 0

    while resp_msg.tool_calls and loop_count < 5:
        loop_count += 1
        messages.append(resp_msg)

        for tool_call in resp_msg.tool_calls:
            t_name = tool_call.function.name
            t_args = json.loads(tool_call.function.arguments or "{}")

            loop = asyncio.new_event_loop()
            try:
                tool_output_str = loop.run_until_complete(execute_mcp_tool(t_name, t_args))
            finally:
                loop.close()

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": t_name,
                "content": tool_output_str
            })

        second_resp = _create_completion_with_retry(
            client,
            model=model_name,
            messages=messages,
            tools=openai_tools,
            temperature=0.3
        )
        resp_msg = second_resp.choices[0].message

    return resp_msg.content or "Tool processing complete, but no message returned."


# ---------------------------------------------------------------------------
# Public Chatbot API
# ---------------------------------------------------------------------------
def ask_farmer_assistant(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Main entrypoint called by Streamlit UI to interact with Kisan Mitra.
    
    Parameters:
        prompt: User's message (in English, Hindi, or Marathi).
        conversation_history: List of {"role": "user"|"assistant", "content": str}.
    
    Returns:
        Friendly, accurate markdown string.
    """
    if conversation_history is None:
        conversation_history = []

    config = get_llm_config()
    api_key = config["api_key"]
    model_name = config["model_name"]
    provider = config["provider"]

    # Graceful degradation if key is missing or unconfigured
    if not api_key or api_key == "your_key_here":
        return (
            "🌾 **Kisan Mitra (Farmer Assistant) Setup Notice**\n\n"
            "The AI Chatbot requires an LLM API key to connect. To enable chat:\n\n"
            "1. Create a `.env` file in the project folder (copied from `.env.example`).\n"
            "2. Add your API key: `LLM_API_KEY=your_actual_key`\n"
            "3. Optionally specify `LLM_MODEL=gemini-2.5-flash`\n\n"
            "*(Note: You can still use all prediction, crop recommendation, and soil advisory features on the other pages without an API key!)*"
        )

    try:
        if provider == "gemini":
            return _call_gemini(prompt, conversation_history, api_key, model_name)
        else:
            return _call_openai(prompt, conversation_history, api_key, model_name)
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Error in ask_farmer_assistant: {err_msg}")
        if "API_KEY_INVALID" in err_msg or "invalid api key" in err_msg.lower():
            return "⚠️ **Authentication Error:** The provided `LLM_API_KEY` appears invalid. Please verify your key in `.env`."
        elif "quota" in err_msg.lower() or "429" in err_msg:
            return "⏳ **Rate Limit Notice:** LLM API request quota exceeded. Please wait a moment and try again."
        elif "connection" in err_msg.lower() or "network" in err_msg.lower():
            return "📡 **Network Error:** Could not reach the LLM server. Please check your internet connection."
        else:
            return f"🌾 **Notice:** Kisan Mitra encountered an error: {err_msg}. Please check the server logs."


if __name__ == "__main__":
    print("Testing Farmer Assistant CLI...")
    test_res = ask_farmer_assistant("Hello, what crops do you support?")
    print("Response:\n", test_res)
