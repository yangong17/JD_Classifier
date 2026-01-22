"""
Web UI for Job Description Classifier.
Flask-based interface for running Phase 1 and Phase 2 classification.
"""

import os
import re
import json
import glob
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, send_file
from dotenv import load_dotenv
from prompts import FACTORS, PHASE1_PROMPT, PHASE2_PROMPT

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Folders for file handling
UPLOAD_FOLDER = "/tmp/classifier_uploads"
OUTPUT_FOLDER = "/tmp/classifier_outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Available models with pricing (per 1M tokens)
MODELS = {
    "gemini-2.0-flash": {
        "provider": "google",
        "name": "Gemini 2.0 Flash (Fast)",
        "input_cost": 0.10,
        "output_cost": 0.40,
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "name": "Gemini 2.5 Flash (Balanced)",
        "input_cost": 0.15,
        "output_cost": 0.60,
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "name": "Gemini 2.5 Pro (Advanced)",
        "input_cost": 1.25,
        "output_cost": 10.00,
    },
    "gemini-3-pro": {
        "provider": "google",
        "name": "Gemini 3 Pro (Most Advanced)",
        "input_cost": 2.50,
        "output_cost": 15.00,
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "name": "Claude Sonnet 4",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "claude-sonnet-4-5-20250514": {
        "provider": "anthropic",
        "name": "Claude Sonnet 4.5",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
}

# Content placeholder marker
CONTENT_PLACEHOLDER = "{content}"

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are a highly seasoned compensation consultant with over 30 years of experience conducting municipal compensation studies. Your task is to classify compensable factors in the following job-related text across nine dimensions. Assign a score (1–5) for each factor.

Guidelines:
- Use only explicitly stated information in the text.

Factors to rate (with brief descriptors):
1. Knowledge and Skills — depth and breadth of expertise required
2. Problem Solving and Complexity — judgment and complexity of work
3. Decision Authority — autonomy and discretion in decisions
4. Impact and Organizational Scope — breadth and significance of impact
5. Stakeholder Interaction and Influence — interaction complexity and influence level
6. Experience Relevance — amount of prior relevant experience needed
7. Supervisory Responsibility — people leadership and accountability for others
8. Budget and Resource Accountability — responsibility for budget, assets, or allocation
9. Working Conditions — environmental or contextual job demands

Text to analyze:
{content}

Instructions:
Rate each factor using the 1–5 scale. Use only explicitly stated information and avoid assumptions. If information is ambiguous or missing, default to the more conservative (lower) rating.

Respond in the following exact format (labels plus number, one per line, no extra text):

Knowledge and Skills: 3
Problem Solving and Complexity: 3
Decision Authority: 3
Impact and Organizational Scope: 3
Stakeholder Interaction and Influence: 3
Experience Relevance: 3
Supervisory Responsibility: 3
Budget and Resource Accountability: 3
Working Conditions: 3"""


# =============================================================================
# LLM Client Functions (from classifier.py)
# =============================================================================


def get_llm_client(model_name: str, google_api_key: str = None, anthropic_api_key: str = None):
    """Initialize the appropriate LLM client based on model name."""
    if model_name.startswith("gemini"):
        from google import genai

        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API Key not provided. Enter it in the UI or set GOOGLE_API_KEY in .env file.")
        client = genai.Client(api_key=api_key)
        return ("google", client)

    elif model_name.startswith("claude"):
        import anthropic

        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API Key not provided. Enter it in the UI or set ANTHROPIC_API_KEY in .env file.")
        client = anthropic.Anthropic(api_key=api_key)
        return ("anthropic", client)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def call_llm(provider: str, client, model: str, prompt: str) -> str:
    """Call the LLM and return the response text."""
    if provider == "google":
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text

    elif provider == "anthropic":
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_phase1_response(response: str) -> dict:
    """Parse Phase 1 response (Yes/No for each factor)."""
    result = {}
    for factor in FACTORS:
        pattern = rf"{re.escape(factor)}:\s*(Yes|No)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result[factor] = 1 if match.group(1).lower() == "yes" else 0
        else:
            result[factor] = -1
    return result


def parse_phase2_response(response: str) -> dict:
    """Parse Phase 2 response (1-5 score for each factor)."""
    result = {}
    for factor in FACTORS:
        pattern = rf"{re.escape(factor)}:\s*\[?(\d)\]?"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            result[factor] = score if 1 <= score <= 5 else -1
        else:
            result[factor] = -1
    return result


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approx 4 chars per token)."""
    return len(text) // 4


def get_available_csvs():
    """Get list of CSV files in the project directory."""
    csv_files = glob.glob("*.csv")
    return sorted(csv_files)


# =============================================================================
# Routes
# =============================================================================


@app.route("/")
def index():
    """Main page."""
    # Get all CSV files in directory
    csv_files = get_available_csvs()
    
    # Get columns from first CSV if available
    columns = []
    first_csv = csv_files[0] if csv_files else None
    if first_csv:
        try:
            df = pd.read_csv(first_csv)
            columns = df.columns.tolist()
        except:
            pass

    return render_template(
        "index.html",
        models=MODELS,
        factors=FACTORS,
        csv_files=csv_files,
        columns=columns,
        default_prompt=DEFAULT_PROMPT_TEMPLATE,
        content_placeholder=CONTENT_PLACEHOLDER,
        google_key_set=bool(os.getenv("GOOGLE_API_KEY")),
        anthropic_key_set=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


@app.route("/get_columns/<path:csv_file>")
def get_columns(csv_file):
    """Get columns for a specific CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return jsonify({"columns": df.columns.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/get_sample", methods=["POST"])
def get_sample():
    """Get sample data and generate preview prompt."""
    data = request.get_json()
    csv_file = data.get("csv_file")
    column = data.get("column")
    prompt_template = data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    
    try:
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found"}), 400
        
        # Get first non-empty value
        sample_content = None
        for val in df[column]:
            if pd.notna(val) and str(val).strip():
                sample_content = str(val)
                break
        
        if not sample_content:
            sample_content = "[No sample data available]"
        
        # Generate preview prompt
        preview_prompt = prompt_template.replace(CONTENT_PLACEHOLDER, sample_content)
        
        return jsonify({
            "sample_content": sample_content,
            "preview_prompt": preview_prompt
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/estimate_cost", methods=["POST"])
def estimate_cost():
    """Estimate cost for processing."""
    data = request.get_json()
    csv_file = data.get("csv_file")
    column = data.get("column")
    prompt_template = data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    row_limit = data.get("row_limit")
    
    try:
        df = pd.read_csv(csv_file)
        if row_limit:
            df = df.head(int(row_limit))
        
        total_rows = len(df)
        
        # Calculate average content length
        if column in df.columns:
            avg_content_len = df[column].astype(str).str.len().mean()
        else:
            avg_content_len = 500
        
        # Estimate tokens per request (approx 4 chars per token)
        prompt_base_len = len(prompt_template) - len(CONTENT_PLACEHOLDER)
        avg_input_tokens = int((prompt_base_len + avg_content_len) / 4)
        avg_output_tokens = 200  # Approximate output tokens
        
        total_input_tokens = avg_input_tokens * total_rows
        total_output_tokens = avg_output_tokens * total_rows
        
        # Calculate cost for each model
        cost_estimates = {}
        for model_id, model_info in MODELS.items():
            input_cost = (total_input_tokens / 1_000_000) * model_info["input_cost"]
            output_cost = (total_output_tokens / 1_000_000) * model_info["output_cost"]
            total_cost = input_cost + output_cost
            cost_estimates[model_id] = {
                "name": model_info["name"],
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(total_cost, 4),
            }
        
        return jsonify({
            "total_rows": total_rows,
            "avg_input_tokens": int(avg_input_tokens),
            "avg_output_tokens": avg_output_tokens,
            "total_input_tokens": int(total_input_tokens),
            "total_output_tokens": total_output_tokens,
            "cost_estimates": cost_estimates
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/process", methods=["POST"])
def process():
    """Process the CSV with the selected settings."""
    data = request.get_json()

    phase = data.get("phase", "1")
    model = data.get("model", "gemini-2.0-flash")
    column = data.get("column", "description")
    row_limit = data.get("row_limit")
    prompt_template = data.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    csv_path = data.get("csv_path", "job_descriptions.csv")
    google_api_key = data.get("google_api_key", "").strip() or None
    anthropic_api_key = data.get("anthropic_api_key", "").strip() or None

    if row_limit:
        row_limit = int(row_limit)

    def generate():
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            if row_limit:
                df = df.head(row_limit)

            total = len(df)
            yield f"data: {json.dumps({'status': 'starting', 'total': total})}\n\n"

            # Initialize LLM
            provider, client = get_llm_client(model, google_api_key, anthropic_api_key)

            # Select parse function based on phase
            parse_func = parse_phase1_response if phase == "1" else parse_phase2_response

            results = []
            for i, row in df.iterrows():
                job_desc = str(row[column])
                title = row.get("title", "N/A")[:40]

                yield f"data: {json.dumps({'status': 'processing', 'current': i + 1, 'total': total, 'title': title})}\n\n"

                # Build prompt by replacing content placeholder
                prompt = prompt_template.replace(CONTENT_PLACEHOLDER, job_desc)

                try:
                    response = call_llm(provider, client, model, prompt)
                    parsed = parse_func(response)
                    parsed["_raw"] = response
                except Exception as e:
                    parsed = {f: -1 for f in FACTORS}
                    parsed["_raw"] = f"ERROR: {e}"

                results.append(parsed)

            # Build output
            output_df = df.copy()
            suffix = "Check" if phase == "1" else "Score"
            for factor in FACTORS:
                output_df[f"{factor} {suffix}"] = [r.get(factor, -1) for r in results]
            output_df[f"Phase{phase}_Raw"] = [r.get("_raw", "") for r in results]

            # Save
            output_file = f"classification_phase{phase}.csv"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)
            output_df.to_csv(output_path, index=False)

            yield f"data: {json.dumps({'status': 'complete', 'file': output_file, 'rows': total})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/download/<filename>")
def download(filename):
    """Download the output file."""
    path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404


if __name__ == "__main__":
    print("Starting Classifier UI...")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)
