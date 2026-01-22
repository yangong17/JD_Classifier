"""
Job Description Classifier using LLMs.
Two-phase classification: Phase 1 (binary check), Phase 2 (detailed scoring).
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
from prompts import FACTORS, PHASE1_PROMPT, PHASE2_PROMPT

# Load environment variables
load_dotenv()

# =============================================================================
# SETTINGS - Modify these values as needed
# =============================================================================

INPUT_CSV = "job_descriptions.csv"
COLUMN_TO_ANALYZE = "description"
ROW_LIMIT = 5  # Set to None to process all rows

# Optional text to wrap around the job description
PRE_PROMPT = ""
POST_PROMPT = ""

# Available models:
# Google: "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro"
# Anthropic: "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250514"

PHASE1_MODEL = "gemini-2.0-flash"
PHASE2_MODEL = "gemini-3-pro"

# Output files
PHASE1_OUTPUT = "classification_check.csv"
PHASE2_OUTPUT = "job_scores.csv"

# =============================================================================
# LLM Client Setup
# =============================================================================


def get_llm_client(model_name: str):
    """Initialize the appropriate LLM client based on model name."""
    if model_name.startswith("gemini"):
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        client = genai.Client(api_key=api_key)
        return ("google", client)

    elif model_name.startswith("claude"):
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
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


# =============================================================================
# Parsing Functions
# =============================================================================


def parse_phase1_response(response: str) -> dict:
    """Parse Phase 1 response (Yes/No for each factor)."""
    result = {}
    for factor in FACTORS:
        # Look for "Factor Name: Yes" or "Factor Name: No"
        pattern = rf"{re.escape(factor)}:\s*(Yes|No)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result[factor] = 1 if match.group(1).lower() == "yes" else 0
        else:
            result[factor] = -1  # Could not parse
    return result


def parse_phase2_response(response: str) -> dict:
    """Parse Phase 2 response (1-5 score for each factor)."""
    result = {}
    for factor in FACTORS:
        # Look for "Factor Name: 3" or "Factor Name: [3]"
        pattern = rf"{re.escape(factor)}:\s*\[?(\d)\]?"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            result[factor] = score if 1 <= score <= 5 else -1
        else:
            result[factor] = -1  # Could not parse
    return result


# =============================================================================
# Phase 1: Classification Check
# =============================================================================


def run_phase1():
    """Run Phase 1: Check if factors are present in job descriptions."""
    print(f"=== Phase 1: Classification Check ===")
    print(f"Model: {PHASE1_MODEL}")
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {PHASE1_OUTPUT}")
    print()

    # Load data
    df = pd.read_csv(INPUT_CSV)
    if ROW_LIMIT:
        df = df.head(ROW_LIMIT)
    print(f"Processing {len(df)} rows...")

    # Initialize LLM client
    provider, client = get_llm_client(PHASE1_MODEL)

    # Process each row
    results = []
    for i, row in df.iterrows():
        job_desc = str(row[COLUMN_TO_ANALYZE])
        print(f"  Row {i + 1}/{len(df)}: {row.get('title', 'N/A')[:50]}...")

        # Build prompt
        prompt = PHASE1_PROMPT.format(
            pre_prompt=PRE_PROMPT, job_description=job_desc, post_prompt=POST_PROMPT
        )

        try:
            response = call_llm(provider, client, PHASE1_MODEL, prompt)
            parsed = parse_phase1_response(response)
            parsed["_raw_response"] = response
        except Exception as e:
            print(f"    Error: {e}")
            parsed = {f: -1 for f in FACTORS}
            parsed["_raw_response"] = f"ERROR: {e}"

        results.append(parsed)

    # Build output dataframe
    output_df = df.copy()
    for factor in FACTORS:
        col_name = f"{factor} Check"
        output_df[col_name] = [r.get(factor, -1) for r in results]
    output_df["Phase1_Raw_Response"] = [r.get("_raw_response", "") for r in results]

    # Save
    output_df.to_csv(PHASE1_OUTPUT, index=False)
    print(f"\nSaved to {PHASE1_OUTPUT}")
    return output_df


# =============================================================================
# Phase 2: Detailed Scoring
# =============================================================================


def run_phase2():
    """Run Phase 2: Score each factor from 1-5."""
    print(f"=== Phase 2: Detailed Scoring ===")
    print(f"Model: {PHASE2_MODEL}")
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {PHASE2_OUTPUT}")
    print()

    # Load data
    df = pd.read_csv(INPUT_CSV)
    if ROW_LIMIT:
        df = df.head(ROW_LIMIT)
    print(f"Processing {len(df)} rows...")

    # Initialize LLM client
    provider, client = get_llm_client(PHASE2_MODEL)

    # Process each row
    results = []
    for i, row in df.iterrows():
        job_desc = str(row[COLUMN_TO_ANALYZE])
        print(f"  Row {i + 1}/{len(df)}: {row.get('title', 'N/A')[:50]}...")

        # Build prompt
        prompt = PHASE2_PROMPT.format(
            pre_prompt=PRE_PROMPT, job_description=job_desc, post_prompt=POST_PROMPT
        )

        try:
            response = call_llm(provider, client, PHASE2_MODEL, prompt)
            parsed = parse_phase2_response(response)
            parsed["_raw_response"] = response
        except Exception as e:
            print(f"    Error: {e}")
            parsed = {f: -1 for f in FACTORS}
            parsed["_raw_response"] = f"ERROR: {e}"

        results.append(parsed)

    # Build output dataframe
    output_df = df.copy()
    for factor in FACTORS:
        col_name = f"{factor} Score"
        output_df[col_name] = [r.get(factor, -1) for r in results]
    output_df["Phase2_Raw_Response"] = [r.get("_raw_response", "") for r in results]

    # Save
    output_df.to_csv(PHASE2_OUTPUT, index=False)
    print(f"\nSaved to {PHASE2_OUTPUT}")
    return output_df


# =============================================================================
# Main
# =============================================================================


def main():
    """Run both phases."""
    print("Job Description Classifier")
    print("=" * 50)
    print()

    print("Select phase to run:")
    print("  1. Phase 1 - Classification Check (Yes/No)")
    print("  2. Phase 2 - Detailed Scoring (1-5)")
    print("  3. Run Both Phases")
    print()

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        run_phase1()
    elif choice == "2":
        run_phase2()
    elif choice == "3":
        run_phase1()
        print()
        run_phase2()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
