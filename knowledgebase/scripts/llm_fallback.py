import json
import os
from datetime import datetime, timezone
from google import genai

# =============================
# Configuration
# =============================

PIPELINE_VERSION = "v1.3.0"
MAX_RETRIES = 3

ALLOWED_BASE = [
    "buzz", "crew", "crop", "fade", "undercut",
    "quiff", "pompadour", "fringe", "slick_back",
    "mohawk", "curtain", "mullet", "afro",
    "dreadlocks", "long_flow", "skater",
    "edgar", "side_part", "business",
    "shaved", "other"
]

ALLOWED_LENGTH = ["ultra_short", "short", "medium", "long"]

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# =============================
# Utilities
# =============================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_prompt_template():
    with open("C:\\Reaserch\\Knowledgebase\\prompts\\llm_taxonomy_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

# =============================
# Prompt Builder
# =============================

def build_prompt(unresolved):
    template = load_prompt_template()

    minimal = [
        {"hairstyle_id": s["hairstyle_id"]}
        for s in unresolved
    ]

    return template + "\n\n" + json.dumps(minimal, indent=2)

# =============================
# JSON Cleaner (CRITICAL FIX)
# =============================

def clean_llm_output(text):
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    # Extract JSON array only
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        text = text[start:end+1]

    return text.strip()

# =============================
# LLM Call with Retry
# =============================

def call_llm_with_retry(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            if not response.candidates:
                raise ValueError("No candidates returned.")

            parts = response.candidates[0].content.parts
            if not parts:
                raise ValueError("No content parts returned.")

            result_text = parts[0].text.strip()

            # Save raw response for debugging
            save_json(
                "C:\\Reaserch\\Knowledgebase\\data_intermediate\\llm_raw_response.json",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pipeline_version": PIPELINE_VERSION,
                    "raw": result_text
                }
            )

            cleaned = clean_llm_output(result_text)

            parsed = json.loads(cleaned)
            return parsed

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

    raise ValueError("LLM failed after maximum retries.")

# =============================
# Schema Validation
# =============================

def validate_schema(style):
    if "hairstyle_id" not in style:
        return False
    if style.get("base_category") not in ALLOWED_BASE:
        return False
    if style.get("length_category") not in ALLOWED_LENGTH:
        return False
    return True

# =============================
# Main
# =============================

def main():
    unresolved = load_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\unresolved_styles.json")

    if not unresolved:
        print("No unresolved styles.")
        return

    prompt = build_prompt(unresolved)
    classified = call_llm_with_retry(prompt)

    if not isinstance(classified, list):
        raise ValueError("LLM output is not a list.")

    validated = []
    rejected = []

    for style in classified:
        if validate_schema(style):
            style["classification_source"] = "llm"
            style["classification_timestamp"] = datetime.now(timezone.utc).isoformat()
            style["pipeline_version"] = PIPELINE_VERSION
            validated.append(style)
        else:
            rejected.append(style)

    save_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\llm_classified.json", validated)

    if rejected:
        save_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\llm_rejected.json", rejected)

    print(f"LLM classified: {len(validated)}")
    print(f"Rejected entries: {len(rejected)}")

if __name__ == "__main__":
    main()
