import json
import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_CATEGORIES = [
    "buzz", "crew", "crop", "fade", "undercut",
    "quiff", "pompadour", "fringe", "slick_back",
    "mohawk", "curtain", "mullet", "afro",
    "dreadlocks", "long_flow", "skater",
    "edgar", "side_part", "business",
    "shaved", "other"
]

PROMPT = f"""
You are a professional barbering and trichology expert.

Generate structured compatibility knowledge for ALL of the following base categories.

Return STRICT JSON only.
Return a single JSON array.
No markdown.
No explanation.
No comments.
No trailing text.

Each object MUST have this exact schema:

- base_category (string)
- face_shape (scores 0.0-1.0 for: oval, round, square, rectangle, diamond, heart, triangle)
- texture (scores 0.0-1.0 for: straight, wavy, curly, coily)
- density (scores 0.0-1.0 for: low, medium, high)
- hair_loss:
    - receding_hairline (0-1)
    - crown_thinning (0-1)
    - norwood:
        - 1, 2, 3, 4, 5 (0-1 each)
- maintenance (scores 0.0-1.0 for: low, medium, high)
- effects (3-5 short tags)

Important constraints:
- Do NOT use the word "category" — use "base_category".
- Do NOT use "oblong".
- Use "rectangle".
- Avoid unrealistic universal 1.0 for all attributes.
- Keep values realistic and varied.
- Ensure exactly {len(BASE_CATEGORIES)} entries.

Base categories:
{BASE_CATEGORIES}
"""

def clean_output(text):
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("JSON array not found in output.")
    return text[start:end+1]

def main():
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT
    )

    result_text = response.candidates[0].content.parts[0].text
    cleaned = clean_output(result_text)
    parsed = json.loads(cleaned)

    with open("C:\\Reaserch\\Knowledgebase\\data_knowledge\\hairstyle_rules.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print("hairstyle_rules.json regenerated successfully.")

if __name__ == "__main__":
    main()
