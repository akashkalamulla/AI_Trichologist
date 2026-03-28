import json

# Fields that do NOT belong in taxonomy
FIELDS_TO_REMOVE = [
    "compatibility",
    "hair_loss_support",
    "density_support",
    "adds_volume",
    "reduces_bulk",
    "maintenance_level",
    "hard_constraints",
    "penalty_rules"
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def clean_entries(data):
    cleaned = []

    for entry in data:
        # Remove unwanted fields
        for field in FIELDS_TO_REMOVE:
            entry.pop(field, None)

        cleaned.append(entry)

    return cleaned

def main():
    input_path = "C:\\Reaserch\\Knowledgebase\\data_intermediate\\canonical_step2.json"
    output_path = "C:\\Reaserch\\Knowledgebase\\data_intermediate\\canonical_step2_clean.json"

    data = load_json(input_path)
    cleaned = clean_entries(data)
    save_json(output_path, cleaned)

    print(f"Cleaned file saved to: {output_path}")
    print(f"Total entries cleaned: {len(cleaned)}")

if __name__ == "__main__":
    main()
