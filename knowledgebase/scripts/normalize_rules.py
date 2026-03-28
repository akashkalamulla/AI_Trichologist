import json

def normalize_value(v):
    if isinstance(v, (int, float)):
        if v >= 1.0:
            return 0.9
        return round(v, 2)
    return v

def normalize_nested(obj):
    if isinstance(obj, dict):
        return {k: normalize_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_nested(v) for v in obj]
    else:
        return normalize_value(obj)

def main():
    path = "C:\\Reaserch\\Knowledgebase\\data_knowledge\\hairstyle_rules.json"

    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    normalized = []

    for rule in rules:
        normalized.append(normalize_nested(rule))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)

    print("Rules normalized successfully.")

if __name__ == "__main__":
    main()
