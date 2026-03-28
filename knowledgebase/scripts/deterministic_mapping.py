import json
import os

BASE_LENGTH_RULES = {
    "buzz": "ultra_short",
    "shaved": "ultra_short",
    "crew": "short",
    "crop": "short",
    "fade": "short",
    "edgar": "short",
    "undercut": "short",
    "quiff": "medium",
    "pompadour": "medium",
    "side_part": "medium",
    "slick_back": "medium",
    "curtain": "long",
    "mullet": "long",
    "dreadlocks": "long",
    "long_flow": "long",
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def build_family_lookup(families):
    lookup = {}
    for fam in families:
        family_id = fam["family_id"]
        for variant in fam["variants"]:
            lookup[variant] = family_id
    return lookup

def map_base_category(family_id):
    return family_id.replace("_cut", "").replace("_", "")

def assign_length(base_category):
    return BASE_LENGTH_RULES.get(base_category)

def main():
    families = load_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\canonical_families.json")
    styles = load_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\canonical_hairstyles.json")

    family_lookup = build_family_lookup(families)
    unresolved = []

    for style in styles:
        style_id = style["hairstyle_id"]

        # Deterministic base category
        if style_id in family_lookup:
            base = map_base_category(family_lookup[style_id])
            style["base_category"] = base
        else:
            style["base_category"] = None

        # Deterministic length
        style["length_category"] = assign_length(style["base_category"])

        if not style["base_category"] or not style["length_category"]:
            unresolved.append(style)

    save_json("Knowledgebase/data_intermediate/canonical_step2.json", styles)
    save_json("Knowledgebase/data_intermediate/unresolved_styles.json", unresolved)

    print(f"Unresolved styles: {len(unresolved)}")

if __name__ == "__main__":
    main()
