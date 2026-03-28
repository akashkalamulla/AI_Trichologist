import json
from collections import Counter
from datetime import datetime

PIPELINE_VERSION = "v1.1.0"

ALLOWED_BASE = [
    "buzz", "crew", "crop", "fade", "undercut",
    "quiff", "pompadour", "fringe", "slick_back",
    "mohawk", "curtain", "mullet", "afro",
    "dreadlocks", "long_flow", "skater",
    "edgar", "side_part", "business",
    "shaved", "other"
]

ALLOWED_LENGTH = ["ultra_short", "short", "medium", "long"]


# =============================
# Utilities
# =============================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =============================
# Validation Functions
# =============================

def validate_entry(style):
    if style.get("base_category") not in ALLOWED_BASE:
        return False
    if style.get("length_category") not in ALLOWED_LENGTH:
        return False
    if not style.get("hairstyle_id"):
        return False
    return True


# =============================
# Main
# =============================

def main():
    canonical = load_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\canonical_step2.json")
    llm_data = load_json("C:\\Reaserch\\Knowledgebase\\data_intermediate\\llm_classified.json")

    # Build LLM lookup
    llm_lookup = {s["hairstyle_id"]: s for s in llm_data}

    seen_ids = set()
    duplicates = []
    unresolved_after_merge = []
    deterministic_count = 0
    llm_count = 0

    for style in canonical:
        sid = style["hairstyle_id"]

        # Duplicate detection
        if sid in seen_ids:
            duplicates.append(sid)
        seen_ids.add(sid)

        # If deterministic assignment exists → protect it
        if style.get("base_category") and style.get("length_category"):
            style["classification_source"] = "deterministic"
            deterministic_count += 1

        # Otherwise attempt LLM fill
        elif sid in llm_lookup:
            style["base_category"] = llm_lookup[sid]["base_category"]
            style["length_category"] = llm_lookup[sid]["length_category"]
            style["classification_source"] = "llm"
            style["classification_timestamp"] = llm_lookup[sid].get(
                "classification_timestamp"
            )
            llm_count += 1

        # Still unresolved
        else:
            unresolved_after_merge.append(sid)

        # Final schema validation
        if not validate_entry(style):
            raise ValueError(f"Invalid entry after merge: {sid}")

    # Reject duplicates
    if duplicates:
        raise ValueError(f"Duplicate hairstyle_ids detected: {duplicates}")

    # Reject unresolved
    if unresolved_after_merge:
        raise ValueError(
            f"Unresolved styles remain after merge: {unresolved_after_merge}"
        )

    # Distribution audit
    base_distribution = Counter(s["base_category"] for s in canonical)
    length_distribution = Counter(s["length_category"] for s in canonical)

    audit_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "total_styles": len(canonical),
        "deterministic_count": deterministic_count,
        "llm_count": llm_count,
        "base_distribution": dict(base_distribution),
        "length_distribution": dict(length_distribution),
    }

    save_json("C:\\Reaserch\\Knowledgebase\\data_final\\taxonomy_audit_log.json", audit_log)
    save_json("C:\\Reaserch\\Knowledgebase\\data_final\\canonical_taxonomy_final.json", canonical)

    print("Final taxonomy saved.")
    print("Audit log generated.")


if __name__ == "__main__":
    main()
