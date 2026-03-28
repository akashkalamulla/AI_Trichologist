import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

WEIGHTS = {"face": 0.40, "texture": 0.25, "density": 0.20, "hair_loss": 0.15}
ENHANCEMENT_WEIGHT = 0.10
MAX_PER_CATEGORY = 2

ALLOWED_FACE = {"oval", "round", "square", "oblong", "heart"}
ALLOWED_TEXTURE = {"straight", "wavy", "curly"}
ALLOWED_DENSITY = {"low", "medium", "high"}

def resolve_kb_dir(kb_dir: Optional[str] = None) -> Path:
    if kb_dir:
        p = Path(kb_dir).expanduser().resolve()
        if p.exists():
            return p
    env = os.environ.get("KB_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    default_drive = Path("/content/drive/MyDrive/AI_Trichologist/knowledgebase").resolve()
    if default_drive.exists():
        return default_drive
    return Path(__file__).resolve().parent.parent

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_taxonomy_and_rules(kb_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    KB = resolve_kb_dir(kb_dir)
    taxonomy_path = KB / "data_final" / "canonical_taxonomy_final.json"
    rules_path = KB / "data_knowledge" / "hairstyle_rules.json"
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Missing taxonomy JSON: {taxonomy_path}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Missing rules JSON: {rules_path}")
    taxonomy = _load_json(taxonomy_path)
    rules = _load_json(rules_path)
    if not isinstance(taxonomy, list) or not isinstance(rules, list):
        raise ValueError("taxonomy and rules must both be JSON arrays (lists).")
    return taxonomy, rules

def validate_weights(weights: Dict[str, float]) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0. Current sum: {total}")

def safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default

def hairloss_score_from_profile(rule: Dict[str, Any], user_profile: Dict[str, Any]) -> Tuple[float, str]:
    hl = safe_get(rule, "hair_loss", {}) or {}

    norwood = user_profile.get("norwood_stage") or user_profile.get("norwood") or user_profile.get("norwoodStage")
    norwood_score = None
    if norwood is not None:
        norwood_map = safe_get(hl, "norwood", {}) or {}
        norwood_score = norwood_map.get(str(norwood))

    pat = user_profile.get("baldness_pattern_label") or user_profile.get("baldness_pattern") or "none"
    pat = str(pat).lower().strip()

    frontal = hl.get("receding_hairline", None)
    crown = hl.get("crown_thinning", None)

    pattern_score = None
    pattern_reason = None

    if pat in ["none", "normal", "no", "0"]:
        pattern_score = None
        pattern_reason = "pattern=none"
    elif pat in ["frontal", "front", "temple", "hairline"]:
        pattern_score = frontal if frontal is not None else None
        pattern_reason = "pattern=frontal→receding_hairline"
    elif pat in ["crown", "vertex"]:
        pattern_score = crown if crown is not None else None
        pattern_reason = "pattern=crown→crown_thinning"
    elif pat in ["diffuse", "diffuse_thinning", "overall"]:
        if frontal is not None and crown is not None:
            pattern_score = (float(frontal) + float(crown)) / 2.0
            pattern_reason = "pattern=diffuse→avg(frontal,crown)"
        elif frontal is not None:
            pattern_score = float(frontal)
            pattern_reason = "pattern=diffuse→fallback frontal"
        elif crown is not None:
            pattern_score = float(crown)
            pattern_reason = "pattern=diffuse→fallback crown"
        else:
            pattern_score = None
            pattern_reason = "pattern=diffuse→no keys"
    else:
        pattern_score = None
        pattern_reason = f"pattern={pat}→ignored"

    if pattern_score is not None:
        return float(pattern_score), pattern_reason
    if norwood_score is not None:
        return float(norwood_score), f"norwood={norwood}"
    return 0.5, "neutral"

def compute_score(style: Dict[str, Any], rule: Dict[str, Any], user_profile: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    face_shape = user_profile.get("face_shape")
    texture = user_profile.get("texture")
    density = user_profile.get("density")

    face_score = safe_get(rule, "face_shape", {}).get(face_shape, 0)
    texture_score = safe_get(rule, "texture", {}).get(texture, 0)
    density_score = safe_get(rule, "density", {}).get(density, 0)

    hairloss_score, hairloss_reason = hairloss_score_from_profile(rule, user_profile)

    compatibility_score = (
        WEIGHTS["face"] * float(face_score) +
        WEIGHTS["texture"] * float(texture_score) +
        WEIGHTS["density"] * float(density_score) +
        WEIGHTS["hair_loss"] * float(hairloss_score)
    )

    maintenance_penalty = float(rule.get("maintenance", 0) or 0)
    compatibility_score *= (1 - (maintenance_penalty * 0.2))

    if face_score >= 0.85:
        improvement_bonus = ENHANCEMENT_WEIGHT
    elif face_score >= 0.75:
        improvement_bonus = ENHANCEMENT_WEIGHT / 2
    else:
        improvement_bonus = 0.0

    final_score = float(compatibility_score) + float(improvement_bonus)

    breakdown = {
        "face_shape": {"value": face_shape, "rule_score": float(face_score), "weight": WEIGHTS["face"]},
        "texture": {"value": texture, "rule_score": float(texture_score), "weight": WEIGHTS["texture"]},
        "density": {"value": density, "rule_score": float(density_score), "weight": WEIGHTS["density"]},
        "hair_loss": {"rule_score": float(hairloss_score), "weight": WEIGHTS["hair_loss"], "reason": hairloss_reason},
        "maintenance_penalty": maintenance_penalty,
        "improvement_bonus": float(improvement_bonus),
    }
    return round(final_score, 4), breakdown

def build_explanation(style: Dict[str, Any], rule: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
    fx = rule.get("effects", [])
    fx_txt = ", ".join(fx) if isinstance(fx, list) else str(fx)

    parts = [
        f"face_shape={breakdown['face_shape']['value']}({breakdown['face_shape']['rule_score']})",
        f"texture={breakdown['texture']['value']}({breakdown['texture']['rule_score']})",
        f"density={breakdown['density']['value']}({breakdown['density']['rule_score']})",
        f"hair_loss={breakdown['hair_loss']['reason']}({breakdown['hair_loss']['rule_score']})",
    ]
    if breakdown["maintenance_penalty"] > 0:
        parts.append(f"maintenance_penalty={breakdown['maintenance_penalty']}")
    if breakdown["improvement_bonus"] > 0:
        parts.append(f"bonus={breakdown['improvement_bonus']}")
    if fx_txt:
        parts.append(f"effects={fx_txt}")
    return " | ".join(parts)

def diversify(ranked: List[Dict[str, Any]], max_per_category: int = MAX_PER_CATEGORY, top_n: int = 5) -> List[Dict[str, Any]]:
    final: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for item in ranked:
        base = item.get("base_category", "unknown")
        counts.setdefault(base, 0)
        if counts[base] < max_per_category:
            final.append(item)
            counts[base] += 1
        if len(final) >= top_n:
            break
    return final

def recommend(user_profile: Dict[str, Any], top_n: int = 5, kb_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    validate_weights(WEIGHTS)
    taxonomy, rules = load_taxonomy_and_rules(kb_dir)
    rule_map = {r.get("base_category"): r for r in rules if isinstance(r, dict) and r.get("base_category")}
    if not rule_map:
        raise ValueError("No valid rules found (missing base_category).")

    scored: List[Dict[str, Any]] = []
    for style in taxonomy:
        if not isinstance(style, dict):
            continue
        base = style.get("base_category")
        if not base or base not in rule_map:
            continue
        rule = rule_map[base]
        score, breakdown = compute_score(style, rule, user_profile)
        explanation = build_explanation(style, rule, breakdown)
        scored.append({
            "hairstyle_id": style.get("hairstyle_id"),
            "display_name": style.get("display_name"),
            "base_category": base,
            "score": score,
            "explanation": explanation,
            "breakdown": breakdown,
            "effects": rule.get("effects", [])
        })

    if not scored:
        raise ValueError("No styles scored (taxonomy/rules base_category mismatch or empty KB).")

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return diversify(ranked, max_per_category=MAX_PER_CATEGORY, top_n=top_n)

def _require_enum(name: str, value: Any, allowed: set):
    if value is None:
        raise ValueError(f"Missing required field '{name}'. Run finalize_profile_meta(meta_path) in your notebook before recommending.")
    v = str(value).lower().strip()
    if v not in allowed:
        raise ValueError(f"Invalid '{name}'='{value}'. Allowed: {sorted(list(allowed))}")
    return v

def recommend_from_meta(meta_json_path: str, top_n: int = 5, kb_dir: Optional[str] = None) -> Dict[str, Any]:
    meta_p = Path(meta_json_path).expanduser().resolve()
    if not meta_p.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_p}")

    meta = _load_json(meta_p)
    if not isinstance(meta, dict):
        raise ValueError("meta.json must be an object/dict.")

    face_shape = meta.get("face_shape") or meta.get("face_shape_label") or meta.get("faceShape")
    texture = meta.get("texture") or meta.get("hair_texture") or meta.get("hair_texture_label") or meta.get("hairTexture")
    density = meta.get("density") or meta.get("hair_density") or meta.get("hair_density_label") or meta.get("hairDensity")

    face_shape = _require_enum("face_shape", face_shape, ALLOWED_FACE)
    texture = _require_enum("texture", texture, ALLOWED_TEXTURE)
    density = _require_enum("density", density, ALLOWED_DENSITY)

    user_profile = {
        "face_shape": face_shape,
        "texture": texture,
        "density": density,
        "baldness_pattern_label": meta.get("baldness_pattern_label") or meta.get("baldness_pattern") or "none",
        "norwood_stage": meta.get("norwood_stage") or meta.get("norwood") or meta.get("norwoodStage"),
    }

    results = recommend(user_profile, top_n=top_n, kb_dir=kb_dir)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["base_category"]] = counts.get(r["base_category"], 0) + 1

    return {
        "top_5_hairstyle_ids": [r["hairstyle_id"] for r in results[:top_n]],
        "results": results[:top_n],
        "diversification_summary": counts,
        "user_profile_used": user_profile
    }
