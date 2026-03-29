from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# Small utilities
# -------------------------
def _read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_write_json(p: str, obj: Dict[str, Any]) -> None:
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _abs_join(base_dir: str, rel_or_abs: str) -> str:
    if not rel_or_abs:
        return ""
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(base_dir, rel_or_abs)

def _clip_xyxy(x1:int, y1:int, x2:int, y2:int, w:int, h:int) -> Tuple[int,int,int,int]:
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1, y1, x2, y2

def _bbox_from_mask(mask_255: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def _expand_bbox(b: Tuple[int,int,int,int], w:int, h:int, margin: float) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = b
    bw = max(1, x2-x1)
    bh = max(1, y2-y1)
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    nbw = bw*(1+margin)
    nbh = bh*(1+margin)
    nx1 = int(round(cx - nbw/2))
    nx2 = int(round(cx + nbw/2))
    ny1 = int(round(cy - nbh/2))
    ny2 = int(round(cy + nbh/2))
    return _clip_xyxy(nx1, ny1, nx2, ny2, w, h)

def _overlay_mask(bgr: np.ndarray, mask_255: np.ndarray, alpha: float = 0.45, color=(0,255,0)) -> np.ndarray:
    out = bgr.copy()
    m = mask_255 > 0
    if m.any():
        out[m] = (out[m].astype(np.float32)*(1-alpha) + np.array(color, dtype=np.float32)*alpha).astype(np.uint8)
    return out


# -------------------------
# Config
# -------------------------
@dataclass
class Phase4Config:
    model_name: str = "baldness_zones_mvp_v3_labels_v2"

    # ROI selection
    roi_margin: float = 0.10          # expand bbox a bit (mask bbox)
    min_roi_area_ratio: float = 0.01  # if ROI too small -> fallback

    # Zone splits over ROI height
    frontal_end: float = 0.40
    mid_end: float = 0.70

    # Scalp proxy (HSV) thresholds (conservative)
    # H in [0..179], S/V in [0..255]
    skin_h_min: int = 0
    skin_h_max: int = 25
    skin_s_min: int = 25
    skin_s_max: int = 140
    skin_v_min: int = 75
    skin_v_max: int = 210

    # Ignore extreme highlights/darks (reduce glare false positives)
    ignore_v_high: int = 235
    ignore_v_low: int = 25

    # Morph cleanup
    scalp_morph_ksize: int = 5

    # Outputs
    save_overlay: bool = True
    overlay_alpha: float = 0.55


# -------------------------
# Core: scalp proxy within mask ROI and zones
# -------------------------
def _scalp_proxy_mask(bgr: np.ndarray, cfg: Phase4Config) -> np.ndarray:
    """Returns binary mask of 'skin-like' pixels (255 = scalp/skin proxy)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    cond = (
        (h >= cfg.skin_h_min) & (h <= cfg.skin_h_max) &
        (s >= cfg.skin_s_min) & (s <= cfg.skin_s_max) &
        (v >= cfg.skin_v_min) & (v <= cfg.skin_v_max) &
        (v >= cfg.ignore_v_low) & (v <= cfg.ignore_v_high)
    )

    out = (cond.astype(np.uint8) * 255)

    k = cfg.scalp_morph_ksize
    if k >= 3:
        if k % 2 == 0: k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out

def _zone_slices(y1:int, y2:int, cfg: Phase4Config) -> Dict[str, Tuple[int,int]]:
    h = max(1, y2-y1)
    f_end = y1 + int(round(cfg.frontal_end * h))
    m_end = y1 + int(round(cfg.mid_end * h))
    return {
        "frontal": (y1, f_end),
        "mid": (f_end, m_end),
        "crown": (m_end, y2),
    }

def _score_zone(scalp_255: np.ndarray, hairmask_255: np.ndarray, roi: Tuple[int,int,int,int], yslice: Tuple[int,int]) -> float:
    x1,y1,x2,y2 = roi
    zy1, zy2 = yslice
    zy1 = max(y1, min(zy1, y2))
    zy2 = max(y1, min(zy2, y2))
    if zy2 <= zy1:
        return 0.0

    scalp = scalp_255[zy1:zy2, x1:x2]
    hairm = hairmask_255[zy1:zy2, x1:x2]

    inside = (hairm > 0)
    denom = int(inside.sum())
    if denom == 0:
        return 0.0

    num = int(((scalp > 0) & inside).sum())
    return float(num / denom)

def _pattern_label(frontal: float, mid: float, crown: float) -> Tuple[str, float]:
    """
    Stricter labeling to reduce overuse of 'diffuse'.
    label: none/frontal/crown/diffuse
    """
    mx = max(frontal, mid, crown)

    # stricter "none" threshold (reduces diffuse spam)
    if mx < 0.14:
        return "none", 0.75

    # dominant pattern (must be clearly higher)
    if frontal > crown + 0.10 and frontal > mid + 0.08:
        return "frontal", min(0.95, 0.55 + frontal)
    if crown > frontal + 0.10 and crown > mid + 0.08:
        return "crown", min(0.95, 0.55 + crown)

    # diffuse only if at least 2 zones are meaningfully high
    elevated = sum([frontal > 0.22, mid > 0.22, crown > 0.22])
    if elevated >= 2:
        return "diffuse", min(0.90, 0.50 + mx)

    # otherwise: mild diffuse
    return "diffuse", min(0.85, 0.45 + mx)


# -------------------------
# Public runner for your pipeline
# -------------------------
def run_phase4_baldness_zones(
    *,
    base_dir: str,
    meta_path: str,
    config: Optional[Phase4Config] = None,
    overwrite: bool = True
) -> Dict[str, Any]:
    cfg = config or Phase4Config()
    meta = _read_json(meta_path)

    # Load image path (processed > raw)
    rel_img = meta.get("processed_path") or meta.get("raw_path") or ""
    img_abs = _abs_join(base_dir, rel_img)
    if not img_abs or not os.path.exists(img_abs):
        meta["baldness_error"] = f"image_not_found: {img_abs}"
        _safe_write_json(meta_path, meta)
        return meta

    bgr = cv2.imread(img_abs, cv2.IMREAD_COLOR)
    if bgr is None:
        meta["baldness_error"] = f"cv2_imread_failed: {img_abs}"
        _safe_write_json(meta_path, meta)
        return meta

    H, W = bgr.shape[:2]

    # Load hair mask (from Phase 3)
    rel_mask = meta.get("mask_path") or ""
    mask_abs = _abs_join(base_dir, rel_mask)
    if not mask_abs or not os.path.exists(mask_abs):
        meta["baldness_error"] = f"mask_not_found: {mask_abs}"
        _safe_write_json(meta_path, meta)
        return meta

    hairmask = cv2.imread(mask_abs, cv2.IMREAD_GRAYSCALE)
    if hairmask is None:
        meta["baldness_error"] = f"mask_imread_failed: {mask_abs}"
        _safe_write_json(meta_path, meta)
        return meta

    # ROI source: mask bbox first (best)
    roi_src = "mask_bbox"
    bbox = _bbox_from_mask(hairmask)
    if bbox is None:
        roi_src = "fallback_top"
        bbox = (0, 0, W-1, int(0.45 * H))

    roi = _expand_bbox(bbox, W, H, cfg.roi_margin)

    # sanity: if roi too tiny, fallback
    x1,y1,x2,y2 = roi
    roi_area = (x2-x1) * (y2-y1)
    if roi_area / float(W*H) < cfg.min_roi_area_ratio:
        roi_src = "fallback_top"
        roi = (0, 0, W-1, int(0.45 * H))

    # Scalp proxy for whole image
    scalp = _scalp_proxy_mask(bgr, cfg)

    # Zone scores
    zs = _zone_slices(roi[1], roi[3], cfg)
    frontal = _score_zone(scalp, hairmask, roi, zs["frontal"])
    mid = _score_zone(scalp, hairmask, roi, zs["mid"])
    crown = _score_zone(scalp, hairmask, roi, zs["crown"])

    label, conf = _pattern_label(frontal, mid, crown)

    # Outputs directory
    image_id = meta.get("image_id") or os.path.basename(meta_path).replace("_meta.json", "")
    out_dir = os.path.join(base_dir, "data", "baldness")
    _ensure_dir(out_dir)

    zones_json_rel = os.path.join("data", "baldness", f"{image_id}_zones.json")
    zones_json_abs = os.path.join(base_dir, zones_json_rel)

    overlay_rel = os.path.join("data", "baldness", f"{image_id}_zones_overlay.jpg")
    overlay_abs = os.path.join(base_dir, overlay_rel)

    # Save zones json
    zones_obj = {
        "image_id": image_id,
        "roi_source": roi_src,
        "roi_xyxy": [int(v) for v in roi],
        "scores": {"frontal": frontal, "mid": mid, "crown": crown},
        "pattern_label": label,
        "confidence": conf,
        "config": {
            "frontal_end": cfg.frontal_end,
            "mid_end": cfg.mid_end,
            "skin_h": [cfg.skin_h_min, cfg.skin_h_max],
            "skin_s": [cfg.skin_s_min, cfg.skin_s_max],
            "skin_v": [cfg.skin_v_min, cfg.skin_v_max],
        },
    }
    _safe_write_json(zones_json_abs, zones_obj)

    # Save overlay
    if cfg.save_overlay:
        vis = bgr.copy()

        # draw ROI box
        cv2.rectangle(vis, (roi[0],roi[1]), (roi[2],roi[3]), (255,0,0), 3)

        # draw zone lines
        cv2.line(vis, (roi[0], zs["frontal"][1]), (roi[2], zs["frontal"][1]), (0,255,255), 3)
        cv2.line(vis, (roi[0], zs["mid"][1]), (roi[2], zs["mid"][1]), (0,255,255), 3)

        # overlay hairmask in green
        vis = _overlay_mask(vis, hairmask, alpha=cfg.overlay_alpha, color=(0,255,0))

        # overlay scalp proxy in red (only within hairmask)
        scalp_in = ((scalp > 0) & (hairmask > 0)).astype(np.uint8) * 255
        red = vis.copy()
        red_mask = scalp_in > 0
        red[red_mask] = (red[red_mask].astype(np.float32)*0.5 + np.array([0,0,255],dtype=np.float32)*0.5).astype(np.uint8)
        vis = red

        # label text
        txt = f"pattern={label} conf={conf:.2f} | F={frontal:.2f} M={mid:.2f} C={crown:.2f} | roi={roi_src}"
        cv2.putText(vis, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)

        cv2.imwrite(overlay_abs, vis)

    # -----------------------
    # Update meta schema (clean + reportable)
    # -----------------------
    meta["baldness_zones_path"] = zones_json_rel
    meta["baldness_zones_overlay_path"] = overlay_rel if cfg.save_overlay else None
    meta["baldness_model"] = cfg.model_name

    meta["baldness_zone_scores"] = {"frontal": float(frontal), "mid": float(mid), "crown": float(crown)}
    meta["baldness_pattern_label"] = label
    meta["baldness_confidence"] = float(conf)
    meta["baldness_roi_source"] = roi_src

    meta["bald_summary"] = {
        "pattern_label": label,
        "confidence": float(conf),
        "roi_source": roi_src,
        "scores": meta["baldness_zone_scores"],
    }

    meta.pop("baldness_error", None)
    _safe_write_json(meta_path, meta)
    return meta
