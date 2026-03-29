import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


# =========================
# Config
# =========================
@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (512, 512)  # (W, H)
    apply_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)

    color_space: str = "rgb"   # "rgb" | "bgr" | "hsv"
    normalize: str = "01"      # "01" | "imagenet" | "none"

    face_crop: bool = True
    face_crop_margin: float = 0.30
    min_face_size_ratio: float = 0.10

    return_preview_rgb: bool = True


# =========================
# Public API
# =========================
def preprocess_image(
    image_path: str,
    cfg: Optional[PreprocessConfig] = None
) -> Tuple[np.ndarray, Dict[str, Union[str, float, int, Tuple[int, int]]], Optional[np.ndarray]]:
    """
    Returns:
      processed: float32 (model-ready)
      meta: dict (debug + evaluation)
      preview_rgb: uint8 RGB (UI preview)
    """
    cfg = cfg or PreprocessConfig()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    meta: Dict[str, Union[str, float, int, Tuple[int, int]]] = {
        "image_path": image_path,
        "original_shape": tuple(bgr.shape[:2]),
        "face_detected": False,
        "face_box_xyxy": None,
        "face_area_ratio": 0.0,
        "clahe": cfg.apply_clahe,
        "color_space": cfg.color_space,
        "normalize": cfg.normalize,
        "target_size": cfg.target_size,
    }

    # -------- Face crop --------
    working = bgr
    if cfg.face_crop:
        crop = _detect_and_crop_face_opencv(working, margin=cfg.face_crop_margin)
        if crop is not None:
            cropped, (x1, y1, x2, y2) = crop
            h, w = working.shape[:2]
            face_area = max(0, x2 - x1) * max(0, y2 - y1)
            ratio = float(face_area / max(1, h * w))

            meta["face_box_xyxy"] = (int(x1), int(y1), int(x2), int(y2))
            meta["face_area_ratio"] = ratio

            if ratio >= cfg.min_face_size_ratio:
                working = cropped
                meta["face_detected"] = True

    meta["post_crop_shape"] = tuple(working.shape[:2])

    # -------- Resize --------
    resized = _smart_resize(working, cfg.target_size)
    meta["resized_shape"] = tuple(resized.shape[:2])

    # Work in RGB for CLAHE correctness
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # -------- CLAHE --------
    if cfg.apply_clahe:
        rgb = _apply_clahe_lab(rgb, cfg.clip_limit, cfg.tile_grid_size)

    # -------- Output color space --------
    if cfg.color_space == "rgb":
        out = rgb
    elif cfg.color_space == "bgr":
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif cfg.color_space == "hsv":
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError("color_space must be 'rgb', 'bgr', or 'hsv'")

    out_f = out.astype(np.float32)

    # -------- Normalize --------
    processed = _normalize(out_f, cfg.color_space, cfg.normalize)

    meta["dtype"] = str(processed.dtype)
    meta["min"] = float(processed.min())
    meta["max"] = float(processed.max())
    meta["mean"] = float(processed.mean())

    preview_rgb = rgb.astype(np.uint8) if cfg.return_preview_rgb else None
    return processed, meta, preview_rgb


def save_phase1_outputs(
    image_id: str,
    processed: np.ndarray,
    meta: Dict,
    preview_rgb: Optional[np.ndarray],
    input_save_path: str,
    processed_dir: str,
    profiles_dir: str
) -> Dict[str, str]:
    """
    Saves:
      - processed preview image -> processed_dir/{image_id}_processed.jpg
      - meta json -> profiles_dir/{image_id}_meta.json
    Always JSON-safe. No numpy serialization crashes.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)

    preview_path = os.path.join(processed_dir, f"{image_id}_processed.jpg")
    if preview_rgb is None:
        preview_rgb = _processed_to_preview_rgb(processed, meta)

    cv2.imwrite(preview_path, cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

    meta_path = os.path.join(profiles_dir, f"{image_id}_meta.json")
    meta_out = dict(meta)
    meta_out["raw_path"] = input_save_path
    meta_out["processed_path"] = preview_path
    meta_out["image_id"] = image_id

    # ✅ critical fix: ensure JSON serializable
    with open(meta_path, "w") as f:
        json.dump(_json_safe(meta_out), f, indent=2)

    return {"raw_path": input_save_path, "processed_path": preview_path, "meta_path": meta_path}


# =========================
# Helpers
# =========================
def _json_safe(obj):
    """Convert numpy types -> plain Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _smart_resize(bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    tw, th = target_size
    h, w = bgr.shape[:2]
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_CUBIC
    return cv2.resize(bgr, (tw, th), interpolation=interp)


def _apply_clahe_lab(rgb: np.ndarray, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _detect_and_crop_face_opencv(bgr: np.ndarray, margin: float = 0.30):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    H, W = bgr.shape[:2]
    pad_w, pad_h = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(W, x + w + pad_w), min(H, y + h + pad_h)

    return bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _normalize(out_f: np.ndarray, color_space: str, normalize: str) -> np.ndarray:
    if normalize == "none":
        return out_f.astype(np.float32)

    if normalize == "01":
        if color_space == "hsv":
            x = out_f.copy().astype(np.float32)
            x[..., 0] /= 179.0
            x[..., 1] /= 255.0
            x[..., 2] /= 255.0
            return x
        return (out_f / 255.0).astype(np.float32)

    if normalize == "imagenet":
        if color_space != "rgb":
            raise ValueError("imagenet normalization requires rgb")
        x = (out_f / 255.0).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std

    raise ValueError("normalize must be '01', 'imagenet', or 'none'")


def _processed_to_preview_rgb(processed: np.ndarray, meta: Dict) -> np.ndarray:
    norm = meta.get("normalize", "01")
    cs = meta.get("color_space", "rgb")
    x = processed.copy()

    if norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x * std) + mean
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    if norm == "01":
        if cs == "hsv":
            hsv = x.copy()
            hsv[..., 0] *= 179.0
            hsv[..., 1] *= 255.0
            hsv[..., 2] *= 255.0
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        rgb = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        if cs == "bgr":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    if cs == "bgr":
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if cs == "hsv":
        return cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return x
