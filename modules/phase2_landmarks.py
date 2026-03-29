from __future__ import annotations

import json, os, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _score01_below(x: float, good_max: float) -> float:
    if good_max <= 0:
        return 0.0
    if x <= good_max:
        return 1.0
    if x >= 2.0 * good_max:
        return 0.0
    return float(1.0 - (x - good_max) / good_max)

def _score01_above(x: float, good_min: float) -> float:
    if good_min <= 0:
        return 1.0
    if x >= good_min:
        return 1.0
    if x <= 0.0:
        return 0.0
    return float(x / good_min)


IDX = {
    "left_cheek": 234,
    "right_cheek": 454,
    "top": 10,
    "chin": 152,
    "jaw_left": 172,
    "jaw_right": 397,
    "cheekbone_left": 93,
    "cheekbone_right": 323,
    "forehead_left": 127,
    "forehead_right": 356,
    "midline": 168,
    "eye_outer_left": 33,
    "eye_outer_right": 263,
}

SYMM_PAIRS = [(33,263),(133,362),(61,291),(234,454),(172,397),(127,356)]


@dataclass
class TaskLandmarkerConfig:
    model_path: Optional[str] = None
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # gate
    max_abs_roll_deg: float = 12.0
    max_yaw_asymmetry_strict: float = 0.18
    max_yaw_asymmetry_relaxed: float = 0.25
    min_bbox_rel_area: float = 0.05
    min_face_width_px: float = 120.0

    # scores
    w_frontality: float = 0.6
    w_size: float = 0.4


def _ensure_model(model_path: Optional[str]) -> str:
    if model_path and os.path.exists(model_path):
        return model_path
    default_path = "/content/face_landmarker.task"
    if os.path.exists(default_path):
        return default_path
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    rc = os.system(f"wget -q -O {default_path} {url}")
    if rc != 0 or not os.path.exists(default_path):
        raise RuntimeError("Failed to download face_landmarker.task")
    return default_path

def _mp_face_bbox(image_bgr):
    """Return bbox in absolute pixels: (x1,y1,x2,y2) or None."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    res = fd.process(rgb)
    if not res.detections:
        return None

    det = res.detections[0]
    bb = det.location_data.relative_bounding_box
    x1 = int(bb.xmin * w)
    y1 = int(bb.ymin * h)
    x2 = int((bb.xmin + bb.width) * w)
    y2 = int((bb.ymin + bb.height) * h)

    # clip
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _expand_bbox(bbox, w, h, margin=0.4):
    x1,y1,x2,y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw2 = bw * (1.0 + margin)
    bh2 = bh * (1.0 + margin)
    nx1 = int(max(0, cx - bw2/2))
    ny1 = int(max(0, cy - bh2/2))
    nx2 = int(min(w, cx + bw2/2))
    ny2 = int(min(h, cy + bh2/2))
    return (nx1, ny1, nx2, ny2)

def extract_face_landmarks_tasks(image_bgr: Any, cfg: TaskLandmarkerConfig):
    if image_bgr is None:
        return None, {"error": "image_bgr is None"}

    H, W = image_bgr.shape[:2]
    if H < 32 or W < 32:
        return None, {"error": f"image too small ({W}x{H})"}

    model_path = _ensure_model(cfg.model_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=cfg.min_face_detection_confidence,
        min_face_presence_confidence=cfg.min_face_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    def _run_landmarker(img_bgr):
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        face_lms = result.face_landmarks[0]
        return [(float(lm.x) * w, float(lm.y) * h) for lm in face_lms]

    # 1) Try full image first
    lms = _run_landmarker(image_bgr)
    if lms is not None:
        return lms, {"width": W, "height": H, "mode": "full"}

    # 2) Fallback: detect bbox → crop → landmarks
    bbox = _mp_face_bbox(image_bgr)
    if bbox is None:
        return None, {"width": W, "height": H, "error": "no_face_detected_bbox"}

    x1,y1,x2,y2 = _expand_bbox(bbox, W, H, margin=0.4)
    crop = image_bgr[y1:y2, x1:x2].copy()
    lms_crop = _run_landmarker(crop)
    if lms_crop is None:
        return None, {"width": W, "height": H, "error": "no_face_on_crop", "crop": [x1,y1,x2,y2]}

    # map to original coords
    lms = [(px + x1, py + y1) for (px, py) in lms_crop]
    return lms, {"width": W, "height": H, "mode": "crop", "crop": [x1,y1,x2,y2]}


def compute_geometry_features(landmarks_px: List[Tuple[float, float]], image_size: Tuple[int, int], symmetry: bool = True) -> Dict[str, Any]:
    w, h = image_size

    def P(name: str) -> Tuple[float, float]:
        return landmarks_px[IDX[name]]

    face_width = _dist(P("left_cheek"), P("right_cheek"))
    face_height = _dist(P("top"), P("chin"))
    jaw_width = _dist(P("jaw_left"), P("jaw_right"))
    cheekbone_width = _dist(P("cheekbone_left"), P("cheekbone_right"))
    forehead_width = _dist(P("forehead_left"), P("forehead_right"))

    eps = 1e-6
    out = {
        "distances_px": {"face_width": face_width, "face_height": face_height, "jaw_width": jaw_width, "cheekbone_width": cheekbone_width, "forehead_width": forehead_width},
        "ratios": {"width_height": face_width / max(face_height, eps), "jaw_cheekbone": jaw_width / max(cheekbone_width, eps), "forehead_cheekbone": forehead_width / max(cheekbone_width, eps)},
        "landmark_indices_used": {k: v for k, v in IDX.items() if k not in ("eye_outer_left", "eye_outer_right")},
        "image_size": {"width": int(w), "height": int(h)},
    }

    if symmetry:
        mid_x = P("midline")[0]
        errs = []
        for li, ri in SYMM_PAIRS:
            dL = abs(landmarks_px[li][0] - mid_x)
            dR = abs(landmarks_px[ri][0] - mid_x)
            errs.append(abs(dL - dR))
        mean_err = float(sum(errs) / max(len(errs), 1))
        norm_err = mean_err / max(face_width, eps)
        out["symmetry"] = {"enabled": True, "midline_index": IDX["midline"], "pair_count": len(SYMM_PAIRS), "mean_mirror_error_px": mean_err, "normalized_error": norm_err, "symmetry_score": _clip01(1.0 - norm_err), "pairs": SYMM_PAIRS}

    return out


def _recommendations(reject_reasons: List[str]) -> List[str]:
    rec = []
    for r in reject_reasons:
        if r.startswith("yaw_too_high"):
            rec.append("Turn your face forward (reduce head turn).")
        elif r.startswith("roll_too_high"):
            rec.append("Keep your head level (reduce tilt).")
        elif r.startswith("face_too_small") or r.startswith("face_width_too_small"):
            rec.append("Move closer to the camera so your face fills more of the frame.")
    out = []
    for x in rec:
        if x not in out:
            out.append(x)
    return out


def compute_pose_quality_scores(landmarks_px: List[Tuple[float, float]], info: Dict[str, Any], cfg: TaskLandmarkerConfig, mode: str = "strict") -> Dict[str, Any]:
    if mode not in ("strict", "relaxed"):
        mode = "strict"
    yaw_thr = cfg.max_yaw_asymmetry_strict if mode == "strict" else cfg.max_yaw_asymmetry_relaxed

    w = int(info.get("width", 0))
    h = int(info.get("height", 0))
    bbox = info.get("bbox") or {}

    (x1, y1) = landmarks_px[IDX["eye_outer_left"]]
    (x2, y2) = landmarks_px[IDX["eye_outer_right"]]
    roll_deg = float(math.atan2((y2 - y1), (x2 - x1)) * 180.0 / math.pi)

    mid_x = landmarks_px[IDX["midline"]][0]
    left_x = landmarks_px[IDX["left_cheek"]][0]
    right_x = landmarks_px[IDX["right_cheek"]][0]
    a = abs(mid_x - left_x)
    b = abs(right_x - mid_x)
    eps = 1e-6
    yaw_asym = float(abs(a - b) / max(a, b, eps))

    x_min = float(bbox.get("x_min", 0.0)); y_min = float(bbox.get("y_min", 0.0))
    x_max = float(bbox.get("x_max", 0.0)); y_max = float(bbox.get("y_max", 0.0))
    bbox_w = max(0.0, x_max - x_min)
    bbox_h = max(0.0, y_max - y_min)
    bbox_area = bbox_w * bbox_h
    img_area = max(float(w * h), 1.0)
    bbox_rel_area = float(bbox_area / img_area)

    face_width_px = float(abs(right_x - left_x))

    reasons: List[str] = []
    if abs(roll_deg) > cfg.max_abs_roll_deg:
        reasons.append(f"roll_too_high(abs>{cfg.max_abs_roll_deg})")
    if yaw_asym > yaw_thr:
        reasons.append(f"yaw_too_high(>{yaw_thr})")
    if bbox_rel_area < cfg.min_bbox_rel_area:
        reasons.append(f"face_too_small(bbox_rel_area<{cfg.min_bbox_rel_area})")
    if face_width_px < cfg.min_face_width_px:
        reasons.append(f"face_width_too_small(px<{cfg.min_face_width_px})")

    quality_pass = (len(reasons) == 0)

    roll_score = _score01_below(abs(roll_deg), cfg.max_abs_roll_deg)
    yaw_score  = _score01_below(yaw_asym, yaw_thr)
    frontality_score = float(0.5 * roll_score + 0.5 * yaw_score)

    size_score_area  = _score01_above(bbox_rel_area, cfg.min_bbox_rel_area)
    size_score_width = _score01_above(face_width_px, cfg.min_face_width_px)
    size_score = float(0.5 * size_score_area + 0.5 * size_score_width)

    quality_score = float(_clip01(cfg.w_frontality * frontality_score + cfg.w_size * size_score))

    return {
        "pose": {"roll_deg": roll_deg, "yaw_asymmetry": yaw_asym},
        "size": {"bbox_rel_area": bbox_rel_area, "face_width_px": face_width_px, "bbox_w_px": bbox_w, "bbox_h_px": bbox_h},
        "scores": {"frontality_score": frontality_score, "size_score": size_score, "quality_score": quality_score, "roll_score": roll_score, "yaw_score": yaw_score},
        "quality_gate": {
            "mode": mode,
            "quality_pass": quality_pass,
            "reject_reasons": reasons,
            "recommendations": _recommendations(reasons),
            "thresholds_used": {"max_abs_roll_deg": cfg.max_abs_roll_deg, "max_yaw_asymmetry": yaw_thr, "min_bbox_rel_area": cfg.min_bbox_rel_area, "min_face_width_px": cfg.min_face_width_px},
        },
    }


def run_phase2_on_meta(meta_path: str, *, processed_dir: str, profiles_dir: str, cfg: TaskLandmarkerConfig, prefer_processed: bool = True, mode: str = "strict") -> Dict[str, Any]:
    meta = _read_json(meta_path)

    image_id = meta.get("image_id") or os.path.basename(meta_path).replace("_meta.json", "").replace(".json", "")
    processed_path = meta.get("processed_path")
    raw_path = meta.get("raw_path")

    if prefer_processed and processed_path and os.path.exists(processed_path):
        img_path = processed_path
    elif raw_path and os.path.exists(raw_path):
        img_path = raw_path
    else:
        img_path = processed_path or raw_path

    geometry_path = os.path.join(profiles_dir, f"{image_id}_geometry.json")

    geometry_obj: Dict[str, Any] = {
        "image_id": image_id,
        "landmark_model": "mediapipe_face_landmarker_tasks",
        "source_image_used": img_path,
        "landmark_count_expected_min": 468,
        "landmark_count_actual": 0,
    }

    image_bgr = cv2.imread(img_path) if (img_path and os.path.exists(img_path)) else None
    if image_bgr is None:
        geometry_obj["error"] = "image_read_failed"
        geometry_obj["landmark_count"] = 0
        _write_json(geometry_path, geometry_obj)

        meta["geometry_path"] = geometry_path
        meta["landmark_model"] = "mediapipe_face_landmarker_tasks"
        meta["landmark_count"] = 0
        _write_json(meta_path, meta)
        return {"ok": False, "meta_path": meta_path, "error": "image_read_failed", "geometry_path": geometry_path}

    try:
        landmarks_px, info = extract_face_landmarks_tasks(image_bgr, cfg)
    except Exception as e:
        geometry_obj["error"] = f"landmarker_failed: {type(e).__name__}: {e}"
        geometry_obj["landmark_count"] = 0
        _write_json(geometry_path, geometry_obj)

        meta["geometry_path"] = geometry_path
        meta["landmark_model"] = "mediapipe_face_landmarker_tasks"
        meta["landmark_count"] = 0
        _write_json(meta_path, meta)
        return {"ok": False, "meta_path": meta_path, "error": geometry_obj["error"], "geometry_path": geometry_path}

    if landmarks_px is None:
        geometry_obj.update(info)
        geometry_obj["error"] = info.get("error", "no_face_detected")
        geometry_obj["landmark_count"] = 0
        _write_json(geometry_path, geometry_obj)

        meta["geometry_path"] = geometry_path
        meta["landmark_model"] = "mediapipe_face_landmarker_tasks"
        meta["landmark_count"] = 0
        _write_json(meta_path, meta)
        return {"ok": False, "meta_path": meta_path, "error": geometry_obj["error"], "geometry_path": geometry_path}

    h, w = image_bgr.shape[:2]
    geom = compute_geometry_features(landmarks_px, (w, h), symmetry=True)

    geometry_obj.update(info)
    geometry_obj["landmark_count"] = len(landmarks_px)
    geometry_obj["landmark_count_actual"] = len(landmarks_px)

    pq = compute_pose_quality_scores(landmarks_px, info, cfg, mode=mode)
    geometry_obj.update(pq)

    geometry_obj["geometry"] = geom
    _write_json(geometry_path, geometry_obj)

    meta["geometry_path"] = geometry_path
    meta["landmark_model"] = "mediapipe_face_landmarker_tasks"
    meta["landmark_count"] = len(landmarks_px)
    meta["quality_mode"] = mode
    _write_json(meta_path, meta)

    return {"ok": True, "meta_path": meta_path, "geometry_path": geometry_path, "quality_pass": geometry_obj["quality_gate"]["quality_pass"], "quality_score": geometry_obj["scores"]["quality_score"]}


def batch_run_phase2(meta_paths: List[str], *, processed_dir: str, profiles_dir: str, cfg: TaskLandmarkerConfig, prefer_processed: bool = True, mode: str = "strict") -> Dict[str, Any]:
    success = 0
    fail = 0
    rejected = 0
    failures = []

    for mpth in meta_paths:
        try:
            res = run_phase2_on_meta(mpth, processed_dir=processed_dir, profiles_dir=profiles_dir, cfg=cfg, prefer_processed=prefer_processed, mode=mode)
            if res.get("ok"):
                success += 1
                if res.get("quality_pass") is False:
                    rejected += 1
            else:
                fail += 1
                failures.append(res)
        except Exception as e:
            fail += 1
            failures.append({"ok": False, "meta_path": mpth, "error": f"exception: {type(e).__name__}: {e}"})

    return {"success": success, "fail": fail, "rejected": rejected, "mode": mode, "failures": failures}
