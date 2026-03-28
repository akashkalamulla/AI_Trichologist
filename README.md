# AI Trichologist

**An Explainable and Constraint-Aware AI Framework for Hairstyle Recommendation and Virtual Try-On**

BSc (Hons) Computing — Final Year Research Project  
NIBM / Coventry University

---

## What This Is

AI Trichologist is an end-to-end computer vision pipeline that analyses a user's facial structure and hair condition from a single photograph, recommends hairstyles through a weighted multi-attribute scoring engine, explains *why* each style suits them using LLM-generated reasoning, and renders a photorealistic virtual try-on via Stable Diffusion inpainting. The entire system runs in Google Colab and is served through a Gradio web interface.

The research contribution is **system-level integration and explainability** — not novel model architecture. Existing pretrained models (MediaPipe, Stable Diffusion, Gemini) are orchestrated into a unified pipeline where every recommendation is traceable back to scored facial attributes, and the user can see exactly why a hairstyle was suggested.

---

## Pipeline Architecture

The system operates across five sequential phases, communicating through a shared JSON profile stored on Google Drive:

**Phase 1 — Preprocessing** (`modules/phase1_preprocessing.py`)  
Input image is resized to 512×512, contrast-enhanced via CLAHE (clip limit 2.0, 8×8 grid), and face-cropped with a 30% margin using MediaPipe Face Detection. A quality gate rejects images where the detected face is below 10% of frame area.

**Phase 2 — Facial Landmark Extraction** (`modules/phase2_landmarks.py`)  
MediaPipe Face Mesh extracts 468 landmarks. Derived measurements include face shape classification (oval, round, square, oblong, heart) from jawline-to-cheekbone ratios, forehead height estimation, and facial symmetry scoring.

**Phase 3 — Hair Segmentation** (`modules/phase3_segmentation.py`)  
MediaPipe Selfie Segmentation isolates the hair region. A face protection mask prevents skin pixels from being misclassified as hair. The segmented region is used to estimate hair density and texture (straight, wavy, curly) via edge frequency analysis.

**Phase 4 — Baldness Zone Analysis** (`modules/phase4_baldness_zones.py`)  
Analyses the scalp-to-hair ratio across defined zones (frontal, temporal, crown). Computes hairline recession via `HairlineRatio` and maps to a Norwood stage estimate. Outputs `ScalpRatio`, `DensityScore`, `ForeheadHeight`, and `RecessionScore` as numerical features.

**Phase 5 — Knowledge-Based Recommendation** (`knowledgebase/scripts/reasoning_engine.py`)  
A weighted scoring formula ranks 197 canonical hairstyles across 21 base categories:

```
CompatibilityScore = (0.40 × FaceScore) + (0.25 × TextureScore) + (0.20 × DensityScore) + (0.15 × HairLossScore)
AdjustedScore = CompatibilityScore + (EnhancementWeight × LearnedBoost)
```

A diversification pass caps output at 2 styles per base category to prevent redundant recommendations. A lightweight feedback loop (`learned_weights.json`) stores user preference boosts that persist across sessions.

**LLM Explanation Layer** (Gemini API)  
Each recommendation is accompanied by a natural-language explanation generated via the Gemini API, grounded in the scored attributes — not hallucinated. Explanation templates (`explanation_templates.json`) provide structured prompts so the LLM output remains faithful to the computed scores.

**Virtual Try-On** (Stable Diffusion Inpainting)  
The segmented hair region is used as an inpainting mask. Gemini generates a style-specific prompt, and `runwayml/stable-diffusion-inpainting` renders the selected hairstyle onto the original photograph. The result is served back through the Gradio dashboard.

---

## Knowledge Base

The recommendation engine is backed by a structured knowledge base built from 11 curated web sources:

| Source | Entries |
|--------|---------|
| thevou.json | 80 |
| haircuts.json | 40 |
| styleseat.json | 29 |
| wimpoleclinic.json | 27 |
| hairstyles_data.json | 25 |
| bosshunting.json | 20 |
| forteseries.json | 17 |
| forteseries_data.json | 17 |
| toppik.json | 15 |
| heygoldie.json | 10 |
| manforhimself.json | 5 |

Raw entries are deduplicated and normalised through a multi-step pipeline (`deterministic_mapping.py` → `llm_fallback.py` → `clean_taxonomy.py` → `validate_taxonomy.py`) into **197 canonical hairstyles** grouped under **21 base categories**: buzz, crew, crop, fade, undercut, quiff, pompadour, fringe, slick back, mohawk, curtain, mullet, afro, dreadlocks, long flow, skater, edgar, side part, business, shaved, and other.

Each category carries per-attribute compatibility scores (face shape × 7, texture × 4, density × 3, hair loss patterns + Norwood stages 1–5), a maintenance rating, and aesthetic effect tags.

---

## Project Structure

```
AI_Trichologist/
├── modules/                          # Core pipeline (Python)
│   ├── phase1_preprocessing.py       # CLAHE, face crop, quality gate
│   ├── phase2_landmarks.py           # MediaPipe 468-landmark extraction
│   ├── phase3_segmentation.py        # Hair segmentation + density/texture
│   ├── phase4_baldness_zones.py      # Scalp ratio, Norwood estimation
│   └── __init__.py
├── knowledgebase/
│   ├── data_raw/                     # 11 scraped source JSONs
│   ├── data_intermediate/            # Dedup & classification stages
│   ├── data_final/                   # 197-entry canonical taxonomy
│   ├── data_knowledge/               # Rules, metadata, learned weights
│   ├── scripts/                      # KB build + reasoning engine
│   └── prompts/                      # LLM taxonomy classification prompt
├── notebooks/                        # Google Colab notebooks
│   ├── phase1_preprocessing.ipynb
│   ├── phase2_landmarks.ipynb
│   ├── phase3_segmentation.ipynb
│   ├── phase4_baldness_zones.ipynb
│   ├── knowledgebase.ipynb           # KB construction & recommendation
│   └── Diffusion.ipynb               # Virtual try-on pipeline
├── config/                           # Runtime config (populated in Colab)
├── models/                           # Model cache (populated at runtime)
└── outputs/generated/                # Try-on output images
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Face detection & landmarks | MediaPipe Face Mesh (468 landmarks) |
| Hair segmentation | MediaPipe Selfie Segmentation |
| Image preprocessing | OpenCV, CLAHE |
| Recommendation engine | Custom weighted scoring (Python) |
| Explainability | Gemini API (structured prompt generation) |
| Virtual try-on | Stable Diffusion Inpainting (`runwayml/stable-diffusion-inpainting`) |
| Feedback persistence | JSON file (`learned_weights.json`) |
| Runtime environment | Google Colab (GPU) |
| User interface | Gradio (public HTTPS link) |
| Data interchange | Shared JSON profile on Google Drive |

---

## How to Run

1. Open any of the phase notebooks in Google Colab
2. Mount Google Drive when prompted
3. Upload a front-facing portrait photograph
4. Run cells sequentially — each phase writes to the shared JSON profile
5. The `knowledgebase.ipynb` notebook runs recommendation and explanation generation
6. The `Diffusion.ipynb` notebook runs virtual try-on rendering
7. The Gradio dashboard provides an integrated interface for end-to-end use

**Requirements**: Google Colab with GPU runtime, Google Drive access, Gemini API key (stored via Colab `userdata`).

---

## Key Research Claims

1. **Multi-attribute constraint-aware recommendation**: Unlike single-factor systems, the scoring engine jointly considers face shape, hair texture, density, and hair loss pattern — including Norwood stage mapping — with empirically set weights.

2. **Explainability by design**: Every recommendation includes a traceable breakdown showing exactly which attributes contributed what score. The LLM explanation layer translates these numerical scores into human-readable reasoning, grounded in computed values rather than generated from scratch.

3. **Integrated virtual try-on**: The same pipeline that analyses and recommends also renders — the segmentation mask from Phase 3 directly becomes the inpainting mask for Stable Diffusion, eliminating manual mask creation.

4. **Preference-adaptive feedback loop**: The `learned_weights.json` mechanism allows the system to adjust future recommendations based on past user ratings, without retraining any model.

5. **Reproducible knowledge base construction**: The 11-source → 197-style taxonomy pipeline is fully scripted and auditable, with deterministic mapping as the primary classification method and LLM fallback only for unresolved styles.

---

## Evaluation Summary

A nine-participant within-subjects user study assessed usability (SUS) and trust (before/after explanation exposure):

- **Mean SUS Score**: 57.22 (acceptable range)
- **Trust before explanation**: 3.33 / 5.0 (SD = 0.87)
- **Trust after explanation**: 3.56 / 5.0 (SD = 0.53)
- The reduction in standard deviation (0.87 → 0.53) indicates that explanations produced **more consistent trust** across participants, even though the mean shift was modest

The evaluation is framed under Design Science Research (DSR) methodology — the artefact itself and its integration novelty are the primary contributions, with the user study providing formative validation rather than summative proof.

---

## Limitations

- The system processes single front-facing photographs only; profile or angled shots are not supported
- Hair texture classification uses edge frequency heuristics rather than a trained texture classifier
- Virtual try-on quality depends on Stable Diffusion inpainting fidelity, which can produce artifacts at hair-skin boundaries
- The knowledge base covers 21 base style categories — niche or culturally specific styles outside these categories are not represented
- Evaluation was conducted with 9 participants under DSR framing; generalisability claims are limited accordingly

---

## License

Academic research project. Not licensed for commercial use.
