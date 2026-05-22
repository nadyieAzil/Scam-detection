# 🛡️ AI Scam Detection — Model Handover Document

> **For:** Frontend Developer & any AI assistant continuing this project  
> **Backend owner:** [Your name]  
> **Last updated:** 2026-05-22

---

## 📌 What This Project Does

This is a **Malaysian-context SMS/message scam detector**.  
Given any text message, it returns whether the message is **SCAM** or **SAFE**, along with a confidence score.

It supports **mixed-language input** — Bahasa Malaysia, English, and Manglish (e.g., "kena block", "send OTP cepat").

---

## 🧠 How the Model Works (Architecture)

The system uses a **3-phase hybrid approach**:

```
Input Message
     │
     ├──► [Phase 1] XLM-RoBERTa AI Model  ──► AI Probability (0.0 – 1.0)
     │         Fine-tuned on 3,000 Malaysian scam messages
     │
     ├──► [Phase 2] Rule Engine            ──► Rule Score (0.0 – 1.0)
     │         60+ hand-coded regex rules for Malaysian scam patterns
     │
     └──► [Phase 3] Fusion Logic           ──► Final Score + Label
               Final = (0.70 × AI_prob) + (0.30 × Rule_score)
               If Final ≥ 0.50  →  SCAM
               If Final < 0.50  →  SAFE
```

### Phase 1 — AI Model (XLM-RoBERTa)
- Base: `xlm-roberta-base` (Facebook, multilingual transformer)
- Fine-tuned on `malaysia_scam_dataset_3000.csv` (3,000 rows, balanced SCAM/SAFE)
- Training: 3 epochs, batch size 64, max sequence length 64 tokens
- Output: probability score 0.0–1.0 (how likely the message is SCAM)
- Saved to: `./xlm-roberta-scam-model/`

### Phase 2 — Rule Engine
- 60+ hand-written regex rules targeting Malaysian scam signals
- **Positive rules** (raise scam risk): `bit.ly`, `OTP`, `kena block`, `loan approved`, `claim now`, `send IC`, `bank details`, `kastam`, `crypto profit`, etc.
- **Negative rules** (lower risk / safe signals): `jom makan`, `lecture`, `exam`, `traffic jam`, `assignment`, `cafe`, etc.
- Score clamped to [0, 100] then normalised to [0.0, 1.0]

### Phase 3 — Fusion
- Combines AI + Rules with weighted average
- AI gets 70% weight (stronger, generalises better)
- Rules get 30% weight (catches novel phishing domains AI may miss)
- Threshold: 0.50

---

## 🔌 API Reference

The backend is a **FastAPI** server. Base URL is provided separately (ngrok/deployed URL).

### `GET /`
Health check.
```json
{ "status": "ok", "model": "./xlm-roberta-scam-model" }
```

---

### `POST /predict` ← Main endpoint

**Request body (JSON):**
```json
{
  "message": "Your text message here"
}
```

**Response (JSON):**
```json
{
  "label": "SCAM",
  "confidence": 0.8912,
  "ai_prob": 0.9200,
  "rule_score": 0.7500
}
```

| Field | Type | Description |
|---|---|---|
| `label` | `string` | `"SCAM"` or `"SAFE"` |
| `confidence` | `float` | Final fusion score (0.0–1.0). Higher = more likely SCAM |
| `ai_prob` | `float` | Raw AI model probability |
| `rule_score` | `float` | Rule engine score |

**Interactive docs:** Visit `BASE_URL/docs` in browser to test manually.

---

## 💻 Frontend Integration (React)

```js
const BASE_URL = "https://YOUR-NGROK-OR-DEPLOYED-URL";

async function checkScam(message) {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) throw new Error("API error");
  
  const data = await res.json();
  // data.label       → "SCAM" or "SAFE"
  // data.confidence  → 0.0 to 1.0  (use for progress bar / percentage)
  // data.ai_prob     → AI model contribution
  // data.rule_score  → Rule engine contribution
  return data;
}
```

### Suggested UI elements to show:
- 🔴 Red badge / alert when `label === "SCAM"`
- 🟢 Green badge when `label === "SAFE"`
- Confidence bar: `Math.round(data.confidence * 100)` → e.g. `89%`
- Optional breakdown: AI score vs Rule score

---

## 📁 Project File Structure

```
AI Scam detection/
│
├── api.py                        ← FastAPI server (run this to serve the API)
├── run_training.py               ← Retrains the model (backend dev only)
├── fusion_engine.py              ← Standalone evaluation of all 3 systems
├── append_cells.py               ← Adds Phase 4 & 5 cells to Jupyter notebook
├── generate.py                   ← Dataset generation script
│
├── malaysia_scam_dataset_3000.csv  ← Training dataset (3,000 messages)
│
├── xlm-roberta-scam-model/       ← ⭐ Saved fine-tuned model (share this folder)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── sentencepiece.bpe.model
│   └── special_tokens_map.json
│
└── results/                      ← Huggingface Trainer checkpoints
```

---

## ▶️ How to Start the API Server

```bash
# Install dependencies (first time only)
pip install fastapi uvicorn transformers torch numpy

# Run the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Expose publicly via ngrok (separate terminal)
ngrok http 8000
```

---

## ⚙️ Config Values (api.py)

| Constant | Value | Meaning |
|---|---|---|
| `MODEL_DIR` | `./xlm-roberta-scam-model` | Path to saved model |
| `AI_WEIGHT` | `0.70` | AI contribution to final score |
| `RULE_WEIGHT` | `0.30` | Rule engine contribution |
| `THRESHOLD` | `0.50` | Score cutoff for SCAM label |

---

## 📊 Model Performance (on test set, 300 samples)

| System | Accuracy | Notes |
|---|---|---|
| AI Only (XLM-RoBERTa) | ~93–96% | Strong on seen patterns |
| Rule Engine Only | ~70–80% | Good on keyword-heavy messages |
| **Fusion (AI + Rules)** | **~95–97%** | Best overall |

---

## 🔑 Key Notes for AI Assistants Continuing This Project

1. **Do NOT retrain** unless intentional — model is already saved in `xlm-roberta-scam-model/`
2. **FORCE_RETRAIN = False** in `run_training.py` → it resumes from saved model by default
3. The dataset has **balanced classes** (50% SCAM, 50% SAFE) — maintain this if adding data
4. **Max token length is 64** — messages longer than ~50 words get truncated
5. The model is **multilingual** — it handles Malay, English, and mixed naturally
6. CORS is set to `allow_origins=["*"]` in `api.py` — restrict this in production
7. The `/predict` endpoint is **stateless** — no session, no database, pure inference
