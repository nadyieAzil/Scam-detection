"""
FastAPI backend — AI Scam Detection
Loads fine-tuned XLM-RoBERTa + Rule Engine and exposes /predict
"""

import re
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR   = "./xlm-roberta-scam-model"
AI_WEIGHT   = 0.70
RULE_WEIGHT = 0.30
THRESHOLD   = 0.50
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ────────────────────────────────────────────────
print(f"Loading model from {MODEL_DIR} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("Model ready.")

# ── Rule Engine ───────────────────────────────────────────────────────────────
RULES = [
    (r"bit\.ly",                    +45),
    (r"bank-secure\.com",           +45),
    (r"sh0pee-help\.net",           +45),
    (r"tng-update\.com",            +45),
    (r"wasap\.my",                  +35),
    (r"http[s]?://\S+",             +20),
    (r"\botp\b",                    +35),
    (r"send.*\botp\b",              +15),
    (r"\burgent\b",                 +20),
    (r"cepat",                      +15),
    (r"segera",                     +15),
    (r"immediately",                +15),
    (r"kena (block|suspend)",       +30),
    (r"dibekukan",                  +30),
    (r"(account|akaun).*(frozen|blocked|suspend)", +20),
    (r"verify sekarang",            +25),
    (r"verify now",                 +25),
    (r"(login|masuk) untuk aktifkan", +25),
    (r"suspicious (activity|login)", +30),
    (r"detected login",             +30),
    (r"confirm identity",           +25),
    (r"reset password",             +25),
    (r"email hacked",               +30),
    (r"you win",                    +35),
    (r"claim now",                  +30),
    (r"sebelum expired",            +25),
    (r"free rm\d+",                 +35),
    (r"rm\d+.*voucher",             +30),
    (r"loan approved",              +35),
    (r"pinjaman.*diluluskan",       +35),
    (r"send.*\bic\b",               +40),
    (r"bank details",               +40),
    (r"send ic",                    +40),
    (r"crypto profit",              +40),
    (r"investment.*guaranteed",     +40),
    (r"profit.*200\s*%",            +40),
    (r"guaranteed",                 +25),
    (r"part.?time.*rm\d+",          +30),
    (r"no experience needed",       +25),
    (r"join (now|link)",            +20),
    (r"parcel stuck",               +30),
    (r"kastam",                     +25),
    (r"release fee",                +35),
    (r"customs hold",               +30),
    (r"bayar rm\d+.*release",       +35),
    (r"police report",              +30),
    (r"bank negara report",         +30),
    (r"involved in (case|kes)",     +30),
    (r"jom makan",                  -25),
    (r"\blecture\b",                -25),
    (r"\bstudy\b",                  -20),
    (r"\bexam\b",                   -20),
    (r"\bassignment\b",             -20),
    (r"traffic jam",                -20),
    (r"\bdinner\b",                 -15),
    (r"\bkelas\b",                  -15),
    (r"\bcafe\b",                   -15),
    (r"tolong aku",                 -15),
    (r"project ai",                 -15),
    (r"\bweekend\b",                -10),
    (r"good morning",               -10),
    (r"bangun cepat",               -10),
    (r"jumpa esok",                 -10),
    (r"lambat sikit",               -10),
    (r"plan.*malam",                -10),
]
COMPILED_RULES = [(re.compile(p, re.IGNORECASE), d) for p, d in RULES]

def rule_score(text: str) -> float:
    score = 0
    for pattern, delta in COMPILED_RULES:
        if pattern.search(text):
            score += delta
    return max(0, min(100, score)) / 100.0

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Scam Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your friend's domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    label: str          # "SCAM" or "SAFE"
    confidence: float   # fusion score 0.0 – 1.0
    ai_prob: float
    rule_score: float

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_DIR}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.message

    # AI probability
    enc = tokenizer(text, truncation=True, padding=True,
                    max_length=64, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    ai_prob = float(torch.softmax(logits, dim=-1)[0, 1])

    # Rule score
    r_score = rule_score(text)

    # Fusion
    fusion = AI_WEIGHT * ai_prob + RULE_WEIGHT * r_score
    label  = "SCAM" if fusion >= THRESHOLD else "SAFE"

    return PredictResponse(
        label=label,
        confidence=round(fusion, 4),
        ai_prob=round(ai_prob, 4),
        rule_score=round(r_score, 4),
    )
