"""
=============================================================================
 PHASE 4 — Rule Engine  +  PHASE 5 — Fusion Logic
=============================================================================
 Loads the fine-tuned XLM-RoBERTa from ./xlm-roberta-scam-model, then:
   • Computes AI SCAM probability for every test sample
   • Computes a rule-based risk score (0-100) for every test sample
   • Fuses:  Final = 0.70 × AI_prob  +  0.30 × (rule_score / 100)
   • Prints a three-way comparison:  AI-only | Rule-only | Fusion
=============================================================================
"""

import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────────────────────────────────────
# 0.  REPRODUCIBILITY + DEVICE
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATASET  →  reproduce the SAME test split used during training
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("malaysia_scam_dataset_3000.csv")
label_map = {"SAFE": 0, "SCAM": 1}
df["label_int"] = df["label"].map(label_map)

_, temp_df = train_test_split(df, test_size=0.20, random_state=42,
                               stratify=df["label_int"])
_, test_df = train_test_split(temp_df, test_size=0.50, random_state=42,
                               stratify=temp_df["label_int"])

test_texts  = test_df["text"].tolist()
test_labels = test_df["label_int"].tolist()
print(f"\nTest set: {len(test_df)} samples  "
      f"(SAFE={test_labels.count(0)}, SCAM={test_labels.count(1)})")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOAD SAVED MODEL + TOKENIZER
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "./xlm-roberta-scam-model"
print(f"\nLoading saved model from '{MODEL_DIR}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print("Model loaded successfully.")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  AI INFERENCE — get SCAM probability for every test sample
# ──────────────────────────────────────────────────────────────────────────────
def get_ai_probs(texts, batch_size=64):
    """Return array of SCAM probabilities (shape: [N,])."""
    scam_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=64, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        scam_probs.extend(probs.tolist())
    return np.array(scam_probs)

print("\nRunning AI inference on test set...")
ai_probs = get_ai_probs(test_texts)
ai_preds  = (ai_probs >= 0.5).astype(int)
print(f"AI inference done.  Sample prob range: "
      f"[{ai_probs.min():.4f}, {ai_probs.max():.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  RULE ENGINE  (Phase 4)
# ──────────────────────────────────────────────────────────────────────────────
#
#   Each entry is  (pattern, score_delta)
#   Positive → raises SCAM risk   |   Negative → lowers risk (safe indicator)
#   Final rule score is CLAMPED to [0, 100].
#
RULES = [
    # ── Suspicious domains / shorteners ─────────────────────────────────────
    (r"bit\.ly",                    +45),
    (r"bank-secure\.com",           +45),
    (r"sh0pee-help\.net",           +45),
    (r"tng-update\.com",            +45),
    (r"wasap\.my",                  +35),
    (r"http[s]?://\S+",             +20),   # any raw URL

    # ── OTP / credential harvesting ─────────────────────────────────────────
    (r"\botp\b",                    +35),
    (r"send.*\botp\b",              +15),   # stacked bonus

    # ── Urgency language ────────────────────────────────────────────────────
    (r"\burgent\b",                 +20),
    (r"cepat",                      +15),
    (r"segera",                     +15),
    (r"immediately",                +15),

    # ── Account / wallet threats ────────────────────────────────────────────
    (r"kena (block|suspend)",       +30),
    (r"dibekukan",                  +30),
    (r"(account|akaun).*(frozen|blocked|suspend)", +20),
    (r"verify sekarang",            +25),
    (r"verify now",                 +25),
    (r"(login|masuk) untuk aktifkan", +25),
    (r"suspicious (activity|login)",  +30),
    (r"detected login",             +30),
    (r"confirm identity",           +25),
    (r"reset password",             +25),
    (r"email hacked",               +30),

    # ── Prize / voucher scams ────────────────────────────────────────────────
    (r"you win",                    +35),
    (r"claim now",                  +30),
    (r"sebelum expired",            +25),
    (r"free rm\d+",                 +35),
    (r"rm\d+.*voucher",             +30),

    # ── Loan / financial scams ──────────────────────────────────────────────
    (r"loan approved",              +35),
    (r"pinjaman.*diluluskan",       +35),
    (r"send.*\bic\b",               +40),
    (r"bank details",               +40),
    (r"send ic",                    +40),

    # ── Investment / crypto fraud ────────────────────────────────────────────
    (r"crypto profit",              +40),
    (r"investment.*guaranteed",     +40),
    (r"profit.*200\s*%",            +40),
    (r"guaranteed",                 +25),

    # ── Job scams ────────────────────────────────────────────────────────────
    (r"part.?time.*rm\d+",          +30),
    (r"no experience needed",       +25),
    (r"join (now|link)",            +20),

    # ── Parcel / customs scams ──────────────────────────────────────────────
    (r"parcel stuck",               +30),
    (r"kastam",                     +25),
    (r"release fee",                +35),
    (r"customs hold",               +30),
    (r"bayar rm\d+.*release",       +35),

    # ── Police / authority impersonation ────────────────────────────────────
    (r"police report",              +30),
    (r"bank negara report",         +30),
    (r"involved in (case|kes)",     +30),

    # ─────────────────────────────────────────────────────────────────────────
    # SAFE INDICATORS — subtract from risk score
    # ─────────────────────────────────────────────────────────────────────────
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

# Pre-compile patterns for speed
COMPILED_RULES = [(re.compile(p, re.IGNORECASE), delta) for p, delta in RULES]

def rule_score(text: str) -> float:
    """
    Return a SCAM risk score in [0.0, 1.0].
    Internally computed on a 0-100 integer scale, then normalised.
    """
    score = 0
    tl = text.lower()
    for pattern, delta in COMPILED_RULES:
        if pattern.search(tl):
            score += delta
    score = max(0, min(100, score))   # clamp to [0, 100]
    return score / 100.0              # normalise to [0, 1]

print("\nRunning Rule Engine on test set...")
rule_scores = np.array([rule_score(t) for t in test_texts])
rule_preds  = (rule_scores >= 0.5).astype(int)
print(f"Rule scoring done.  Score range: "
      f"[{rule_scores.min():.4f}, {rule_scores.max():.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  FUSION LOGIC  (Phase 5)
#     Final Score = 0.70 × AI_prob  +  0.30 × rule_score
# ──────────────────────────────────────────────────────────────────────────────
AI_WEIGHT   = 0.70
RULE_WEIGHT = 0.30
THRESHOLD   = 0.50

fusion_scores = AI_WEIGHT * ai_probs + RULE_WEIGHT * rule_scores
fusion_preds  = (fusion_scores >= THRESHOLD).astype(int)
print(f"\nFusion scores range: "
      f"[{fusion_scores.min():.4f}, {fusion_scores.max():.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 6.  COMPARATIVE EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
DIVIDER = "=" * 60

def print_report(name, preds):
    acc = accuracy_score(test_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary")
    cm = confusion_matrix(test_labels, preds)
    print(f"\n{DIVIDER}")
    print(f"  {name}")
    print(DIVIDER)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {p*100:.2f}%")
    print(f"  Recall    : {r*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted SAFE  Predicted SCAM")
    print(f"  True SAFE      {cm[0,0]:^14}  {cm[0,1]:^14}")
    print(f"  True SCAM      {cm[1,0]:^14}  {cm[1,1]:^14}")
    print(f"\n  Full Classification Report:")
    print(classification_report(test_labels, preds,
                                target_names=["SAFE", "SCAM"],
                                digits=4))

print_report("SYSTEM 1 — AI Only  (XLM-RoBERTa)",  ai_preds)
print_report("SYSTEM 2 — Rule Engine Only",          rule_preds)
print_report("SYSTEM 3 — FUSION  (AI 70% + Rules 30%)", fusion_preds)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SAMPLE PREDICTIONS — live inference demo
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_MSGS = [
    # Should be SCAM
    "TNG eWallet kena suspend, login untuk aktifkan tng-update.com",
    "You win RM4500 cash, claim now sebelum expired bit.ly/verify",
    "OTP needed to verify transaction, send cepat",
    "Bank kau kena block, click link ni cepat bank-secure.com/login",
    "Loan approved instant RM3500, send IC & bank details",
    # Should be SAFE
    "Jom makan dekat cafe kampus petang ni bro",
    "Assignment dah siap belum? Aku lambat sikit traffic jam",
    "Kita jumpa esok kelas 10am, jangan lupa bawa notes",
    "Good morning bro, bangun cepat ada lecture pagi ni",
    "Aku tengah buat project AI ni, susah gila bro",
]

print(f"\n{DIVIDER}")
print("  LIVE DEMO — Fusion Inference on Sample Messages")
print(DIVIDER)

for msg in SAMPLE_MSGS:
    ai_p  = float(get_ai_probs([msg])[0])
    r_p   = rule_score(msg)
    final = AI_WEIGHT * ai_p + RULE_WEIGHT * r_p
    label = "🔴 SCAM" if final >= THRESHOLD else "🟢 SAFE"
    print(f"\n  Msg   : {msg[:70]}")
    print(f"  AI    : {ai_p*100:5.1f}%  |  Rule: {r_p*100:5.1f}%  "
          f"|  Fusion: {final*100:5.1f}%  →  {label}")

print(f"\n{DIVIDER}")
print("  Fusion engine evaluation complete.")
print(DIVIDER)
