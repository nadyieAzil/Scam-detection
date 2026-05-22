"""
Appends Phase 4 (Rule Engine) + Phase 5 (Fusion Logic) cells
to the existing scam_detection_xlm_roberta.ipynb notebook.
"""
import json

NOTEBOOK_PATH = "scam_detection_xlm_roberta.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── helper ────────────────────────────────────────────────────────────────────
def md_cell(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
new_cells = []

new_cells.append(md_cell([
    "---\n",
    "## 🟣 Phase 4 — Rule-Based Risk Engine\n",
    "We manually code a **Rule Engine** that assigns a risk score (0–100) based on "
    "known scam signals in Malaysian messages.\n\n",
    "Each rule is a compiled regex pattern paired with a score delta:\n",
    "- **Positive delta** → raises scam risk (e.g. `bit.ly`, `OTP`, `kena block`)\n",
    "- **Negative delta** → lowers risk (safe phrases like `jom makan`, `lecture`, `exam`)\n\n",
    "The score is **clamped to [0, 100]** and then **normalised to [0.0, 1.0]** "
    "for seamless fusion with the AI probability."
]))

new_cells.append(code_cell([
    "import re\n",
    "\n",
    "# ── Rule table: (regex_pattern, score_delta) ──────────────────────────\n",
    "# Positive delta → SCAM indicator   |   Negative → SAFE indicator\n",
    "RULES = [\n",
    "    # Suspicious domains / shorteners\n",
    "    (r'bit\\.ly',                    +45),\n",
    "    (r'bank-secure\\.com',           +45),\n",
    "    (r'sh0pee-help\\.net',           +45),\n",
    "    (r'tng-update\\.com',            +45),\n",
    "    (r'wasap\\.my',                  +35),\n",
    "    (r'http[s]?://\\S+',             +20),  # any raw URL\n",
    "\n",
    "    # OTP / credential harvesting\n",
    "    (r'\\botp\\b',                    +35),\n",
    "    (r'send.*\\botp\\b',              +15),  # stacked bonus\n",
    "\n",
    "    # Urgency language\n",
    "    (r'\\burgent\\b',                 +20),\n",
    "    (r'cepat',                       +15),\n",
    "    (r'segera',                      +15),\n",
    "    (r'immediately',                 +15),\n",
    "\n",
    "    # Account / wallet threats\n",
    "    (r'kena (block|suspend)',        +30),\n",
    "    (r'dibekukan',                   +30),\n",
    "    (r'(account|akaun).*(frozen|blocked|suspend)', +20),\n",
    "    (r'verify sekarang',             +25),\n",
    "    (r'verify now',                  +25),\n",
    "    (r'(login|masuk) untuk aktifkan',+25),\n",
    "    (r'suspicious (activity|login)', +30),\n",
    "    (r'detected login',              +30),\n",
    "    (r'confirm identity',            +25),\n",
    "    (r'reset password',              +25),\n",
    "    (r'email hacked',                +30),\n",
    "\n",
    "    # Prize / voucher scams\n",
    "    (r'you win',                     +35),\n",
    "    (r'claim now',                   +30),\n",
    "    (r'sebelum expired',             +25),\n",
    "    (r'free rm\\d+',                  +35),\n",
    "    (r'rm\\d+.*voucher',              +30),\n",
    "\n",
    "    # Loan / financial scams\n",
    "    (r'loan approved',               +35),\n",
    "    (r'pinjaman.*diluluskan',        +35),\n",
    "    (r'send.*\\bic\\b',               +40),\n",
    "    (r'bank details',                +40),\n",
    "    (r'send ic',                     +40),\n",
    "\n",
    "    # Investment / crypto fraud\n",
    "    (r'crypto profit',               +40),\n",
    "    (r'investment.*guaranteed',      +40),\n",
    "    (r'profit.*200\\s*%',             +40),\n",
    "    (r'guaranteed',                  +25),\n",
    "\n",
    "    # Job scams\n",
    "    (r'part.?time.*rm\\d+',           +30),\n",
    "    (r'no experience needed',        +25),\n",
    "    (r'join (now|link)',             +20),\n",
    "\n",
    "    # Parcel / customs scams\n",
    "    (r'parcel stuck',                +30),\n",
    "    (r'kastam',                      +25),\n",
    "    (r'release fee',                 +35),\n",
    "    (r'customs hold',                +30),\n",
    "    (r'bayar rm\\d+.*release',        +35),\n",
    "\n",
    "    # Authority impersonation\n",
    "    (r'police report',               +30),\n",
    "    (r'bank negara report',          +30),\n",
    "    (r'involved in (case|kes)',      +30),\n",
    "\n",
    "    # ── SAFE INDICATORS ─────────────────────────────────────────────────\n",
    "    (r'jom makan',                   -25),\n",
    "    (r'\\blecture\\b',                 -25),\n",
    "    (r'\\bstudy\\b',                   -20),\n",
    "    (r'\\bexam\\b',                    -20),\n",
    "    (r'\\bassignment\\b',              -20),\n",
    "    (r'traffic jam',                 -20),\n",
    "    (r'\\bdinner\\b',                  -15),\n",
    "    (r'\\bkelas\\b',                   -15),\n",
    "    (r'\\bcafe\\b',                    -15),\n",
    "    (r'tolong aku',                  -15),\n",
    "    (r'project ai',                  -15),\n",
    "    (r'\\bweekend\\b',                 -10),\n",
    "    (r'good morning',                -10),\n",
    "    (r'bangun cepat',                -10),\n",
    "    (r'jumpa esok',                  -10),\n",
    "    (r'lambat sikit',                -10),\n",
    "]\n",
    "\n",
    "# Pre-compile all patterns for performance\n",
    "COMPILED_RULES = [(re.compile(p, re.IGNORECASE), d) for p, d in RULES]\n",
    "\n",
    "def rule_score(text: str) -> float:\n",
    "    \"\"\"Return normalised SCAM risk in [0.0, 1.0].\"\"\"\n",
    "    score = 0\n",
    "    for pattern, delta in COMPILED_RULES:\n",
    "        if pattern.search(text):\n",
    "            score += delta\n",
    "    score = max(0, min(100, score))   # clamp\n",
    "    return score / 100.0             # normalise\n",
    "\n",
    "# Quick smoke-test\n",
    "test_cases = [\n",
    "    (\"OTP needed to verify transaction, send cepat\", 1),\n",
    "    (\"Jom makan dekat cafe kampus\",                 0),\n",
    "    (\"Bank kau kena block, click link ni cepat bit.ly/verify\", 1),\n",
    "    (\"Assignment dah siap belum?\",                  0),\n",
    "]\n",
    "print('Rule engine smoke-test:')\n",
    "for txt, expected in test_cases:\n",
    "    rs = rule_score(txt)\n",
    "    verdict = 'SCAM' if rs >= 0.5 else 'SAFE'\n",
    "    ok = '✅' if (1 if rs >= 0.5 else 0) == expected else '❌'\n",
    "    print(f'  {ok} [{rs:.2f}] {verdict:4s} | {txt[:55]}')\n",
]))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — FUSION
# ══════════════════════════════════════════════════════════════════════════════
new_cells.append(md_cell([
    "---\n",
    "## 🔴 Phase 5 — Fusion Logic\n",
    "Combine the **AI probability** and the **rule-based score** into a single "
    "final decision:\n\n",
    "```\n",
    "Final Score = 0.70 × AI_prob_scam  +  0.30 × rule_score\n",
    "```\n\n",
    "- **Threshold = 0.50**: scores ≥ 0.50 → `SCAM`, below → `SAFE`\n",
    "- Weights chosen so the strong AI model dominates, while rules "
    "catch edge cases the model might miss (e.g. novel phishing domains).\n\n",
    "We compare **three systems** side-by-side on the isolated test set:\n",
    "| System | Description |\n",
    "|--------|-------------|\n",
    "| AI Only | Raw XLM-RoBERTa threshold |\n",
    "| Rule Only | Hand-coded heuristics |\n",
    "| **Fusion** | **AI 70% + Rules 30%** |"
]))

new_cells.append(code_cell([
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.metrics import confusion_matrix, precision_recall_fscore_support\n",
    "\n",
    "# ── Constants ──────────────────────────────────────────────────────────────\n",
    "AI_WEIGHT   = 0.70\n",
    "RULE_WEIGHT = 0.30\n",
    "THRESHOLD   = 0.50\n",
    "\n",
    "# ── AI probabilities on test set (reuse classify_message) ──────────────────\n",
    "def get_ai_probs_batch(texts, batch_size=64):\n",
    "    \"\"\"Return np.array of SCAM probabilities for a list of texts.\"\"\"\n",
    "    all_probs = []\n",
    "    model.eval()\n",
    "    for i in range(0, len(texts), batch_size):\n",
    "        batch = texts[i:i+batch_size]\n",
    "        enc = tokenizer(batch, truncation=True, padding=True,\n",
    "                        max_length=64, return_tensors='pt')\n",
    "        enc = {k: v.to(model.device) for k, v in enc.items()}\n",
    "        with torch.no_grad():\n",
    "            logits = model(**enc).logits\n",
    "        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()\n",
    "        all_probs.extend(probs.tolist())\n",
    "    return np.array(all_probs)\n",
    "\n",
    "print('Computing AI probabilities on test set...')\n",
    "ai_probs    = get_ai_probs_batch(test_texts)\n",
    "ai_preds    = (ai_probs >= THRESHOLD).astype(int)\n",
    "\n",
    "print('Computing rule scores on test set...')\n",
    "rule_scores = np.array([rule_score(t) for t in test_texts])\n",
    "rule_preds  = (rule_scores >= THRESHOLD).astype(int)\n",
    "\n",
    "# ── Fusion ─────────────────────────────────────────────────────────────────\n",
    "fusion_scores = AI_WEIGHT * ai_probs + RULE_WEIGHT * rule_scores\n",
    "fusion_preds  = (fusion_scores >= THRESHOLD).astype(int)\n",
    "\n",
    "print('Done.\\n')\n",
]))

new_cells.append(md_cell([
    "### 📊 Three-Way Comparison Results"
]))

new_cells.append(code_cell([
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.gridspec as gridspec\n",
    "\n",
    "DIVIDER = '=' * 55\n",
    "\n",
    "def print_report(name, preds):\n",
    "    acc = accuracy_score(test_labels, preds)\n",
    "    p, r, f1, _ = precision_recall_fscore_support(\n",
    "        test_labels, preds, average='binary')\n",
    "    print(f'\\n{DIVIDER}')\n",
    "    print(f'  {name}')\n",
    "    print(DIVIDER)\n",
    "    print(f'  Accuracy  : {acc*100:.2f}%')\n",
    "    print(f'  Precision : {p*100:.2f}%')\n",
    "    print(f'  Recall    : {r*100:.2f}%')\n",
    "    print(f'  F1-Score  : {f1*100:.2f}%')\n",
    "    print(classification_report(\n",
    "        test_labels, preds, target_names=['SAFE','SCAM'], digits=4))\n",
    "\n",
    "print_report('SYSTEM 1 — AI Only  (XLM-RoBERTa)', ai_preds)\n",
    "print_report('SYSTEM 2 — Rule Engine Only',         rule_preds)\n",
    "print_report('SYSTEM 3 — FUSION  (AI 70% + Rules 30%)', fusion_preds)\n",
]))

new_cells.append(md_cell([
    "### 🖼️ Confusion Matrix Comparison (All 3 Systems)"
]))

new_cells.append(code_cell([
    "systems = [\n",
    "    ('AI Only',           ai_preds),\n",
    "    ('Rule Engine Only',  rule_preds),\n",
    "    ('Fusion (70/30)',    fusion_preds),\n",
    "]\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "fig.suptitle('Confusion Matrix Comparison — AI vs Rules vs Fusion',\n",
    "             fontsize=14, fontweight='bold', y=1.02)\n",
    "\n",
    "for ax, (name, preds) in zip(axes, systems):\n",
    "    cm = confusion_matrix(test_labels, preds)\n",
    "    acc = accuracy_score(test_labels, preds)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
    "                xticklabels=['SAFE','SCAM'],\n",
    "                yticklabels=['SAFE','SCAM'],\n",
    "                linewidths=0.5, linecolor='grey')\n",
    "    ax.set_title(f'{name}\\nAcc: {acc*100:.2f}%', fontweight='bold')\n",
    "    ax.set_xlabel('Predicted')\n",
    "    ax.set_ylabel('True')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

new_cells.append(md_cell([
    "### 🧪 Live Fusion Inference Demo"
]))

new_cells.append(code_cell([
    "SAMPLE_MSGS = [\n",
    "    # Expected SCAM\n",
    "    'TNG eWallet kena suspend, login untuk aktifkan tng-update.com',\n",
    "    'You win RM4500 cash, claim now sebelum expired bit.ly/verify',\n",
    "    'OTP needed to verify transaction, send cepat',\n",
    "    'Loan approved instant RM3500, send IC & bank details',\n",
    "    'Investment crypto profit 200% guaranteed, join now sh0pee-help.net',\n",
    "    # Expected SAFE\n",
    "    'Jom makan dekat cafe kampus petang ni bro',\n",
    "    'Assignment dah siap belum? Aku lambat sikit traffic jam',\n",
    "    'Kita jumpa esok kelas 10am, jangan lupa bawa notes',\n",
    "    'Good morning bro, bangun cepat ada lecture pagi ni',\n",
    "    'Aku tengah buat project AI ni, susah gila bro',\n",
    "]\n",
    "\n",
    "print(f'{\\'=\\'*60}')\n",
    "print('  LIVE FUSION DEMO — Sample Message Inference')\n",
    "print(f'{\\'=\\'*60}')\n",
    "\n",
    "ai_batch  = get_ai_probs_batch(SAMPLE_MSGS)\n",
    "for i, msg in enumerate(SAMPLE_MSGS):\n",
    "    ai_p   = float(ai_batch[i])\n",
    "    r_p    = rule_score(msg)\n",
    "    final  = AI_WEIGHT * ai_p + RULE_WEIGHT * r_p\n",
    "    label  = '🔴 SCAM' if final >= THRESHOLD else '🟢 SAFE'\n",
    "    print(f'\\n  Msg    : {msg[:65]}')\n",
    "    print(f'  AI     : {ai_p*100:5.1f}%  |  '\n",
    "          f'Rule : {r_p*100:5.1f}%  |  '\n",
    "          f'Fusion: {final*100:5.1f}%  →  {label}')\n",
]))

# Append all new cells to notebook
nb["cells"].extend(new_cells)

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(new_cells)} new cells to {NOTEBOOK_PATH}")
