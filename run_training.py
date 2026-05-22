import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Set FORCE_RETRAIN = True  →  always load from HuggingFace (fresh base model)
# Set FORCE_RETRAIN = False →  resume from saved model if it exists (default)
FORCE_RETRAIN    = False
BASE_MODEL_NAME  = "xlm-roberta-base"
SAVED_MODEL_DIR  = "./xlm-roberta-scam-model"
OUTPUT_DIR       = "./results"
NUM_EPOCHS       = 3        # ← increase epochs for a proper retrain
BATCH_SIZE       = 64
MAX_SEQ_LEN      = 64
SEED             = 42
# ─────────────────────────────────────────────────────────────────────────────

# 1. Reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# 2. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 3. Load Dataset
dataset_path = "malaysia_scam_dataset_3000.csv"
df = pd.read_csv(dataset_path)
print(f"Dataset successfully loaded! Total rows: {len(df)}")

# 4. Encoding & Split
label_mapping = {"SAFE": 0, "SCAM": 1}
df['label_int'] = df['label'].map(label_mapping)

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df['label_int']
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df['label_int']
)
print(f"Splits - Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 5. Decide which checkpoint to load from
saved_model_exists = (
    os.path.isdir(SAVED_MODEL_DIR) and
    os.path.isfile(os.path.join(SAVED_MODEL_DIR, "config.json"))
)

if FORCE_RETRAIN or not saved_model_exists:
    load_from = BASE_MODEL_NAME
    if FORCE_RETRAIN:
        print(f"[FORCE_RETRAIN=True] Loading fresh base model from HuggingFace: {BASE_MODEL_NAME}")
    else:
        print(f"No saved model found. Downloading base model from HuggingFace: {BASE_MODEL_NAME}")
else:
    load_from = SAVED_MODEL_DIR
    print(f"Saved model found → resuming fine-tuning from: {SAVED_MODEL_DIR}")

# 6. Tokenization
print(f"Loading tokenizer from: {load_from}")
tokenizer = AutoTokenizer.from_pretrained(load_from)

train_texts  = train_df['text'].tolist()
train_labels = train_df['label_int'].tolist()
val_texts    = val_df['text'].tolist()
val_labels   = val_df['label_int'].tolist()
test_texts   = test_df['text'].tolist()
test_labels  = test_df['label_int'].tolist()

print("Tokenizing texts...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN)
val_encodings   = tokenizer(val_texts,   truncation=True, padding=True, max_length=MAX_SEQ_LEN)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_SEQ_LEN)

# 7. PyTorch Dataset
class ScamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ScamDataset(train_encodings, train_labels)
val_dataset   = ScamDataset(val_encodings,   val_labels)
test_dataset  = ScamDataset(test_encodings,  test_labels)

# 8. Load Model
print(f"Loading model from: {load_from}")
model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy':  acc,
        'f1':        f1,
        'precision': precision,
        'recall':    recall
    }

# 9. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 10. Train
print(f"Starting training ({NUM_EPOCHS} epoch(s), batch size {BATCH_SIZE}) ...")
trainer.train()
print("Training completed!")

# 11. Evaluate on test set
print("Evaluating on isolated test set...")
test_results = trainer.predict(test_dataset)
print(f"Test Metrics: {test_results.metrics}")

test_preds = np.argmax(test_results.predictions, axis=-1)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(test_labels, test_preds, target_names=["SAFE", "SCAM"]))

cm = confusion_matrix(test_labels, test_preds)
print("\n=== CONFUSION MATRIX ===")
print(cm)

# 12. Save (always overwrite so next run picks up the latest weights)
print(f"Saving fine-tuned model to '{SAVED_MODEL_DIR}' ...")
trainer.model.save_pretrained(SAVED_MODEL_DIR)
tokenizer.save_pretrained(SAVED_MODEL_DIR)
print("Model and tokenizer saved successfully!")
print(f"  → Run fusion_engine.py next to evaluate the Fusion system.")
