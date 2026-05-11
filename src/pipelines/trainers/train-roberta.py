import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import transformers
transformers.AdamW = torch.optim.AdamW

from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== SETTINGS/CONFIG ==========
DATA_DIR = "/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/data"
SAVE_DIR = "/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
MAX_LEN = 256
LR = 2e-5
EPOCHS = 0
FP16 = True  # Mixed precision training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ========== DATASET CLASS ==========
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item

# ========== LOAD & PREPARE DATA ==========
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# we only use the single binary column "toxic"
df["label"] = df["toxic"].fillna(0).astype(int)

train_df, val_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df["label"])
concept_df, val_df = train_test_split(val_df, test_size=0.5, random_state=42, stratify=val_df["label"])

def print_label_distribution(name, df):
    counts = df["label"].value_counts().sort_index()
    total = len(df)

    print(f"\n{name}")
    for label, count in counts.items():
        label_name = "non-toxic" if label == 0 else "toxic"
        print(f"  {label_name:10s}: {count:6d} ({count/total:.2%})")
    print(f"  TOTAL      : {total:6d}")

print_label_distribution("TRAIN", train_df)
print_label_distribution("CONCEPT", concept_df)
print_label_distribution("VAL", val_df)


tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

train_dataset = ToxicDataset(train_df["comment_text"].tolist(),
                             train_df["label"].tolist(),
                             tokenizer)

val_dataset = ToxicDataset(val_df["comment_text"].tolist(),
                           val_df["label"].tolist(),
                           tokenizer)


                     

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ========== MODEL ==========
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

scaler = torch.amp.GradScaler(enabled=FP16)

# ========== TRAIN + EVAL ==========
def train_epoch():
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc="Train")
    for batch in loop:
        optimizer.zero_grad()
        with torch.amp.autocast(enabled=FP16):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels
            )
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1))

    return total_loss / len(train_loader)

from sklearn.metrics import accuracy_score, confusion_matrix

def eval_model(return_preds=False):
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn
            )
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            preds.extend(pred.cpu().tolist())
            truths.extend(labels.cpu().tolist())

    acc = accuracy_score(truths, preds)

    if return_preds:
        return acc, np.array(truths), np.array(preds)
    else:
        return acc


best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
    train_loss = train_epoch()
    val_acc = eval_model()

    print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # SAVE BEST CHECKPOINT
    ckpt_path = os.path.join(SAVE_DIR, f"best_model.pth")
    if val_acc > best_acc:
        best_acc = val_acc
        print(f"Saving best model (acc={best_acc:.4f})")
        torch.save(model.state_dict(), ckpt_path)

model.load_state_dict(torch.load("/netscratch/aslam/TCAV/fast-tcav/nlp/wiki/checkpoints/best_model.pth"))
# model.load_state_dict(torch.load(ckpt_path))
model.to(DEVICE)
print("\nEvaluating best model on validation set...")
val_acc, y_true, y_pred = eval_model(return_preds=True)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix (rows = true, cols = pred):")
print("           pred_non_toxic   pred_toxic")
print(f"true_non_toxic     {cm[0,0]:6d}         {cm[0,1]:6d}")
print(f"true_toxic         {cm[1,0]:6d}         {cm[1,1]:6d}")

tn, fp, fn, tp = cm.ravel()
print("\nDerived metrics:")
print(f"  Precision (toxic): {tp / (tp + fp + 1e-8):.4f}")
print(f"  Recall    (toxic): {tp / (tp + fn + 1e-8):.4f}")
print(f"  F1        (toxic): {2*tp / (2*tp + fp + fn + 1e-8):.4f}")