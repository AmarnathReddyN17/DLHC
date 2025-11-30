import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast, GradScaler

# --- CONFIG FOR 10GB VRAM ---
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 128
BATCH_SIZE = 8       
ACCUM_STEPS = 4      
EPOCHS = 3           
LR = 2e-5            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Full Fine-Tuning on {device} (10GB Slice)")

# --- 1. LOAD DATA ---
print("⏳ Loading Data...")
try:
    cohort = pd.read_csv('cohort.csv')
    notes = pd.read_csv('clinical_notes.csv')
    df = pd.merge(cohort[['HADM_ID', 'MORTALITY']], notes, on='HADM_ID', how='inner')
    df = df.dropna(subset=['TEXT'])
    df = df[df['TEXT'].str.strip() != '']
    print(f"✅ Data Loaded: {len(df)} rows")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- 2. DATASET ---
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

class NotesDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tokenizer.encode_plus(
            str(self.texts[i]), max_length=MAX_LEN, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'lbl': torch.tensor(self.labels[i], dtype=torch.float)
        }

train_ds = NotesDataset(train_df['TEXT'].values, train_df['MORTALITY'].values)
test_ds = NotesDataset(test_df['TEXT'].values, test_df['MORTALITY'].values)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, num_workers=2) 

# --- 3. MODEL (Output Raw Logits) ---
class FullBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.bert.gradient_checkpointing_enable() # Save Memory
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        # REMOVED SIGMOID HERE for stability
        
    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)[1] 
        return self.out(self.drop(out))

model = FullBert().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler() 

# --- 4. TRAIN LOOP ---
print(f"🔥 Starting End-to-End Fine-Tuning...")
print(f"   Batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS}")

best_auc = 0
# Use BCEWithLogitsLoss for FP16 stability
criterion = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        lbl = batch['lbl'].to(device).unsqueeze(1)
        
        with autocast():
            logits = model(ids, mask)
            loss = criterion(logits, lbl)
            loss = loss / ACCUM_STEPS 
            
        scaler.scale(loss).backward()
        
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * ACCUM_STEPS
        
        if step % 200 == 0 and step > 0:
            print(f"   Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {running_loss/200:.4f}")
            running_loss = 0

    # --- EVALUATE ---
    print(f"⏳ Evaluating Epoch {epoch+1}...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            target = batch['lbl'].to(device)
            
            with autocast():
                logits = model(ids, mask)
                # Apply Sigmoid manually here for metrics
                probs = torch.sigmoid(logits)
            
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy())
            
    auc = roc_auc_score(targets, preds)
    print(f"✅ Epoch {epoch+1} AUROC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_bert_finetuned.pth")

print(f"🏆 Best Full-BERT AUROC: {best_auc:.4f}")
