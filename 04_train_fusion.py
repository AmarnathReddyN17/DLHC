import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

# --- CONFIG ---
BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 6 # Sufficient for convergence
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Final Run on {device}")

# --- 1. DATA ---
print("⏳ Loading Advanced Data...")
try:
    cohort = pd.read_csv('advanced_cohort.csv')
    notes = pd.read_csv('clinical_notes.csv')
    df = pd.merge(cohort, notes, on='HADM_ID', how='inner')
    df = df.dropna(subset=['TEXT'])
    df = df[df['TEXT'].str.strip() != '']
except:
    print("❌ Error loading data.")
    exit()

# Features
drop_cols = ['SUBJECT_ID', 'HADM_ID', 'MORTALITY', 'TEXT', 'label']
feature_cols = [c for c in df.columns if c not in drop_cols]
X_tab = df[feature_cols].values
X_text = df['TEXT'].values
y = df['MORTALITY'].values

# Split (80/20)
X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_tab, X_text, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(X_tab_train)
X_tab_test = scaler.transform(X_tab_test)

# --- 2. DATASET ---
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

class FusionDataset(Dataset):
    def __init__(self, texts, tabular, labels):
        self.texts = texts
        self.tabular = tabular
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
            'tab': torch.tensor(self.tabular[i], dtype=torch.float),
            'lbl': torch.tensor(self.labels[i], dtype=torch.float)
        }

train_ds = FusionDataset(X_text_train, X_tab_train, y_train)
test_ds = FusionDataset(X_text_test, X_tab_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, num_workers=2)

# --- 3. MODEL ---
class FusionNetwork(nn.Module):
    def __init__(self, n_tab_features):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL)
        self.bert_drop = nn.Dropout(0.3)
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(768 + 64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, ids, mask, tab):
        bert_out = self.bert_drop(self.bert(ids, mask)[1])
        tab_out = self.tab_mlp(tab)
        return self.fusion_head(torch.cat((bert_out, tab_out), dim=1))

model = FusionNetwork(n_tab_features=X_tab.shape[1]).to(device)

# Load Weights (Optimized)
try:
    pt_dict = torch.load("best_bert_finetuned.pth")
    model_dict = model.state_dict()
    pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and 'out' not in k}
    model_dict.update(pt_dict)
    model.load_state_dict(model_dict)
    print("✅ Loaded fine-tuned BERT.")
except:
    print("⚠️ Using Base BERT.")

for p in model.bert.parameters(): p.requires_grad = False
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.BCELoss()

# --- 4. TRAIN ---
print("🔥 Training Final Model...")
for epoch in range(EPOCHS):
    model.train()
    for i, batch in enumerate(train_loader):
        ids, mask = batch['ids'].to(device), batch['mask'].to(device)
        tab, lbl = batch['tab'].to(device), batch['lbl'].to(device).unsqueeze(1)
        
        opt.zero_grad()
        loss = crit(model(ids, mask, tab), lbl)
        loss.backward()
        opt.step()
        
    print(f"   Epoch {epoch+1}/{EPOCHS} Complete")

# --- 5. SAVE RESULTS ---
print("⏳ Generating Predictions...")
model.eval()
preds, targets = [], []
with torch.no_grad():
    for batch in test_loader:
        ids, mask = batch['ids'].to(device), batch['mask'].to(device)
        tab = batch['tab'].to(device)
        preds.extend(model(ids, mask, tab).cpu().numpy().flatten())
        targets.extend(batch['lbl'].numpy())

auc = roc_auc_score(targets, preds)
print(f"\n🚀 FINAL ADVANCED AUROC: {auc:.4f}")

# Save for Plotting
res_df = pd.DataFrame({'True': targets, 'Pred': preds})
res_df.to_csv('advanced_results.csv', index=False)
print("💾 Saved 'advanced_results.csv'")
