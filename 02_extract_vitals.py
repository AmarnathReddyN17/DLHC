import pandas as pd
import numpy as np
import os

# --- CONFIG ---
if os.path.exists("DATASET/CHARTEVENTS.csv"): DATA_PATH = "DATASET/"
elif os.path.exists("DLHC/DATASET/CHARTEVENTS.csv"): DATA_PATH = "DLHC/DATASET/"
else: DATA_PATH = "DATASET/"

print(f"🚀 Starting Advanced Vitals Mining from {DATA_PATH}")

# 1. LOAD COHORT (Target Admissions)
print("⏳ Loading cohort.csv...")
try:
    cohort = pd.read_csv('cohort.csv')
    target_hadms = set(cohort['HADM_ID'].unique())
    print(f"   Targeting {len(target_hadms)} admissions.")
except:
    print("❌ Error: cohort.csv not found.")
    exit()

# 2. DEFINE VITALS & ITEMIDs
# MIMIC-III uses different codes for CareVue and Metavision systems
# We map them all to common labels
vitals_map = {
    # Heart Rate
    211: 'HR', 220045: 'HR',
    # Systolic BP
    51: 'SYS_BP', 442: 'SYS_BP', 455: 'SYS_BP', 6701: 'SYS_BP', 220179: 'SYS_BP', 220050: 'SYS_BP',
    # Diastolic BP
    8368: 'DIA_BP', 8440: 'DIA_BP', 8441: 'DIA_BP', 8555: 'DIA_BP', 220180: 'DIA_BP', 220051: 'DIA_BP',
    # Mean BP
    456: 'MEAN_BP', 52: 'MEAN_BP', 6702: 'MEAN_BP', 443: 'MEAN_BP', 220052: 'MEAN_BP', 220181: 'MEAN_BP', 225312: 'MEAN_BP',
    # Respiratory Rate
    615: 'RESP_RATE', 618: 'RESP_RATE', 220210: 'RESP_RATE', 224690: 'RESP_RATE',
    # SpO2
    646: 'SPO2', 220277: 'SPO2',
    # Temperature (F) - Will convert
    223761: 'TEMP_F', 678: 'TEMP_F',
    # Temperature (C)
    223762: 'TEMP_C', 676: 'TEMP_C'
}

all_itemids = set(vitals_map.keys())

# 3. PROCESS CHARTEVENTS (Chunked)
print("⏳ Scanning CHARTEVENTS.csv (This is 33GB, please be patient)...")

chunk_size = 5000000 # 5 Million rows per chunk
use_cols = ['HADM_ID', 'ITEMID', 'VALUENUM']

vitals_data = []

try:
    chunk_iter = pd.read_csv(os.path.join(DATA_PATH, 'CHARTEVENTS.csv'), usecols=use_cols, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iter):
        # Filter 1: Relevant Admissions
        chunk = chunk[chunk['HADM_ID'].isin(target_hadms)]

        # Filter 2: Relevant Items
        chunk = chunk[chunk['ITEMID'].isin(all_itemids)]

        if not chunk.empty:
            chunk = chunk.dropna(subset=['VALUENUM'])
            chunk['LABEL'] = chunk['ITEMID'].map(vitals_map)

            # Normalize Temperature (F -> C)
            f_mask = chunk['LABEL'] == 'TEMP_F'
            if f_mask.any():
                chunk.loc[f_mask, 'VALUENUM'] = (chunk.loc[f_mask, 'VALUENUM'] - 32) * 5/9
                chunk.loc[f_mask, 'LABEL'] = 'TEMP_C'

            # Aggregate per chunk to save memory
            agg = chunk.groupby(['HADM_ID', 'LABEL'])['VALUENUM'].agg(['mean', 'min', 'max']).reset_index()
            vitals_data.append(agg)

        print(f"   Processed chunk {i+1}...")

except Exception as e:
    print(f"⚠️ Warning: {e}")

# 4. MERGE & AGGREGATE
print("🧩 Aggregating Vitals...")
if vitals_data:
    vitals_full = pd.concat(vitals_data)
    # Re-aggregate (since same ID might appear in multiple chunks)
    vitals_final = vitals_full.groupby(['HADM_ID', 'LABEL']).agg(
        {'mean': 'mean', 'min': 'min', 'max': 'max'}
    ).reset_index()

    # Pivot to Wide Format
    vitals_pivot = vitals_final.pivot_table(index='HADM_ID', columns='LABEL', values=['mean', 'min', 'max'])
    # Flatten columns (e.g., 'mean', 'HR' -> 'HR_mean')
    vitals_pivot.columns = [f'{col[1]}_{col[0]}' for col in vitals_pivot.columns]
    vitals_pivot = vitals_pivot.reset_index()

    print(f"   Extracted vitals for {len(vitals_pivot)} patients.")

    # Merge with Cohort
    advanced_cohort = pd.merge(cohort, vitals_pivot, on='HADM_ID', how='left')

    # Fill NaNs (Vital signs shouldn't be 0, use median)
    advanced_cohort = advanced_cohort.fillna(advanced_cohort.median(numeric_only=True))
    advanced_cohort = advanced_cohort.fillna(0) # Fallback

    advanced_cohort.to_csv('advanced_cohort.csv', index=False)
    print(f"✅ SUCCESS! Saved 'advanced_cohort.csv' with shape {advanced_cohort.shape}")
else:
    print("❌ No vitals found.")
