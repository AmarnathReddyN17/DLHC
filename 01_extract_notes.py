import pandas as pd
import numpy as np
import os
import re

# --- CONFIG ---
if os.path.exists("DATASET/NOTEEVENTS.csv"): DATA_PATH = "DATASET/"
elif os.path.exists("DLHC/DATASET/NOTEEVENTS.csv"): DATA_PATH = "DLHC/DATASET/"
else: DATA_PATH = "DATASET/"

print(f"🚀 Extracting Notes using path: {DATA_PATH}")

# 1. LOAD COHORT (To know which patients to keep)
print("⏳ Loading cohort.csv to get Patient IDs...")
try:
    cohort = pd.read_csv('cohort.csv')
    
    # Load Admissions to get ADMITTIME
    # Note: We read only necessary columns to save memory
    adm = pd.read_csv(os.path.join(DATA_PATH, 'ADMISSIONS.csv'), usecols=['HADM_ID', 'ADMITTIME'])
    adm['ADMITTIME'] = pd.to_datetime(adm['ADMITTIME'])
    
    # Merge to get Admit Time for our cohort
    cohort_dates = pd.merge(cohort[['HADM_ID']], adm, on='HADM_ID', how='inner')
    
    # Valid HADM_IDs set for fast filtering
    valid_hadm_ids = set(cohort['HADM_ID'].unique())
    # Lookup for Admit Time
    admit_time_map = cohort_dates.set_index('HADM_ID')['ADMITTIME'].to_dict()
    
    print(f"   Targeting {len(valid_hadm_ids)} admissions.")

except Exception as e:
    print(f"❌ Error loading cohort: {e}")
    exit()

# 2. PROCESS NOTES (Chunked)
print("⏳ Processing NOTEEVENTS.csv (This is large, please wait)...")
# FIX: Use Uppercase column names and remove ROW_ID to prevent errors
note_cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'TEXT']
chunk_size = 1000000
processed_notes = []

try:
    chunk_iter = pd.read_csv(os.path.join(DATA_PATH, 'NOTEEVENTS.csv'), usecols=note_cols, chunksize=chunk_size, low_memory=False)
    
    for i, chunk in enumerate(chunk_iter):
        # Filter 1: Keep only our cohort
        chunk = chunk[chunk['HADM_ID'].isin(valid_hadm_ids)].copy()
        
        # Filter 2: Remove Discharge Summaries (Leakage Risk!)
        chunk = chunk[chunk['CATEGORY'] != 'Discharge summary']
        
        if chunk.empty: continue

        # Filter 3: Time Window (First 48 Hours)
        # Handle missing ChartTime by using ChartDate
        chunk['CHARTTIME'] = chunk['CHARTTIME'].fillna(chunk['CHARTDATE'])
        chunk['NOTE_TIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
        
        # Map Admit Time
        chunk['ADMIT_TIME'] = chunk['HADM_ID'].map(admit_time_map)
        chunk['ADMIT_TIME'] = pd.to_datetime(chunk['ADMIT_TIME'])
        
        # Calculate time difference
        chunk['HOURS_SINCE_ADMIT'] = (chunk['NOTE_TIME'] - chunk['ADMIT_TIME']).dt.total_seconds() / 3600
        
        # KEEP: Notes from -24h (pre-admit data) to +48h
        mask = (chunk['HOURS_SINCE_ADMIT'] >= -24) & (chunk['HOURS_SINCE_ADMIT'] <= 48)
        chunk = chunk[mask]
        
        if not chunk.empty:
            # Simple Cleaning
            def clean_text(text):
                if not isinstance(text, str): return ""
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = re.sub(r'\[\*\*.*?\*\*\]', '', text) # Remove de-id brackets
                return text.strip()

            chunk['TEXT'] = chunk['TEXT'].apply(clean_text)
            
            # Select relevant cols
            processed_notes.append(chunk[['HADM_ID', 'TEXT']])
            
        if (i+1) % 5 == 0: print(f"   Processed chunk {i+1}...")

except Exception as e:
    print(f"⚠️ Error reading notes: {e}")

# 3. AGGREGATE
if processed_notes:
    print("🧩 Aggregating notes per patient...")
    all_notes = pd.concat(processed_notes)
    # Group by admission and join all texts
    final_notes = all_notes.groupby('HADM_ID')['TEXT'].apply(lambda x: " ".join(x)).reset_index()
    
    # Save
    final_notes.to_csv('clinical_notes.csv', index=False)
    print(f"✅ SUCCESS! Saved 'clinical_notes.csv' with {len(final_notes)} rows.")
else:
    print("❌ No relevant notes found.")
