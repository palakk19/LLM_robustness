"""
STRATIFIED DATASET BUILDER — FLEXIBLE TARGET SIZE
=================================================

Just change TOTAL_TARGET:

  TOTAL_TARGET = 1200   -> ~1000–1200 final
  TOTAL_TARGET = 1400   -> safer for ~1200
  TOTAL_TARGET = 3000   -> large dataset

Run ONCE. Use same CSV for all experiments.
DO NOT change RANDOM_SEED after building.
"""

import pandas as pd
import numpy as np
import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ── Configuration ─────────────────────────────────────────────────────────────
NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

OUTPUT_CSV    = "palak_final/stratified_dataset.csv"
OUTPUT_REPORT = "palak_final/dataset_summary.html"

RANDOM_SEED   = 42

# 🔥 CHANGE THIS ONLY
TOTAL_TARGET = 1200

# Relative weights (same logic as before)
BASE_WEIGHTS = {
    'A': 1.0, 'B': 0.8, 'C': 1.0, 'D': 0.8, 'E': 1.0,
    'F': 1.0, 'G': 1.0, 'H': 0.8, 'I': 1.0, 'J': 1.0,
    'K': 1.0, 'L': 0.8, 'M': 1.0, 'N': 1.0, 'O': 1.0,
    'P': 0.8, 'Q': 0.8, 'R': 1.0, 'S': 1.0, 'T': 0.8,
    'Z': 1.0,
}

CHAPTER_NAMES = {
    'A': 'Infectious & Parasitic',
    'B': 'Infectious (B codes)',
    'C': 'Neoplasms (Cancer)',
    'D': 'Blood & Immune',
    'E': 'Endocrine & Metabolic',
    'F': 'Mental & Behavioural',
    'G': 'Nervous System',
    'H': 'Eye & Ear',
    'I': 'Circulatory System',
    'J': 'Respiratory System',
    'K': 'Digestive System',
    'L': 'Skin & Subcutaneous',
    'M': 'Musculoskeletal',
    'N': 'Genitourinary',
    'O': 'Pregnancy & Childbirth',
    'P': 'Perinatal',
    'Q': 'Congenital Malformations',
    'R': 'Symptoms & Signs',
    'S': 'Injuries & Trauma',
    'T': 'Poisoning & External',
    'Z': 'Health Status & Contact',
}

MIN_TEXT_LEN = 200

# ── Convert weights → targets ────────────────────────────────────────────────
total_weight = sum(BASE_WEIGHTS.values())

TARGET_PER_CHAPTER = {
    ch: max(1, int(round((BASE_WEIGHTS[ch] / total_weight) * TOTAL_TARGET)))
    for ch in BASE_WEIGHTS
}

# ── Text Processing ──────────────────────────────────────────────────────────
def clean_noise(text):
    text = str(text)
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"_{2,}", " ", text)
    text = text.replace("=", " ").replace("*", " ").replace("#", " ")
    return " ".join(text.split())

def strip_leaking_sections(text):
    parts = re.split(
        r"\n\s*(?:Assessment(?:\s+and\s+Plan)?|A\s*/\s*P|A\s*&\s*P"
        r"|Discharge\s+(?:Diagnosis|Diagnoses|Condition|Medications?"
        r"|Instructions?|Summary)|Primary\s+Diagnosis|Secondary\s+Diagnos"
        r"|Discharge\s+Disposition|Follow[\s\-]?up)\s*:? ",
        text, maxsplit=1, flags=re.I)
    return parts[0].strip()

def extract_clinical_info(text):
    _NEXT = (
        r"History of Present Illness|HPI|Past Medical History|PMH|"
        r"Social History|Family History|Review of Systems|"
        r"Physical Exam(?:ination)?|Vital Signs|Labs?|Laboratory|"
        r"Imaging|Studies|Medications? on Admission|Brief Hospital Course"
    )
    parts = []
    for label, pattern, minlen, maxlen in [
        ("Chief Complaint", rf"Chief Complaint\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 3, 300),
        ("HPI", rf"(?:History of Present Illness|HPI)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 100, 1200),
        ("Hospital Course", rf"Brief Hospital Course\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 100, 1200),
        ("PMH", rf"(?:Past Medical History|PMH)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 50, 600),
        ("Physical Exam", rf"Physical Exam(?:ination)?\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 50, 600),
    ]:
        m = re.search(pattern, text, re.S | re.I)
        if m:
            content = m.group(1).strip()
            if len(content) >= minlen:
                parts.append(f"{label}: {content[:maxlen]}")
    return "\n\n".join(parts)[:3000] if parts else text[-2000:].strip()

def process_note(text):
    return extract_clinical_info(strip_leaking_sections(clean_noise(text)))

def get_chapter(icd_code):
    if not icd_code or pd.isna(icd_code):
        return None
    code = re.sub(r'[^A-Z0-9]', '', str(icd_code).strip().upper())
    return code[0] if code and code[0] in TARGET_PER_CHAPTER else None

# ── Main ─────────────────────────────────────────────────────────────────────
def build_dataset():

    theoretical_max = sum(TARGET_PER_CHAPTER.values())

    print("=" * 65)
    print("STRATIFIED DATASET BUILDER")
    print(f"Requested total: {TOTAL_TARGET}")
    print(f"Theoretical max: {theoretical_max}")
    print("=" * 65)

    print("\n[1/6] Loading diagnoses...")
    diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip', low_memory=False)
    diags = diags[diags['icd_version'] == 10]
    primary = diags[diags['seq_num'] == 1].copy()

    primary['chapter'] = primary['icd_code'].apply(get_chapter)
    primary = primary[primary['chapter'].notna()]

    if os.path.exists(DICT_PATH):
        d = pd.read_csv(DICT_PATH, compression='gzip')
        d = d[d['icd_version'] == 10]
        primary = pd.merge(primary, d[['icd_code','icd_version','long_title']],
                           on=['icd_code','icd_version'], how='left')

    print("\n[2/6] Sampling candidates...")
    chapter_candidate_dfs = {}

    for ch in TARGET_PER_CHAPTER:
        ch_df = primary[primary['chapter'] == ch]
        target = TARGET_PER_CHAPTER[ch]
        n_cand = min(len(ch_df), target * 8)

        if n_cand == 0:
            chapter_candidate_dfs[ch] = pd.DataFrame()
            continue

        chapter_candidate_dfs[ch] = ch_df.sample(n=n_cand, random_state=RANDOM_SEED)

    all_cands = pd.concat(chapter_candidate_dfs.values()).drop_duplicates('hadm_id')
    needed_ids = set(all_cands['hadm_id'])

    print("\n[3/6] Scanning notes...")
    note_chunks = []

    for chunk in pd.read_csv(NOTES_PATH, compression='gzip',
                             usecols=['subject_id','hadm_id','text'],
                             chunksize=5000):
        matching = chunk[chunk['hadm_id'].isin(needed_ids)]
        if len(matching):
            note_chunks.append(matching)

    notes = pd.concat(note_chunks).drop_duplicates('hadm_id')

    print("\n[4/6] Processing text...")
    df = pd.merge(notes, all_cands, on='hadm_id')
    df['processed_text'] = df['text'].apply(process_note)
    df['text_length'] = df['processed_text'].str.len()
    df = df[df['text_length'] >= MIN_TEXT_LEN]

    print("\n[5/6] Stratified sampling...")
    final_parts = []

    for ch in TARGET_PER_CHAPTER:
        ch_df = df[df['chapter'] == ch]
        n_take = min(len(ch_df), TARGET_PER_CHAPTER[ch])
        if n_take > 0:
            final_parts.append(ch_df.sample(n=n_take, random_state=RANDOM_SEED))

    final_df = pd.concat(final_parts)

    print("\n[6/6] Saving...")
    final_df = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    final_df['dataset_index'] = range(len(final_df))
    final_df['chapter_name'] = final_df['chapter'].map(CHAPTER_NAMES)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 65)
    print(f"FINAL DATASET SIZE: {len(final_df)}")
    print(f"Requested: {TOTAL_TARGET}")
    print("=" * 65)

    return final_df


if __name__ == "__main__":
    build_dataset()