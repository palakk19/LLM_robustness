import os
import re
import pandas as pd

from src.config import (
    TOTAL_EXPERIMENT_SAMPLES,
    RANDOM_SEED,
    NOTES_PATH,
    DIAGNOSES_PATH,
    DICT_PATH,
)
from src.utils import debug_print

# ============================================================
# SECTION 2: DATA LOADER
# Loads from stratified_dataset.csv (built by build_stratified_dataset.py)
# Falls back to MIMICLoader if CSV not found
# ============================================================
def load_data():
    """Load stratified dataset CSV if available, otherwise use MIMICLoader."""
    csv_path = "data/stratified_dataset.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # ---  Only sample if a specific number is provided(Useful for debugging and to run on smaller sample size) ---
        if TOTAL_EXPERIMENT_SAMPLES is not None and len(df) > TOTAL_EXPERIMENT_SAMPLES:
            df = df.sample(n=TOTAL_EXPERIMENT_SAMPLES, random_state=RANDOM_SEED)
        # ---------------------------------------------------------

        debug_print("DATA", f"Loaded stratified dataset: {len(df)} notes "
                    f"| {df['chapter'].nunique()} chapters")
        counts = df.groupby('chapter').size()
        for ch, n in sorted(counts.items()):
            debug_print("DATA", f"  Chapter {ch}: {n} notes")
        return df

    else:
        debug_print("DATA", "stratified_dataset.csv not found, using MIMICLoader")

        # If set to None, tell the fallback loader to grab a massive number of samples
        # so it effectively processes everything it can find.
        fallback_limit = TOTAL_EXPERIMENT_SAMPLES if TOTAL_EXPERIMENT_SAMPLES is not None else 1_000_000
        return MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, n_samples=fallback_limit)

class MIMICLoader:
    _LEAKING_HEADERS_RE = re.compile(
        r"\n\s*(?:"
        r"Assessment(?:\s+and\s+Plan)?|A\s*/\s*P|A\s*&\s*P"
        r"|Discharge\s+(?:Diagnosis|Diagnoses|Condition|Medications?|Instructions?|Summary)"
        r"|Primary\s+Diagnosis|Secondary\s+Diagnos"
        r"|Discharge\s+Disposition|Follow[\s\-]?up"
        r")\s*:?",
        re.I,
    )

    @staticmethod
    def clean_noise(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @classmethod
    def strip_leaking_sections(cls, text):
        parts = cls._LEAKING_HEADERS_RE.split(text, maxsplit=1)
        return parts[0].strip()

    @staticmethod
    def extract_blind_clinical_info(text):
        _NEXT = (
            r"History of Present Illness|HPI|Past Medical History|PMH|"
            r"Social History|Family History|Review of Systems|"
            r"Physical Exam(?:ination)?|Vital Signs|Labs?|Laboratory|"
            r"Imaging|Studies|Medications? on Admission|Brief Hospital Course"
        )
        parts = []
        for label, pattern, minlen, maxlen in [
            ("Chief Complaint",
             rf"Chief Complaint\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 3, 300),
            ("HPI",
             rf"(?:History of Present Illness|HPI)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 100, 1200),
            ("Hospital Course",
             rf"Brief Hospital Course\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 100, 1200),
            ("PMH",
             rf"(?:Past Medical History|PMH)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 50, 600),
            ("Physical Exam",
             rf"Physical Exam(?:ination)?\s*:?\s*(.*?)(?:{_NEXT}|\Z)", 50, 600),
        ]:
            m = re.search(pattern, text, re.S | re.I)
            if m:
                content = m.group(1).strip()
                if len(content) >= minlen:
                    parts.append(f"{label}: {content[:maxlen]}")
        return "\n\n".join(parts)[:3000] if parts else text[-2000:].strip()

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading MIMIC (ICD-10 only)...")
        diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
        diags = diags[diags['icd_version'] == 10].copy()
        hadm_ids = set(diags['hadm_id'].unique())
        chunks, found = [], 0
        for chunk in pd.read_csv(notes_path, compression='gzip',
                                  usecols=['subject_id', 'hadm_id', 'text'],
                                  chunksize=5000):
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids)].copy()
            if len(chunk):
                chunks.append(chunk)
                found += len(chunk)
            if found >= n_samples * 50:
                break
        notes = pd.concat(chunks, ignore_index=True)
        if os.path.exists(dict_path):
            d = pd.read_csv(dict_path, compression='gzip')
            d = d[d['icd_version'] == 10]
            diags = pd.merge(diags, d[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = 'Unknown'
        primary = diags[diags['seq_num'] == 1]
        df = pd.merge(notes, primary[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')
        df['text'] = df['text'].apply(cls.clean_noise)
        df['text'] = df['text'].apply(cls.strip_leaking_sections)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)
        df['chapter'] = df['icd_code'].apply(lambda c: c[0] if c and not pd.isna(c) else 'X')
        result = df.sample(n=min(len(df), n_samples), random_state=42)
        debug_print("DATA", f"Sampled {len(result)} notes")
        return result