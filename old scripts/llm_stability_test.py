"""
LLM STABILITY / TEST-RETEST RELIABILITY EXPERIMENT
====================================================

Research question:
  How stable is the LLM's ICD-10 prediction on identical inputs?
  This must be established BEFORE any robustness experiment results
  can be interpreted.

Why this matters:
  If the LLM gives different ICD codes on repeated runs of the SAME
  note (without any perturbation), then measured "drift" in the main
  experiment may be noise from the model, not caused by the attack.

What this script does:
  1. Loads N notes from MIMIC (same pipeline as main experiment)
  2. Predicts ICD on each note K times WITHOUT any perturbation
  3. Measures:
     - Consistency rate: % of notes where all K predictions agree
     - Flip rate: % of notes where at least one prediction differs
     - Agreement matrix: how often each pair of runs agrees
  4. Saves results to CSV and generates a stability report

Recommended settings:
  N_NOTES = 50   (enough for statistical confidence)
  N_RUNS  = 5    (5 repeated predictions per note)
  
  Run with temperature=0.0 first, then temperature=0.1
  to show the professor the difference temperature makes.

How to cite in your paper:
  "We established LLM prediction stability by running each of N=50
   notes through [MODEL] K=5 times without perturbation. The model
   showed X% consistency rate at temperature=0.0 and Y% at
   temperature=0.1, confirming that measured drift rates above Z%
   can be attributed to perturbation rather than model stochasticity."
"""

import torch
import pandas as pd
import numpy as np
import re
import json
import requests
import warnings
from tqdm import tqdm
from collections import Counter

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME     = "llama3.1:8b"
API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

N_NOTES        = 100   # number of unique notes to test
N_RUNS         = 5      # how many times to predict each note

# Test both temperatures to show the difference
TEMPERATURES   = [0.0, 0.1]

OUTPUT_CSV     = "palak_final/stability_results.csv"
OUTPUT_REPORT  = "palak_final/stability_report.html"


# ── LLM Connector ─────────────────────────────────────────────────────────────
def query_llm(prompt, temperature=0.0, max_tokens=15):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model"  : MODEL_NAME,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.post(API_URL, headers=headers,
                            data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


def get_icd_prediction(text, temperature=0.0):
    prompt = (
        f"Read the clinical note below and return the single most likely "
        f"ICD-10-CM diagnosis code.\n"
        f"Output format: one uppercase letter followed by 2 digits, then "
        f"optionally a dot and 1-2 more digits.\n"
        f"Valid examples: J18.9  I21.0  K56.9  N39.0  M17.11  A41.9\n"
        f"Do NOT output a disease name, sentence, or any explanation.\n"
        f"Output the ICD-10 code ONLY.\n\n"
        f"Clinical note:\n{text[:2500]}\n\nICD-10 code:"
    )
    resp = query_llm(prompt, temperature=temperature, max_tokens=15)
    if resp:
        # Strip DeepSeek thinking tags if present
        resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.S).strip()
        m = re.search(r'[A-Z]\d{2}\.\d+', resp.upper())
        if m:
            return m.group(0)
        m = re.search(r'[A-Z]\d{2,4}', resp.upper())
        if m:
            return m.group(0)
    return "UNKNOWN"


def normalise_icd(code):
    return re.sub(r'[^A-Z0-9]', '', code.upper())


# ── Data Loader (same as main pipeline) ───────────────────────────────────────
def load_notes(n_samples):
    print(f"Loading {n_samples} notes from MIMIC...")
    import os

    diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip', low_memory=False)
    diags = diags[diags['icd_version'] == 10].copy()
    hadm_ids = set(diags['hadm_id'].unique())

    chunks = []
    notes_found = 0
    for chunk in pd.read_csv(NOTES_PATH, compression='gzip',
                              usecols=['subject_id', 'hadm_id', 'text'],
                              chunksize=5000):
        chunk = chunk[chunk['hadm_id'].isin(hadm_ids)].copy()
        if len(chunk):
            chunks.append(chunk)
            notes_found += len(chunk)
        if notes_found >= n_samples * 20:
            break

    notes = pd.concat(chunks, ignore_index=True)

    if os.path.exists(DICT_PATH):
        d = pd.read_csv(DICT_PATH, compression='gzip')
        d = d[d['icd_version'] == 10]
        diags = pd.merge(diags, d[['icd_code', 'icd_version', 'long_title']],
                         on=['icd_code', 'icd_version'], how='left')

    primary = diags[diags['seq_num'] == 1]
    df = pd.merge(notes,
                  primary[['hadm_id', 'icd_code', 'long_title']],
                  on='hadm_id')

    # Clean and extract (same as main pipeline)
    def clean(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    def extract(text):
        # Remove leaking sections
        text = re.split(
            r"\n\s*(?:Discharge\s+Diagnosis|Assessment(?:\s+and\s+Plan)?|"
            r"Primary\s+Diagnosis|Follow[\s-]?up)\s*:?",
            text, maxsplit=1, flags=re.I)[0]
        # Extract HPI and hospital course
        parts = []
        for label, pattern in [
            ("Chief Complaint",
             r"Chief Complaint\s*:?\s*(.*?)(?:History of Present Illness|HPI|\Z)"),
            ("HPI",
             r"(?:History of Present Illness|HPI)\s*:?\s*(.*?)(?:Past Medical|Social|\Z)"),
            ("Hospital Course",
             r"Brief Hospital Course\s*:?\s*(.*?)(?:Medications|Discharge|\Z)"),
        ]:
            m = re.search(pattern, text, re.S | re.I)
            if m and len(m.group(1).strip()) > 50:
                parts.append(f"{label}: {m.group(1).strip()[:1200]}")
        return "\n\n".join(parts)[:3000] if parts else text[-2000:]

    df['text']    = df['text'].apply(clean)
    df['note']    = df['text'].apply(extract)
    result = df.sample(n=min(len(df), n_samples), random_state=42)
    print(f"Loaded {len(result)} notes")
    return result


# ── Core Stability Experiment ──────────────────────────────────────────────────
def run_stability_experiment(df, temperature):
    """
    For each note, run ICD prediction N_RUNS times at the given temperature.
    Returns a list of result dicts.
    """
    print(f"\n{'='*60}")
    print(f"Running stability experiment | temp={temperature} | {N_RUNS} runs per note")
    print(f"{'='*60}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Stability temp={temperature}"):
        hadm_id  = row['hadm_id']
        note     = row['note']
        gt_icd   = row['icd_code']
        gt_desc  = row.get('long_title', '')

        predictions = []
        for run_idx in range(N_RUNS):
            pred = get_icd_prediction(note, temperature=temperature)
            predictions.append(pred)

        # Normalise all predictions
        norm_preds = [normalise_icd(p) for p in predictions]
        norm_gt    = normalise_icd(gt_icd)

        # Consistency: all predictions agree with each other
        unique_preds   = set(p for p in norm_preds if p != 'UNKNOWN')
        all_same       = len(unique_preds) <= 1
        most_common    = Counter(norm_preds).most_common(1)[0][0]
        agreement_rate = norm_preds.count(most_common) / N_RUNS

        # Prefix consistency (3-char ICD prefix)
        norm_prefixes   = [p[:3] for p in norm_preds if len(p) >= 3]
        unique_prefixes = set(norm_prefixes)
        prefix_all_same = len(unique_prefixes) <= 1

        # How often does most common prediction match ground truth
        gt_match_rate = sum(
            1 for p in norm_preds
            if len(p) >= 3 and len(norm_gt) >= 3 and p[:3] == norm_gt[:3]
        ) / N_RUNS

        print(f"  {hadm_id}: GT={gt_icd} | preds={predictions} | "
              f"consistent={'YES' if all_same else 'NO'} | "
              f"agreement={agreement_rate:.0%}")

        results.append({
            'hadm_id'        : hadm_id,
            'gt_icd'         : gt_icd,
            'gt_desc'        : gt_desc,
            'temperature'    : temperature,
            'predictions'    : json.dumps(predictions),
            'norm_predictions': json.dumps(norm_preds),
            'unique_pred_count': len(unique_preds),
            'most_common_pred' : most_common,
            'agreement_rate'   : round(agreement_rate, 4),
            'all_same'         : all_same,
            'prefix_all_same'  : prefix_all_same,
            'gt_prefix_match_rate': round(gt_match_rate, 4),
            'has_unknown'      : 'UNKNOWN' in norm_preds,
        })

    return results


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_statistics(results, temperature):
    df_r = pd.DataFrame(results)
    df_r = df_r[df_r['temperature'] == temperature]

    n = len(df_r)
    consistent    = df_r['all_same'].sum()
    prefix_consist = df_r['prefix_all_same'].sum()
    avg_agree     = df_r['agreement_rate'].mean()
    avg_gt_match  = df_r['gt_prefix_match_rate'].mean()

    # Noise floor: proportion of runs where prediction differs from most common
    flip_rate = 1 - avg_agree

    # 95% CI on consistency rate
    p = consistent / n
    ci = 1.96 * ((p * (1-p) / n) ** 0.5)

    print(f"\n  Temperature = {temperature}")
    print(f"  Notes tested:               {n}")
    print(f"  Exact code consistent:      {consistent}/{n} ({100*p:.1f}%) "
          f"[95% CI: {100*(p-ci):.1f}% - {100*(p+ci):.1f}%]")
    print(f"  3-char prefix consistent:   {prefix_consist}/{n} ({100*prefix_consist/n:.1f}%)")
    print(f"  Avg agreement rate:         {avg_agree:.3f}")
    print(f"  Noise floor (flip rate):    {flip_rate:.3f} ({100*flip_rate:.1f}%)")
    print(f"  Avg GT prefix match rate:   {avg_gt_match:.3f}")
    print()

    if temperature == 0.0:
        if consistent == n:
            print("  CONCLUSION: Model is fully deterministic at temp=0.0")
            print("  All measured drift in main experiment is caused by perturbation")
        else:
            print(f"  WARNING: {n - consistent} notes were inconsistent even at temp=0.0")
            print(f"  Noise floor = {100*flip_rate:.1f}% -- drift below this is unreliable")
    else:
        print(f"  CONCLUSION: At temp={temperature}, noise floor = {100*flip_rate:.1f}%")
        print(f"  Only drift rates > {100*flip_rate:.1f}% can be attributed to perturbation")

    return {
        'temperature'       : temperature,
        'n'                 : n,
        'consistent_count'  : int(consistent),
        'consistency_rate'  : round(p, 4),
        'ci_lower'          : round(max(0, p-ci), 4),
        'ci_upper'          : round(min(1, p+ci), 4),
        'prefix_consistency': round(prefix_consist/n, 4),
        'avg_agreement'     : round(avg_agree, 4),
        'noise_floor'       : round(flip_rate, 4),
        'avg_gt_match'      : round(avg_gt_match, 4),
    }


# ── HTML Report ────────────────────────────────────────────────────────────────
def generate_stability_report(all_results, stats_by_temp):
    rows = ""
    for r in sorted(all_results, key=lambda x: (x['temperature'], x['hadm_id'])):
        preds     = json.loads(r['predictions'])
        norm_pred = json.loads(r['norm_predictions'])
        consistent = r['all_same']
        bg = '#ccffcc' if consistent else '#ffcccc'

        pred_cells = "".join(
            f'<td style="padding:4px 8px;border:1px solid #ddd;text-align:center;'
            f'background:{"#ccffcc" if p == r["most_common_pred"] else "#fff3cd"};">'
            f'{preds[i]}</td>'
            for i, p in enumerate(norm_pred)
        )

        rows += f"""
        <tr style="background:{bg};">
          <td style="padding:4px 8px;border:1px solid #ddd;">{r['hadm_id']}</td>
          <td style="padding:4px 8px;border:1px solid #ddd;">{r['temperature']}</td>
          <td style="padding:4px 8px;border:1px solid #ddd;">{r['gt_icd']}</td>
          <td style="padding:4px 8px;border:1px solid #ddd;font-size:0.85em;">{r['gt_desc'][:40]}</td>
          {pred_cells}
          <td style="padding:4px 8px;border:1px solid #ddd;text-align:center;">
            {r['agreement_rate']:.0%}</td>
          <td style="padding:4px 8px;border:1px solid #ddd;text-align:center;">
            {"YES" if consistent else "<b style='color:red'>NO</b>"}</td>
        </tr>"""

    run_headers = "".join(
        f'<th style="padding:6px;border:1px solid #ccc;background:#ddd;">Run {i+1}</th>'
        for i in range(N_RUNS)
    )

    stats_rows = ""
    for s in stats_by_temp:
        temp_color = '#ccffcc' if s['temperature'] == 0.0 else '#fff3cd'
        stats_rows += f"""
        <tr style="background:{temp_color};">
          <td style="padding:6px 12px;border:1px solid #ccc;"><b>{s['temperature']}</b></td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;">{s['n']}</td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;">
            <b>{100*s['consistency_rate']:.1f}%</b></td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;">
            [{100*s['ci_lower']:.1f}% - {100*s['ci_upper']:.1f}%]</td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;">
            {100*s['prefix_consistency']:.1f}%</td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;color:red;">
            <b>{100*s['noise_floor']:.1f}%</b></td>
          <td style="padding:6px 12px;border:1px solid #ccc;text-align:center;">
            {100*s['avg_gt_match']:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><title>LLM Stability Report</title>
<meta charset="utf-8">
<style>
  body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;
        padding:20px;background:#fafafa;}}
  h1,h2{{color:#1F4E79;}}
  .summary{{background:#f0f0f0;padding:15px;border-radius:8px;margin-bottom:20px;}}
  .key-finding{{background:#e8f4f8;padding:14px;border-radius:8px;
                margin:20px 0;border-left:4px solid #2E75B6;}}
</style>
</head><body>

<h1>LLM Prediction Stability Report</h1>
<p style="color:#666;">Model: {MODEL_NAME} | {N_NOTES} notes | {N_RUNS} runs per note</p>

<div class="key-finding">
  <b>Why this experiment matters:</b> Before interpreting robustness results,
  we must establish that measured ICD drift is caused by text perturbation
  and not by inherent LLM stochasticity. This report quantifies the model's
  baseline prediction noise across repeated identical inputs.
  <br><br>
  <b>Interpretation:</b> The noise floor (flip rate) represents the proportion
  of runs where the model gives a different answer on the SAME note.
  Only drift rates ABOVE this floor can be confidently attributed to
  perturbation.
</div>

<h2>Summary Statistics</h2>
<table style="border-collapse:collapse;font-size:0.9em;width:100%;">
  <tr style="background:#2E75B6;color:white;">
    <th style="padding:8px 12px;border:1px solid #ccc;">Temperature</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">Notes</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">Consistency Rate</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">95% CI</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">Prefix Consistency</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">Noise Floor</th>
    <th style="padding:8px 12px;border:1px solid #ccc;">GT Match Rate</th>
  </tr>
  {stats_rows}
</table>

<div class="key-finding" style="margin-top:20px;">
  <b>How to use these numbers in your presentation:</b><br>
  If temp=0.0 consistency = 100%: the model is fully deterministic.
  All drift in your main experiment is caused by the perturbation.<br><br>
  If temp=0.1 noise floor = 8%: any measured drift rate below 8% is
  within the noise range and cannot be attributed to the attack.
  Your main experiment drift rates must exceed this floor to be meaningful.
</div>

<h2>Per-Note Results</h2>
<table style="border-collapse:collapse;font-size:0.88em;width:100%;">
  <tr style="background:#ddd;">
    <th style="padding:6px;border:1px solid #ccc;">Patient ID</th>
    <th style="padding:6px;border:1px solid #ccc;">Temp</th>
    <th style="padding:6px;border:1px solid #ccc;">GT ICD</th>
    <th style="padding:6px;border:1px solid #ccc;">GT Description</th>
    {run_headers}
    <th style="padding:6px;border:1px solid #ccc;">Agreement</th>
    <th style="padding:6px;border:1px solid #ccc;">Consistent?</th>
  </tr>
  {rows}
</table>
<p style="font-size:0.85em;color:#888;margin-top:8px;">
  Green rows = all {N_RUNS} runs gave the same code. Red rows = at least one run differed.
  Within a row, green cells = matches the most common prediction, yellow = differs.
</p>
</body></html>"""

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nReport saved: {OUTPUT_REPORT}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print(f"LLM Stability Experiment")
    print(f"Model: {MODEL_NAME} | Notes: {N_NOTES} | Runs per note: {N_RUNS}")
    print(f"Temperatures: {TEMPERATURES}")

    # Test connection
    test = query_llm("Reply with OK", temperature=0.0)
    if not test:
        print("ERROR: Cannot connect to LLM server")
        return
    print(f"LLM connection OK: {test}")

    # Load notes
    df = load_notes(N_NOTES)

    # Run experiment at each temperature
    all_results  = []
    stats_by_temp = []

    for temp in TEMPERATURES:
        results = run_stability_experiment(df, temperature=temp)
        all_results.extend(results)

        print(f"\n--- Statistics for temperature={temp} ---")
        stats = compute_statistics(all_results, temperature=temp)
        stats_by_temp.append(stats)

    # Save CSV
    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV saved: {OUTPUT_CSV}")

    # Generate HTML report
    generate_stability_report(all_results, stats_by_temp)

    # Final summary for presentation
    print("\n" + "="*60)
    print("COPY THIS INTO YOUR PRESENTATION:")
    print("="*60)
    for s in stats_by_temp:
        print(f"\nTemperature = {s['temperature']}:")
        print(f"  Consistency rate:  {100*s['consistency_rate']:.1f}% "
              f"[95% CI: {100*s['ci_lower']:.1f}-{100*s['ci_upper']:.1f}%]")
        print(f"  Noise floor:       {100*s['noise_floor']:.1f}%")
        if s['temperature'] == 0.0 and s['consistency_rate'] == 1.0:
            print(f"  -> Model is fully deterministic at temp=0.0")
            print(f"  -> All drift in main experiment is caused by perturbation")
        elif s['temperature'] > 0:
            print(f"  -> Main experiment drift rates must exceed {100*s['noise_floor']:.1f}%")
            print(f"     to be attributed to perturbation (not model noise)")
    print("="*60)


if __name__ == "__main__":
    main()
