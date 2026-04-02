"""
=============================================================================
ROBUSTNESS PIPELINE — LEVEL 0 (LEXICAL SYNONYM ATTACK)
=============================================================================

RESEARCH QUESTION:
  When medical terms in a clinical discharge note are replaced with their
  validated synonyms (same meaning, different words), does the LLM still
  assign the correct ICD-10 diagnosis code?

ATTACK DEFINITION — LEVEL 0:
  Lexical synonym substitution. Each replacement is guaranteed to mean
  exactly the same thing as the original because both terms share the same
  UMLS Concept Unique Identifier (CUI).

  Example:
    "myocardial infarction" → "heart attack"
    Both map to UMLS CUI C0027051. Same concept. Different surface form.

  This is the most conservative possible attack. If the model fails here,
  it means it is sensitive to writing style, not to clinical content.

PIPELINE FLOW:
  Load MIMIC notes (ICD-10 only)
    ↓
  Predict ICD on original note (Llama)
    ↓
  Baseline check: is prediction correct? (ICD prefix match)
    ↓ skip if wrong
  Extract medical entities (scispaCy NER)
    ↓
  UMLS filter: keep only terms with a medical CUI (no LLM involved)
    ↓
  Get UMLS synonyms for each validated entity (UMLS atoms lookup)
    ↓  NO LLM IN THIS STEP — synonyms come from UMLS directly
  Apply up to MAX_CHANGES replacements to note
    ↓
  Predict ICD on perturbed note (Llama)
    ↓
  Classify drift: stable / subcategory / categorical
    ↓
  Save CSV + HTML report + plots

WHY UMLS FOR SYNONYMS (not LLM):
  Previous versions used Llama to generate synonyms, then validated them.
  This was circular — the model being tested was also generating the test.
  UMLS is an independent medical knowledge base. Every synonym it provides
  is guaranteed to share the same CUI as the original term. No LLM opinion
  is needed or used in the perturbation step.

AUTHOR: Research project — RWTH Aachen
=============================================================================
"""

import torch
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import random
import warnings
import spacy
import requests
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION — change these to run
# ============================================================
MODEL_NAME   = "llama3.1:8b"
API_URL      = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

UMLS_API_KEY   = "379d16de-4bda-4cf5-98b6-6cac578452c1"   # ← put your key here
UMLS_BASE_URL  = "https://uts-ws.nlm.nih.gov/rest"

FINAL_RESULTS  = "level0_robustness.csv"
HTML_REPORT    = "level0_report.html"
PLOT_FILE      = "level0_plots.png"

MAX_SAMPLES    = 20   # increase to 500 for final run
MAX_CHANGES    = 8      # max synonyms applied per note
DEVICE         = "cpu"

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR
# Llama is used ONLY for:
#   (a) predicting ICD on original note
#   (b) predicting ICD on perturbed note
#   (c) baseline equivalence check (secondary gate only)
# Llama is NOT used for synonym generation in Level 0.
# ============================================================
def query_llm(prompt, max_tokens=50):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model"  : MODEL_NAME,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.0, "num_predict": max_tokens}
        # temperature=0 → fully deterministic, no stochastic drift
    }
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.post(API_URL, headers=headers,
                            data=json.dumps(payload), timeout=300)
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except Exception as e:
        debug_print("LLM_ERROR", str(e))
        return ""


# ============================================================
# SECTION 2: UMLS CLIENT
#
# What UMLS does in this pipeline:
#
#   (A) Entity validation — checks whether a scispaCy-detected
#       entity is actually a known medical concept. If UMLS does
#       not recognise a term, it is not a medical concept and we
#       skip it. This replaces the stop-list approach.
#
#       bathroom    → not in UMLS → skip
#       morning     → not in UMLS → skip
#       heparin     → CUI C0019134, Pharmacologic Substance → keep
#       dyspnea     → CUI C0013404, Sign or Symptom → keep
#
#   (B) Synonym retrieval — for each validated entity, UMLS
#       returns all known synonyms (called "atoms") for its CUI.
#       We randomly pick one as the replacement.
#
#       dyspnea → CUI C0013404 → atoms include:
#         "shortness of breath", "breathlessness",
#         "difficulty breathing", "SOB"
#       We pick one randomly. No LLM involved.
#
# UMLS semantic types accepted for Level 0:
#   Only types that directly drive ICD coding decisions.
#   Administrative types like "Health Care Activity" are excluded
#   because they include section headers (Chief Complaint, HPI etc.)
# ============================================================

CLINICAL_SEM_TYPES = {
    # What the patient has — core ICD drivers
    'Disease or Syndrome',
    'Neoplastic Process',
    'Pathologic Function',
    'Injury or Poisoning',
    'Mental or Behavioral Dysfunction',
    'Congenital Abnormality',
    # What the patient is given
    'Pharmacologic Substance',
    'Antibiotic',
    'Clinical Drug',
    # What was done
    'Therapeutic or Preventive Procedure',
    'Diagnostic Procedure',
    'Laboratory Procedure',
    # What the patient feels/shows
    'Sign or Symptom',
    'Laboratory or Test Result',
}


class UMLSClient:
    """
    UMLS REST API client with caching.

    Authentication uses CAS (two-step):
      Step 1: API key → Ticket Granting Ticket (TGT), valid 8 hours
      Step 2: TGT → Service Ticket (ST), single use per API call

    Caching: every term is looked up at most once per run.
    Across 500 notes with repeated terms, cache hit rate is ~80%.
    """

    def __init__(self, api_key):
        self.api_key    = api_key
        self.tgt        = None
        self._cache     = {}   # term → (cui, sem_types, atoms)
        self._auth()

    def _auth(self):
        url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        try:
            resp = requests.post(url, data={"apikey": self.api_key}, timeout=30)
            m = re.search(r'action="(.*?)"', resp.text)
            if m:
                self.tgt = m.group(1)
                debug_print("UMLS", "Authentication OK")
            else:
                debug_print("UMLS", "Auth FAILED — check API key")
        except Exception as e:
            debug_print("UMLS", f"Auth error: {e}")

    def _ticket(self):
        if not self.tgt:
            return None
        try:
            resp = requests.post(
                self.tgt,
                data={"service": "http://umlsks.nlm.nih.gov"},
                timeout=15)
            return resp.text.strip()
        except:
            return None

    def lookup(self, term):
        """
        Look up a term in UMLS.
        Returns (cui, sem_types, atoms) or (None, [], []).

        cui       — Concept Unique Identifier e.g. "C0013404"
        sem_types — list of semantic type strings
        atoms     — list of all known synonyms for this concept
        """
        key = term.lower().strip()
        if key in self._cache:
            return self._cache[key]

        st = self._ticket()
        if not st:
            self._cache[key] = (None, [], [])
            return (None, [], [])

        try:
            # Search for the term
            resp = requests.get(
                f"{UMLS_BASE_URL}/search/current",
                params={
                    "string"      : term,
                    "ticket"      : st,
                    "searchType"  : "normalizedString",
                    "returnIdType": "concept",
                    "pageSize"    : 3
                },
                timeout=15)
            results = resp.json().get("result", {}).get("results", [])
            if not results or results[0].get("ui") == "NONE":
                self._cache[key] = (None, [], [])
                return (None, [], [])

            cui = results[0]["ui"]

            # Get semantic types
            st2 = self._ticket()
            resp2 = requests.get(
                f"{UMLS_BASE_URL}/content/current/CUI/{cui}",
                params={"ticket": st2},
                timeout=15)
            sem_list = resp2.json().get("result", {}).get("semanticTypes", [])
            sem_types = [s.get("name", "") for s in sem_list]

            # Get atoms (synonyms)
            atoms = self._get_atoms(cui)

            result = (cui, sem_types, atoms)
            self._cache[key] = result
            return result

        except Exception as e:
            debug_print("UMLS_LOOKUP", f"Failed for '{term}': {e}")
            self._cache[key] = (None, [], [])
            return (None, [], [])

    def _get_atoms(self, cui):
        """
        Get all English synonym strings for a CUI.
        These are the atoms — every term that maps to the same concept.

        For CUI C0013404 (Dyspnea), atoms include:
          "Dyspnea", "dyspnoea", "shortness of breath",
          "breathlessness", "difficulty breathing", "SOB", ...

        We filter to English only and deduplicate.
        Returns list of strings.
        """
        st = self._ticket()
        if not st:
            return []
        try:
            resp = requests.get(
                f"{UMLS_BASE_URL}/content/current/CUI/{cui}/atoms",
                params={
                    "ticket"  : st,
                    "language": "ENG",
                    "pageSize": 25
                },
                timeout=15)
            atoms = resp.json().get("result", [])
            names = list({a.get("name", "").strip()
                          for a in atoms
                          if a.get("name", "").strip()})
            return names
        except:
            return []


# ============================================================
# SECTION 3: DATA LOADER
# ============================================================
class MIMICLoader:

    # These sections give away the diagnosis — must be excluded
    # so the model cannot just read the answer from the note
    _ANSWER_SECTIONS = re.compile(
        r'(?:Discharge Diagnosis(?:es)?|Primary Diagnosis|Final Diagnosis|'
        r'Discharge Condition|Discharge Medications|Discharge Instructions|'
        r'Follow.?up|Transitional Issues|Medications on (?:Admission|Discharge))'
        r'\s*:?.*',
        re.S | re.I
    )

    @staticmethod
    def clean(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)   # remove MIMIC redactions
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @classmethod
    def extract(cls, text):
        """
        Extract only the clinically informative sections.
        EXCLUDES discharge diagnosis to prevent answer leakage.
        The model must reason from symptoms/history, not read the answer.
        """
        # Truncate at any answer-leaking section
        text = cls._ANSWER_SECTIONS.sub('', text)

        parts = []
        for label, pattern in [
            ("Chief Complaint",
             r"Chief Complaint\s*:?\s*(.*?)(?:History of Present Illness|HPI|\Z)"),
            ("HPI",
             r"(?:History of Present Illness|HPI)\s*:?\s*(.*?)(?:Past Medical History|Social History|Physical Exam|\Z)"),
            ("Hospital Course",
             r"Brief Hospital Course\s*:?\s*(.*?)(?:Medications on Admission|Discharge|\Z)"),
            ("Assessment",
             r"(?:Assessment(?:\s+and\s+Plan)?)\s*:?\s*(.*?)(?:Discharge|Follow|\Z)"),
            ("PMH",
             r"(?:Past Medical History|PMH)\s*:?\s*(.*?)(?:Social History|Family History|\Z)"),
        ]:
            m = re.search(pattern, text, re.S | re.I)
            if m:
                content = m.group(1).strip()
                if len(content) > 50:
                    parts.append(f"{label}: {content[:1200]}")

        return "\n\n".join(parts)[:3000] if parts else text[-2000:]

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n):
        debug_print("DATA", "Loading MIMIC (ICD-10 only)...")

        diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
        diags = diags[diags['icd_version'] == 10].copy()
        debug_print("DATA", f"ICD-10 diagnoses: {len(diags):,}")

        hadm_ids = set(diags['hadm_id'].unique())
        chunks, found = [], 0
        for chunk in pd.read_csv(notes_path, compression='gzip',
                                  usecols=['subject_id', 'hadm_id', 'text'],
                                  chunksize=5000):
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
            if len(chunk):
                chunks.append(chunk)
                found += len(chunk)
            if found >= n * 50:
                break
        notes = pd.concat(chunks, ignore_index=True)

        if os.path.exists(dict_path):
            d = pd.read_csv(dict_path, compression='gzip')
            d = d[d['icd_version'] == 10]
            diags = pd.merge(diags,
                             d[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown"

        primary = diags[diags['seq_num'] == 1]
        df = pd.merge(notes,
                      primary[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')
        df['text']      = df['text'].apply(cls.clean)
        df['note_text'] = df['text'].apply(cls.extract)
        result = df.sample(n=min(len(df), n), random_state=42)
        debug_print("DATA", f"Sampled {len(result)} notes")
        return result


# ============================================================
# SECTION 4: ICD PREDICTION
# ============================================================
def predict_icd(text):
    """
    Ask Llama to predict the primary ICD-10 code from a clinical note.
    Two attempts with different prompt styles.
    Returns ICD code string or 'UNKNOWN'.
    """
    prompts = [
        (f"Read the clinical note. Return ONLY the single most likely ICD-10 "
         f"diagnosis code (format: J18.9). Do not explain. Code only.\n\n"
         f"Note:\n{text}\n\nICD-10 Code:"),
        (f"Output a single ICD-10 code for this note. "
         f"Format: letter + 2 digits + optional dot + digits.\n\n"
         f"Note:\n{text[:800]}\n\nCode:")
    ]
    for prompt in prompts:
        resp = query_llm(prompt, max_tokens=15)
        if resp:
            # Strip DeepSeek-R1 thinking tags if present
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.S).strip()
            m = re.search(r'[A-Z]\d{2}\.\d+', resp.upper())
            if m:
                return m.group(0)
            m = re.search(r'[A-Z]\d{2,4}', resp.upper())
            if m:
                return m.group(0)
    return "UNKNOWN"


def describe_icd(code):
    """Ask Llama to describe an ICD code in 2-5 words."""
    if code == "UNKNOWN":
        return "Unknown"
    resp = query_llm(
        f"What condition does ICD-10 code '{code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words.",
        max_tokens=20)
    return resp or "Unknown"


# ============================================================
# SECTION 5: BASELINE CORRECTNESS CHECK
#
# We check whether Llama's prediction on the ORIGINAL note
# matches the ground truth BEFORE applying any attack.
# Only correct predictions proceed to the robustness test.
#
# Two gates in order:
#
# Gate 1 — ICD prefix match (objective, no LLM):
#   Compare first 3 characters of predicted vs ground truth.
#   GT=C3412 → prefix C34 | Pred=C34.91 → prefix C34 → MATCH ✓
#   GT=I130  → prefix I13 | Pred=I50.9  → prefix I50 → NO MATCH ✗
#   This handles the MIMIC format difference:
#     MIMIC stores C3412 (no dot), Llama outputs C34.2 (with dot)
#     Both normalise to prefix C34 → correctly matched
#
# Gate 2 — LLM equivalence check (secondary, handles edge cases):
#   Ask Llama: "Is your prediction the same condition as the GT?"
#   Handles ICD-9 vs ICD-10 edge cases and near-equivalent codes.
#   This gate is secondary — prefix match alone is sufficient.
#
# A case passes if EITHER gate agrees.
# ============================================================
def normalise_icd(code):
    return re.sub(r'[^A-Z0-9]', '', code.upper())


def llm_equivalence(pred_icd, gt_icd, gt_desc):
    """Ask LLM whether prediction matches ground truth condition."""
    if pred_icd == "UNKNOWN":
        return False
    resp = query_llm(
        f"Ground truth: '{gt_desc}' (ICD-10: {gt_icd})\n"
        f"Predicted: {pred_icd}\n\n"
        f"Does the predicted code represent the same primary condition "
        f"as the ground truth? Minor specificity differences are OK.\n"
        f"Answer YES or NO only.",
        max_tokens=5)
    return resp.strip().upper().startswith("YES")


def baseline_correct(pred_icd, gt_icd, gt_desc):
    """
    Returns (passed, gate_string, prefix_ok, llm_ok).

    gate_string shows exactly which gate fired, for transparency
    in the HTML report.
    """
    if pred_icd == "UNKNOWN":
        return False, "unknown prediction", False, False

    gt_norm   = normalise_icd(gt_icd)
    pred_norm = normalise_icd(pred_icd)

    # Gate 1: ICD prefix match (objective)
    prefix_ok = (len(gt_norm) >= 3 and
                 len(pred_norm) >= 3 and
                 gt_norm[:3] == pred_norm[:3])

    # Gate 2: LLM equivalence (secondary — only call if prefix fails)
    # This saves API calls — if prefix matches we do not need LLM
    llm_ok = False
    if not prefix_ok:
        llm_ok = llm_equivalence(pred_icd, gt_icd, gt_desc)

    passed = prefix_ok or llm_ok
    gate_str = (f"prefix={gt_norm[:3]}{'==' if prefix_ok else '!='}"
                f"{pred_norm[:3]} | llm={'YES' if llm_ok else 'NO'}")
    return passed, gate_str, prefix_ok, llm_ok


# ============================================================
# SECTION 6: LEVEL 0 PERTURBATION
#
# This is the core of the experiment.
#
# Level 0 = Lexical Synonym Substitution using UMLS atoms.
#
# For every medical entity in the note:
#   1. Look it up in UMLS
#   2. If it has a clinical CUI → it is a valid medical term
#   3. Get all known synonyms (atoms) from UMLS for that CUI
#   4. Pick a synonym that is different from the original
#   5. Apply it to the note
#
# The synonym is guaranteed to mean exactly the same thing
# because it shares the same CUI as the original term.
# This is the definition of a synonym in medical terminology.
#
# What makes this "Level 0":
#   - Meaning is IDENTICAL (same CUI)
#   - Only the surface form changes
#   - No clinical information is altered
#   - If the model drifts, it is purely due to linguistic sensitivity
# ============================================================

# Minimum synonym length — avoid single-letter atoms
MIN_ATOM_LEN = 3

# Hard safety blocks — these should never be replaced regardless
# of whether they appear in UMLS
NEVER_REPLACE = re.compile(
    r'^(no|not|denies|denied|without|negative|none|absent|absence|'
    r'left|right|bilateral|elevated|decreased|increased|'
    r'mild|moderate|severe|stable|benign|malignant|'
    r'positive|negative|normal|abnormal)$',
    re.IGNORECASE
)


def entity_safe_to_replace(ent):
    """
    Basic safety check before UMLS lookup.
    Rejects: negated terms, polarity words, single chars, numbers.
    Returns True if safe to proceed to UMLS.
    """
    text = ent.text.strip()
    if len(text) < 3:
        return False
    if NEVER_REPLACE.match(text):
        return False
    # Reject if entity starts with a negation in context
    neg_terms = {'no', 'not', 'denies', 'without', 'negative', 'none'}
    window = [t.text.lower() for t in
              list(ent.doc)[max(0, ent.start - 3):ent.start]]
    if any(w in neg_terms for w in window):
        return False
    # Reject verbs, determiners, pronouns
    if ent[0].pos_ in {'VERB', 'ADP', 'DET', 'PRON',
                        'CCONJ', 'SCONJ', 'PUNCT', 'NUM'}:
        return False
    return True


def pick_synonym(original, atoms):
    """
    Pick a random UMLS synonym that:
    - Is different from the original
    - Is not just a capitalisation variant of the original
    - Has meaningful length (>= MIN_ATOM_LEN chars)
    - Is not longer than 5 words (avoid sentence-length atoms)
    Returns chosen synonym string or None.
    """
    original_lower = original.lower().strip()
    candidates = []
    for atom in atoms:
        atom = atom.strip()
        if not atom or len(atom) < MIN_ATOM_LEN:
            continue
        if atom.lower() == original_lower:
            continue
        if len(atom.split()) > 5:
            continue
        # Skip if atom contains the original as a substring
        # (e.g. "acute myocardial infarction" when original is "MI")
        # — we want genuinely different surface forms
        candidates.append(atom)

    if not candidates:
        return None
    return random.choice(candidates)


class Level0Perturber:
    """
    Level 0 perturbation engine.

    Process per note:
      1. Run scispaCy NER to find entity candidates
      2. For each candidate, look up in UMLS
         - If not in UMLS or wrong semantic type → skip
         - If in UMLS with clinical type → keep
      3. Get UMLS atoms (synonyms) for each validated entity
      4. Pick a synonym using pick_synonym()
      5. Apply up to MAX_CHANGES replacements

    All synonyms come from UMLS — no LLM involvement.
    """

    def __init__(self, umls_client):
        self.umls = umls_client
        for model in ["en_core_sci_lg", "en_core_sci_md", "en_core_sci_sm"]:
            try:
                self.nlp = spacy.load(model)
                debug_print("NER", f"Loaded {model}")
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                "No scispaCy model found.\n"
                "Install: pip install scispacy\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                "/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz")

    def perturb(self, text, max_changes=MAX_CHANGES):
        """
        Apply Level 0 attack to a note.

        Returns:
          perturbed_text  — note with synonyms applied
          changes_log     — list of (original, synonym, cui, sem_type)
        """
        doc  = self.nlp(text)
        seen = set()

        # Phase 1: collect all candidate entities
        candidates = []
        for ent in doc.ents:
            key = ent.text.strip().lower()
            if key in seen:
                continue
            if not entity_safe_to_replace(ent):
                continue
            seen.add(key)
            sent = ent.sent.text if ent.sent else text[max(0,ent.start_char-60):ent.end_char+60]
            candidates.append({
                'text'      : ent.text,
                'start_char': ent.start_char,
                'end_char'  : ent.end_char,
                'context'   : sent,
            })

        print(f"\n   > scispaCy found {len(candidates)} entity candidates")

        # Phase 2: UMLS validation
        validated = []
        for ent in candidates:
            cui, sem_types, atoms = self.umls.lookup(ent['text'])
            if cui is None:
                continue   # not a known medical concept
            if not any(t in CLINICAL_SEM_TYPES for t in sem_types):
                continue   # wrong semantic type (admin/structural concept)
            ent['cui']      = cui
            ent['sem_type'] = sem_types[0] if sem_types else 'Unknown'
            ent['atoms']    = atoms
            validated.append(ent)

        print(f"   > {len(validated)} passed UMLS clinical filter")

        if not validated:
            return text, []

        # Phase 3: pick synonyms and apply (up to max_changes)
        # Shuffle to get random selection across the note
        random.shuffle(validated)
        selected = validated[:max_changes]

        print(f"   > Selecting up to {max_changes}, attempting {len(selected)}")

        accepted = []
        for ent in selected:
            synonym = pick_synonym(ent['text'], ent['atoms'])
            if not synonym:
                print(f"       NO SYNONYM: '{ent['text']}' "
                      f"({len(ent['atoms'])} atoms, none usable)")
                continue

            accepted.append({
                'original'  : ent['text'],
                'synonym'   : synonym,
                'cui'       : ent['cui'],
                'sem_type'  : ent['sem_type'],
                'start_char': ent['start_char'],
                'end_char'  : ent['end_char'],
            })
            print(f"       ACCEPTED [{ent['sem_type'][:25]:25s}] "
                  f"'{ent['text']}' → '{synonym}' "
                  f"(CUI {ent['cui']})")

        if not accepted:
            return text, []

        # Apply replacements right-to-left to preserve char offsets
        perturbed = text
        for item in sorted(accepted, key=lambda x: x['start_char'], reverse=True):
            perturbed = (perturbed[:item['start_char']]
                         + item['synonym']
                         + perturbed[item['end_char']:])

        changes_log = [(i['original'], i['synonym'],
                        i['cui'], i['sem_type'])
                       for i in accepted]
        return perturbed, changes_log


# ============================================================
# SECTION 7: DRIFT CLASSIFICATION
# ============================================================
def classify_drift(orig_icd, pert_icd):
    """
    Compare original and perturbed ICD predictions.

    Same code         → STABLE (no drift)
    Same 3-char prefix → SUBCATEGORY DRIFT (same disease family, different specificity)
    Different prefix   → CATEGORICAL DRIFT (different disease category)

    The 3-character prefix defines the disease category in ICD-10.
    E.g. I50 = all heart failure codes. Drift within I50 is subcategory.
    Drift from I50 to J18 (pneumonia) is categorical.

    Categorical drift is the primary robustness metric.
    """
    o = normalise_icd(orig_icd)
    p = normalise_icd(pert_icd)

    if "UNKNOWN" in (o, p):
        return {"drift_level": "unknown",
                "categorical_drift": False,
                "subcategory_drift": False}

    exact  = (o == p)
    prefix = (len(o) >= 3 and len(p) >= 3 and o[:3] == p[:3])

    level = "none" if exact else ("subcategory" if prefix else "categorical")
    return {
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 8: SEMANTIC SIMILARITY (ClinicalBERT)
# Used as a quality check — confirms perturbations are minimal.
# Not used as a decision gate.
# ============================================================
class SimilarityScorer:
    def __init__(self):
        debug_print("BERT", "Loading ClinicalBERT...")
        self.tok = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.mdl = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.mdl.eval()

    def score(self, t1, t2):
        with torch.no_grad():
            inp = self.tok([t1, t2], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(DEVICE)
            out = self.mdl(**inp)
            mask = inp['attention_mask'].unsqueeze(-1).float()
            e1 = (out.last_hidden_state[0] * mask[0]).sum(0) / mask[0].sum()
            e2 = (out.last_hidden_state[1] * mask[1]).sum(0) / mask[1].sum()
            return float(cosine_similarity(
                e1.unsqueeze(0).numpy(),
                e2.unsqueeze(0).numpy())[0][0])


# ============================================================
# SECTION 9: HTML REPORT
# ============================================================
def generate_html(df, filename=HTML_REPORT):

    COLORS = {
        'none'       : '#ccffcc',
        'subcategory': '#fff3cd',
        'categorical': '#ffcccc',
        'unknown'    : '#eeeeee',
        'skipped'    : '#f9f9f9',
    }
    LABELS = {
        'none'       : 'STABLE',
        'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
        'unknown'    : 'UNKNOWN ICD',
        'skipped'    : 'SKIPPED',
    }

    def highlight(orig_text, pert_text, changes_str):
        """
        Highlight only the specific changed positions.
        Uses offset-aware replacement — does NOT highlight all
        occurrences of a word throughout the note.
        """
        if not changes_str or changes_str in ('SKIPPED', '', 'None'):
            return orig_text, pert_text

        oh, ph = orig_text, pert_text
        consumed = []

        pairs = []
        for part in changes_str.split('|'):
            part = part.strip().replace('\u2192', '->')
            m = re.match(r'\[.*?\]\s*(.*?)\s*->\s*(.*)', part)
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))

        for orig_w, new_w in pairs:
            # Find first non-consumed occurrence of orig_w in oh
            pattern = re.compile(r'\b' + re.escape(orig_w) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(orig_text):
                s, e = match.start(), match.end()
                if not any(not (e <= cs or s >= ce) for cs, ce in consumed):
                    consumed.append((s, e))
                    oh = oh[:s] + (
                        f'<span style="background:#FF6B6B;color:white;'
                        f'padding:1px 4px;border-radius:3px;font-weight:bold;">'
                        f'{orig_w}</span>') + oh[e:]
                    break

            # Find first occurrence of new_w in ph
            pat2 = re.compile(r'\b' + re.escape(new_w) + r'\b', re.IGNORECASE)
            m2 = pat2.search(pert_text)
            if m2:
                s, e = m2.start(), m2.end()
                ph = ph[:s] + (
                    f'<span style="background:#51C878;color:white;'
                    f'padding:1px 4px;border-radius:3px;font-weight:bold;">'
                    f'{new_w}</span>') + ph[e:]

        return oh, ph

    rows = ""
    for _, row in df.iterrows():
        skipped    = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift      = row.get('drift_level', 'none')
        dc         = COLORS.get(drift, '#ccffcc')
        dl         = LABELS.get(drift, 'STABLE')
        sim_d      = f"{row['semantic_similarity']:.4f}" if row.get('semantic_similarity') else 'N/A'
        gate_d     = row.get('baseline_gate', '')
        pred_desc  = row.get('pred_desc', '')
        chg_str    = row.get('changes_made', '')

        if skipped:
            body = (f'<p style="color:#999;font-style:italic;padding:10px;'
                    f'background:#f9f9f9;border-radius:4px;">'
                    f'Skipped — baseline did not match ground truth.<br>'
                    f'<small>Gates: {gate_d}</small></p>')
        else:
            oh, ph = highlight(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                chg_str)
            body = f"""
            <div style="margin-top:10px;">
              <div style="font-size:0.85em;margin-bottom:8px;">
                <span style="background:#FF6B6B;color:white;padding:2px 8px;
                border-radius:3px;">Replaced (original)</span>&nbsp;
                <span style="background:#51C878;color:white;padding:2px 8px;
                border-radius:3px;">UMLS synonym (Level 0)</span>
              </div>
              <table style="width:100%;border-collapse:collapse;">
                <tr>
                  <th style="width:50%;background:#e8f4f8;padding:8px;
                  border:1px solid #ccc;">Original Note</th>
                  <th style="width:50%;background:#fff8e8;padding:8px;
                  border:1px solid #ccc;">Perturbed Note</th>
                </tr>
                <tr>
                  <td style="padding:10px;border:1px solid #ccc;vertical-align:top;
                  white-space:pre-wrap;font-size:0.83em;line-height:1.55em;">{oh}</td>
                  <td style="padding:10px;border:1px solid #ccc;vertical-align:top;
                  white-space:pre-wrap;font-size:0.83em;line-height:1.55em;">{ph}</td>
                </tr>
                <tr>
                  <td style="padding:8px;border:1px solid #ccc;background:#f0f0f0;
                  text-align:center;"><b>ICD-10: {row.get('orig_icd_pred','?')}</b></td>
                  <td style="padding:8px;border:1px solid #ccc;background:#f0f0f0;
                  text-align:center;"><b>ICD-10: {row.get('pert_icd_pred','?')}</b></td>
                </tr>
              </table>
            </div>"""

        rows += f"""
        <div style="border:1px solid #ccc;margin:20px 0;padding:15px;border-radius:8px;">
          <h3 style="margin:0 0 10px 0;">
            Patient ID: {row['hadm_id']}
            <span style="background:{dc};padding:4px 10px;
            border-radius:4px;font-size:0.9em;">{dl}</span>
          </h3>
          <table style="font-size:0.9em;border-collapse:collapse;margin-bottom:10px;">
            <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                <td>{row.get('gt_desc','')} ({row.get('gt_icd','')} ICD-10)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                <td>{row.get('orig_icd_pred','')} — {pred_desc}</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                <td><small>{gate_d}</small></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                <td>{chg_str if chg_str not in ('SKIPPED','','None') else 'None'}
                ({row.get('n_changes',0)} terms)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                <td>{sim_d}</td></tr>
          </table>
          {body}
        </div>"""

    vdf = df[df['baseline_correct'] == True]
    cat_rate = vdf['categorical_drift'].mean()*100 if len(vdf) else 0
    sub_rate = vdf['subcategory_drift'].mean()*100  if len(vdf) else 0
    avg_sim  = f"{vdf['semantic_similarity'].mean():.4f}" if len(vdf) else 'N/A'
    avg_chg  = f"{vdf['n_changes'].mean():.1f}" if len(vdf) else 'N/A'

    html = f"""<!DOCTYPE html><html><head><title>Level 0 Robustness Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa;}}
h1{{color:#1F4E79;}} h3{{margin:0 0 10px 0;}}
.summary{{background:#f0f0f0;padding:15px;border-radius:8px;margin-bottom:30px;}}
.summary p{{margin:4px 0;}}
.method{{background:#e8f4f8;padding:12px;border-radius:6px;margin-bottom:20px;font-size:0.9em;color:#555;}}
.legend{{display:flex;gap:16px;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;}}
.legend span{{padding:3px 10px;border-radius:4px;}}
</style></head><body>
<h1>ICD-10 Robustness Report — Level 0 (Lexical Synonym Attack)</h1>
<p style="color:#666;">Model: {MODEL_NAME} | Attack: Level 0 — UMLS synonym substitution</p>
<div class="method">
  <b>Level 0 Attack Method:</b> Medical entities are identified using scispaCy NER
  and validated against UMLS. For each validated entity, UMLS returns all known
  synonyms (atoms) sharing the same Concept Unique Identifier (CUI). A random
  synonym is selected. No LLM is used for synonym generation — all substitutions
  are guaranteed to be semantically identical by UMLS definition.
  <br><br>
  <b>Baseline gate:</b> ICD-10 prefix matching (objective) + LLM equivalence (secondary).
  Cases pass if predicted and ground truth share the same 3-character ICD prefix.
</div>
<div class="legend">
  <span style="background:#ccffcc;">STABLE — same ICD code</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT — same disease family</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease</span>
</div>
<div class="summary">
  <h2>Summary</h2>
  <p><b>Model:</b> {MODEL_NAME} | <b>Attack Level:</b> 0 (Lexical synonyms via UMLS)</p>
  <p><b>Total Samples:</b> {len(df)}</p>
  <p><b>Valid for Analysis:</b> {len(vdf)}
     <span style="color:#888;">(baseline: ICD prefix match OR LLM equivalence)</span></p>
  <p><b>Skipped (wrong baseline):</b> {len(df)-len(vdf)}</p>
  <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}%
     <span style="color:#888;">(primary metric — different disease category)</span></p>
  <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%</p>
  <p><b>Avg Semantic Similarity:</b> {avg_sim}</p>
  <p><b>Avg UMLS Synonyms Applied:</b> {avg_chg}</p>
</div>
{rows}
</body></html>"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML saved: {filename}")


# ============================================================
# SECTION 10: PLOTS
# ============================================================
def plot_results(df):
    vdf = df[df['baseline_correct'] == True]
    if not len(vdf):
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Plot 1: Semantic similarity distribution
    axes[0].hist(vdf['semantic_similarity'].dropna(), bins=15,
                 color='teal', edgecolor='black')
    axes[0].set_title('Semantic Preservation\n(valid cases only)')
    axes[0].set_xlabel('Cosine Similarity (ClinicalBERT)')
    axes[0].set_ylabel('Count')

    # Plot 2: Drift breakdown
    counts = vdf['drift_level'].value_counts()
    colors = {'none':'#90EE90', 'subcategory':'#FFD700',
              'categorical':'#FF6B6B', 'unknown':'#CCCCCC'}
    axes[1].bar(counts.index,
                counts.values,
                color=[colors.get(k,'grey') for k in counts.index])
    axes[1].set_title(f"Drift Breakdown\nCategorical: "
                      f"{vdf['categorical_drift'].mean()*100:.1f}%")
    axes[1].set_xlabel('Drift Level')
    axes[1].tick_params(axis='x', rotation=0)

    # Plot 3: Changes vs similarity scatter
    for level, color in [('none','green'), ('subcategory','orange'),
                          ('categorical','red')]:
        mask = vdf['drift_level'] == level
        axes[2].scatter(vdf[mask]['n_changes'],
                        vdf[mask]['semantic_similarity'],
                        label=level, color=color, alpha=0.6)
    axes[2].set_title('Changes vs Similarity')
    axes[2].set_xlabel('UMLS Synonyms Applied')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].legend()

    # Plot 4: Baseline quality pie
    bc = df['baseline_correct'].value_counts()
    labels = []
    vals   = []
    clrs   = []
    if True in bc.index:
        labels.append(f'Correct (n={bc[True]})')
        vals.append(bc[True])
        clrs.append('#90EE90')
    if False in bc.index:
        labels.append(f'Skipped (n={bc[False]})')
        vals.append(bc[False])
        clrs.append('#FFB6C1')
    axes[3].pie(vals, labels=labels, colors=clrs, autopct='%1.1f%%')
    axes[3].set_title('Baseline Quality\n(ICD prefix + LLM gate)')

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 11: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Level 0 Attack | Model: {MODEL_NAME}")

    # Test LLM connection
    if not query_llm("Say hello"):
        debug_print("ERROR", "Cannot reach LLM server. Check VPN/connection.")
        return
    print("LLM OK")

    # Initialise UMLS
    debug_print("MAIN", "Connecting to UMLS...")
    umls = UMLSClient(UMLS_API_KEY)
    if not umls.tgt:
        debug_print("ERROR",
                    "UMLS auth failed. Get your key at "
                    "https://uts.nlm.nih.gov/uts/profile")
        return
    print("UMLS OK")

    # Sanity check UMLS — verify a known synonym pair
    debug_print("MAIN", "UMLS sanity check...")
    cui_d, st_d, atoms_d = umls.lookup("dyspnea")
    cui_s, st_s, atoms_s = umls.lookup("shortness of breath")
    if cui_d and cui_s and cui_d == cui_s:
        print(f"  ✓ dyspnea and 'shortness of breath' share CUI {cui_d} — UMLS working")
    else:
        print(f"  ⚠ dyspnea CUI={cui_d}, 'shortness of breath' CUI={cui_s}")
        print(f"    dyspnea atoms sample: {atoms_d[:3]}")

    # Load data
    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        import traceback
        debug_print("ERROR", f"Data load failed: {e}")
        traceback.print_exc()
        return

    perturber = Level0Perturber(umls)
    scorer    = SimilarityScorer()
    results   = []

    print("\n" + "="*80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id  = row['hadm_id']
            note     = row['note_text']
            gt_desc  = row['long_title']
            gt_icd   = row['icd_code']
            icd_ver  = row['icd_version']

            print(f"\n[{hadm_id}] GT: {gt_desc[:50]} ({gt_icd})")

            # Step 1: predict on original note
            orig_pred = predict_icd(note)
            pred_desc = describe_icd(orig_pred)
            print(f"  Pred: {orig_pred} — {pred_desc}")

            # Step 2: baseline check
            passed, gate_str, prefix_ok, llm_ok = baseline_correct(
                orig_pred, gt_icd, gt_desc)
            print(f"  Gate: {gate_str} → {'PASS' if passed else 'SKIP'}")

            if not passed:
                results.append({
                    'hadm_id'         : hadm_id,
                    'gt_icd'          : gt_icd,
                    'gt_desc'         : gt_desc,
                    'original_text'   : note,
                    'perturbed_text'  : 'SKIPPED',
                    'changes_made'    : 'SKIPPED',
                    'n_changes'       : 0,
                    'orig_icd_pred'   : orig_pred,
                    'pred_desc'       : pred_desc,
                    'pert_icd_pred'   : 'SKIPPED',
                    'baseline_gate'   : gate_str,
                    'baseline_correct': False,
                    'drift_level'     : 'skipped',
                    'categorical_drift': False,
                    'subcategory_drift': False,
                    'semantic_similarity': 0.0,
                })
                continue

            # Step 3: Level 0 perturbation
            print(f"  Perturbing (UMLS synonyms, max {MAX_CHANGES} changes)...")
            pert_note, changes = perturber.perturb(note)
            print(f"  Applied {len(changes)} UMLS synonyms")

            # Step 4: predict on perturbed note
            pert_pred = predict_icd(pert_note)
            print(f"  Pert pred: {pert_pred}")

            # Step 5: classify drift
            drift = classify_drift(orig_pred, pert_pred)
            sim   = scorer.score(note, pert_note)
            print(f"  Drift: {drift['drift_level']} | Sim: {sim:.4f}")

            # Format changes for report
            # Format: [semantic_type] original → synonym (CUI)
            chg_str = " | ".join(
                f"[{sem}] {o} → {s} ({cui})"
                for o, s, cui, sem in changes
            )

            results.append({
                'hadm_id'            : hadm_id,
                'gt_icd'             : gt_icd,
                'gt_desc'            : gt_desc,
                'original_text'      : note,
                'perturbed_text'     : pert_note,
                'changes_made'       : chg_str,
                'n_changes'          : len(changes),
                'orig_icd_pred'      : orig_pred,
                'pred_desc'          : pred_desc,
                'pert_icd_pred'      : pert_pred,
                'baseline_gate'      : gate_str,
                'baseline_correct'   : True,
                **drift,
                'semantic_similarity': sim,
            })

        except Exception as e:
            import traceback
            debug_print("ERROR", f"Row {idx} failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No results.")
        return

    final_df = pd.DataFrame(results)

    # Save CSV (without full note text)
    csv_df = final_df.drop(columns=['original_text','perturbed_text'],
                            errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)

    plot_results(final_df)
    generate_html(final_df)

    vdf = final_df[final_df['baseline_correct'] == True]
    print("\n" + "="*80)
    print(f"LEVEL 0 RESULTS — {MODEL_NAME}")
    print(f"  Total samples        : {len(final_df)}")
    print(f"  Valid (correct base) : {len(vdf)} "
          f"({100*len(vdf)//len(final_df)}%)")
    print(f"  Skipped              : {len(final_df)-len(vdf)}")
    if len(vdf):
        print(f"  Categorical drift    : "
              f"{vdf['categorical_drift'].mean()*100:.1f}%  ← primary metric")
        print(f"  Subcategory drift    : "
              f"{vdf['subcategory_drift'].mean()*100:.1f}%")
        print(f"  Avg similarity       : "
              f"{vdf['semantic_similarity'].mean():.4f}")
        print(f"  Avg synonyms applied : "
              f"{vdf['n_changes'].mean():.1f}")
    print(f"  CSV  → {FINAL_RESULTS}")
    print(f"  HTML → {HTML_REPORT}")
    print(f"  Plot → {PLOT_FILE}")
    print("="*80)


if __name__ == "__main__":
    main()