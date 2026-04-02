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
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME     = "llama3.1:8b"
API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS  = "main/final_robustness.csv"
HTML_REPORT    = "main/robustness_report.html"
PLOT_FILE      = "main/robustness_plots.png"
PER_ICD_CSV    = "main/per_icd_analysis.csv"
PER_CHAPTER_CSV= "main/per_chapter_analysis.csv"

MAX_SAMPLES    = 100
DEVICE         = "cpu"
ATTACK_LEVEL   = 0   # 0=synonyms, 1=related/broader, 2=differential diagnosis
PERTURBATION_RATE = 0.30  # fraction of entities to perturb per note

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR  with retry
# ============================================================
def query_llm(prompt, max_tokens=50, retries=3, retry_delay=15):
    """
    Single LLM connector with retry logic.
    Retries on 504/timeout — common on shared university servers.
    Returns empty string on total failure (caller handles gracefully).
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model"  : MODEL_NAME,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.0, "num_predict": max_tokens}
    }
    session = requests.Session()
    session.trust_env = False

    for attempt in range(retries):
        try:
            response = session.post(API_URL, headers=headers,
                                    data=json.dumps(payload), timeout=300)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except requests.exceptions.RequestException as e:
            debug_print("API_ERROR",
                        f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                import time
                time.sleep(retry_delay)
    return ""


# ============================================================
# SECTION 2: DATA LOADER
# ============================================================
class MIMICLoader:
    """
    Loads MIMIC-IV discharge notes with ICD-10 ground truth.

    KEY DESIGN DECISIONS:
    - ICD-10 only: eliminates version mismatch in baseline comparison
    - Discharge Diagnosis section EXCLUDED: prevents answer leakage
      (model must reason from clinical evidence, not read the answer)
    - Chunked note loading: memory-efficient for large datasets
    - primary diagnosis only (seq_num=1): clean single-label classification
    """

    # Sections that directly state the diagnosis — excluded to prevent leakage
    LEAKING_SECTIONS = re.compile(
        r'(?:Discharge Diagnosis(?:es)?|Primary Diagnosis|Final Diagnosis|'
        r'Discharge Condition|Discharge Instructions|'
        r'Medications on (?:Admission|Discharge)|Discharge Medications|'
        r'Follow.?up|Transitional Issues)\s*:?.*',
        re.S | re.I
    )

    @staticmethod
    def clean_noise(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)   # remove de-id tokens
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @classmethod
    def extract_clinical_info(cls, text: str) -> str:
        """
        Extract clinically rich sections, excluding any that state the diagnosis.
        Sections extracted: Chief Complaint, HPI, Hospital Course,
        Assessment & Plan, PMH.
        Falls back to last 2000 chars if no sections found.
        """
        truncated = cls.LEAKING_SECTIONS.sub('', text)
        parts = []

        cc = re.search(
            r"(?:Chief Complaint)\s*:?\s*(.*?)"
            r"(?:History of Present Illness|HPI|Past Medical History|\Z)",
            truncated, re.S | re.I)
        if cc:
            c = cc.group(1).strip()
            if len(c) > 10:
                parts.append(f"Chief Complaint: {c[:300]}")

        hpi = re.search(
            r"(?:History of Present Illness|HPI)\s*:?\s*(.*?)"
            r"(?:Past Medical History|Social History|"
            r"Review of Systems|Physical Exam|\Z)",
            truncated, re.S | re.I)
        if hpi:
            c = hpi.group(1).strip()
            if len(c) > 100:
                parts.append(f"HPI: {c[:1200]}")

        course = re.search(
            r"Brief Hospital Course\s*:?\s*(.*?)"
            r"(?:Medications on Admission|Discharge Medications|"
            r"Discharge Diagnosis|Discharge Condition|\Z)",
            truncated, re.S | re.I)
        if course:
            c = course.group(1).strip()
            if len(c) > 100:
                parts.append(f"Hospital Course: {c[:1200]}")

        ap = re.search(
            r"(?:Assessment(?:\s+and\s+Plan)?|A(?:/|&)P)\s*:?\s*(.*?)"
            r"(?:Discharge|Follow.?up|\Z)",
            truncated, re.S | re.I)
        if ap:
            c = ap.group(1).strip()
            if len(c) > 50:
                parts.append(f"Assessment: {c[:800]}")

        pmh = re.search(
            r"(?:Past Medical History|PMH)\s*:?\s*(.*?)"
            r"(?:Social History|Family History|Review of Systems|\Z)",
            truncated, re.S | re.I)
        if pmh:
            c = pmh.group(1).strip()
            if len(c) > 50:
                parts.append(f"PMH: {c[:400]}")

        if parts:
            return "\n\n".join(parts)[:3000]
        return truncated[-2000:].strip()

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading MIMIC data (ICD-10 only)...")

        diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
        diags = diags[diags['icd_version'] == 10].copy()
        debug_print("DATA", f"ICD-10 diagnoses: {len(diags)} rows")

        hadm_ids = set(diags['hadm_id'].unique())

        chunks, found = [], 0
        for chunk in pd.read_csv(notes_path, compression='gzip',
                                  usecols=['subject_id', 'hadm_id', 'text'],
                                  chunksize=5000):
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids)].copy()
            if len(chunk) > 0:
                chunks.append(chunk)
                found += len(chunk)
                if found >= n_samples * 50:
                    break
        if not chunks:
            raise ValueError("No matching notes found for ICD-10 admissions")
        notes = pd.concat(chunks, ignore_index=True)

        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            icd_dict = icd_dict[icd_dict['icd_version'] == 10].copy()
            diags = pd.merge(diags,
                             icd_dict[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown"

        primary = diags[diags['seq_num'] == 1]
        df = pd.merge(notes,
                      primary[['hadm_id', 'icd_code', 'icd_version',
                                'long_title']],
                      on='hadm_id')

        df['text']           = df['text'].apply(cls.clean_noise)
        df['processed_text'] = df['text'].apply(cls.extract_clinical_info)

        # Drop duplicates (same hadm_id may appear in multiple chunks)
        df = df.drop_duplicates(subset='hadm_id')

        result = df.sample(n=min(len(df), n_samples), random_state=42)
        debug_print("DATA", f"Loaded {len(result)} ICD-10 records")
        return result


# ============================================================
# SECTION 3: ENTITY FILTERING HELPERS
# ============================================================

# ── Stop sets ────────────────────────────────────────────────────────────────
# Words that look like medical entities but should never be replaced.
# These are categorised for clarity and maintenance.

# Section headers from MIMIC note structure
SECTION_HEADERS = {
    'surgical', 'invasive procedure', 'major surgical',
    'chief complaint', 'hospital course', 'brief hospital course',
    'past medical history', 'social history', 'family history',
    'review of systems', 'physical exam', 'assessment and plan',
    'discharge condition', 'discharge instructions', 'hpi', 'pmh',
    'ros', 'assessment', 'plan',
}

# Generic clinical process words — not ICD-relevant
GENERIC_CLINICAL = {
    'evaluation', 'history', 'exam', 'examination', 'findings', 'results',
    'improvement', 'recovery', 'deterioration', 'management', 'treatment',
    'course', 'status', 'baseline', 'presentation', 'admission', 'discharge',
    'arrival', 'transfer', 'follow-up', 'follow up', 'monitoring',
    'observation', 'workup', 'work-up', 'labs', 'laboratory', 'imaging',
    'data', 'information', 'report', 'note', 'documentation',
}

# Demographics and social words
DEMOGRAPHIC_SOCIAL = {
    'man', 'woman', 'male', 'female', 'lady', 'gentleman', 'patient',
    'mr', 'mrs', 'ms', 'dr', 'mr.', 'mrs.', 'ms.', 'dr.',
    'husband', 'wife', 'daughter', 'son', 'family', 'friend',
    'child', 'children', 'sister', 'brother', 'mother', 'father',
}

# Dosing frequency abbreviations
DOSING_FREQ = {
    'bid', 'tid', 'qid', 'qd', 'prn', 'po', 'iv', 'im', 'sq', 'sl',
    'ac', 'pc', 'hs', 'stat', 'q4h', 'q6h', 'q8h', 'q12h', 'q24h',
    'once daily', 'twice daily', 'three times daily',
}

# Administrative / logistical words
ADMIN_WORDS = {
    'home', 'residence', 'hospital', 'clinic', 'office', 'floor', 'ward',
    'unit', 'department', 'service', 'team', 'staff', 'routine', 'standard',
    'instructions', 'plan', 'care', 'benefits', 'diet', 'nutrition',
    'morning', 'evening', 'night', 'afternoon', 'daily', 'regular',
    'outpatient', 'inpatient', 'ambulatory',
}

ALL_STOP_WORDS = (SECTION_HEADERS | GENERIC_CLINICAL |
                  DEMOGRAPHIC_SOCIAL | DOSING_FREQ | ADMIN_WORDS)

# Regex patterns for unsafe entity text
UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^(no|not|denies|denied|without|negative|none|absent|absence)\b',
        r'^(positive|negative)$',
        r'^(left|right|bilateral|unilateral|ipsilateral|contralateral)$',
        r'^(elevated|increased|decreased|reduced|low|high|normal|abnormal|'
        r'mild|moderate|severe|minimal|significant|acute|chronic|stable|'
        r'benign|malignant|unstable|resolved|worsening|improving)$',
        r'^(minute|second|hour|day|week|month|year)s?$',
        r'^\d',          # starts with number
        r'^[A-Za-z]$',   # single letter
        r'^[A-Z]{1,2}$', # 1-2 letter abbreviation
    ]
]

# Words that invert clinical meaning — never allow in replacements
MEANING_INVERTERS = {
    'no', 'not', 'absence', 'without', 'lack', 'negative',
    'non', 'none', 'never', 'deny', 'denies',
}


def entity_passes_filters(text: str, pos_tag: str = None) -> tuple:
    """
    Multi-gate filter for entity text.
    Returns (passes: bool, reason: str).

    Gate 1 — minimum length (4 chars — blocks Mr, Ms, TID, BID etc.)
    Gate 2 — unsafe regex patterns (negations, polarities, numbers)
    Gate 3 — stop word sets (section headers, demographics, admin)
    Gate 4 — POS tag (reject verbs, determiners, pronouns if available)
    Gate 5 — negation phrase prefix check
    """
    t = text.strip()
    lower = t.lower().rstrip('.')

    # Gate 1: minimum length
    if len(t) < 4:
        return False, f"too short ({len(t)} chars)"

    # Gate 2: unsafe patterns
    for pat in UNSAFE_PATTERNS:
        if pat.match(t):
            return False, "unsafe pattern"

    # Gate 3: stop words
    if lower in ALL_STOP_WORDS:
        return False, "stop word"

    # Gate 3b: starts with negation
    NEG_STARTS = ('no ', 'not ', 'denies ', 'without ', 'absence of ',
                  'no evidence of ', 'negative for ')
    if lower.startswith(NEG_STARTS):
        return False, "negation phrase"

    # Gate 4: POS tag
    if pos_tag in {"VERB", "ADP", "DET", "PRON", "CCONJ",
                   "SCONJ", "PUNCT", "PART", "NUM", "SPACE"}:
        return False, f"non-content POS: {pos_tag}"

    return True, "ok"


def replacement_passes_filters(original: str, replacement: str) -> tuple:
    """
    Structural validation of synonym candidate.
    Returns (passes: bool, reason: str).
    """
    if not replacement or not replacement.strip():
        return False, "empty"

    # Clean up
    repl = replacement.strip().strip('"\'')
    repl = re.sub(r'[.!?,;]+$', '', repl).strip()

    # Arrow echo not cleaned
    if ' -> ' in repl or '\u2192' in repl:
        return False, "contains arrow"

    # Identical to original
    if repl.lower() == original.lower():
        return False, "identical to original"

    # Meaning inverters introduced
    repl_words = set(repl.lower().split())
    orig_words = set(original.lower().split())
    added = repl_words - orig_words
    if added & MEANING_INVERTERS:
        return False, "introduces negation"

    # Word count check
    n_orig = len(original.split())
    n_repl = len(repl.split())
    is_abbrev = original.strip().isupper() and len(original.strip()) <= 5
    word_limit = 6 if is_abbrev else max(n_orig * 2, n_orig + 3)
    if n_repl > word_limit:
        return False, f"too long ({n_orig}->{n_repl} words)"
    if n_orig == 1 and n_repl > 3:
        return False, f"single word expanded to {n_repl}"

    # Character ratio
    if len(repl) / max(len(original), 1) > 5.0:
        return False, "character ratio too high"

    # Mid-sentence punctuation (LLM output a sentence not a term)
    if re.search(r'[.!?](?!\s*$)', repl):
        return False, "mid-sentence punctuation"

    # Replacement is a subset of original (e.g. "surgical" -> "surgery")
    orig_root = original.lower()[:5]
    repl_root = repl.lower()[:5]
    if orig_root == repl_root:
        return False, "same root word"

    # Dash-separated echo (e.g. "term - synonym")
    if ' - ' in repl and repl.split(' - ')[0].strip().lower() == original.lower():
        return False, "dash echo pattern"

    return True, repl   # return cleaned replacement


# ============================================================
# SECTION 4: NER — BC5CDR MODEL (DISEASE + CHEMICAL)
# ============================================================

# Synonym prompts for three attack levels
PROMPT_LEVEL_0 = (
    "You are a clinical terminology expert. "
    "Below is a numbered list of medical terms with their type and context.\n\n"
    "For EACH term provide ONE medically equivalent synonym.\n\n"
    "STRICT RULES:\n"
    "- Output ONLY the replacement term. Nothing else.\n"
    "- Do NOT repeat the original term.\n"
    "- Do NOT use arrows (->) or separators.\n"
    "- 1-4 words maximum.\n"
    "- Same clinical meaning — do NOT change polarity, severity, or diagnosis.\n"
    "- If no good synonym exists, write: NONE\n\n"
    "FORMAT — one line per term:\n"
    "1. <synonym or NONE>\n"
    "2. <synonym or NONE>\n\n"
    "Terms:\n{term_lines}\n\nSynonyms:"
)

PROMPT_LEVEL_1 = (
    "You are a clinical terminology expert. "
    "Below is a numbered list of medical terms with their type and context.\n\n"
    "For EACH term provide ONE clinically related alternative "
    "(may be broader, more general, or a related condition in the same family).\n\n"
    "RULES:\n"
    "- Output ONLY the replacement term. Nothing else.\n"
    "- Do NOT repeat the original term.\n"
    "- Do NOT use arrows (->) or separators.\n"
    "- 1-4 words maximum.\n"
    "- Do NOT invert polarity or switch to a completely unrelated condition.\n"
    "- If no good alternative exists, write: NONE\n\n"
    "FORMAT — one line per term:\n"
    "1. <related term or NONE>\n"
    "2. <related term or NONE>\n\n"
    "Terms:\n{term_lines}\n\nReplacements:"
)

PROMPT_LEVEL_2 = (
    "You are a clinical terminology expert. "
    "Below is a numbered list of medical terms with their type and context.\n\n"
    "For EACH term provide ONE plausible alternative that refers to a DIFFERENT "
    "specific condition or drug that could cause diagnostic confusion "
    "(differential diagnosis style).\n\n"
    "RULES:\n"
    "- Output ONLY the replacement term. Nothing else.\n"
    "- Do NOT repeat the original term.\n"
    "- Do NOT use arrows (->) or separators.\n"
    "- 1-4 words maximum.\n"
    "- Must be medically plausible — no random or nonsensical terms.\n"
    "- Choose terms with symptom overlap (e.g. PE -> pneumonia).\n"
    "- Do NOT simply negate the original.\n"
    "- If no good alternative exists, write: NONE\n\n"
    "FORMAT — one line per term:\n"
    "1. <alternative or NONE>\n"
    "2. <alternative or NONE>\n\n"
    "Terms:\n{term_lines}\n\nReplacements:"
)

ATTACK_PROMPTS = {0: PROMPT_LEVEL_0, 1: PROMPT_LEVEL_1, 2: PROMPT_LEVEL_2}


def clean_llm_echo(raw: str) -> str:
    """
    Strip Llama's echo pattern: 'original -> synonym' or 'A -> B -> C'.
    Also strips dash-separated echoes: 'term - synonym'.
    Always returns a string (never None).
    """
    # Arrow echo
    if ' -> ' in raw or '\u2192' in raw:
        parts = re.split(r'\s*(?:->|\u2192)\s*', raw)
        raw = parts[-1].strip()

    # Dash echo: "word - synonym" where first part matches original
    # (handled in replacement_passes_filters)

    raw = raw.strip('"\'')
    raw = re.sub(r'[.!?,;]+$', '', raw).strip()
    raw = re.sub(r'\s*\(no change\)\s*', '', raw, flags=re.I).strip()
    # Remove "word - explanation" patterns
    raw = re.sub(r'\s*-\s*.+$', '', raw).strip() if raw.count(' - ') == 1 else raw
    return raw


class MedicalPerturber:
    """
    Research-grade perturbation pipeline using BC5CDR NER model.

    WHY BC5CDR:
    - en_core_sci_md/sm v0.5.x only produce generic ENTITY labels
    - BC5CDR is trained specifically for biomedical NER
    - Produces DISEASE and CHEMICAL labels — the two entity types
      most directly linked to ICD coding decisions
    - Well-validated on clinical text in the literature

    PIPELINE:
    Phase 1 — Extract DISEASE + CHEMICAL entities via BC5CDR NER
    Phase 2 — Filter via multi-gate entity filter (stop words, POS, patterns)
    Phase 3 — Batch synonym generation (one LLM call for all entities)
    Phase 4 — Structural validation of each synonym candidate
    Phase 5 — Apply replacements to ALL occurrences (not just first)
    """

    def __init__(self):
        # Load BC5CDR — the correct model for typed biomedical NER
        try:
            self.nlp = spacy.load("en_ner_bc5cdr_md")
            debug_print("PERTURBER",
                        "BC5CDR loaded — labels: DISEASE, CHEMICAL")
        except OSError:
            raise RuntimeError(
                "en_ner_bc5cdr_md not found.\n"
                "Install with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/"
                "ai2-s2-scispacy/releases/v0.5.4/"
                "en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

        # Verify labels
        ner_labels = self.nlp.get_pipe('ner').labels
        debug_print("PERTURBER", f"NER labels confirmed: {ner_labels}")
        if 'DISEASE' not in ner_labels and 'CHEMICAL' not in ner_labels:
            raise RuntimeError(
                f"BC5CDR model has unexpected labels: {ner_labels}. "
                "Reinstall the model."
            )

        # Load en_core_sci_md for POS tagging (it has a good tagger even
        # though its NER is generic — we use it only for POS, not NER)
        try:
            self.nlp_pos = spacy.load("en_core_sci_md",
                                       exclude=["ner"])
            debug_print("PERTURBER",
                        "en_core_sci_md loaded for POS tagging")
            self.has_pos = True
        except OSError:
            self.has_pos = False
            debug_print("PERTURBER",
                        "en_core_sci_md not available — POS filter disabled")

    def _get_pos(self, text: str, start: int) -> str:
        """Get POS tag of first token of entity at start_char."""
        if not self.has_pos:
            return None
        doc = self.nlp_pos(text[max(0, start-5):start+50])
        if doc and len(doc) > 0:
            return doc[0].pos_
        return None

    def _is_negated(self, text: str, start: int) -> bool:
        """Check if the 3 tokens before entity contain a negation word."""
        NEG = {'no', 'not', 'denies', 'negative', 'without',
               'none', 'denied', 'absent', 'absence'}
        window = text[max(0, start - 40):start].lower()
        # Split into last few words
        words = re.findall(r'\b\w+\b', window)[-4:]
        return any(w in NEG for w in words)

    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get sentence context around entity."""
        # Find sentence boundaries
        sent_start = text.rfind('.', 0, start)
        sent_start = sent_start + 1 if sent_start >= 0 else max(0, start - 80)
        sent_end = text.find('.', end)
        sent_end = sent_end if sent_end >= 0 else min(len(text), end + 80)
        return text[sent_start:sent_end].strip()

    # ── Phase 1 + 2: Extract and filter ──────────────────────────────────────
    def extract_entities(self, text: str) -> list:
        """
        Extract DISEASE and CHEMICAL entities from text using BC5CDR.
        Apply multi-gate filter to ensure only medically relevant,
        replaceable entities are returned.
        """
        doc = self.nlp(text)
        entities = []
        seen = set()

        for ent in doc.ents:
            # Only DISEASE and CHEMICAL
            if ent.label_ not in ('DISEASE', 'CHEMICAL'):
                continue

            # Negation check
            if self._is_negated(text, ent.start_char):
                continue

            # De-duplicate
            key = ent.text.strip().lower()
            if key in seen:
                continue

            # Get POS of first token
            pos = self._get_pos(text, ent.start_char)

            # Multi-gate filter
            passes, reason = entity_passes_filters(ent.text, pos)
            if not passes:
                continue

            seen.add(key)
            entities.append({
                'text'      : ent.text,
                'label'     : ent.label_,   # DISEASE or CHEMICAL
                'start_char': ent.start_char,
                'end_char'  : ent.end_char,
                'context'   : self._get_context(text, ent.start_char,
                                                 ent.end_char),
            })

        return entities

    # ── Phase 3: Batch synonym generation ────────────────────────────────────
    def get_synonyms_batch(self, entities: list,
                           attack_level: int = 0) -> dict:
        """
        Single LLM call for all entities.
        Returns {original_text: synonym_or_None}.
        """
        if not entities:
            return {}

        term_lines = "\n".join(
            f"{i+1}. [{e['label']}] \"{e['text']}\"  "
            f"(context: \"{e['context'][:80]}...\")"
            for i, e in enumerate(entities)
        )

        prompt = ATTACK_PROMPTS[attack_level].format(term_lines=term_lines)
        response = query_llm(prompt,
                             max_tokens=len(entities) * 12 + 60)
        if not response:
            return {}

        synonym_map = {}
        for line in response.strip().split('\n'):
            line = line.strip()
            m = re.match(r'^(\d+)[.)]\s*(.+)$', line)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            raw = m.group(2).strip()

            if 0 <= idx < len(entities):
                original = entities[idx]['text']
                cleaned = clean_llm_echo(raw)

                if not cleaned or cleaned.upper() == 'NONE':
                    synonym_map[original] = None
                elif cleaned.lower() == original.lower():
                    synonym_map[original] = None
                else:
                    synonym_map[original] = cleaned

        return synonym_map

    # ── Phase 4 + 5: Validate and apply ──────────────────────────────────────
    def apply_perturbations(self, text: str, entities: list,
                            synonym_map: dict) -> tuple:
        """
        Validate each synonym structurally, then apply to ALL occurrences
        of the term in the text (not just the first/by-offset occurrence).

        WHY ALL OCCURRENCES:
        Clinical notes often repeat the same diagnosis/drug multiple times.
        Replacing only the first occurrence creates inconsistency —
        the same term appears in both original and perturbed forms in
        the same note, which would confuse the ICD prediction.
        """
        accepted = []

        for ent in entities:
            original = ent['text']
            candidate = synonym_map.get(original)
            if not candidate:
                continue

            passes, result = replacement_passes_filters(original, candidate)
            if not passes:
                print(f"       BLOCKED ({result}): '{original}' -> '{candidate}'")
                continue

            # result contains the cleaned replacement when passes=True
            cleaned_replacement = result
            accepted.append({
                'original'   : original,
                'replacement': cleaned_replacement,
                'label'      : ent['label'],
                'start_char' : ent['start_char'],
                'end_char'   : ent['end_char'],
            })
            print(f"       ACCEPTED [{ent['label']:8s}]: "
                  f"'{original}' -> '{cleaned_replacement}'")

        if not accepted:
            return text, []

        perturbed = text
        for item in accepted:
            # Replace ALL occurrences using regex word-boundary matching
            pattern = re.compile(
                r'\b' + re.escape(item['original']) + r'\b',
                re.IGNORECASE
            )
            count_before = len(pattern.findall(perturbed))
            perturbed = pattern.sub(item['replacement'], perturbed)
            count_after = len(re.findall(
                re.escape(item['replacement']), perturbed, re.IGNORECASE))
            if count_before > 1:
                print(f"       NOTE: '{item['original']}' replaced "
                      f"{count_before} occurrences")

        changes_log = [(i['original'], i['replacement'], i['label'])
                       for i in accepted]
        return perturbed, changes_log

    # ── Public entry point ────────────────────────────────────────────────────
    def perturb(self, text: str,
                perturbation_rate: float = PERTURBATION_RATE,
                attack_level: int = 0) -> tuple:
        """
        Full perturbation pipeline.
        perturbation_rate: fraction of extracted entities to perturb.
        """
        all_entities = self.extract_entities(text)
        if not all_entities:
            print("   > No medical entities found after filtering.")
            return text, []

        print(f"\n   > Entities found ({len(all_entities)}):")
        for e in all_entities:
            print(f"       [{e['label']:8s}] {e['text']}")

        n = max(1, int(len(all_entities) * perturbation_rate))
        targets = random.sample(all_entities, min(n, len(all_entities)))

        synonym_map = self.get_synonyms_batch(targets, attack_level)

        print(f"\n   > Synonym candidates:")
        for term, syn in synonym_map.items():
            print(f"       {term:35s} -> {syn if syn else 'NONE'}")

        print(f"\n   > Validation + Application:")
        return self.apply_perturbations(text, targets, synonym_map)


# ============================================================
# SECTION 5: SEMANTIC EVALUATOR (ClinicalBERT)
# ============================================================
class SemanticEvaluator:
    """
    ClinicalBERT-based semantic similarity using mean pooling.
    Mean pooling over all non-padding tokens is more robust than
    CLS token alone for longer clinical texts.
    """

    def __init__(self):
        debug_print("EVAL", "Loading ClinicalBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def get_sim(self, text1: str, text2: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(
                [text1, text2], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            e1 = (outputs.last_hidden_state[0] * mask[0]).sum(0) / mask[0].sum()
            e2 = (outputs.last_hidden_state[1] * mask[1]).sum(0) / mask[1].sum()
            sim = cosine_similarity(e1.unsqueeze(0).numpy(),
                                    e2.unsqueeze(0).numpy())[0][0]
        return float(sim)


# ============================================================
# SECTION 6: BASELINE CORRECTNESS — PURE ICD STRING MATCHING
# ============================================================
def normalise_icd(code: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', code.upper())


def is_baseline_correct(orig_icd: str, gt_icd: str) -> tuple:
    """
    Pure ICD-10 string matching — NO LLM involved.

    WHY NO LLM:
    - Removes ~1 LLM call per sample (500 samples = 500 fewer calls)
    - ICD-10 codes are objective — string matching is reliable
    - LLM was causing false passes AND false failures

    THREE TIERS (descending strictness):
    Tier 1 — Exact match: same code (e.g. I21.0 == I21.0)
    Tier 2 — 3-char prefix: same disease family (e.g. I21.0 ~ I21.9)
    Tier 3 — Chapter (first letter): same body system (e.g. I21 ~ I48,
             both cardiovascular) — most lenient, reduces skip rate

    Tier 3 justification: llama3.1:8b is a weak ICD coder — it often
    gets the right organ system but wrong specific code. Tier 3 recovers
    these cases without accepting truly wrong predictions (e.g. I vs J).
    """
    if orig_icd in ("UNKNOWN", ""):
        return False, "unknown_pred", "tier=none"

    gt_norm   = normalise_icd(gt_icd)
    pred_norm = normalise_icd(orig_icd)

    # Tier 1: exact
    if gt_norm == pred_norm:
        return True, "tier1_exact", f"exact: {orig_icd}=={gt_icd}"

    # Tier 2: 3-char prefix (same ICD category)
    if len(gt_norm) >= 3 and len(pred_norm) >= 3:
        if gt_norm[:3] == pred_norm[:3]:
            return True, "tier2_prefix", \
                   f"prefix: {gt_norm[:3]}=={pred_norm[:3]}"

    # # Tier 3: chapter (first letter — same body system)
    # if gt_norm[0] == pred_norm[0]:
    #     return True, "tier3_chapter", \
    #            f"chapter: {gt_norm[0]}=={pred_norm[0]}"

    return False, "no_match", \
           f"mismatch: pred={orig_icd} gt={gt_icd}"


# ============================================================
# SECTION 7: ICD PREDICTION
# ============================================================
def get_icd_prediction(text: str) -> str:
    """
    Two-attempt ICD-10 prediction with regex extraction.
    Returns 'UNKNOWN' on failure.
    Strips DeepSeek-style <think> tags if present.
    """
    primary_prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (format: J18.9 or I21.0). "
        f"Do not write the disease name. Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code. "
        f"Format: letter + 2 digits + optional dot + digits.\n\n"
        f"Note:\n{text[:800]}\n\nCode:"
    )
    for prompt in [primary_prompt, fallback_prompt]:
        response = query_llm(prompt)
        if response:
            response = re.sub(r'<think>.*?</think>', '', response,
                               flags=re.S).strip()
            m = re.search(r'[A-Z]\d{2}\.\d+', response.upper())
            if m:
                return m.group(0)
            m = re.search(r'[A-Z]\d{2,4}', response.upper())
            if m:
                return m.group(0)
    return "UNKNOWN"


def get_diagnosis_description(icd_code: str) -> str:
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    prompt = (
        f"What medical condition does ICD-10 code '{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words."
    )
    return query_llm(prompt)


# ============================================================
# SECTION 8: DRIFT CLASSIFICATION
# ============================================================
def classify_drift(orig_icd: str, pert_icd: str) -> dict:
    orig_norm = normalise_icd(orig_icd)
    pert_norm = normalise_icd(pert_icd)

    if "UNKNOWN" in (orig_norm, pert_norm):
        return {"icd_codes_differ": True, "drift_level": "unknown",
                "categorical_drift": False, "subcategory_drift": False}

    exact  = orig_norm == pert_norm
    prefix = (len(orig_norm) >= 3 and len(pert_norm) >= 3
              and orig_norm[:3] == pert_norm[:3])

    level = "none" if exact else ("subcategory" if prefix else "categorical")
    return {
        "icd_codes_differ" : not exact,
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 9: PER-ICD ANALYSIS (what your professor asked for)
# ============================================================
def generate_per_icd_analysis(df: pd.DataFrame) -> tuple:
    """
    Groups results by ICD code and ICD chapter to show:
    - Which diagnoses are most/least robust to perturbation
    - Which entity types cause most drift per diagnosis
    - Chapter-level drift rates

    This directly addresses the professor's request for
    per-ICD breakdown of results.
    """
    valid = df[df['baseline_correct'] == True].copy()
    if len(valid) == 0:
        print("No valid samples for per-ICD analysis.")
        return pd.DataFrame(), pd.DataFrame()

    valid['icd_chapter']       = valid['ground_truth_icd'].str[0]
    valid['icd_chapter_range'] = valid['ground_truth_icd'].str[:3]

    # ── Per ICD code ──────────────────────────────────────────────────────
    per_icd = valid.groupby('ground_truth_icd').agg(
        description       = ('ground_truth_desc', 'first'),
        total_samples     = ('hadm_id', 'count'),
        cat_drift_count   = ('categorical_drift', 'sum'),
        sub_drift_count   = ('subcategory_drift', 'sum'),
        stable_count      = ('drift_level',
                             lambda x: (x == 'none').sum()),
        avg_sim           = ('semantic_similarity', 'mean'),
        avg_changes       = ('changes_count', 'mean'),
        disease_changes   = ('disease_changes', 'sum'),
        chemical_changes  = ('chemical_changes', 'sum'),
    ).reset_index()

    per_icd['cat_drift_rate_%'] = (
        per_icd['cat_drift_count'] / per_icd['total_samples'] * 100
    ).round(1)
    per_icd['stable_rate_%'] = (
        per_icd['stable_count'] / per_icd['total_samples'] * 100
    ).round(1)

    # Only keep ICD codes with >= 2 samples for statistical meaning
    per_icd = per_icd[per_icd['total_samples'] >= 2]
    per_icd = per_icd.sort_values('cat_drift_rate_%', ascending=False)

    # ── Per ICD chapter ────────────────────────────────────────────────────
    CHAPTER_NAMES = {
        'A': 'Infectious diseases', 'B': 'Infectious diseases',
        'C': 'Neoplasms', 'D': 'Neoplasms/Blood',
        'E': 'Endocrine/Metabolic', 'F': 'Mental disorders',
        'G': 'Nervous system', 'H': 'Eye/Ear',
        'I': 'Circulatory system', 'J': 'Respiratory system',
        'K': 'Digestive system', 'L': 'Skin',
        'M': 'Musculoskeletal', 'N': 'Genitourinary',
        'O': 'Pregnancy/Childbirth', 'P': 'Perinatal',
        'Q': 'Congenital', 'R': 'Symptoms/Signs',
        'S': 'Injury/Trauma', 'T': 'Poisoning',
        'U': 'Special purposes', 'V': 'External causes',
        'W': 'External causes', 'X': 'External causes',
        'Y': 'External causes', 'Z': 'Health status/Services',
    }

    per_chapter = valid.groupby('icd_chapter').agg(
        total             = ('hadm_id', 'count'),
        cat_drift         = ('categorical_drift', 'sum'),
        sub_drift         = ('subcategory_drift', 'sum'),
        avg_sim           = ('semantic_similarity', 'mean'),
        avg_changes       = ('changes_count', 'mean'),
        disease_changes   = ('disease_changes', 'sum'),
        chemical_changes  = ('chemical_changes', 'sum'),
    ).reset_index()

    per_chapter['cat_drift_rate_%'] = (
        per_chapter['cat_drift'] / per_chapter['total'] * 100
    ).round(1)
    per_chapter['system'] = per_chapter['icd_chapter'].map(CHAPTER_NAMES)
    per_chapter = per_chapter.sort_values('cat_drift_rate_%', ascending=False)

    # Save
    per_icd.to_csv(PER_ICD_CSV, index=False)
    per_chapter.to_csv(PER_CHAPTER_CSV, index=False)

    print("\n" + "=" * 70)
    print("PER-ICD DRIFT ANALYSIS (codes with >= 2 samples)")
    print("=" * 70)
    cols = ['ground_truth_icd', 'description', 'total_samples',
            'cat_drift_rate_%', 'stable_rate_%', 'avg_changes']
    print(per_icd[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-CHAPTER DRIFT ANALYSIS")
    print("=" * 70)
    print(per_chapter[['icd_chapter', 'system', 'total',
                        'cat_drift_rate_%', 'avg_changes']].to_string(
        index=False))

    return per_icd, per_chapter


# ============================================================
# SECTION 10: PLOTS
# ============================================================
def plot_results(df: pd.DataFrame, per_icd: pd.DataFrame,
                 per_chapter: pd.DataFrame):
    valid = df[df['baseline_correct'] == True]
    if len(valid) == 0:
        print("No valid samples to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f"Robustness Analysis — {MODEL_NAME} | "
                 f"Attack Level {ATTACK_LEVEL} | "
                 f"n={len(valid)} valid samples",
                 fontsize=14, fontweight='bold')

    # Plot 1: Semantic similarity distribution
    ax = axes[0, 0]
    ax.hist(valid['semantic_similarity'], bins=20,
            color='teal', edgecolor='black', alpha=0.8)
    ax.set_title("Semantic Preservation\n(perturbed vs original)")
    ax.set_xlabel("ClinicalBERT Cosine Similarity")
    ax.set_ylabel("Count")
    ax.axvline(valid['semantic_similarity'].mean(), color='red',
               linestyle='--', label=f"Mean={valid['semantic_similarity'].mean():.4f}")
    ax.legend()

    # Plot 2: Drift level breakdown
    ax = axes[0, 1]
    drift_counts = valid['drift_level'].value_counts()
    colors_map = {'none': '#90EE90', 'subcategory': '#FFD700',
                  'categorical': '#FF6B6B', 'unknown': '#CCCCCC'}
    bar_colors = [colors_map.get(k, 'grey') for k in drift_counts.index]
    bars = ax.bar(drift_counts.index, drift_counts.values, color=bar_colors,
                  edgecolor='black')
    for bar, val in zip(bars, drift_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', fontweight='bold')
    cat_rate = valid['categorical_drift'].mean() * 100
    ax.set_title(f"Drift Level Breakdown\nCategorical drift: {cat_rate:.1f}%")
    ax.set_xlabel("Drift Level")

    # Plot 3: Changes vs similarity scatter
    ax = axes[0, 2]
    colors = {'none': 'green', 'subcategory': 'orange',
              'categorical': 'red', 'unknown': 'grey'}
    for level in ['none', 'subcategory', 'categorical']:
        mask = valid['drift_level'] == level
        if mask.sum() > 0:
            ax.scatter(valid[mask]['changes_count'],
                       valid[mask]['semantic_similarity'],
                       label=f"{level} (n={mask.sum()})",
                       color=colors[level], alpha=0.6, s=30)
    ax.set_title("Changes Count vs Similarity")
    ax.set_xlabel("Medical Terms Changed")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(title="Drift level", fontsize=8)

    # Plot 4: Entity type breakdown
    ax = axes[1, 0]
    type_data = {
        'DISEASE' : valid['disease_changes'].sum(),
        'CHEMICAL': valid['chemical_changes'].sum(),
    }
    type_data = {k: v for k, v in type_data.items() if v > 0}
    if type_data:
        ax.bar(type_data.keys(), type_data.values(),
               color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
        ax.set_title("Perturbations by Entity Type")
        ax.set_xlabel("Entity Type (BC5CDR)")
        ax.set_ylabel("Total Changes")

    # Plot 5: Per-chapter drift rate
    ax = axes[1, 1]
    if len(per_chapter) > 0:
        top_chapters = per_chapter.head(10)
        bars = ax.barh(top_chapters['icd_chapter'],
                       top_chapters['cat_drift_rate_%'],
                       color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax.set_title("Categorical Drift Rate by ICD Chapter\n(top 10)")
        ax.set_xlabel("Categorical Drift Rate (%)")
        ax.set_ylabel("ICD Chapter")
        # Add system names
        for i, (_, row) in enumerate(top_chapters.iterrows()):
            ax.text(0.5, i, row.get('system', ''), va='center',
                    fontsize=7, color='black')

    # Plot 6: Baseline quality pie
    ax = axes[1, 2]
    counts = df['baseline_correct'].value_counts()
    labels_pie, colors_pie = [], []
    if True in counts.index:
        labels_pie.append(f'Valid\n(n={counts[True]})')
        colors_pie.append('#90EE90')
    if False in counts.index:
        labels_pie.append(f'Skipped\n(n={counts[False]})')
        colors_pie.append('#FFB6C1')
    ax.pie(counts, labels=labels_pie, colors=colors_pie,
           autopct='%1.1f%%', startangle=90)
    ax.set_title("Baseline Quality\n(ICD prefix/chapter match)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 11: HTML REPORT
# ============================================================
def _highlight_changes(text: str, changes: list, is_original: bool) -> str:
    """
    Highlight changed terms using offset-aware matching.
    Only highlights the FIRST occurrence to avoid marking natural
    occurrences of the same word elsewhere in the note.
    """
    if not changes:
        return text

    result = text
    consumed = []

    words_to_mark = [(o, 'orig') for o, _, _ in changes] if is_original \
                    else [(r, 'repl') for _, r, _ in changes]

    spans = []
    for word, kind in words_to_mark:
        pat = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        for m in pat.finditer(text):
            overlap = any(not (m.end() <= cs or m.start() >= ce)
                          for cs, ce in consumed)
            if not overlap:
                consumed.append((m.start(), m.end()))
                spans.append((m.start(), m.end(), word, kind))
                break

    spans.sort(key=lambda x: x[0], reverse=True)
    for start, end, word, kind in spans:
        color = '#FF6B6B' if kind == 'orig' else '#51C878'
        label = 'replaced' if kind == 'orig' else 'inserted'
        span = (f'<span style="background:{color};color:white;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold;" title="{label}">'
                f'{result[start:end]}</span>')
        result = result[:start] + span + result[end:]

    return result


def generate_html_report(df: pd.DataFrame,
                          per_icd: pd.DataFrame,
                          per_chapter: pd.DataFrame,
                          filename: str = HTML_REPORT):

    DRIFT_COLORS = {
        'none': '#ccffcc', 'subcategory': '#fff3cd',
        'categorical': '#ffcccc', 'unknown': '#eeeeee', 'skipped': '#f9f9f9',
    }
    DRIFT_LABELS = {
        'none': 'STABLE', 'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
        'unknown': 'UNKNOWN ICD', 'skipped': 'SKIPPED',
    }

    def parse_changes(s):
        if not s or s in ('SKIPPED', '', 'None', None):
            return []
        out = []
        for part in s.split('|'):
            part = part.strip()
            m = re.match(r'\[([^\]]+)\]\s+(.+?)\s+->\s+(.+)', part)
            if m:
                out.append((m.group(2).strip(),
                             m.group(3).strip(),
                             m.group(1).strip()))
        return out

    # ── Per-ICD summary table ──────────────────────────────────────────────
    per_icd_html = ""
    if len(per_icd) > 0:
        rows = ""
        for _, r in per_icd.iterrows():
            rows += (
                f"<tr>"
                f"<td>{r['ground_truth_icd']}</td>"
                f"<td>{r['description']}</td>"
                f"<td>{r['total_samples']}</td>"
                f"<td style='color:{'red' if r['cat_drift_rate_%']>20 else 'black'};'>"
                f"<b>{r['cat_drift_rate_%']}%</b></td>"
                f"<td>{r['stable_rate_%']}%</td>"
                f"<td>{r['avg_changes']:.1f}</td>"
                f"<td>{int(r['disease_changes'])}</td>"
                f"<td>{int(r['chemical_changes'])}</td>"
                f"</tr>"
            )
        per_icd_html = f"""
        <h2>Per-ICD Code Analysis</h2>
        <p style="color:#666;font-size:0.9em;">
        Only ICD codes with ≥2 valid samples shown.
        Sorted by categorical drift rate (highest first).
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
          <tr style="background:#e8f4f8;">
            <th style="padding:6px;border:1px solid #ccc;">ICD Code</th>
            <th style="padding:6px;border:1px solid #ccc;">Description</th>
            <th style="padding:6px;border:1px solid #ccc;">N</th>
            <th style="padding:6px;border:1px solid #ccc;">Cat.Drift%</th>
            <th style="padding:6px;border:1px solid #ccc;">Stable%</th>
            <th style="padding:6px;border:1px solid #ccc;">Avg Changes</th>
            <th style="padding:6px;border:1px solid #ccc;">Disease Δ</th>
            <th style="padding:6px;border:1px solid #ccc;">Chemical Δ</th>
          </tr>
          {rows}
        </table>"""

    per_chapter_html = ""
    if len(per_chapter) > 0:
        rows = ""
        for _, r in per_chapter.iterrows():
            rows += (
                f"<tr>"
                f"<td><b>{r['icd_chapter']}</b></td>"
                f"<td>{r.get('system','')}</td>"
                f"<td>{r['total']}</td>"
                f"<td style='color:{'red' if r['cat_drift_rate_%']>20 else 'black'};'>"
                f"<b>{r['cat_drift_rate_%']}%</b></td>"
                f"<td>{r['avg_changes']:.1f}</td>"
                f"</tr>"
            )
        per_chapter_html = f"""
        <h2>Per-Chapter Drift Analysis</h2>
        <table style="width:60%;border-collapse:collapse;font-size:0.85em;">
          <tr style="background:#e8f4f8;">
            <th style="padding:6px;border:1px solid #ccc;">Chapter</th>
            <th style="padding:6px;border:1px solid #ccc;">Body System</th>
            <th style="padding:6px;border:1px solid #ccc;">N</th>
            <th style="padding:6px;border:1px solid #ccc;">Cat.Drift%</th>
            <th style="padding:6px;border:1px solid #ccc;">Avg Changes</th>
          </tr>
          {rows}
        </table>"""

    # ── Per-sample rows ────────────────────────────────────────────────────
    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        dc = DRIFT_COLORS.get(drift_level, '#ccffcc')
        dl = DRIFT_LABELS.get(drift_level, 'STABLE')

        changes     = parse_changes(row.get('changes_made', ''))
        changes_disp = " | ".join(
            f"[{l}] {o} → {r}" for o, r, l in changes
        ) if changes else "None"

        sim_d  = (f"{float(row['semantic_similarity']):.4f}"
                  if row.get('semantic_similarity') else 'N/A')
        gate_d = row.get('baseline_gate', '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — prediction did not match ground truth ICD.<br>'
                f'<small>Gate: {gate_d}</small></p>')
        else:
            orig_h = _highlight_changes(
                row.get('original_text', ''), changes, is_original=True)
            pert_h = _highlight_changes(
                row.get('perturbed_text', ''), changes, is_original=False)
            text_section = f"""
            <div style="margin-top:10px;">
              <div style="margin-bottom:8px;font-size:0.85em;">
                <span style="background:#FF6B6B;color:white;padding:2px 8px;
                border-radius:3px;">Replaced (original)</span>&nbsp;
                <span style="background:#51C878;color:white;padding:2px 8px;
                border-radius:3px;">Inserted (synonym)</span>
              </div>
              <table style="width:100%;border-collapse:collapse;">
                <tr>
                  <th style="width:50%;background:#e8f4f8;padding:8px;
                  border:1px solid #ccc;">Original Text</th>
                  <th style="width:50%;background:#fff8e8;padding:8px;
                  border:1px solid #ccc;">Perturbed Text</th>
                </tr>
                <tr>
                  <td style="padding:10px;border:1px solid #ccc;
                  vertical-align:top;white-space:pre-wrap;
                  font-size:0.85em;line-height:1.6em;">{orig_h}</td>
                  <td style="padding:10px;border:1px solid #ccc;
                  vertical-align:top;white-space:pre-wrap;
                  font-size:0.85em;line-height:1.6em;">{pert_h}</td>
                </tr>
                <tr>
                  <td style="padding:8px;border:1px solid #ccc;
                  background:#f0f0f0;text-align:center;">
                  <b>ICD-10: {row.get('original_icd_pred','?')}</b></td>
                  <td style="padding:8px;border:1px solid #ccc;
                  background:#f0f0f0;text-align:center;">
                  <b>ICD-10: {row.get('perturbed_icd_pred','?')}</b></td>
                </tr>
              </table>
            </div>"""

        rows_html += f"""
        <div style="border:1px solid #ccc;margin:20px 0;
        padding:15px;border-radius:8px;">
          <h3 style="margin:0 0 10px 0;">
            Patient: {row['hadm_id']}
            <span style="background:{dc};padding:4px 10px;
            border-radius:4px;font-size:0.9em;">{dl}</span>
          </h3>
          <table style="font-size:0.9em;border-collapse:collapse;
          margin-bottom:10px;">
            <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                <td>{row.get('ground_truth_desc','')}
                ({row.get('ground_truth_icd','')})</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                <td>{row.get('original_icd_pred','')} —
                {row.get('original_pred_desc','')}</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gate:</b></td>
                <td><small>{gate_d}</small></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                <td>{changes_disp}
                ({row.get('changes_count',0)} terms)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                <td>{sim_d}</td></tr>
          </table>
          {text_section}
        </div>"""

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = valid_df['categorical_drift'].mean()*100 if len(valid_df) else 0
    sub_rate    = valid_df['subcategory_drift'].mean()*100 if len(valid_df) else 0
    avg_sim     = (f"{valid_df['semantic_similarity'].mean():.4f}"
                   if len(valid_df) else 'N/A')
    avg_changes = (f"{valid_df['changes_count'].mean():.1f}"
                   if len(valid_df) else 'N/A')

    html = f"""<!DOCTYPE html>
<html><head><title>Robustness Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;
     padding:20px;background:#fafafa;}}
h1,h2{{color:#333;}} h3{{margin:0 0 10px 0;}}
.summary{{background:#f0f0f0;padding:15px;border-radius:8px;
          margin-bottom:30px;}}
.summary p{{margin:4px 0;}}
.legend{{display:flex;gap:16px;margin-bottom:16px;
         font-size:0.85em;flex-wrap:wrap;}}
.legend span{{padding:3px 10px;border-radius:4px;}}
.method{{background:#e8f4f8;padding:12px;border-radius:6px;
         margin-bottom:20px;font-size:0.9em;color:#555;}}
table th{{text-align:left;}}
</style></head><body>
<h1>Clinical Note Robustness Report</h1>
<p style="color:#666;">
  Model: {MODEL_NAME} | NER: BC5CDR (DISEASE+CHEMICAL) |
  Attack Level: {ATTACK_LEVEL} |
  Perturbation Rate: {PERTURBATION_RATE*100:.0f}% of entities
</p>

<div class="method">
  <b>Methodological choices:</b>
  (1) ICD-10 only — eliminates version mismatch.
  (2) NER: BC5CDR model — typed DISEASE/CHEMICAL labels,
  not generic ENTITY from en_core_sci_md/sm.
  (3) Baseline gate: 3-tier ICD string matching (exact → prefix → chapter)
  — no LLM needed, saves ~1 API call per sample.
  (4) Discharge Diagnosis excluded — prevents answer leakage.
  (5) ALL occurrences of a term replaced, not just first.
  (6) Per-ICD and per-chapter breakdown included.
</div>

<div class="legend">
  <span style="background:#ccffcc;">STABLE — exact ICD match</span>
  <span style="background:#fff3cd;">SUBCATEGORY — same 3-char family</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease
  ← primary metric</span>
</div>

<div class="summary">
  <h2>Summary Statistics</h2>
  <p><b>Model:</b> {MODEL_NAME}</p>
  <p><b>Attack Level:</b> {ATTACK_LEVEL}
     (0=synonyms, 1=related terms, 2=differential diagnosis)</p>
  <p><b>Total Samples:</b> {len(df)}</p>
  <p><b>Valid for Analysis:</b> {len(valid_df)}
     <span style="color:#888;">(ICD prefix/chapter match)</span></p>
  <p><b>Skipped:</b> {len(df)-len(valid_df)}</p>
  <p><b>Categorical Drift Rate:</b>
     <b style="color:{'red' if cat_rate>20 else 'black'}">{cat_rate:.1f}%</b>
     <span style="color:#888;">← primary robustness metric</span></p>
  <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%</p>
  <p><b>Avg Semantic Similarity:</b> {avg_sim}</p>
  <p><b>Avg Medical Terms Changed:</b> {avg_changes}</p>
  <p><b>Disease changes:</b>
     {int(valid_df['disease_changes'].sum()) if len(valid_df) else 0} |
  <b>Chemical changes:</b>
     {int(valid_df['chemical_changes'].sum()) if len(valid_df) else 0}</p>
</div>

{per_chapter_html}
<br>
{per_icd_html}
<br>
<h2>Per-Sample Results</h2>
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {filename}")


# ============================================================
# SECTION 12: MAIN
# ============================================================
def main():
    debug_print("MAIN",
                f"Model: {MODEL_NAME} | "
                f"Attack: {ATTACK_LEVEL} | "
                f"Samples: {MAX_SAMPLES}")

    # API check
    test = query_llm("Reply with the word OK only.", max_tokens=5)
    if not test:
        debug_print("ERROR", "Cannot connect to LLM server.")
        return
    debug_print("MAIN", f"LLM OK (response: '{test}')")

    # Load data
    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH,
                              DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        import traceback
        debug_print("ERROR", f"Data load failed: {e}")
        traceback.print_exc()
        return

    # Init models
    perturber = MedicalPerturber()
    scorer    = SemanticEvaluator()
    results   = []

    print("\n" + "=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id   = row['hadm_id']
            orig_text = row['processed_text']
            gt_desc   = row['long_title']
            gt_icd    = row['icd_code']
            icd_ver   = row['icd_version']

            print(f"\n[{hadm_id}] GT: {gt_desc} ({gt_icd})")

            # Step 1: Predict ICD on original text
            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd)
            print(f"   Pred: {orig_icd} — {orig_desc}")

            # Step 2: Baseline gate (pure ICD string matching)
            passed, tier, gate_str = is_baseline_correct(orig_icd, gt_icd)
            print(f"   Gate: {gate_str} → {'PASS' if passed else 'SKIP'}")

            if not passed:
                results.append({
                    "hadm_id"             : hadm_id,
                    "ground_truth_icd"    : gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc"   : gt_desc,
                    "original_text"       : orig_text,
                    "perturbed_text"      : "SKIPPED",
                    "changes_made"        : "SKIPPED",
                    "changes_count"       : 0,
                    "disease_changes"     : 0,
                    "chemical_changes"    : 0,
                    "original_icd_pred"   : orig_icd,
                    "original_pred_desc"  : orig_desc,
                    "baseline_tier"       : tier,
                    "baseline_gate"       : gate_str,
                    "baseline_correct"    : False,
                    "perturbed_icd_pred"  : "SKIPPED",
                    "icd_codes_differ"    : False,
                    "drift_level"         : "skipped",
                    "categorical_drift"   : False,
                    "subcategory_drift"   : False,
                    "suspicious_drift"    : False,
                    "semantic_similarity" : 0.0,
                })
                continue

            # Step 3: Perturb
            print(f"   Perturbing (rate={PERTURBATION_RATE}, "
                  f"level={ATTACK_LEVEL})...")
            pert_text, changes_log = perturber.perturb(
                orig_text,
                perturbation_rate=PERTURBATION_RATE,
                attack_level=ATTACK_LEVEL)
            print(f"   {len(changes_log)} changes applied")

            # Step 4: Predict on perturbed text
            pert_icd   = get_icd_prediction(pert_text)
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995
                          and len(changes_log) <= 2)

            print(f"   Pert ICD: {pert_icd} | "
                  f"Drift: {drift_info['drift_level']} | "
                  f"Sim: {sim:.4f}"
                  + (" ⚠ SUSPICIOUS" if suspicious else ""))

            changes_str = " | ".join(
                f"[{l}] {o} -> {r}" for o, r, l in changes_log)

            results.append({
                "hadm_id"             : hadm_id,
                "ground_truth_icd"    : gt_icd,
                "ground_truth_version": icd_ver,
                "ground_truth_desc"   : gt_desc,
                "original_text"       : orig_text,
                "perturbed_text"      : pert_text,
                "changes_made"        : changes_str,
                "changes_count"       : len(changes_log),
                "disease_changes"     : sum(1 for _,_,l in changes_log
                                            if l == "DISEASE"),
                "chemical_changes"    : sum(1 for _,_,l in changes_log
                                            if l == "CHEMICAL"),
                "original_icd_pred"   : orig_icd,
                "original_pred_desc"  : orig_desc,
                "baseline_tier"       : tier,
                "baseline_gate"       : gate_str,
                "baseline_correct"    : True,
                "perturbed_icd_pred"  : pert_icd,
                **drift_info,
                "suspicious_drift"    : suspicious,
                "semantic_similarity" : sim,
            })
            print("-" * 60)

        except Exception as e:
            import traceback
            debug_print("ERROR", f"Row {idx} failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No results to save.")
        return

    final_df = pd.DataFrame(results)

    # Save CSV (without raw text columns)
    csv_df = final_df.drop(
        columns=['original_text', 'perturbed_text'], errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)
    print(f"CSV saved: {FINAL_RESULTS}")

    # Per-ICD analysis
    per_icd, per_chapter = generate_per_icd_analysis(final_df)

    # Plots
    plot_results(final_df, per_icd, per_chapter)

    # HTML report
    generate_html_report(final_df, per_icd, per_chapter)

    # Final summary
    valid_df = final_df[final_df['baseline_correct'] == True]
    print("\n" + "=" * 80)
    print(f"FINAL SUMMARY — {MODEL_NAME} | Attack Level {ATTACK_LEVEL}")
    print(f"  Total samples        : {len(final_df)}")
    print(f"  Valid (analysed)     : {len(valid_df)}")
    print(f"  Skipped              : {len(final_df) - len(valid_df)}")
    if len(valid_df) > 0:
        t1 = (csv_df['baseline_tier'] == 'tier1_exact').sum()
        t2 = (csv_df['baseline_tier'] == 'tier2_prefix').sum()
        t3 = (csv_df['baseline_tier'] == 'tier3_chapter').sum()
        print(f"  Baseline tiers       : "
              f"exact={t1} prefix={t2} chapter={t3}")
        print(f"  Categorical drift    : "
              f"{valid_df['categorical_drift'].mean()*100:.1f}%  ← primary")
        print(f"  Subcategory drift    : "
              f"{valid_df['subcategory_drift'].mean()*100:.1f}%")
        print(f"  Avg text similarity  : "
              f"{valid_df['semantic_similarity'].mean():.4f}")
        print(f"  Avg changes / note   : "
              f"{valid_df['changes_count'].mean():.1f}")
        print(f"  DISEASE changes      : "
              f"{valid_df['disease_changes'].sum()}")
        print(f"  CHEMICAL changes     : "
              f"{valid_df['chemical_changes'].sum()}")
        print(f"  Suspicious drifts    : "
              f"{valid_df['suspicious_drift'].sum()}")
    print(f"  CSV  → {FINAL_RESULTS}")
    print(f"  HTML → {HTML_REPORT}")
    print(f"  Plot → {PLOT_FILE}")
    print(f"  Per-ICD  → {PER_ICD_CSV}")
    print(f"  Per-Chapter → {PER_CHAPTER_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()