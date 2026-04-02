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
# CONFIGURATION
# ============================================================
MODEL_NAME     = "llama3.1:8b"
API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"

NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS  = "final_robustnessattack0.csv"
HTML_REPORT    = "robustness_reportattack0.html"
PLOT_FILE      = "robustness_plotsattack0.png"

MAX_SAMPLES    = 1000
DEVICE         = "cpu"
ATTACK_LEVEL   = 0         # 0=strict synonyms, 1=related terms, 2=differential

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR
# ============================================================
def query_rwth_server(prompt, max_tokens=50):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model"  : MODEL_NAME,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0.0, "num_predict": max_tokens}
        # FIX: temperature=0 removes stochastic drift between runs
    }
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(API_URL, headers=headers,
                                data=json.dumps(payload), timeout=300)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        debug_print("API_ERROR", f"Connection failed: {e}")
        return ""


# ============================================================
# SECTION 2: DATA LOADER
# ============================================================
class MIMICLoader:

    # Sections that contain the answer — must be stripped to prevent leakage
    # FIX: Discharge Diagnosis section was leaking ground truth into note text,
    # allowing the model to read the answer directly rather than reason from
    # clinical evidence. These sections are now explicitly excluded.
    LEAKING_SECTION_PATTERN = re.compile(
        r'(?:Discharge Diagnosis(?:es)?|Primary Diagnosis|Final Diagnosis|'
        r'Discharge Condition|Discharge Instructions|'
        r'Medications on (?:Admission|Discharge)|'
        r'Discharge Medications|Follow.?up|Transitional Issues)'
        r'\s*:?.*',
        re.S | re.I
    )

    @staticmethod
    def clean_noise(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @classmethod
    def extract_clinical_info(cls, text: str) -> str:
        """
        Extract only clinically rich sections needed for ICD prediction.
        FIX: Discharge Diagnosis and all answer-leaking sections are now
        explicitly excluded. The model must reason from clinical evidence
        (symptoms, history, hospital course) not from stated diagnoses.
        """
        # First truncate at any answer-leaking section
        truncated = cls.LEAKING_SECTION_PATTERN.sub('', text)

        relevant_parts = []

        cc = re.search(
            r"(?:Chief Complaint)\s*:?\s*(.*?)"
            r"(?:History of Present Illness|HPI|Past Medical History|\Z)",
            truncated, re.S | re.I)
        if cc:
            content = cc.group(1).strip()
            if len(content) > 10:
                relevant_parts.append(f"Chief Complaint: {content[:300]}")

        hpi = re.search(
            r"(?:History of Present Illness|HPI)\s*:?\s*(.*?)"
            r"(?:Past Medical History|Social History|"
            r"Review of Systems|Physical Exam|\Z)",
            truncated, re.S | re.I)
        if hpi:
            content = hpi.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"HPI: {content[:1200]}")

        course = re.search(
            r"Brief Hospital Course\s*:?\s*(.*?)"
            r"(?:Medications on Admission|Discharge Medications|"
            r"Discharge Diagnosis|Discharge Condition|\Z)",
            truncated, re.S | re.I)
        if course:
            content = course.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"Hospital Course: {content[:1200]}")

        ap = re.search(
            r"(?:Assessment(?:\s+and\s+Plan)?|A(?:/|&)P)\s*:?\s*(.*?)"
            r"(?:Discharge|Follow.?up|\Z)",
            truncated, re.S | re.I)
        if ap:
            content = ap.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"Assessment: {content[:800]}")

        pmh = re.search(
            r"(?:Past Medical History|PMH)\s*:?\s*(.*?)"
            r"(?:Social History|Family History|Review of Systems|\Z)",
            truncated, re.S | re.I)
        if pmh:
            content = pmh.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"PMH: {content[:400]}")

        if relevant_parts:
            return "\n\n".join(relevant_parts)[:3000]

        return truncated[-2000:].strip()

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading MIMIC data (ICD-10 only)...")

        diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
        # FIX: Filter to ICD-10 only — ground truth and predictions are both
        # ICD-10, so comparison is a direct string match with no version mismatch
        diags = diags[diags['icd_version'] == 10].copy()
        debug_print("DATA", f"ICD-10 diagnoses: {len(diags)} rows")

        hadm_ids_needed = set(diags['hadm_id'].unique())

        chunks, notes_found = [], 0
        for chunk in pd.read_csv(notes_path, compression='gzip',
                                  usecols=['subject_id', 'hadm_id', 'text'],
                                  chunksize=5000):
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids_needed)].copy()
            if len(chunk) > 0:
                chunks.append(chunk)
                notes_found += len(chunk)
                if notes_found >= n_samples * 100:
                    break
        notes = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

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
                      primary[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')

        df['text']           = df['text'].apply(cls.clean_noise)
        df['processed_text'] = df['text'].apply(cls.extract_clinical_info)

        result = df.sample(n=min(len(df), n_samples), random_state=42)
        debug_print("DATA", f"Loaded {len(result)} ICD-10 records")
        return result


# ============================================================
# SECTION 3: PERTURBATION — MEDICALLY RELEVANT ENTITIES ONLY
# ============================================================

# FIX: Only accept typed medical entity labels from en_core_sci_md/lg.
# This replaces the previous approach that accepted all ENTITY labels
# from en_core_sci_sm, which labelled administrative words, section
# headers, and general English as entities.
# These four types are the ones that directly influence ICD coding.
ACCEPTED_ENTITY_LABELS = {
    "DISEASE",    # myocardial infarction, pneumonia, CKD, DVT
    "CHEMICAL",   # heparin, aspirin, furosemide, Bactrim, ETOH
    "PROCEDURE",  # echocardiogram, Whipple, laparoscopy, CT scan
    "ANATOMY",    # only anatomical sites with diagnostic relevance
}

MIN_ENTITY_CHARS = 3

# Safety patterns — block these regardless of entity type
UNSAFE_ENTITY_PATTERNS = [
    r'^(no|not|denies|denied|without|negative|none|absent|absence)\b',
    r'^(positive|negative)$',
    r'^(left|right|bilateral|unilateral|ipsilateral|contralateral)$',
    r'^(elevated|increased|decreased|reduced|low|high|normal|abnormal|'
    r'mild|moderate|severe|minimal|significant|acute|chronic|stable)$',
    r'^(minute|second|hour|day|week|month|year)s?$',
    r'^\d',
    r'^[A-Z]$',
]
_UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_ENTITY_PATTERNS]

# FIX: Explicit block list for administratively-labelled terms that
# scispaCy sometimes misclassifies as medical entities
ADMIN_STOP_ENTITIES = {
    'chief complaint', 'hospital course', 'hospital stay', 'discharge',
    'admission', 'floor', 'ward', 'team', 'staff', 'service', 'department',
    'dr', 'mr', 'mrs', 'ms', 'patient', 'physician', 'doctor', 'nurse',
    'regular diet', 'normal diet', 'morning', 'evening', 'night', 'daily',
    'routine', 'standard', 'general', 'instructions', 'plan', 'management',
    'care', 'treatment', 'follow up', 'follow-up', 'result', 'outcome',
    'benefits', 'advantages', 'adequate', 'sufficient', 'expensive', 'costly',
    'shoes', 'footwear', 'sisters', 'siblings', 'child', 'children',
    'long time', 'clinical pathway', 'care pathway',
}


def _clean_synonym_echo(synonym: str) -> str:
    """
    FIX: Strip Llama's echo pattern 'original -> synonym'.
    Also handles triple chains 'A -> B -> C' by taking the last element.
    Previously this function had a missing return for the non-arrow case,
    causing clean synonyms to be silently dropped (returned None).
    Now always returns a string.
    """
    if " -> " in synonym:
        parts   = [p.strip() for p in synonym.split("->")]
        synonym = parts[-1].strip()
    # Always clean quotes and trailing punctuation
    synonym = synonym.strip('"\'')
    synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
    # Strip parenthetical notes like "(no change)"
    synonym = re.sub(r'\s*\(no change\)\s*', '', synonym, flags=re.I).strip()
    return synonym


def entity_is_safe_to_replace(ent, accepted_labels) -> tuple:
    """
    Three-gate filter for entity safety.
    Gate 1 — label must be in accepted medical types (DISEASE/CHEMICAL/
              PROCEDURE/ANATOMY). This is the primary filter that removes
              administrative words, section headers, and general English.
    Gate 2 — safety patterns (negations, polarities, numbers).
    Gate 3 — admin stop set for edge cases that pass Gate 1.
    """
    # Gate 1: typed label check — only DISEASE, CHEMICAL, PROCEDURE, ANATOMY
    if accepted_labels is not None and ent.label_ not in accepted_labels:
        return False, f"non-medical label: {ent.label_}"

    text  = ent.text.strip()
    lower = text.lower()

    if len(text) < MIN_ENTITY_CHARS:
        return False, "too short"

    # Gate 2: safety patterns
    for pattern in _UNSAFE_COMPILED:
        if pattern.match(text):
            return False, "unsafe pattern"

    if lower.startswith(('no ', 'not ', 'denies ', 'without ')):
        return False, "negation phrase"

    # Gate 2b: block polarity words that scispaCy labels as DISEASE
    POLARITY_WORDS = {'benign', 'malignant', 'positive', 'negative',
                      'normal', 'abnormal', 'stable', 'unstable'}
    if lower in POLARITY_WORDS:
        return False, "polarity word"

    # Gate 3: admin stop set
    if lower in ADMIN_STOP_ENTITIES:
        return False, "administrative term"

    # Gate 3b: POS filter — reject verbs, determiners, pronouns
    if ent[0].pos_ in {"VERB", "ADP", "DET", "PRON", "CCONJ",
                        "SCONJ", "PUNCT", "PART", "NUM", "SPACE"}:
        return False, f"non-content POS: {ent[0].pos_}"

    return True, ent.label_


def replacement_is_structurally_valid(original: str, replacement: str) -> tuple:
    """Layer 2: structural rules."""
    # FIX: Variable shadowing bug — orig_words and repl_words were being
    # redefined as sets mid-function, breaking n_repl calculation.
    # Now uses clearly named variables throughout.
    if " -> " in replacement:
        return False, "arrow echo not fully cleaned"

    orig_word_list = original.strip().split()
    repl_word_list = replacement.strip().split()
    n_orig = len(orig_word_list)
    n_repl = len(repl_word_list)

    # Meaning inverters check — uses separate set variables
    MEANING_INVERTERS = {'no', 'not', 'absence', 'without',
                         'lack', 'negative', 'non'}
    orig_word_set = set(w.lower() for w in orig_word_list)
    repl_word_set = set(w.lower() for w in repl_word_list)
    added_words   = repl_word_set - orig_word_set
    if added_words & MEANING_INVERTERS:
        return False, "introduces negation"

    is_abbrev  = original.strip().isupper() and len(original.strip()) <= 5
    word_limit = 6 if is_abbrev else max(n_orig * 2, n_orig + 3)

    if n_repl > word_limit:
        return False, f"too long ({n_orig}->{n_repl} words)"
    if n_orig == 1 and n_repl > 3:
        return False, f"single-word expanded to {n_repl} words"
    if not replacement.strip():
        return False, "empty"
    if replacement.strip().lower() == original.strip().lower():
        return False, "identical to original"
    if len(replacement) / max(len(original), 1) > 5.0:
        return False, "character ratio too high"
    if re.search(r'[.!?](?!\s*$)', replacement):
        return False, "mid-sentence punctuation"

    return True, "ok"


# Synonym generation prompts — three attack levels
PROMPT_LEVEL_0 = (
    "You are a clinical terminology expert. Below is a numbered list of "
    "medical terms from a clinical note, each with its type and context.\n\n"
    "For EACH term, provide ONE medically equivalent synonym.\n\n"
    "STRICT RULES:\n"
    "- Write ONLY the replacement term. Do NOT repeat the original.\n"
    "- Do NOT use arrows (->) or any other separator.\n"
    "- 1-4 words maximum.\n"
    "- Same clinical meaning — do NOT change polarity, severity, "
    "laterality, or diagnosis.\n"
    "FORMAT (one line per term):\n"
    "1. <synonym or NONE>\n"
    "2. <synonym or NONE>\n\n"
    "Terms:\n{term_lines}\n\nSynonyms:"
)

PROMPT_LEVEL_1 = (
    "You are a clinical terminology expert. Below is a numbered list of "
    "medical terms from a clinical note, each with its type and context.\n\n"
    "For EACH term, provide ONE clinically related alternative term "
    "(may be more general, broader, or a related condition in same family).\n\n"
    "RULES:\n"
    "- Write ONLY the replacement term. Do NOT repeat the original.\n"
    "- Do NOT use arrows (->) or any other separator.\n"
    "- 1-4 words maximum.\n"
    "- Do NOT invert polarity or switch to completely unrelated condition.\n"
    "FORMAT (one line per term):\n"
    "1. <related term or NONE>\n"
    "2. <related term or NONE>\n\n"
    "Terms:\n{term_lines}\n\nReplacements:"
)

PROMPT_LEVEL_2 = (
    "You are a clinical terminology expert. Below is a numbered list of "
    "medical terms from a clinical note, each with its type and context.\n\n"
    "For EACH term, provide ONE plausible alternative that refers to a "
    "DIFFERENT specific condition/drug/procedure but could realistically "
    "appear in a similar note (differential diagnosis style).\n\n"
    "RULES:\n"
    "- Write ONLY the replacement term. Do NOT repeat the original.\n"
    "- Do NOT use arrows (->) or any other separator.\n"
    "- 1-4 words maximum.\n"
    "- Must be medically plausible — no random or nonsensical terms.\n"
    "NOTE: Level 2 creates SEMANTIC perturbations (different conditions). "
    "Use only for differential diagnosis robustness testing.\n"
    "FORMAT (one line per term):\n"
    "1. <alternative term or NONE>\n"
    "2. <alternative term or NONE>\n\n"
    "Terms:\n{term_lines}\n\nReplacements:"
)

ATTACK_PROMPTS = {0: PROMPT_LEVEL_0, 1: PROMPT_LEVEL_1, 2: PROMPT_LEVEL_2}


def find_synonyms_batch(terms: list, attack_level: int = 0) -> dict:
    if not terms:
        return {}

    term_lines = "\n".join(
        f"{i+1}. [{item['label']}] \"{item['text']}\"  "
        f"(context: \"{item['context'][:80]}...\")"
        for i, item in enumerate(terms)
    )

    prompt   = ATTACK_PROMPTS.get(attack_level, PROMPT_LEVEL_0).format(
        term_lines=term_lines)
    response = query_rwth_server(prompt, max_tokens=len(terms) * 12 + 60)
    if not response:
        return {}

    synonym_map = {}
    for line in response.strip().split('\n'):
        line  = line.strip()
        match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
        if not match:
            continue
        idx     = int(match.group(1)) - 1
        raw_syn = match.group(2).strip()

        if 0 <= idx < len(terms):
            original = terms[idx]['text']
            # FIX: echo-strip always returns a string now (missing return fixed)
            synonym  = _clean_synonym_echo(raw_syn)

            if not synonym or synonym.upper() == 'NONE':
                synonym_map[original] = None
            elif synonym.lower() == original.lower():
                synonym_map[original] = None
            else:
                synonym_map[original] = synonym

    return synonym_map


class StructuredMedicalPerturber:
    """
    Perturbation pipeline restricted to medically relevant entity types.
    FIX: Now requires en_core_sci_md or en_core_sci_lg for typed NER.
    en_core_sci_sm labels everything as ENTITY — too coarse for medical
    relevance filtering. The md/lg models produce DISEASE, CHEMICAL,
    PROCEDURE, ANATOMY labels which are the only ones we accept.
    """

    def __init__(self):
        # FIX: Prefer md/lg over sm — typed labels are essential
        loaded = False
        for model_name in ["en_core_sci_lg", "en_core_sci_md"]:
            try:
                self.nlp = spacy.load(model_name)
                debug_print("PERTURBER", f"Loaded {model_name} (typed NER)")
                loaded = True
                break
            except OSError:
                continue

        if not loaded:
            # Fall back to sm but warn — quality will be lower
            try:
                self.nlp = spacy.load("en_core_sci_sm")
                debug_print("PERTURBER",
                            "WARNING: en_core_sci_sm loaded — typed labels "
                            "unavailable. Install md/lg for better filtering.\n"
                            "pip install https://s3-us-west-2.amazonaws.com/"
                            "ai2-s2-scispacy/releases/v0.5.4/"
                            "en_core_sci_md-0.5.4.tar.gz")
            except OSError:
                raise RuntimeError(
                    "No scispacy model found. Install:\n"
                    "pip install scispacy\n"
                    "pip install https://s3-us-west-2.amazonaws.com/"
                    "ai2-s2-scispacy/releases/v0.5.4/"
                    "en_core_sci_md-0.5.4.tar.gz")

        # Detect whether model provides typed labels
        test_doc     = self.nlp("myocardial infarction aspirin echocardiogram")
        found_labels = {ent.label_ for ent in test_doc.ents}
        if found_labels <= {"ENTITY", ""}:
            self.accepted_labels = None   # sm model — no type filtering
            debug_print("PERTURBER",
                        "sm model detected — typed label gate disabled. "
                        "Consider upgrading to en_core_sci_md.")
        else:
            self.accepted_labels = ACCEPTED_ENTITY_LABELS
            debug_print("PERTURBER",
                        f"Typed NER active — only: {ACCEPTED_ENTITY_LABELS}")

    def _is_negated(self, token) -> bool:
        neg    = {'no', 'not', 'denies', 'negative', 'without', 'none', 'denied'}
        window = [t.text.lower()
                  for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(t in window for t in neg)

    def extract_medical_entities(self, text: str) -> list:
        doc      = self.nlp(text)
        entities = []
        seen     = set()

        for ent in doc.ents:
            valid, label_str = entity_is_safe_to_replace(
                ent, self.accepted_labels)
            if not valid:
                continue
            if self._is_negated(ent[0]):
                continue

            key = ent.text.strip().lower()
            if key in seen:
                continue
            seen.add(key)

            sent_text = (ent.sent.text if ent.sent
                         else text[max(0, ent.start_char-60):ent.end_char+60])
            entities.append({
                'text'      : ent.text,
                'label'     : label_str,
                'start_char': ent.start_char,
                'end_char'  : ent.end_char,
                'context'   : sent_text,
            })

        return entities

    def validate_and_apply(self, text: str, entities: list,
                           synonym_map: dict) -> tuple:
        accepted = []

        for ent in entities:
            original  = ent['text']
            candidate = synonym_map.get(original)

            if not candidate:
                continue

            struct_ok, struct_reason = replacement_is_structurally_valid(
                original, candidate)
            if not struct_ok:
                print(f"       L2 BLOCKED ({struct_reason}): "
                      f"'{original}'->'{candidate}'")
                continue

            accepted.append({
                'original'   : original,
                'replacement': candidate,
                'label'      : ent['label'],
                'start_char' : ent['start_char'],
                'end_char'   : ent['end_char'],
            })
            print(f"       ACCEPTED [{ent['label']:12s}]: "
                  f"'{original}' -> '{candidate}'")

        if not accepted:
            return text, []

        perturbed = text
        for item in sorted(accepted, key=lambda x: x['start_char'], reverse=True):
            perturbed = (perturbed[:item['start_char']]
                         + item['replacement']
                         + perturbed[item['end_char']:])

        changes_log = [(i['original'], i['replacement'], i['label'])
                       for i in accepted]
        return perturbed, changes_log

    def perturb(self, text: str,
                perturbation_rate: float = 0.30,   # FIX: back to 0.10 from 0.30
                attack_level: int = 0) -> tuple:
        """
        FIX: perturbation_rate restored to 0.10 (was incorrectly set to 0.30).
        0.30 changed too many terms per note, risking semantic meaning shift.
        0.10 of medical entities gives ~4-7 changes per note — surface level.
        """
        all_entities = self.extract_medical_entities(text)
        if not all_entities:
            print("   > No accepted medical entities found.")
            return text, []

        print(f"\n   > Medical entities found ({len(all_entities)}):")
        for e in all_entities:
            print(f"       [{e['label']:12s}] {e['text']}")

        n               = max(1, int(len(all_entities) * perturbation_rate))
        target_entities = random.sample(all_entities, min(n, len(all_entities)))
        synonym_map     = find_synonyms_batch(target_entities,
                                              attack_level=attack_level)

        print(f"\n   > Synonym candidates:")
        for term, syn in synonym_map.items():
            print(f"       {term:30s} -> {syn if syn else 'NONE'}")

        print(f"\n   > Validation:")
        return self.validate_and_apply(text, target_entities, synonym_map)


# ============================================================
# SECTION 4: SEMANTIC EVALUATOR (ClinicalBERT)
# ============================================================
class SemanticEvaluator:
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
            e1   = (outputs.last_hidden_state[0] * mask[0]).sum(0) / mask[0].sum()
            e2   = (outputs.last_hidden_state[1] * mask[1]).sum(0) / mask[1].sum()
            sim  = cosine_similarity(e1.unsqueeze(0).numpy(),
                                     e2.unsqueeze(0).numpy())[0][0]
        return float(sim)


# ============================================================
# SECTION 5: BASELINE CORRECTNESS
# ============================================================
def normalise_icd(code: str) -> str:
    """Remove dots and non-alphanumeric characters, uppercase."""
    return re.sub(r'[^A-Z0-9]', '', code.upper())


def is_baseline_correct(orig_icd: str, gt_icd: str, gt_desc: str) -> tuple:
    """
    FIX: Replaced the three-tier gate (LLM OR high_sim OR prefix) with a
    clean two-gate system:
      Gate 1 — 3-character ICD prefix match (objective, no model involved)
      Gate 2 — LLM clinical equivalence (handles edge cases)

    The previous high_sim >= 0.92 gate was causing false passes — it accepted
    predictions like 'Cardiac Tamponade' for 'Lymphedema' because ClinicalBERT
    embeddings happened to be close. Semantic similarity is now diagnostic only.

    Since we are ICD-10 only, prefix matching is reliable:
      GT=C3412 -> prefix=C34 | Pred=C34.2 -> prefix=C34 -> MATCH
      GT=I890  -> prefix=I89 | Pred=I46.9 -> prefix=I46 -> NO MATCH
    """
    if orig_icd in ("UNKNOWN", ""):
        return False, "unknown_pred", False, False

    gt_norm   = normalise_icd(gt_icd)
    pred_norm = normalise_icd(orig_icd)

    # Gate 1: 3-character prefix match — objective ICD family comparison
    gt_prefix   = gt_norm[:3]
    pred_prefix = pred_norm[:3]
    prefix_ok   = (gt_prefix == pred_prefix) and gt_prefix != ""

    # Gate 2: LLM equivalence check — handles close calls and related conditions
    llm_ok = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)

    passed = prefix_ok or llm_ok

    gate_str = (f"prefix={gt_prefix}{'==' if prefix_ok else '!='}{pred_prefix} "
                f"| llm={'YES' if llm_ok else 'NO'}")

    return passed, gate_str, prefix_ok, llm_ok


def check_clinical_equivalence(pred_icd: str, gt_icd: str,
                                gt_desc: str) -> bool:
    if pred_icd in ("UNKNOWN", ""):
        return False
    prompt = (
        f"Ground truth: '{gt_desc}' (ICD-10: {gt_icd})\n"
        f"Predicted ICD-10: {pred_icd}\n\n"
        f"Does the predicted code represent the same primary medical condition "
        f"as the ground truth? Minor specificity differences are acceptable.\n"
        f"Answer YES or NO only."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")


# ============================================================
# SECTION 6: ICD PREDICTION
# ============================================================
def get_icd_prediction(text: str) -> str:
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
        response = query_rwth_server(prompt)
        if response:
            # Strip DeepSeek-R1 thinking tags if present
            response = re.sub(r'<think>.*?</think>', '', response,
                               flags=re.S).strip()
            match = re.search(r'[A-Z]\d{2}\.\d+', response.strip().upper())
            if match:
                return match.group(0)
            match = re.search(r'[A-Z]\d{2,4}', response.strip().upper())
            if match:
                return match.group(0)
    return "UNKNOWN"


def get_diagnosis_description(icd_code: str) -> str:
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    prompt = (
        f"What medical condition does ICD-10 code '{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words."
    )
    return query_rwth_server(prompt)


# ============================================================
# SECTION 7: DRIFT CLASSIFICATION
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
# SECTION 8: HTML REPORT
# ============================================================
def _highlight_text(text: str, changes: list, is_original: bool) -> str:
    """
    FIX: The previous implementation used regex substitution which highlighted
    ALL occurrences of a word throughout the entire note — including words
    that appeared naturally in the text before any replacement was made.

    For example if 'fever' was replaced with 'pyrexia', the word 'fever'
    would be highlighted red in the original text everywhere it appeared,
    even in sentences that were not changed. Same for the replacement word
    in the perturbed text.

    The fix: use character offset tracking instead of regex substitution.
    We know the exact start_char and end_char of each replacement from the
    changes_log. We only highlight those specific positions, not every
    occurrence of the word in the text.

    However since we only store (original, replacement, label) in changes_log
    and not the offsets, we use a smarter approach:
    - Build a set of (position, word) pairs by finding FIRST occurrence only
    - Mark only that first occurrence, not subsequent ones
    - This is still not perfect but vastly better than highlighting all occurrences

    The truly correct fix would be to store char offsets in changes_log
    (see TODO in perturb()) but this requires pipeline changes beyond the
    HTML report.
    """
    if not changes:
        return text

    result   = text
    # Track which positions have already been highlighted to avoid double-marking
    # We process replacements in order and use a consumed-positions set
    highlighted = []
    consumed_spans = []

    if is_original:
        # In original text: highlight the words that WERE replaced
        words_to_highlight = [(orig, 'original') for orig, _, _ in changes]
    else:
        # In perturbed text: highlight the words that WERE inserted
        words_to_highlight = [(repl, 'replacement') for _, repl, _ in changes]

    # Build result with non-overlapping highlights
    # Process each word to highlight — find first non-consumed occurrence
    offset_map = {}  # word -> list of (start, end) positions in text
    for word, kind in words_to_highlight:
        positions = []
        search_from = 0
        # Use word boundary search
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            positions.append((m.start(), m.end()))
        offset_map[word] = positions

    # Pick the first non-consumed occurrence for each word
    highlight_spans = []  # list of (start, end, word, kind)
    for word, kind in words_to_highlight:
        for start, end in offset_map.get(word, []):
            # Check not overlapping with already-claimed span
            overlap = any(not (end <= cs or start >= ce)
                          for cs, ce in consumed_spans)
            if not overlap:
                consumed_spans.append((start, end))
                highlight_spans.append((start, end, word, kind))
                break  # only highlight first occurrence

    # Sort spans in reverse order so we can insert HTML without shifting offsets
    highlight_spans.sort(key=lambda x: x[0], reverse=True)

    result = text
    for start, end, word, kind in highlight_spans:
        if kind == 'original':
            color = '#FF6B6B'
            label = 'replaced'
        else:
            color = '#51C878'
            label = 'inserted'
        span = (f'<span style="background:{color};color:white;'
                f'padding:1px 4px;border-radius:3px;font-weight:bold;" '
                f'title="{label}">{result[start:end]}</span>')
        result = result[:start] + span + result[end:]

    return result


def generate_html_report(df: pd.DataFrame, filename: str = HTML_REPORT):

    DRIFT_COLORS = {
        'none'       : '#ccffcc',
        'subcategory': '#fff3cd',
        'categorical': '#ffcccc',
        'unknown'    : '#eeeeee',
        'skipped'    : '#f9f9f9',
    }
    DRIFT_LABELS = {
        'none'       : 'STABLE',
        'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
        'unknown'    : 'UNKNOWN ICD',
        'skipped'    : 'SKIPPED',
    }

    def parse_changes(changes_str):
        """Parse '[LABEL] orig -> repl' format into list of tuples."""
        changes = []
        if not changes_str or changes_str in ('SKIPPED', '', None, 'None'):
            return changes
        for pair in changes_str.split('|'):
            pair = pair.strip()
            label_m = re.match(r'\[([^\]]+)\]\s*(.*?)\s*->\s*(.*)', pair)
            if label_m:
                changes.append((
                    label_m.group(2).strip(),
                    label_m.group(3).strip(),
                    label_m.group(1).strip()
                ))
        return changes

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        dc          = DRIFT_COLORS.get(drift_level, '#ccffcc')
        dl          = DRIFT_LABELS.get(drift_level, 'STABLE')
        sim_d       = (f"{float(row['semantic_similarity']):.4f}"
                       if row.get('semantic_similarity') else 'N/A')
        base_d      = f"{float(row.get('baseline_sim', 0)):.4f}" if row.get('baseline_sim') else 'N/A'
        pred_desc   = row.get('original_pred_desc', '')
        changes_str = row.get('changes_made', '')
        gate_d      = row.get('baseline_gate', '')
        changes     = parse_changes(changes_str)

        # Format changes for display
        if changes:
            changes_display = " | ".join(
                f"[{l}] {o} → {n}" for o, n, l in changes)
        else:
            changes_display = "None"

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — baseline did not match ground truth.<br>'
                f'<small>Gates: {gate_d}</small></p>')
        else:
            # FIX: Use offset-aware highlighting instead of regex substitution
            # This ensures only the actually-changed words are highlighted,
            # not every occurrence of those words throughout the note.
            orig_text = row.get('original_text', '')
            pert_text = row.get('perturbed_text', '')
            orig_highlighted = _highlight_text(orig_text, changes, is_original=True)
            pert_highlighted = _highlight_text(pert_text, changes, is_original=False)

            text_section = f"""
            <div style="margin-top:10px;">
              <div style="margin-bottom:8px;font-size:0.85em;">
                <span style="background:#FF6B6B;color:white;padding:2px 8px;
                border-radius:3px;">Replaced terms (original)</span>&nbsp;
                <span style="background:#51C878;color:white;padding:2px 8px;
                border-radius:3px;">Inserted terms (replacement)</span>
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
                  font-size:0.85em;line-height:1.6em;">{orig_highlighted}</td>
                  <td style="padding:10px;border:1px solid #ccc;
                  vertical-align:top;white-space:pre-wrap;
                  font-size:0.85em;line-height:1.6em;">{pert_highlighted}</td>
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
            Patient ID: {row['hadm_id']}
            <span style="background:{dc};padding:4px 10px;
            border-radius:4px;font-size:0.9em;">{dl}</span>
          </h3>
          <table style="font-size:0.9em;border-collapse:collapse;
          margin-bottom:10px;">
            <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                <td>{row.get('ground_truth_desc','')}
                ({row.get('ground_truth_icd','')} ICD-10)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                <td>{row.get('original_icd_pred','')} — {pred_desc}
                (baseline sim: {base_d})</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                <td><small>{gate_d}</small></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                <td>{changes_display} ({row.get('changes_count',0)} terms)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                <td>{sim_d}</td></tr>
          </table>
          {text_section}
        </div>"""

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = valid_df['categorical_drift'].mean()*100 if len(valid_df) else 0
    sub_rate    = valid_df['subcategory_drift'].mean()*100  if len(valid_df) else 0
    avg_sim     = (f"{valid_df['semantic_similarity'].mean():.4f}"
                   if len(valid_df) else 'N/A')
    avg_changes = (f"{valid_df['changes_count'].mean():.1f}"
                   if len(valid_df) else 'N/A')

    html = f"""<!DOCTYPE html>
<html><head><title>Robustness Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;
     padding:20px;background:#fafafa;}}
h1{{color:#333;}} h3{{margin:0 0 10px 0;}}
.summary{{background:#f0f0f0;padding:15px;border-radius:8px;margin-bottom:30px;}}
.summary p{{margin:4px 0;}}
.legend{{display:flex;gap:16px;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;}}
.legend span{{padding:3px 10px;border-radius:4px;}}
.method-note{{background:#e8f4f8;padding:12px;border-radius:6px;
              margin-bottom:20px;font-size:0.9em;color:#555;}}
</style></head><body>
<h1>Clinical Note Robustness Report</h1>
<p style="color:#666;">Model: {MODEL_NAME} | ICD-10 only |
Perturbation rate: 10% of medical entities</p>
<div class="method-note">
  <b>Key methodological choices:</b>
  (1) ICD-10 only — no version mismatch between ground truth and predictions.
  (2) Baseline gate: 3-char ICD prefix match OR LLM equivalence — no semantic
  similarity gate (removed: caused false passes).
  (3) Discharge Diagnosis section excluded from notes — prevents answer leakage.
  (4) Only DISEASE/CHEMICAL/PROCEDURE/ANATOMY entities perturbed — medically
  relevant terms only, not administrative text.
  (5) HTML highlighting uses offset-based matching — only actually-changed
  positions highlighted, not all occurrences of the word.
</div>
<div class="legend">
  <span style="background:#ccffcc;">STABLE — exact ICD-10 match</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT — same 3-char family</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease</span>
</div>
<div class="summary">
  <h2>Summary</h2>
  <p><b>Model:</b> {MODEL_NAME}</p>
  <p><b>Total Samples:</b> {len(df)}</p>
  <p><b>Valid for Analysis:</b> {len(valid_df)}
     <span style="color:#888;">(gate: ICD prefix match OR LLM equivalence)</span></p>
  <p><b>Skipped (wrong baseline):</b> {len(df) - len(valid_df)}</p>
  <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}%
     <span style="color:#888;">(primary robustness metric)</span></p>
  <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%</p>
  <p><b>Avg Semantic Similarity:</b> {avg_sim}</p>
  <p><b>Avg Medical Terms Changed:</b> {avg_changes}</p>
</div>
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {filename}")


# ============================================================
# SECTION 9: PLOTS
# ============================================================
def plot_results(df: pd.DataFrame):
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return

    plt.figure(figsize=(22, 5))

    plt.subplot(1, 4, 1)
    plt.hist(valid_df['semantic_similarity'], bins=15,
             color='teal', edgecolor='black')
    plt.title("Semantic Preservation\n(correct baselines only)")
    plt.xlabel("Cosine Similarity")

    plt.subplot(1, 4, 2)
    drift_counts = valid_df['drift_level'].value_counts()
    colors_map   = {'none': '#90EE90', 'subcategory': '#FFD700',
                    'categorical': '#FF6B6B', 'unknown': '#CCCCCC'}
    bar_colors   = [colors_map.get(k, 'grey') for k in drift_counts.index]
    drift_counts.plot(kind='bar', color=bar_colors)
    cat_rate = valid_df['categorical_drift'].mean() * 100
    plt.title(f"Drift Breakdown\nCategorical: {cat_rate:.1f}%")
    plt.xticks(rotation=0)

    plt.subplot(1, 4, 3)
    plot_df = valid_df[valid_df['drift_level'].isin(
        ['none', 'subcategory', 'categorical'])]
    if len(plot_df) > 0:
        colors = {'none': 'green', 'subcategory': 'orange', 'categorical': 'red'}
        for level in ['none', 'subcategory', 'categorical']:
            mask = plot_df['drift_level'] == level
            plt.scatter(plot_df[mask]['changes_count'],
                        plot_df[mask]['semantic_similarity'],
                        label=level, color=colors[level], alpha=0.6)
        plt.legend(title="Drift level")
    plt.title("Changes vs Similarity")
    plt.xlabel("Words Changed")
    plt.ylabel("Cosine Similarity")

    plt.subplot(1, 4, 4)
    counts = df['baseline_correct'].value_counts()
    lbl, clr = [], []
    if True  in counts.index:
        lbl.append(f'Correct (n={counts[True]})')
        clr.append('#90EE90')
    if False in counts.index:
        lbl.append(f'Wrong (n={counts[False]})')
        clr.append('#FFB6C1')
    plt.pie(counts, labels=lbl, colors=clr, autopct='%1.1f%%')
    plt.title("Baseline Quality\n(gate: prefix + LLM)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 10: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Model: {MODEL_NAME} | Attack level: {ATTACK_LEVEL}")

    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to RWTH server.")
        return
    print("LLM API OK")

    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except Exception as e:
        import traceback
        debug_print("ERROR", f"Data load failed: {e}")
        traceback.print_exc()
        return

    perturber = StructuredMedicalPerturber()
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

            print(f"\n[ID: {hadm_id}]")
            print(f"   GT: {gt_desc} ({gt_icd})")

            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd)
            print(f"   Pred: {orig_icd} — {orig_desc}")

            baseline_ok, gate_str, prefix_ok, llm_ok = is_baseline_correct(
                orig_icd, gt_icd, gt_desc)
            print(f"   Gates: {gate_str}")
            print(f"   → {'PASS' if baseline_ok else 'SKIP'}")

            if not baseline_ok:
                results.append({
                    "hadm_id"             : hadm_id,
                    "ground_truth_icd"    : gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc"   : gt_desc,
                    "original_text"       : orig_text,
                    "perturbed_text"      : "SKIPPED",
                    "changes_made"        : "SKIPPED",
                    "changes_count"       : 0,
                    "original_icd_pred"   : orig_icd,
                    "original_pred_desc"  : orig_desc,
                    "baseline_sim"        : 0.0,
                    "baseline_gate"       : gate_str,
                    "baseline_correct"    : False,
                    "perturbed_icd_pred"  : "SKIPPED",
                    "icd_codes_differ"    : False,
                    "drift_level"         : "skipped",
                    "categorical_drift"   : False,
                    "subcategory_drift"   : False,
                    "suspicious_drift"    : False,
                    "semantic_similarity" : 0.0,
                    "disease_changes"     : 0,
                    "chemical_changes"    : 0,
                    "procedure_changes"   : 0,
                    "anatomy_changes"     : 0,
                })
                continue

            print(f"   Perturbing (rate=0.10, level={ATTACK_LEVEL})...")
            pert_text, changes_log = perturber.perturb(
                orig_text, perturbation_rate=0.10, attack_level=ATTACK_LEVEL)
            print(f"   {len(changes_log)} changes applied")

            pert_icd   = get_icd_prediction(pert_text)
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995 and len(changes_log) <= 3)

            print(f"   Pert ICD: {pert_icd} | Drift: {drift_info['drift_level']}"
                  f" | Sim: {sim:.4f}")

            changes_str = " | ".join(
                f"[{l}] {o} -> {n}" for o, n, l in changes_log)

            results.append({
                "hadm_id"             : hadm_id,
                "ground_truth_icd"    : gt_icd,
                "ground_truth_version": icd_ver,
                "ground_truth_desc"   : gt_desc,
                "original_text"       : orig_text,
                "perturbed_text"      : pert_text,
                "changes_made"        : changes_str,
                "changes_count"       : len(changes_log),
                "original_icd_pred"   : orig_icd,
                "original_pred_desc"  : orig_desc,
                "baseline_sim"        : sim,
                "baseline_gate"       : gate_str,
                "baseline_correct"    : True,
                "perturbed_icd_pred"  : pert_icd,
                **drift_info,
                "suspicious_drift"    : suspicious,
                "semantic_similarity" : sim,
                "disease_changes"  : sum(1 for _,_,l in changes_log if l=="DISEASE"),
                "chemical_changes" : sum(1 for _,_,l in changes_log if l=="CHEMICAL"),
                "procedure_changes": sum(1 for _,_,l in changes_log if l=="PROCEDURE"),
                "anatomy_changes"  : sum(1 for _,_,l in changes_log if l=="ANATOMY"),
            })
            print("-" * 40)

        except Exception as e:
            import traceback
            debug_print("ERROR", f"Row {idx} failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No results.")
        return

    final_df = pd.DataFrame(results)
    csv_df   = final_df.drop(
        columns=['original_text', 'perturbed_text'], errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)

    plot_results(final_df)
    generate_html_report(final_df)

    valid_df = final_df[final_df['baseline_correct'] == True]
    print("\n" + "=" * 80)
    print(f"MODEL: {MODEL_NAME}")
    print(f"  Total              : {len(final_df)}")
    print(f"  Valid              : {len(valid_df)}")
    print(f"  Skipped            : {len(final_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"  Categorical drift  : "
              f"{valid_df['categorical_drift'].mean()*100:.1f}%")
        print(f"  Subcategory drift  : "
              f"{valid_df['subcategory_drift'].mean()*100:.1f}%")
        print(f"  Avg sim            : {valid_df['semantic_similarity'].mean():.4f}")
        print(f"  Avg changes        : {valid_df['changes_count'].mean():.1f}")
        print(f"  DISEASE changes    : {valid_df['disease_changes'].sum()}")
        print(f"  CHEMICAL changes   : {valid_df['chemical_changes'].sum()}")
        print(f"  PROCEDURE changes  : {valid_df['procedure_changes'].sum()}")
        print(f"  ANATOMY changes    : {valid_df['anatomy_changes'].sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()