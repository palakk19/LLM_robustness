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

FINAL_RESULTS  = "palak_final/final_robustness.csv"
HTML_REPORT    = "palak_final/robustness_report.html"
PLOT_FILE      = "palak_final/robustness_plots.png"
MAX_SAMPLES    = 500    # samples PER attack level (feature 5)
DEVICE         = "cpu"

BASELINE_SIM_THRESHOLD = 0.88

# Entity type weights for substitution_semantic_similarity (feature 1)
ENTITY_WEIGHTS = {
    "DISEASE"  : 1.0,   # directly maps to ICD code
    "CHEMICAL" : 0.6,   # affects treatment, not primary diagnosis
    "ANATOMY"  : 0.4,   # contextual/modifier
    "PROCEDURE": 0.5,   # contextual
    "OBSERVATION": 0.15
}

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR
# ============================================================
def query_rwth_server(prompt, max_tokens=50):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens}
    }
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(API_URL, headers=headers,
                                data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        debug_print("API_ERROR", f"Connection failed: {e}")
        return ""


# ============================================================
# SECTION 2: DATA LOADER
# ============================================================
class MIMICLoader:
    _LEAKING_HEADERS_RE = re.compile(
        r"\n\s*(?:"
        r"Assessment(?:\s+and\s+Plan)?|A\s*/\s*P|A\s*&\s*P"
        r"|Discharge\s+(?:Diagnosis|Diagnoses|Condition|Medications?|Instructions?|Summary)"
        r"|Primary\s+Diagnosis|Secondary\s+Diagnos"
        r"|Discharge\s+Disposition"
        r"|Follow[\s\-]?up"
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
    def strip_leaking_sections(cls, text: str) -> str:
        parts = cls._LEAKING_HEADERS_RE.split(text, maxsplit=1)
        return parts[0].strip()

    @staticmethod
    def extract_blind_clinical_info(text: str) -> str:
        _NEXT = (
            r"History of Present Illness|HPI|"
            r"Past Medical History|PMH|"
            r"Social History|Family History|"
            r"Review of Systems|"
            r"Physical Exam(?:ination)?|"
            r"Vital Signs|"
            r"Labs?|Laboratory|Imaging|Studies|"
            r"Medications? on Admission|"
            r"Brief Hospital Course"
        )
        relevant_parts = []

        cc = re.search(rf"Chief Complaint\s*:?\s*(.*?)(?:{_NEXT}|\Z)", text, re.S | re.I)
        if cc:
            content = cc.group(1).strip()
            if 3 <= len(content) <= 300:
                relevant_parts.append(f"Chief Complaint: {content}")

        hpi = re.search(rf"(?:History of Present Illness|HPI)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", text, re.S | re.I)
        if hpi:
            content = hpi.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"HPI: {content[:1200]}")

        course = re.search(rf"Brief Hospital Course\s*:?\s*(.*?)(?:{_NEXT}|\Z)", text, re.S | re.I)
        if course:
            content = course.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"Hospital Course: {content[:1200]}")

        pmh = re.search(rf"(?:Past Medical History|PMH)\s*:?\s*(.*?)(?:{_NEXT}|\Z)", text, re.S | re.I)
        if pmh:
            content = pmh.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"PMH: {content[:600]}")

        pe = re.search(rf"Physical Exam(?:ination)?\s*:?\s*(.*?)(?:{_NEXT}|\Z)", text, re.S | re.I)
        if pe:
            content = pe.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"Physical Exam: {content[:600]}")

        if relevant_parts:
            return "\n\n".join(relevant_parts)[:3000]
        return text[-2000:].strip()

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading Data Tables...")

        diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
        diags = diags[diags['icd_version'] == 10].copy()
        debug_print("DATA", f"ICD-10 diagnoses: {len(diags)} rows")

        hadm_ids_needed = set(diags['hadm_id'].unique())

        chunks, notes_found = [], 0
        target_notes = min(n_samples * 100, 50000)
        for chunk in pd.read_csv(notes_path, compression='gzip',
                                 usecols=['subject_id', 'hadm_id', 'text'],
                                 chunksize=5000):
            chunk = chunk[chunk['hadm_id'].isin(hadm_ids_needed)].copy()
            if len(chunk) > 0:
                chunks.append(chunk)
                notes_found += len(chunk)
                if notes_found >= target_notes:
                    break

        if not chunks:
            raise ValueError("No matching notes found for any ICD-10 admission")
        notes = pd.concat(chunks, ignore_index=True)
        debug_print("DATA", f"Notes loaded: {len(notes)} rows")

        if os.path.exists(dict_path):
            icd_dict = pd.read_csv(dict_path, compression='gzip')
            icd_dict = icd_dict[icd_dict['icd_version'] == 10].copy()
            diags = pd.merge(diags, icd_dict[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown Description"

        primary_diags = diags[diags['seq_num'] == 1]
        df = pd.merge(notes, primary_diags[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')

        df['text'] = df['text'].apply(cls.clean_noise)
        df['text'] = df['text'].apply(cls.strip_leaking_sections)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)

        result = df.sample(n=min(len(df), n_samples), random_state=42)
        debug_print("DATA", f"Sampled {len(result)} records")
        return result


# ============================================================
# SECTION 3: STRUCTURED MEDICAL PERTURBATION
# ============================================================
ACCEPTED_ENTITY_LABELS = {
    "DISEASE", "CHEMICAL", "ANATOMY", "PROCEDURE",
    "ORGANISM", "CELL_TYPE", "CELL_LINE", "PROTEIN", "GENE_OR_GENE_PRODUCT",
}

MIN_ENTITY_CHARS = 5

UNSAFE_ENTITY_PATTERNS = [
    r'^(no|not|denies|denied|without|negative|none|absent|absence)\b',
    r'^(positive|negative)$',
    r'^(left|right|bilateral|unilateral|ipsilateral|contralateral)$',
    r'^(elevated|increased|decreased|reduced|low|high|normal|abnormal|'
    r'mild|moderate|severe|minimal|significant|acute|chronic|stable)$',
    r'^(minute|second|hour|day|week|month|year)s?$',
    r'^\d',
    r'^[A-Z]$',
    r'^(mr|mrs|ms|dr|sir)\.?$',
    r'^(anterior|posterior|medial|lateral|proximal|distal|superior|inferior|dorsal|ventral)$',
    r'^(sharp|dull|adequate|frequent|complete|general|intact|patent)$',
    r'^(morning|evening|nighttime|overnight|recent|daily)$',
]
_UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_ENTITY_PATTERNS]

GENERIC_MODEL_STOP_ENTITIES = {
    "admitted", "hospitalized", "monitored", "tolerated", "resolved",
    "transitioned", "irrigated", "presented", "transferred", "discharged",
    "ambulated", "voided", "ambulating", "tolerating", "weaned",
    "pedestrian", "gentleman", "patient", "history", "details", "reports",
    "services", "intake", "arrival", "recovery", "presentation", "appointment",
    "cardiologist", "outpatient", "findings", "status",
    "hpi", "pmh", "ros", "pacu", "npo", "mr", "mrs", "ms", "dr",
    "hospital course", "operative report", "invasive procedure", "history of", "history", "hx", "primary cardiologist", "cardiologist", 
"dosing", "dose", "baseline creatinine", "creatinine", "baseline", 
"biopsies", "biopsy", "grade ii", "grade", "volume overloaded", "volume", 
    "discharge", "admit",
    "spontaneously", "naturally", "resolved", "appropriate", "suitable",
    "unchanged", "stable", "multiple", "several", "various", "toilet",
    "soft", "flat", "tender", "normal", "abnormal",
    "states", "reports", "denies", "endorses", "notes",
    "age", "disease", "condition", "illness", "problem", "issue",
    "procedure", "operation", "surgery", "surgical", "operative",
    "woman", "man", "male", "female", "person", "individual",
    "anterior", "posterior", "medial", "lateral", "proximal", "distal",
    "superior", "inferior", "dorsal", "ventral",
    "sharp", "dull", "adequate", "frequent", "intact", "patent",
    "morning", "evening", "nighttime", "overnight",
    "examination", "observation", "course", "period", "level",
    "plan", "note", "team", "floor", "ward", "clinic", "office",
    "placement", "therapy", "treatment", "response", "assessment"
}


def _clean_synonym_echo(synonym: str, original: str) -> str:
    synonym = synonym.replace('\u2192', '->').replace('\u27A1', '->')
    if " -> " in synonym:
        parts   = [p.strip() for p in synonym.split("->")]
        synonym = parts[-1].strip()
    orig_escaped = re.escape(original.strip())
    punct_sep    = r'[\s]*[.;:\-]+[\s]+'
    prefix_echo  = re.match(rf'^{orig_escaped}{punct_sep}(.+)$', synonym, re.IGNORECASE)
    if prefix_echo:
        synonym = prefix_echo.group(1).strip()
    else:
        orig_tokens = original.strip().lower().split()
        syn_tokens  = synonym.lower().split()
        n = len(orig_tokens)
        if len(syn_tokens) > n and syn_tokens[:n] == orig_tokens:
            synonym = " ".join(synonym.split()[n:]).strip()
    synonym = synonym.strip('"\'\u2018\u2019\u201c\u201d')
    synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
    return synonym


def _assign_heuristic_label(ent) -> str:
    text_lower = ent.text.lower().strip()
    anatomy_keywords = {
        "artery", "vein", "lobe", "spine", "abdomen", "chest", "hip",
        "knee", "shoulder", "femur", "tibia", "fibula", "humerus", "pelvis",
        "thorax", "lung", "liver", "kidney", "bowel", "colon", "rectum",
        "bladder", "uterus", "ovary", "prostate", "brain", "cortex",
        "ventricle", "aorta", "trachea", "esophagus", "stomach",
    }
    procedure_keywords = {
        "resection", "biopsy", "catheterization", "angioplasty", "bypass",
        "arthroplasty", "hemiarthroplasty", "cholecystectomy", "appendectomy",
        "colectomy", "cystectomy", "prostatectomy", "hysterectomy",
        "lobectomy", "pneumonectomy", "nephrectomy", "transplant",
        "intubation", "extubation", "dialysis", "chemotherapy", "radiation",
        "stenting", "thrombectomy", "embolectomy", "endarterectomy",
        "imaging", "scan", "mri", "ct", "echocardiogram", "ultrasound",
        "x-ray", "endoscopy", "colonoscopy", "bronchoscopy",
    }
    if ent[0].pos_ == "PROPN" or (ent.text.isupper() and len(ent.text) <= 6):
        return "CHEMICAL"
    for word in anatomy_keywords:
        if word in text_lower:
            return "ANATOMY"
    for word in procedure_keywords:
        if word in text_lower:
            return "PROCEDURE"
    return "OBSERVATION"


def entity_is_medical(ent, accepted_labels, generic_stop_set) -> tuple:
    if accepted_labels is not None:
        if ent.label_ not in accepted_labels:
            return False, f"non-medical label: {ent.label_}"
    text = ent.text.strip()
    if len(text) < MIN_ENTITY_CHARS:
        return False, "too short"
    for pattern in _UNSAFE_COMPILED:
        if pattern.match(text):
            return False, "unsafe pattern"
    lower = text.lower()
    if lower.startswith(('no ', 'not ', 'denies ', 'without ')):
        return False, "negation phrase"
    if accepted_labels is None and lower in generic_stop_set:
        return False, "generic non-medical word"
    # 1. Block general non-content words at the start of the entity
    if ent[0].pos_ in {"VERB", "ADP", "DET", "PRON", "CCONJ", "SCONJ",
                        "PUNCT", "PART", "NUM", "SPACE"}:
        return False, f"non-content POS: {ent[0].pos_}"

    # 2. Block descriptive modifiers at the NER level
    # If the grammatical root of the entity is an Adjective or Adverb (e.g., "warm", "regular", "mild")
    # it is a descriptor, not a disease or chemical. Drop it.
    if ent.root.pos_ in {"ADJ", "ADV"}:
        return False, f"standalone modifier: {ent.root.pos_}"
    label_str = _assign_heuristic_label(ent) if accepted_labels is None else (ent.label_ or "ENTITY")
    return True, label_str


# ============================================================
# SECTION 3b: ATTACK PROMPTS
# ============================================================
PROMPT_LEVEL_0 = """\
You are a clinical NLP expert preparing a paraphrase test set.

TASK: For each numbered medical term, output ONE synonym that means the
same thing. This tests whether the same clinical concept can be expressed
differently without changing any diagnosis.

RULES:
- Output ONLY the replacement. No arrows (->), no original term, no explanation.
- 1-4 words. Same clinical meaning. Do not change polarity or severity.
- If no good synonym exists, write NONE.

EXAMPLES (format only - do not reuse these):
1. myocardial infarction  -> heart attack
2. hypertension           -> high blood pressure
3. aspirin                -> acetylsalicylic acid
4. dyspnea                -> shortness of breath
5. cholecystectomy        -> gallbladder removal

Terms:
{term_lines}

Synonyms (one per line, number then dot then term):"""

PROMPT_LEVEL_1 = """\
You are a clinical NLP expert preparing a semantic generalisation test set.

TASK: For each numbered medical term, output ONE broader or related term
from the same clinical category.

RULES:
- Output ONLY the replacement. No arrows, no original, no explanation.
- 1-4 words. Stay in the same organ system / drug class / procedure family.
- Use a superordinate (broader) term, a drug class, or a procedure category.
- Do NOT negate, do NOT change laterality.

EXAMPLES:
1. STEMI                  -> acute coronary syndrome
2. heparin                -> anticoagulant
3. echocardiogram         -> cardiac imaging
4. appendicitis           -> abdominal pathology
5. clopidogrel            -> antiplatelet agent
6. lobar pneumonia        -> pulmonary infection

Terms:
{term_lines}

Generalisations (one per line):"""

PROMPT_LEVEL_2 = """\
You are a clinical NLP expert building an adversarial ICD-coding test set.

TASK: For each numbered medical term, output ONE plausible differential
diagnosis or clinically similar alternative.

RULES:
- The replacement must come from a different ICD-10 subcategory (different 3-character code prefix) than the original term
- 1-4 words. Must be medically plausible in a similar clinical context.
- Choose the closest differential (overlapping symptoms / same presentation).
- Do NOT simply negate. Write NONE if no plausible differential exists.

EXAMPLES:
1. pulmonary embolism      -> pneumonia
2. appendicitis            -> mesenteric adenitis
3. myocardial infarction   -> aortic dissection
4. ischemic stroke         -> hemorrhagic stroke
5. Crohn disease           -> ulcerative colitis
6. bacterial pneumonia     -> viral pneumonitis
7. deep vein thrombosis    -> cellulitis

Terms:
{term_lines}

Differentials (one per line):"""

PROMPT_LEVEL_3 = """\
You are a clinical adversarial NLP researcher generating a cross-category
ICD confusion test set for a robustness study.

TASK: For each numbered medical term, output ONE term from a DIFFERENT
ICD-10 chapter that shares enough surface-level symptom overlap that an
automated coding model could be misled.

RULES:
- Output ONLY the replacement. No arrows, no original, no explanation.
- 1-4 words. The replacement must come from a different ICD-10 major category.
- Choose terms that share presenting symptoms but have a different root cause.
- Do NOT use generic terms like "disease" or "condition". Be specific.
- Write NONE only if no cross-category confusion is possible.

EXAMPLES (showing cross-chapter drift):
1. heart failure           -> COPD exacerbation
2. bacterial pneumonia     -> lung adenocarcinoma
3. hepatic encephalopathy  -> alcohol withdrawal
4. acute kidney injury     -> diabetic ketoacidosis
5. septic shock            -> pulmonary embolism
6. hypertensive crisis     -> panic disorder
7. bowel obstruction       -> mesenteric ischemia
8. community pneumonia     -> congestive heart failure

Terms:
{term_lines}

Cross-category alternatives (one per line):"""

ATTACK_PROMPTS = {0: PROMPT_LEVEL_0, 1: PROMPT_LEVEL_1, 2: PROMPT_LEVEL_2, 3: PROMPT_LEVEL_3}
ATTACK_PERTURBATION_RATES = {
    0: 0.20,  
    1: 0.30,  
    2: 0.45,  
    3: 0.60   
}

_NONE_SENTINELS = {
    "none", "(no change)", "no change", "n/a", "\u2014", "-", "",
    "unchanged", "same", "keep", "as is", "no synonym", "not applicable",
}


def find_synonyms_batch(terms: list, attack_level: int = 0) -> dict:
    if not terms:
        return {}
    term_lines = "\n".join(
        f"{i+1}. [{item['label']}] \"{item['text']}\"  "
        f"(context: \"{item['context'][:80]}...\")"
        for i, item in enumerate(terms)
    )
    prompt = ATTACK_PROMPTS.get(attack_level, PROMPT_LEVEL_0).format(term_lines=term_lines)
    response = query_rwth_server(prompt, max_tokens=len(terms) * 14 + 80)
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
            synonym  = _clean_synonym_echo(raw_syn, original)
            synonym  = synonym.strip('"\'')
            synonym  = re.sub(r'[.!?,;]+$', '', synonym).strip()
            if not synonym or synonym.lower() in _NONE_SENTINELS:
                synonym_map[original] = None
            elif synonym.lower() == original.lower():
                synonym_map[original] = None
            else:
                synonym_map[original] = synonym
    return synonym_map


def replacement_is_structurally_valid(original: str, replacement: str) -> tuple:
    MEANING_INVERTERS = {'no', 'not', 'absence', 'without', 'lack', 'negative'}
    if " -> " in replacement:
        return False, "contains arrow"
    if replacement.lower() in _NONE_SENTINELS:
        return False, "no-change placeholder"
    orig_word_list = original.strip().split()
    repl_word_list = replacement.strip().split()
    n_orig = len(orig_word_list)
    n_repl = len(repl_word_list)
    is_abbreviation = original.strip().isupper() and len(original.strip()) <= 5
    word_limit = 6 if is_abbreviation else max(n_orig * 2, n_orig + 3)
    repl_word_set = set(replacement.lower().split())
    orig_word_set = set(original.lower().split())
    if (repl_word_set - orig_word_set) & MEANING_INVERTERS:
        return False, "replacement introduces negation"
    if n_repl > word_limit:
        return False, f"too long ({n_orig}->{n_repl} words)"
    if n_orig == 1 and n_repl > 3:
        return False, f"single-word expanded to {n_repl} words"
    if not replacement.strip():
        return False, "empty replacement"
    if replacement.strip().lower() == original.strip().lower():
        return False, "identical to original"
    if len(replacement) / max(len(original), 1) > 5.0:
        return False, "character ratio too high"
    if re.search(r'[.!?](?!\s*$)', replacement):
        return False, "mid-sentence punctuation"
    if len(repl_word_list) < len(orig_word_list) * 0.4:
        return False, "replacement too sparse"
    return True, "ok"


def validate_replacement_with_llm(original: str, replacement: str, context: str) -> tuple:
    prompt = (
        f"Context: \"{context}\"\n"
        f"Original: '{original}' | Proposed replacement: '{replacement}'\n\n"
        f"Answer YES unless the replacement would change the clinical meaning, "
        f"polarity, or diagnosis. Synonyms, abbreviation expansions, and "
        f"equivalent phrasings should always be YES.\n"
        f"Answer YES or NO only."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    is_safe  = response.strip().upper().startswith("YES")
    return is_safe, ("approved" if is_safe else "rejected")


class StructuredMedicalPerturber:
    _SAFETY_GATE_LEVELS = {0, 1}

    def __init__(self):
        for model_name in ["en_core_sci_lg", "en_core_sci_md"]:
            try:
                self.nlp = spacy.load(model_name)
                debug_print("PERTURBER", f"Loaded {model_name}")
                break
            except OSError:
                continue
        else:
            raise RuntimeError("No scispacy model found.")

        test_doc     = self.nlp("The patient had myocardial infarction and was given aspirin.")
        found_labels = {ent.label_ for ent in test_doc.ents}
        self.effective_labels = None if (not found_labels or found_labels <= {"ENTITY", ""}) else ACCEPTED_ENTITY_LABELS
        debug_print("PERTURBER", f"Labels: {found_labels} | Mode: {'typed' if self.effective_labels else 'heuristic'}")

    def _is_negated(self, token) -> bool:
        neg    = {'no', 'not', 'denies', 'negative', 'without', 'none', 'denied'}
        window = [t.text.lower() for t in token.doc[max(0, token.i - 3):token.i]]
        return any(t in window for t in neg)

    def extract_medical_entities(self, text: str) -> list:
        doc      = self.nlp(text)
        entities = []
        seen     = set()
        for ent in doc.ents:
            valid, label_str = entity_is_medical(ent, self.effective_labels, GENERIC_MODEL_STOP_ENTITIES)
            if not valid or self._is_negated(ent[0]):
                continue
            key = ent.text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            sent_text = ent.sent.text if ent.sent else text[max(0, ent.start_char-60):ent.end_char+60]
            entities.append({
                'text': ent.text, 'label': label_str,
                'start_char': ent.start_char, 'end_char': ent.end_char,
                'context': sent_text,
            })
        return entities

    def validate_and_apply(self, text: str, entities: list,
                       synonym_map: dict,
                       attack_level: int = 0,
                       scorer=None) -> tuple:          # ← add scorer parameter
        accepted = []
        apply_safety_gate = attack_level in self._SAFETY_GATE_LEVELS

        # Minimum ClinicalBERT cosine similarity between original and replacement
        
        MIN_SUBST_SIM = {
            0: 0.94,  # ULTRA-STRICT: Must be a near-perfect synonym.
            1: 0.88,  # STRICT: Broadening should still be highly similar.
            2: 0.75,  # MODERATE: Differentials will score around here.
            3: 0.00,  # OFF
        }
        sim_threshold = MIN_SUBST_SIM.get(attack_level, 0.0)

        for ent in entities:
            original  = ent['text']
            candidate = synonym_map.get(original)

            if not candidate:
                print(f"       SKIP (no synonym)   : '{original}'")
                continue

            # Layer 2: structural validation
            struct_ok, struct_reason = replacement_is_structurally_valid(
                original, candidate)
            if not struct_ok:
                print(f"       LAYER2 BLOCKED ({struct_reason}): "
                    f"'{original}' -> '{candidate}'")
                continue

            # Layer 2.5: per-substitution semantic similarity gate
            # Runs on the short term pair (1-4 words) where ClinicalBERT
            # pooling is accurate — unlike full-note pooling which gives ~0.999.
            # Skipped entirely when scorer is None or threshold is 0 (L3).
            if scorer is not None and sim_threshold > 0:
                pair_sim = scorer._bert_cosine(original, candidate)
                if pair_sim < sim_threshold:
                    print(f"       LAYER2.5 BLOCKED (sim={pair_sim:.3f} "
                        f"< threshold={sim_threshold}): "
                        f"'{original}' -> '{candidate}'")
                    continue
                else:
                    print(f"       LAYER2.5 PASSED (sim={pair_sim:.3f} "
                        f">= threshold={sim_threshold}): "
                        f"'{original}' -> '{candidate}'")

            # Layer 3: LLM clinical safety gate (L0 and L1 only)
            if apply_safety_gate:
                llm_ok, _ = validate_replacement_with_llm(
                    original, candidate, ent['context'])
                if not llm_ok:
                    print(f"       LAYER3 BLOCKED (clinical): "
                        f"'{original}' -> '{candidate}'")
                    continue
            else:
                print(f"       LAYER3 SKIPPED (adversarial level {attack_level}): "
                    f"'{original}' -> '{candidate}'")

            accepted.append({
                'original'   : original,
                'replacement': candidate,
                'label'      : ent['label'],
                'start_char' : ent['start_char'],
                'end_char'   : ent['end_char'],
            })
            print(f"       ACCEPTED [{ent['label']:20s}]: "
                f"'{original}' -> '{candidate}'")

        if not accepted:
            return text, []

        # Replace ALL occurrences of each accepted term (not just first NER span)
        def _replace_all(t: str, original: str, replacement: str) -> str:
            return re.sub(rf'\b{re.escape(original)}\b', replacement, t,
                        flags=re.IGNORECASE)

        perturbed = text
        for item in accepted:
            perturbed = _replace_all(perturbed, item['original'], item['replacement'])

        changes_log = [(i['original'], i['replacement'], i['label'])
                    for i in accepted]
        return perturbed, changes_log

    def perturb(self, text: str,
            perturbation_rate: float = None,
            attack_level: int = 0,
            scorer=None) -> tuple:              # ← add scorer parameter
        if perturbation_rate is None:
            perturbation_rate = ATTACK_PERTURBATION_RATES.get(attack_level, 0.30)

        all_entities = self.extract_medical_entities(text)
        if not all_entities:
            return text, []

        n               = max(1, int(len(all_entities) * perturbation_rate))
        target_entities = random.sample(all_entities, min(n, len(all_entities)))
        synonym_map     = find_synonyms_batch(target_entities, attack_level=attack_level)

        return self.validate_and_apply(text, target_entities, synonym_map,
                                    attack_level=attack_level,
                                    scorer=scorer)              # ← pass throug


# ============================================================
# SECTION 4: SEMANTIC EVALUATOR
# ============================================================
def _normalized_edit_distance(s1: str, s2: str) -> float:
    w1 = s1.lower().split()
    w2 = s2.lower().split()
    m, n = len(w1), len(w2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] if w1[i-1] == w2[j-1] else 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, [0] * (n + 1)
    return 1.0 - prev[n] / max(m, n)


class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT (CPU)...")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model     = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def _bert_cosine(self, text1: str, text2: str) -> float:
        with torch.no_grad():
            inputs  = self.tokenizer([text1, text2], padding=True, truncation=True,
                                     max_length=512, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            mask    = inputs['attention_mask'].unsqueeze(-1).float()
            e1 = (outputs.last_hidden_state[0] * mask[0]).sum(0) / mask[0].sum()
            e2 = (outputs.last_hidden_state[1] * mask[1]).sum(0) / mask[1].sum()
            sim = cosine_similarity(e1.unsqueeze(0).numpy(), e2.unsqueeze(0).numpy())[0][0]
        return float(sim)

    def get_sim(self, text1: str, text2: str) -> tuple:
        """Returns (bert_cosine_sim, edit_similarity)."""
        return self._bert_cosine(text1, text2), _normalized_edit_distance(text1, text2)

    def descriptions_match(self, desc1: str, desc2: str,
                            threshold: float = BASELINE_SIM_THRESHOLD) -> tuple:
        sim = self._bert_cosine(desc1, desc2)
        return sim >= threshold, sim

    # -- FEATURE 1: substitution_semantic_similarity --------------------------
    def substitution_semantic_similarity(self, changes_log: list,
                                         weights: dict = None) -> float:
        """
        Average ClinicalBERT cosine similarity between each (original, replacement)
        term pair, weighted by entity type importance.

        This is the correct context-preservation metric for clinical notes because:
          - It operates on short 1-4 word spans where BERT pooling works correctly
            (unlike full-note pooling which gives ~0.999 for everything)
          - It measures semantic distance in the clinical embedding space, not
            surface character overlap
          - Weighting by entity type (DISEASE > PROCEDURE > CHEMICAL > ANATOMY)
            reflects that disease-term swaps matter more for ICD coding

        Expected ranges by attack level:
          Level 0 (paraphrase)    : 0.80-0.95  - same concept, different surface
          Level 1 (broadening)    : 0.60-0.80  - same category, less specific
          Level 2 (differential)  : 0.35-0.60  - related but distinct condition
          Level 3 (adversarial)   : 0.15-0.40  - different ICD chapter entirely
        """
        if not changes_log:
            return 1.0   # no changes = perfect preservation

        w = weights or ENTITY_WEIGHTS
        total_sim, total_weight = 0.0, 0.0

        for original, replacement, label in changes_log:
            sim    = self._bert_cosine(original, replacement)
            weight = w.get(label, 0.7)
            total_sim    += sim * weight
            total_weight += weight

        return total_sim / total_weight if total_weight > 0 else 1.0
    # -------------------------------------------------------------------------


# ============================================================
# SECTION 5: BASELINE CORRECTNESS
# ============================================================
def check_clinical_equivalence(pred_icd: str, gt_icd: str, gt_desc: str) -> bool:
    if pred_icd in ("UNKNOWN", ""):
        return False
    prompt = (
        f"Ground truth diagnosis: '{gt_desc}' (ICD-10 code: {gt_icd})\n"
        f"Predicted ICD-10 code: {pred_icd}\n\n"
        f"Does the predicted code represent the same primary medical condition "
        f"as the ground truth? Minor specificity differences are acceptable.\n"
        f"Answer YES or NO only."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")


def _icd_string_match(pred_icd: str, gt_icd: str) -> tuple:
    if pred_icd in ("UNKNOWN", "") or gt_icd in ("UNKNOWN", ""):
        return False, False
    pred_norm = re.sub(r'[^A-Z0-9]', '', pred_icd.strip().upper())
    gt_norm   = re.sub(r'[^A-Z0-9]', '', gt_icd.strip().upper())
    exact_match  = pred_norm == gt_norm
    prefix_match = (len(pred_norm) >= 3 and len(gt_norm) >= 3
                    and pred_norm[:3] == gt_norm[:3])
    return exact_match, prefix_match


def is_baseline_correct(scorer, orig_desc, gt_desc, orig_icd, gt_icd) -> tuple:
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    exact_match, prefix_match = _icd_string_match(orig_icd, gt_icd)

    if exact_match or prefix_match:
        gate_parts = [
            f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'}",
            f"exact_icd={'YES' if exact_match else 'NO'}",
            f"prefix_icd={'YES' if prefix_match else 'NO'}",
            "llm=SKIPPED",
        ]
        return True, baseline_sim, None, " | ".join(gate_parts)

    # if prefix_match and baseline_sim >= 0.85:
    #     gate_parts = [
    #         f"sem_sim={baseline_sim:.4f}v",
    #         "exact_icd=NO", "prefix_icd=YES", "desc_sim=YES (>=0.85)", "llm=SKIPPED",
    #     ]
    #     return True, baseline_sim, None, " | ".join(gate_parts)

    llm_ok      = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)
    high_sim_ok = baseline_sim >= 0.80
    passed      = llm_ok and high_sim_ok   # AND, not OR

    gate_parts = [
        f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'}",
        "exact_icd=NO", "prefix_icd=NO", "desc_sim=NO",
        f"llm={'YES' if llm_ok else 'NO'}",
        f"high_sim={'YES' if high_sim_ok else 'NO'}",
        f"passed={'YES' if passed else 'NO'}",
    ]
    return passed, baseline_sim, llm_ok, " | ".join(gate_parts)


# ============================================================
# SECTION 6: ICD-10 PREDICTION & DESCRIPTION
# ============================================================
def get_icd_prediction(text: str) -> str:
    primary_prompt = (
        f"Read the clinical note below and return the single most likely ICD-10-CM diagnosis code.\n"
        f"Output format: one uppercase letter followed by 2 digits, then optionally a dot and 1-2 more digits.\n"
        f"Valid examples: J18.9  I21.0  K56.9  N39.0  M17.11  A41.9\n"
        f"Do NOT output a disease name, sentence, or any explanation.\n"
        f"Output the ICD-10 code ONLY - nothing else.\n\n"
        f"Clinical note:\n{text[:2500]}\n\nICD-10 code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code for this note. "
        f"Format: letter + 2 digits + optional dot + digits. Examples: J18.9, I21.0, K56.9\n\n"
        f"Note:\n{text[:800]}\n\nCode:"
    )
    for prompt in [primary_prompt, fallback_prompt]:
        response = query_rwth_server(prompt)
        if response:
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
        f"Reply with ONLY the condition name in 2-5 words. No explanation."
    )
    return query_rwth_server(prompt)


def get_icd_chapter(icd_code: str) -> str:
    """Map an ICD-10 code to its chapter letter/range for chapter-wise drift plots."""
    if not icd_code or icd_code == "UNKNOWN":
        return "Unknown"
    code = re.sub(r'[^A-Z0-9]', '', icd_code.strip().upper())
    if not code:
        return "Unknown"
    letter = code[0]
    # ICD-10-CM chapter mapping (first letter of 3-char category)
    chapter_map = {
        'A': 'Infectious', 'B': 'Infectious',
        'C': 'Neoplasms',  'D': 'Neoplasms/Blood',
        'E': 'Endocrine',
        'F': 'Mental',
        'G': 'Nervous',
        'H': 'Eye/Ear',
        'I': 'Circulatory',
        'J': 'Respiratory',
        'K': 'Digestive',
        'L': 'Skin',
        'M': 'Musculoskeletal',
        'N': 'Genitourinary',
        'O': 'Pregnancy',
        'P': 'Perinatal',
        'Q': 'Congenital',
        'R': 'Symptoms/Signs',
        'S': 'Injury', 'T': 'Injury/Poisoning',
        'V': 'External', 'W': 'External', 'X': 'External', 'Y': 'External',
        'Z': 'Health Status',
    }
    return chapter_map.get(letter, f"Chapter {letter}")


# ============================================================
# SECTION 7: DRIFT CLASSIFICATION
# ============================================================
def classify_drift(orig_icd: str, pert_icd: str) -> dict:
    orig_norm = re.sub(r'[^A-Z0-9]', '', orig_icd.upper())
    pert_norm = re.sub(r'[^A-Z0-9]', '', pert_icd.upper())
    if "UNKNOWN" in (orig_norm, pert_norm):
        return {"icd_codes_differ": True, "drift_level": "unknown",
                "categorical_drift": False, "subcategory_drift": False}
    exact  = orig_norm == pert_norm
    prefix = len(orig_norm) >= 3 and len(pert_norm) >= 3 and orig_norm[:3] == pert_norm[:3]
    level  = "none" if exact else ("subcategory" if prefix else "categorical")
    return {
        "icd_codes_differ" : not exact,
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 8: PLOTS  (features 2, 3, 4 added)
# ============================================================
"""
FIXED plot_results function - drop-in replacement for Section 8.

Changes from original:
  - Plots 1-5 (Figure 1) now loop over each attack level separately
    and generate one figure PER level (level0_plots.png etc.)
  - Chapter drift plot also runs per level
  - Feature 4 (cross-level comparison) kept as-is - it was already correct
  - A combined summary figure is added at the end

Replace the entire plot_results function in your pipeline with this.
"""

def plot_results(df: pd.DataFrame):
    debug_print("PLOT", "Generating plots...")
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return

    # Decide which similarity column to use
    if 'substitution_sim' in valid_df.columns:
        sim_col   = 'substitution_sim'
        sim_label = 'Substitution Cosine Sim (term-pair)'
    elif 'edit_similarity' in valid_df.columns:
        sim_col   = 'edit_similarity'
        sim_label = 'Token Edit Similarity'
    else:
        sim_col   = 'semantic_similarity'
        sim_label = 'BERT Cosine Similarity'

    DRIFT_COLORS = {
        'none'       : '#90EE90',
        'subcategory': '#FFD700',
        'categorical': '#FF6B6B',
        'unknown'    : '#CCCCCC',
    }
    LEVEL_LABELS = {
        0: 'L0 - Surface Synonyms',
        1: 'L1 - Clinical Broadening',
        2: 'L2 - Differential Swap',
        3: 'L3 - Cross-Chapter Adversarial',
    }
    LEVEL_COLORS = {0: '#4ECDC4', 1: '#45B7D1', 2: '#FFD700', 3: '#FF6B6B'}

    levels = sorted(valid_df['attack_level'].unique()) \
             if 'attack_level' in valid_df.columns else [None]

    # -- FIGURE PER LEVEL (Plots 1-5) -----------------------------------------
    # Each attack level gets its own clean figure.
    # This replaces the original Figure 1 which mixed all levels.
    for level in levels:
        if level is not None:
            lvl_df   = valid_df[valid_df['attack_level'] == level]
            lvl_tag  = f"L{level}"
            lvl_name = LEVEL_LABELS.get(level, f"Level {level}")
        else:
            lvl_df   = valid_df
            lvl_tag  = "all"
            lvl_name = "All levels combined"

        if len(lvl_df) == 0:
            continue

        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        fig.suptitle(f"Robustness Analysis - {lvl_name}  (n={len(lvl_df)})",
                     fontsize=13, y=1.02)

        # -- Plot 1: Semantic similarity distribution ----------------------
        ax = axes[0]
        sim_vals = lvl_df[sim_col].dropna()
        ax.hist(sim_vals, bins=15, color=LEVEL_COLORS.get(level, 'teal'),
                edgecolor='black', alpha=0.85)
        ax.set_title(f"{sim_label}\n{lvl_tag}: mean={sim_vals.mean():.4f}")
        ax.set_xlabel(sim_label)
        ax.set_ylabel("Count")
        if len(sim_vals) > 0:
            ax.set_xlim(max(0.0, sim_vals.min() - 0.05), 1.01)

        # -- Plot 2: Drift breakdown ---------------------------------------
        ax = axes[1]
        drift_order  = ['none', 'subcategory', 'categorical', 'unknown']
        drift_counts = (lvl_df['drift_level']
                        .value_counts()
                        .reindex(drift_order)
                        .dropna())
        drift_counts.plot(
            kind='bar',
            color=[DRIFT_COLORS.get(k, 'grey') for k in drift_counts.index],
            ax=ax, edgecolor='black', alpha=0.85)
        cat_pct = lvl_df['categorical_drift'].mean() * 100
        sub_pct = lvl_df['subcategory_drift'].mean() * 100
        ax.set_title(f"Drift Breakdown - {lvl_tag}\nCat: {cat_pct:.1f}%  Sub: {sub_pct:.1f}%")
        ax.set_xlabel("Drift Level")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=0)
        # Annotate bars with count
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                        str(int(h)), ha='center', va='bottom', fontsize=9)

        # -- Plot 3: Changes vs Similarity scatter -------------------------
        ax = axes[2]
        for dlevel in ['none', 'subcategory', 'categorical']:
            mask = lvl_df['drift_level'] == dlevel
            if mask.sum() == 0:
                continue
            ax.scatter(
                lvl_df[mask]['changes_count'],
                lvl_df[mask][sim_col],
                label=dlevel,
                color={'none':'green', 'subcategory':'orange',
                       'categorical':'red'}[dlevel],
                alpha=0.7, s=60)
        ax.set_title(f"Terms Changed vs {sim_label}\n{lvl_tag}")
        ax.set_xlabel("Medical Terms Changed")
        ax.set_ylabel(sim_label)
        ax.legend(fontsize=8)

        # -- Plot 4: Entity type breakdown ---------------------------------
        ax = axes[3]
        type_cols = ['disease_changes', 'chemical_changes',
                     'anatomy_changes', 'procedure_changes']
        available = [c for c in type_cols
                     if c in lvl_df.columns and lvl_df[c].sum() > 0]
        if available:
            totals = lvl_df[available].sum()
            totals.index = [c.replace('_changes', '').capitalize()
                             for c in totals.index]
            totals.plot(kind='bar',
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                        ax=ax, edgecolor='black', alpha=0.85)
            ax.set_title(f"Perturbations by Entity Type\n{lvl_tag}")
            ax.set_xlabel("Entity Type")
            ax.set_ylabel("Total Changes")
            ax.tick_params(axis='x', rotation=0)
        else:
            ax.text(0.5, 0.5, 'No entity type data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Entity Types - {lvl_tag}")

        # -- Plot 5: Similarity box plot by drift level --------------------
        ax = axes[4]
        drift_groups = [
            lvl_df[lvl_df['drift_level'] == dlvl][sim_col].dropna().values
            for dlvl in ['none', 'subcategory', 'categorical']
            if dlvl in lvl_df['drift_level'].values
        ]
        drift_labels_bp = [
            dlvl for dlvl in ['none', 'subcategory', 'categorical']
            if dlvl in lvl_df['drift_level'].values
        ]
        if drift_groups:
            bp = ax.boxplot(drift_groups, labels=drift_labels_bp,
                            patch_artist=True)
            for patch, dlvl in zip(bp['boxes'], drift_labels_bp):
                patch.set_facecolor(DRIFT_COLORS.get(dlvl, 'grey'))
                patch.set_alpha(0.8)
        ax.set_title(f"{sim_label} by Drift Level\n{lvl_tag}")
        ax.set_xlabel("Drift Level")
        ax.set_ylabel(sim_label)

        plt.tight_layout()
        out_path = PLOT_FILE.replace('.png', f'_{lvl_tag}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # -- CHAPTER DRIFT - per level ---------------------------------------------
    if 'orig_chapter' in valid_df.columns:
        for level in levels:
            if level is not None:
                lvl_df   = valid_df[valid_df['attack_level'] == level]
                lvl_tag  = f"L{level}"
                lvl_name = LEVEL_LABELS.get(level, f"Level {level}")
            else:
                lvl_df   = valid_df
                lvl_tag  = "all"
                lvl_name = "All levels"

            if len(lvl_df) == 0:
                continue

            chap_drift = (lvl_df.groupby('orig_chapter')['categorical_drift']
                          .agg(['mean', 'count'])
                          .rename(columns={'mean': 'drift_rate', 'count': 'n'})
                          .sort_values('drift_rate', ascending=False))
            chap_drift = chap_drift[chap_drift['n'] >= 2]

            if len(chap_drift) == 0:
                continue

            fig2, ax2 = plt.subplots(figsize=(max(10, len(chap_drift) * 0.9), 5))
            bars = ax2.bar(chap_drift.index,
                           chap_drift['drift_rate'] * 100,
                           color='#FF6B6B', edgecolor='black', alpha=0.8)
            for bar, (_, row) in zip(bars, chap_drift.iterrows()):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.5,
                         f"n={int(row['n'])}", ha='center', va='bottom', fontsize=8)
            ax2.set_title(
                f"Categorical Drift Rate by ICD-10 Chapter\n"
                f"{lvl_name}  (n={len(lvl_df)} valid cases)")
            ax2.set_xlabel("ICD-10 Chapter")
            ax2.set_ylabel("Categorical Drift Rate (%)")
            ax2.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            chap_path = PLOT_FILE.replace('.png', f'_chapter_{lvl_tag}.png')
            fig2.savefig(chap_path, dpi=150)
            plt.close(fig2)
            print(f"  Saved: {chap_path}")

    # -- ICD CODE DRIFT - per level --------------------------------------------
    if 'ground_truth_icd' in valid_df.columns:
        for level in levels:
            if level is not None:
                lvl_df  = valid_df[valid_df['attack_level'] == level]
                lvl_tag = f"L{level}"
                lvl_name = LEVEL_LABELS.get(level, f"Level {level}")
            else:
                lvl_df  = valid_df
                lvl_tag = "all"
                lvl_name = "All levels"

            if len(lvl_df) == 0:
                continue

            icd_drift = (lvl_df.groupby('ground_truth_icd')['categorical_drift']
                         .agg(['mean', 'count'])
                         .rename(columns={'mean': 'drift_rate', 'count': 'n'})
                         .sort_values('drift_rate', ascending=False))
            icd_drift = icd_drift[icd_drift['n'] >= 2].head(20)

            if len(icd_drift) == 0:
                continue

            fig3, ax3 = plt.subplots(figsize=(max(10, len(icd_drift) * 0.8), 5))
            colors3 = [
                '#FF6B6B' if r > 0.5 else
                '#FFD700' if r > 0.2 else
                '#90EE90'
                for r in icd_drift['drift_rate']
            ]
            bars = ax3.barh(icd_drift.index,
                            icd_drift['drift_rate'] * 100,
                            color=colors3, edgecolor='black', alpha=0.85)
            for bar, (_, row) in zip(bars, icd_drift.iterrows()):
                ax3.text(bar.get_width() + 0.5,
                         bar.get_y() + bar.get_height()/2,
                         f"n={int(row['n'])}", va='center', fontsize=8)
            ax3.set_title(
                f"Categorical Drift Rate by ICD-10 Code\n"
                f"{lvl_name} - top 20 codes with n>=2")
            ax3.set_xlabel("Categorical Drift Rate (%)")
            ax3.set_ylabel("Ground Truth ICD-10 Code")
            plt.tight_layout()
            icd_path = PLOT_FILE.replace('.png', f'_icd_{lvl_tag}.png')
            fig3.savefig(icd_path, dpi=150)
            plt.close(fig3)
            print(f"  Saved: {icd_path}")

    # -- FEATURE 4: Cross-level comparison ------------------------------------
    # This was already correct - kept as-is
    if 'attack_level' in valid_df.columns and valid_df['attack_level'].nunique() > 1:
        level_stats = (valid_df.groupby('attack_level')
                       .agg(
                           cat_drift=('categorical_drift', 'mean'),
                           sub_drift=('subcategory_drift', 'mean'),
                           avg_sim=(sim_col, 'mean'),
                           n=('categorical_drift', 'count'),
                       ).reset_index())

        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

        # Drift rate per level
        ax = axes4[0]
        ax.bar(level_stats['attack_level'],
               level_stats['cat_drift'] * 100,
               color=[LEVEL_COLORS.get(l, 'grey')
                      for l in level_stats['attack_level']],
               edgecolor='black', alpha=0.85)
        for _, row in level_stats.iterrows():
            ax.text(row['attack_level'],
                    row['cat_drift'] * 100 + 0.5,
                    f"{row['cat_drift']*100:.1f}%",
                    ha='center', fontsize=9)
        ax.set_title("Categorical Drift Rate\nby Attack Level")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel("Categorical Drift Rate (%)")
        ax.set_xticks(level_stats['attack_level'])
        ax.set_xticklabels([
            f"L{l}\n{LEVEL_LABELS.get(l,'').split('-')[-1].strip()}\n(n={int(r)})"
            for l, r in zip(level_stats['attack_level'], level_stats['n'])
        ], fontsize=8)

        # Substitution similarity per level
        ax = axes4[1]
        ax.plot(level_stats['attack_level'],
                level_stats['avg_sim'],
                marker='o', color='teal', linewidth=2, markersize=8)
        for _, row in level_stats.iterrows():
            ax.annotate(f"{row['avg_sim']:.3f}",
                        (row['attack_level'], row['avg_sim']),
                        textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8)
        ax.set_title(f"Avg {sim_label}\nby Attack Level (v = more semantic shift)")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel(sim_label)
        ax.set_xticks(level_stats['attack_level'])

        # Stacked drift breakdown per level
        ax = axes4[2]
        width = 0.35
        x = np.array(level_stats['attack_level'])
        ax.bar(x - width/2, level_stats['cat_drift'] * 100, width,
               label='Categorical', color='#FF6B6B',
               edgecolor='black', alpha=0.85)
        ax.bar(x + width/2, level_stats['sub_drift'] * 100, width,
               label='Subcategory', color='#FFD700',
               edgecolor='black', alpha=0.85)
        ax.set_title("Categorical vs Subcategory Drift\nby Attack Level")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel("Drift Rate (%)")
        ax.set_xticks(x)
        ax.legend()

        plt.suptitle("Attack Level Comparison (L0 -> L3)", fontsize=13, y=1.02)
        plt.tight_layout()
        level_path = PLOT_FILE.replace('.png', '_attack_levels.png')
        fig4.savefig(level_path, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Saved: {level_path}")

    plt.close('all')
    print("All plots saved.")


# ============================================================
# SECTION 9: HTML REPORT  (feature 7: only marked words perturbed)
# ============================================================
def generate_html_report(df: pd.DataFrame, filename: str = HTML_REPORT):
    DRIFT_COLORS = {
        'none': '#ccffcc', 'subcategory': '#fff3cd',
        'categorical': '#ffcccc', 'unknown': '#eeeeee', 'skipped': '#f9f9f9',
    }
    DRIFT_LABELS = {
        'none': 'STABLE', 'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT', 'unknown': 'UNKNOWN ICD', 'skipped': 'SKIPPED',
    }

    # -- FEATURE 7: only highlight the specific substituted words -------------
    def highlight_changes(original_text: str, perturbed_text: str,
                          changes_str: str) -> tuple:
        """
        Instead of regex-scanning the full note for any occurrence of every
        changed word (which marks unrelated words), we locate the EXACT
        character positions of the substitutions by diffing the two texts at
        word level, then wrap only those positions.

        Algorithm:
          1. Tokenise both texts into (word, start, end) triples.
          2. Find non-matching word positions.
          3. Wrap them in <span> tags.
        This ensures only the actually-perturbed tokens are highlighted,
        even when the same word appears elsewhere unchanged.
        """
        if not changes_str or changes_str in ('None', '', 'SKIPPED'):
            return original_text, perturbed_text

        # Build sets of (original, replacement) pairs from changes_str
        pairs = []
        for pair in changes_str.split("|"):
            pair = re.sub(r'^\[.*?\]\s*', '', pair.strip())
            if " -> " not in pair:
                continue
            ow, nw = [x.strip() for x in pair.split(" -> ", 1)]
            pairs.append((ow, nw))

        if not pairs:
            return original_text, perturbed_text

        def _tokenize(text):
            """Return list of (token, start, end) preserving whitespace."""
            return [(m.group(), m.start(), m.end())
                    for m in re.finditer(r'\S+', text)]

        def _apply_highlights(text, targets, bg_color):
            """
            Highlight only the FIRST occurrence (per original/replacement token)
            that has not already been highlighted, scanning left-to-right.
            """
            tokens  = _tokenize(text)
            result  = []
            pos     = 0
            used    = set()   # (target_lower, occurrence_index)
            counts  = {}      # how many times we've seen each target

            for token, start, end in tokens:
                result.append(text[pos:start])  # whitespace/gap before token
                token_lower = token.lower()
                match_target = None
                for t in targets:
                    if token_lower == t.lower():
                        match_target = t
                        break

                if match_target is not None:
                    idx = counts.get(match_target.lower(), 0)
                    key = (match_target.lower(), idx)
                    if key not in used:
                        used.add(key)
                        counts[match_target.lower()] = idx + 1
                        result.append(
                            f'<span style="background:{bg_color};color:white;'
                            f'padding:1px 4px;border-radius:3px;'
                            f'font-weight:bold;">{token}</span>'
                        )
                    else:
                        counts[match_target.lower()] = idx + 1
                        result.append(token)
                else:
                    result.append(token)
                pos = end

            result.append(text[pos:])   # trailing whitespace
            return "".join(result)

        orig_targets = [p[0] for p in pairs]
        pert_targets = [p[1] for p in pairs]

        orig_h = _apply_highlights(original_text,  orig_targets, '#FF6B6B')
        pert_h = _apply_highlights(perturbed_text,  pert_targets, '#51C878')
        return orig_h, pert_h
    # -------------------------------------------------------------------------

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = valid_df['categorical_drift'].mean() * 100 if len(valid_df) else 0
    sub_rate    = valid_df['subcategory_drift'].mean() * 100 if len(valid_df) else 0
    avg_changes = f"{valid_df['changes_count'].mean():.1f}" if len(valid_df) else 'N/A'

    sim_col_html   = ('substitution_sim'   if 'substitution_sim'   in valid_df.columns else
                      'edit_similarity'    if 'edit_similarity'    in valid_df.columns else
                      'semantic_similarity')
    sim_label_html = ('Substitution Cosine Sim' if sim_col_html == 'substitution_sim' else
                      'Token Edit Similarity'   if sim_col_html == 'edit_similarity'  else
                      'BERT Cosine Similarity')
    avg_sim = f"{valid_df[sim_col_html].mean():.4f}" if len(valid_df) else 'N/A'

    # per-level summary table
    level_summary_html = ""
    if 'attack_level' in valid_df.columns and valid_df['attack_level'].nunique() > 1:
        level_summary_html = "<h2>Per-Attack-Level Summary</h2><table style='border-collapse:collapse;font-size:0.9em;margin-bottom:20px;'>"
        level_summary_html += "<tr style='background:#ddd;'><th style='padding:6px 12px;border:1px solid #ccc;'>Level</th><th style='padding:6px 12px;border:1px solid #ccc;'>Description</th><th style='padding:6px 12px;border:1px solid #ccc;'>n</th><th style='padding:6px 12px;border:1px solid #ccc;'>Cat. Drift %</th><th style='padding:6px 12px;border:1px solid #ccc;'>Sub. Drift %</th><th style='padding:6px 12px;border:1px solid #ccc;'>Avg Sim</th><th style='padding:6px 12px;border:1px solid #ccc;'>Avg Terms Changed</th></tr>"
        level_descs = {0:'Surface paraphrase', 1:'Clinical broadening', 2:'Differential swap', 3:'Adversarial cross-chapter'}
        for lvl in sorted(valid_df['attack_level'].unique()):
            lvl_df  = valid_df[valid_df['attack_level'] == lvl]
            cd      = lvl_df['categorical_drift'].mean() * 100
            sd      = lvl_df['subcategory_drift'].mean() * 100
            sm      = lvl_df[sim_col_html].mean() if sim_col_html in lvl_df.columns else 0
            tc      = lvl_df['changes_count'].mean()
            level_summary_html += (
                f"<tr><td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'><b>L{lvl}</b></td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;'>{level_descs.get(lvl,'')}</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{len(lvl_df)}</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;color:#c00;'><b>{cd:.1f}%</b></td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{sd:.1f}%</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{sm:.4f}</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{tc:.1f}</td></tr>"
            )
        level_summary_html += "</table>"

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        dc          = DRIFT_COLORS.get(drift_level, '#ccffcc')
        dl          = DRIFT_LABELS.get(drift_level, 'STABLE')

        sim_val = row.get(sim_col_html)
        sim_d   = f"{float(sim_val):.4f}" if sim_val not in (None, '') else 'N/A'
        base_d  = f"{float(row.get('baseline_sim', 0)):.4f}"
        subst_d = (f"{float(row['substitution_sim']):.4f}"
                   if row.get('substitution_sim') not in (None, '') else 'N/A')

        changes_d = row['changes_made'] if row.get('changes_made') not in ['SKIPPED', '', None] else 'None'
        gate_d    = row.get('baseline_gate', '')
        attack_lv = row.get('attack_level', 'N/A')
        level_desc = {0:'Surface paraphrase', 1:'Clinical broadening',
                      2:'Differential swap', 3:'Adversarial cross-chapter'}.get(attack_lv, '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;background:#f9f9f9;border-radius:4px;">'
                f'Skipped - baseline did not match ground truth.<br>'
                f'<small>Gates: {gate_d}</small></p>')
        else:
            oh, ph = highlight_changes(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                row.get('changes_made', ''))
            text_section = f"""
            <div style="margin-top:10px;">
              <div style="margin-bottom:8px;font-size:0.85em;">
                <span style="background:#FF6B6B;color:white;padding:2px 8px;border-radius:3px;">Original terms (changed)</span>&nbsp;
                <span style="background:#51C878;color:white;padding:2px 8px;border-radius:3px;">Replacement terms</span>
              </div>
              <table style="width:100%;border-collapse:collapse;">
                <tr>
                  <th style="width:50%;background:#e8f4f8;padding:8px;border:1px solid #ccc;">Original Text</th>
                  <th style="width:50%;background:#fff8e8;padding:8px;border:1px solid #ccc;">Perturbed Text</th>
                </tr>
                <tr>
                  <td style="padding:10px;border:1px solid #ccc;vertical-align:top;white-space:pre-wrap;font-size:0.85em;line-height:1.6em;">{oh}</td>
                  <td style="padding:10px;border:1px solid #ccc;vertical-align:top;white-space:pre-wrap;font-size:0.85em;line-height:1.6em;">{ph}</td>
                </tr>
                <tr>
                  <td style="padding:8px;border:1px solid #ccc;background:#f0f0f0;text-align:center;"><b>ICD-10: {row['original_icd_pred']}</b></td>
                  <td style="padding:8px;border:1px solid #ccc;background:#f0f0f0;text-align:center;"><b>ICD-10: {row['perturbed_icd_pred']}</b></td>
                </tr>
              </table>
            </div>"""

        rows_html += f"""
        <div style="border:1px solid #ccc;margin:20px 0;padding:15px;border-radius:8px;">
          <h3 style="margin:0 0 10px 0;">
            Patient ID: {row['hadm_id']}
            <span style="background:{dc};padding:4px 10px;border-radius:4px;font-size:0.9em;">{dl}</span>
            <span style="background:#e8e8e8;padding:4px 8px;border-radius:4px;font-size:0.8em;margin-left:6px;">L{attack_lv} - {level_desc}</span>
          </h3>
          <table style="font-size:0.9em;border-collapse:collapse;margin-bottom:10px;">
            <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                <td>{row['ground_truth_desc']} ({row['ground_truth_icd']} ICD-10)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                <td>{row['original_icd_pred']} - {row.get('original_pred_desc','')} (baseline sim: {base_d})</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                <td><small>{gate_d}</small></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                <td>{changes_d} ({row['changes_count']} terms)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Substitution Cosine Sim:</b></td>
                <td>{subst_d} <span style="color:#888;font-size:0.85em;">(ClinicalBERT on term pairs - primary context metric)</span></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>{sim_label_html}:</b></td>
                <td>{sim_d}</td></tr>
          </table>
          {text_section}
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><title>Robustness Report - ICD-10</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#fafafa;}}
h1,h2{{color:#333;}} h3{{margin:0 0 10px 0;}}
.summary{{background:#f0f0f0;padding:15px;border-radius:8px;margin-bottom:20px;}}
.summary p{{margin:4px 0;}}
.legend{{display:flex;gap:16px;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;}}
.legend span{{padding:3px 10px;border-radius:4px;}}
.method-note{{background:#e8f4f8;padding:12px;border-radius:6px;margin-bottom:20px;font-size:0.9em;color:#555;}}
</style></head><body>
<h1>Clinical Note Robustness Report (ICD-10)</h1>
<p style="color:#666;">Model: {MODEL_NAME} | Attack levels 0-3 | Primary similarity: substitution cosine sim</p>
<div class="method-note">
  <b>Perturbation:</b> scispaCy NER + heuristic labels + 3-layer validation (structural + LLM for L0/L1).<br>
  <b>Baseline gate:</b> ICD string match (exact/prefix) OR (LLM=YES AND sim&gt;=0.80).<br>
  <b>Primary similarity metric:</b> Weighted ClinicalBERT cosine similarity on (original, replacement) term pairs.
  Weights: DISEASE=1.0, PROCEDURE=0.5, CHEMICAL=0.6, ANATOMY=0.4.
  This correctly measures context preservation because it operates on short spans (1-4 words)
  where BERT pooling is accurate, unlike full-note pooling which gives ~0.999 for everything.<br>
  <b>Feature 6:</b> All occurrences of each perturbed term replaced (not just first NER span).<br>
  <b>Feature 7:</b> HTML highlights only the exact substituted word positions, not unrelated occurrences.
</div>
<div class="legend">
  <span style="background:#ccffcc;">STABLE</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT</span>
</div>
<div class="summary">
  <h2>Overall Summary</h2>
  <p><b>Total Samples:</b> {len(df)}</p>
  <p><b>Valid for Analysis:</b> {len(valid_df)}</p>
  <p><b>Skipped (wrong baseline):</b> {len(df) - len(valid_df)}</p>
  <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}% <span style="color:#888;">(primary robustness metric)</span></p>
  <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%</p>
  <p><b>Avg {sim_label_html}:</b> {avg_sim}</p>
  <p><b>Avg Medical Terms Changed Per Note:</b> {avg_changes}</p>
</div>
{level_summary_html}
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {filename}")


# ============================================================
# SECTION 10: SANITY CHECK
# ============================================================
def sanity_check(perturber: StructuredMedicalPerturber):
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    sample   = ("The patient had myocardial infarction. "
                "Aspirin and heparin were administered. "
                "Echocardiogram showed reduced ejection fraction.")
    entities = perturber.extract_medical_entities(sample)
    print(f"  Entity extraction: {len(entities)} entities")
    for e in entities:
        print(f"    [{e['label']}] {e['text']}")
    labels_found = {e['label'] for e in entities}
    print(f"  Typed labels: {'PASS' if labels_found - {'ENTITY',''} else 'WARNING: all ENTITY'}")

    test_cases = [
        ("nausea",              "nausea -> VOMITING",                              "VOMITING"),
        ("LACTOSE INT",         "LACTOSE INT -> LACTOSE SENSITIVITY",               "LACTOSE SENSITIVITY"),
        ("vital signs",         "vital signs -> VITALS",                            "VITALS"),
        ("left knee arthritis", "left knee arthritis  . left knee osteoarthritis", "left knee osteoarthritis"),
        ("chest pain",          "chest pain chest discomfort",                     "chest discomfort"),
    ]
    all_pass = True
    for original, raw, expected in test_cases:
        result = _clean_synonym_echo(raw, original)
        ok     = result == expected
        all_pass = all_pass and ok
        print(f"  Echo-strip: '{raw}' -> '{result}' ({'PASS' if ok else 'FAIL'})")
    print(f"  Echo-strip overall: {'PASS' if all_pass else 'SOME FAILED'}")

    exact, prefix = _icd_string_match("I21.0", "I21.0")
    print(f"  ICD exact  I21.0==I21.0 : {'PASS' if exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "I21.9")
    print(f"  ICD prefix I21.0~I21.9  : {'PASS' if prefix and not exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "J18.9")
    print(f"  ICD no-match I21.0/J18.9: {'PASS' if not exact and not prefix else 'FAIL'}")
    print("=" * 60 + "\n")


# ============================================================
# SECTION 11: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Starting | LLM: {MODEL_NAME} | All 4 attack levels")

    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to RWTH server. Check VPN.")
        return
    print("LLM API OK")

    try:
        df = pd.read_csv("palak_final/stratified_dataset.csv")
    except Exception as e:
        import traceback
        debug_print("ERROR", f"Data load failed: {e}")
        traceback.print_exc()
        return

    perturber = StructuredMedicalPerturber()
    scorer    = SemanticEvaluator()
    sanity_check(perturber)

    # -- PHASE 1: predict baseline ONCE per note, cache results ---------------
    # The LLM has temperature=0.1, so re-predicting the same note on every
    # attack level can return slightly different codes, causing a note to pass
    # the baseline gate at level 0 but fail at level 2.  By caching here we
    # guarantee identical baseline treatment across all 4 levels and save
    # 3 x N_SAMPLES LLM calls (300 calls for 100 samples).
    #
    # Cache schema per hadm_id:
    #   orig_icd      - predicted ICD-10 code for the unperturbed note
    #   orig_desc     - LLM description of that code
    #   baseline_ok   - whether this note passes the correctness gate
    #   baseline_sim  - BERT cosine sim of (orig_desc, gt_desc)
    #   llm_ok        - LLM equivalence answer (None if skipped by string match)
    #   gate_str      - human-readable gate trace for the HTML report
    #   orig_chapter  - ICD-10 chapter letter group
    # -------------------------------------------------------------------------
    debug_print("MAIN", f"Phase 1: baseline prediction for {len(df)} notes...")
    baseline_cache = {}   # hadm_id -> dict

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Baseline"):
        hadm_id   = row['hadm_id']
        orig_text = row['processed_text']
        gt_desc   = row['long_title']
        gt_icd    = row['icd_code']
        try:
            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd)
            baseline_ok, baseline_sim, llm_ok, gate_str = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd)

            baseline_cache[hadm_id] = {
                "orig_icd"    : orig_icd,
                "orig_desc"   : orig_desc,
                "baseline_ok" : baseline_ok,
                "baseline_sim": baseline_sim,
                "llm_ok"      : llm_ok,
                "gate_str"    : gate_str,
                "orig_chapter": get_icd_chapter(orig_icd),
            }
            status = "PASS" if baseline_ok else "SKIP"
            print(f"  [{status}] {hadm_id}: pred={orig_icd} gt={gt_icd} | {gate_str}")

        except Exception as e:
            import traceback
            debug_print("ERROR", f"Baseline prediction failed for {hadm_id}: {e}")
            traceback.print_exc()
            # Mark as failed so all levels skip this note consistently
            baseline_cache[hadm_id] = {
                "orig_icd": "UNKNOWN", "orig_desc": "Unknown",
                "baseline_ok": False, "baseline_sim": 0.0,
                "llm_ok": False, "gate_str": "ERROR",
                "orig_chapter": "Unknown",
            }

    n_valid   = sum(1 for v in baseline_cache.values() if v["baseline_ok"])
    n_skipped = len(baseline_cache) - n_valid
    debug_print("MAIN", f"Phase 1 done: {n_valid} valid, {n_skipped} skipped")

    # -- PHASE 2: run all 4 attack levels using the cached baseline -----------
    ALL_ATTACK_LEVELS = [0, 1, 2, 3]
    all_results       = []

    for ATTACK_LEVEL in ALL_ATTACK_LEVELS:
        debug_print("MAIN", f"Phase 2: attack level {ATTACK_LEVEL}...")
        results = []
        print("\n" + "=" * 80)
        print(f"ATTACK LEVEL {ATTACK_LEVEL}")
        print("=" * 80)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Level {ATTACK_LEVEL}"):
            try:
                hadm_id = row['hadm_id']
                orig_text = row['processed_text']
                gt_desc   = row['long_title']
                gt_icd    = row['icd_code']
                icd_ver   = row['icd_version']

                # -- Look up cached baseline - never re-predict ----------------
                cache = baseline_cache[hadm_id]
                orig_icd     = cache["orig_icd"]
                orig_desc    = cache["orig_desc"]
                baseline_ok  = cache["baseline_ok"]
                baseline_sim = cache["baseline_sim"]
                llm_ok       = cache["llm_ok"]
                gate_str     = cache["gate_str"]
                orig_chapter = cache["orig_chapter"]
                # -------------------------------------------------------------

                print(f"\n[L{ATTACK_LEVEL}|ID:{hadm_id}] GT: {gt_desc} ({gt_icd}) | "
                      f"Pred: {orig_icd} | {'VALID' if baseline_ok else 'SKIP'}")

                if not baseline_ok:
                    results.append({
                        "hadm_id": hadm_id, "ground_truth_icd": gt_icd,
                        "ground_truth_version": icd_ver, "ground_truth_desc": gt_desc,
                        "original_text": orig_text, "perturbed_text": "SKIPPED",
                        "changes_made": "SKIPPED", "changes_count": 0,
                        "disease_changes": 0, "chemical_changes": 0,
                        "anatomy_changes": 0, "procedure_changes": 0,
                        "original_icd_pred": orig_icd, "original_pred_desc": orig_desc,
                        "baseline_sim": baseline_sim, "baseline_llm_ok": llm_ok,
                        "baseline_gate": gate_str, "baseline_correct": False,
                        "perturbed_icd_pred": "SKIPPED",
                        "icd_codes_differ": False, "drift_level": "skipped",
                        "categorical_drift": False, "subcategory_drift": False,
                        "suspicious_drift": False, "semantic_similarity": 0.0,
                        "edit_similarity": 0.0, "substitution_sim": 1.0,
                        "orig_chapter": orig_chapter,
                        "attack_level": ATTACK_LEVEL,
                    })
                    continue

                pert_text, changes_log = perturber.perturb(orig_text, attack_level=ATTACK_LEVEL)
                pert_icd               = get_icd_prediction(pert_text)
                drift_info             = classify_drift(orig_icd, pert_icd)
                bert_sim, edit_sim     = scorer.get_sim(orig_text, pert_text)
                subst_sim              = scorer.substitution_semantic_similarity(changes_log)

                suspicious = (drift_info['categorical_drift']
                              and edit_sim > 0.995 and len(changes_log) <= 3)

                print(f"   Perturbed ICD: {pert_icd} | Drift: {drift_info['drift_level']} "
                      f"| Subst-sim: {subst_sim:.4f} | Edit-sim: {edit_sim:.4f}")

                changes_str = " | ".join(f"[{l}] {o} -> {n}" for o, n, l in changes_log)

                results.append({
                    "hadm_id": hadm_id, "ground_truth_icd": gt_icd,
                    "ground_truth_version": icd_ver, "ground_truth_desc": gt_desc,
                    "original_text": orig_text, "perturbed_text": pert_text,
                    "changes_made": changes_str, "changes_count": len(changes_log),
                    "disease_changes"  : sum(1 for _, _, l in changes_log if l == "DISEASE"),
                    "chemical_changes" : sum(1 for _, _, l in changes_log if l == "CHEMICAL"),
                    "anatomy_changes"  : sum(1 for _, _, l in changes_log if l == "ANATOMY"),
                    "procedure_changes": sum(1 for _, _, l in changes_log if l == "PROCEDURE"),
                    "original_icd_pred": orig_icd, "original_pred_desc": orig_desc,
                    "baseline_sim": baseline_sim, "baseline_llm_ok": llm_ok,
                    "baseline_gate": gate_str, "baseline_correct": True,
                    "perturbed_icd_pred": pert_icd,
                    **drift_info,
                    "suspicious_drift": suspicious,
                    "semantic_similarity": bert_sim,
                    "edit_similarity"   : edit_sim,
                    "substitution_sim"  : subst_sim,
                    "orig_chapter": orig_chapter,
                    "attack_level": ATTACK_LEVEL,
                })
                print("-" * 40)

            except Exception as e:
                import traceback
                debug_print("ERROR", f"Row {idx} L{ATTACK_LEVEL} failed: {e}")
                traceback.print_exc()
                continue

        all_results.extend(results)
        debug_print("MAIN", f"Level {ATTACK_LEVEL} done: {len(results)} rows")

    if not all_results:
        print("No results to save.")
        return

    final_df = pd.DataFrame(all_results)
    csv_df   = final_df.drop(columns=['original_text', 'perturbed_text'], errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)
    print(f"CSV saved: {FINAL_RESULTS}")

    plot_results(final_df)
    generate_html_report(final_df)

    valid_df = final_df[final_df['baseline_correct'] == True]
    print("\n" + "=" * 80)
    print("FINAL SUMMARY (all levels combined)")
    print(f"  Total samples             : {len(final_df)}")
    print(f"  Valid (ICD-10, matched)   : {len(valid_df)}")
    print(f"  Skipped (wrong baseline)  : {len(final_df) - len(valid_df)}")
    print(f"  Unique notes              : {df['hadm_id'].nunique()} "
          f"(x4 attack levels = {len(final_df)} total rows)")
    if len(valid_df) > 0:
        print(f"\n  Per-level breakdown:")
        for lvl in sorted(valid_df['attack_level'].unique()):
            lvl_df = valid_df[valid_df['attack_level'] == lvl]
            print(f"    L{lvl}: n={len(lvl_df)}  "
                  f"cat_drift={lvl_df['categorical_drift'].mean()*100:.1f}%  "
                  f"subst_sim={lvl_df['substitution_sim'].mean():.4f}  "
                  f"edit_sim={lvl_df['edit_similarity'].mean():.4f}  "
                  f"terms_changed={lvl_df['changes_count'].mean():.1f}")
        print(f"\n  Overall categorical drift : {valid_df['categorical_drift'].mean()*100:.1f}%")
        print(f"  Avg substitution sim      : {valid_df['substitution_sim'].mean():.4f}  <- primary metric")
        print(f"  Avg edit similarity       : {valid_df['edit_similarity'].mean():.4f}")
        print(f"  Avg BERT cosine sim       : {valid_df['semantic_similarity'].mean():.4f}  (reference only)")
    print(f"\n  CSV  -> {FINAL_RESULTS}")
    print(f"  HTML -> {HTML_REPORT}")
    print(f"  Plots-> {PLOT_FILE} (+ _chapter_drift, _icd_drift, _attack_levels)")
    print("=" * 80)


if __name__ == "__main__":
    main()