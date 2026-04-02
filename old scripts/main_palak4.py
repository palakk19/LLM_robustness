import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
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
ORACLE_MODEL   = "llama3.1:8b"  # The Judge/Attacker (Generates & Validates)
TARGET_MODEL   = "qwen2.5:7b"    # The Patient/Student (Predicts ICD codes)

API_URL        = "http://ollama.warhol.informatik.rwth-aachen.de/api/generate"
NOTES_PATH     = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
DIAGNOSES_PATH = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/diagnoses_icd.csv.gz"
DICT_PATH      = r"C:/Users/palak/rfc/dataset/physionet.org/files/mimic-iv-note/2.2/note/d_icd_diagnoses.csv.gz"

FINAL_RESULTS  = "palak_final2/new/inal_robustness.csv"
HTML_REPORT    = "palak_final2/new/robustness_report.html"
PLOT_FILE      = "palak_final2/new/robustness_plots.png"
DEVICE         = "cpu"

BASELINE_SIM_THRESHOLD = 0.88

RANDOM_SEED = 42
TOTAL_EXPERIMENT_SAMPLES = 100

MAX_CHANGES = 8
ATTACK_PERTURBATION_RATES = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30}

MIN_CHAPTER_N = 20
MIN_ICD_N = 5           
MAX_ICD_PLOT_COUNT = 20

ENTITY_WEIGHTS = {
    "DISEASE"    : 1.0,
    "CHEMICAL"   : 0.6,
    "ANATOMY"    : 0.4,
    "PROCEDURE"  : 0.5,
    "OBSERVATION": 0.15,
}

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"[{stage}] {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR
# ============================================================
def query_rwth_server(prompt, max_tokens=50, model_to_use=ORACLE_MODEL):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model" : model_to_use,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
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
def load_data():
    """Load stratified dataset CSV if available, otherwise use MIMICLoader."""
    csv_path = "palak_final/stratified_dataset.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        if TOTAL_EXPERIMENT_SAMPLES is not None and len(df) > TOTAL_EXPERIMENT_SAMPLES:
            df = df.sample(n=TOTAL_EXPERIMENT_SAMPLES, random_state=RANDOM_SEED)
            
        debug_print("DATA", f"Loaded stratified dataset: {len(df)} notes "
                    f"| {df['chapter'].nunique()} chapters")
        counts = df.groupby('chapter').size()
        for ch, n in sorted(counts.items()):
            debug_print("DATA", f"  Chapter {ch}: {n} notes")
        return df
        
    else:
        debug_print("DATA", "stratified_dataset.csv not found, using MIMICLoader")
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
        df['text']          = df['text'].apply(cls.clean_noise)
        df['text']          = df['text'].apply(cls.strip_leaking_sections)
        df['processed_text'] = df['text'].apply(cls.extract_blind_clinical_info)
        df['chapter']       = df['icd_code'].apply(lambda c: c[0] if c and not pd.isna(c) else 'X')
        result = df.sample(n=min(len(df), n_samples), random_state=42)
        debug_print("DATA", f"Sampled {len(result)} notes")
        return result


# ============================================================
# SECTION 3: PERTURBATION
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
    r'^\d', r'^[A-Z]$', r'^(mr|mrs|ms|dr|sir)\.?$',
    r'^(anterior|posterior|medial|lateral|proximal|distal|superior|inferior|dorsal|ventral)$',
    r'^(sharp|dull|adequate|frequent|complete|general|intact|patent)$',
    r'^(morning|evening|nighttime|overnight|recent|daily)$',
]
_UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_ENTITY_PATTERNS]

GENERIC_MODEL_STOP_ENTITIES = {
    "admitted", "hospitalized", "monitored", "tolerated", "resolved",
    "transitioned", "presented", "transferred", "discharged", "ambulated",
    "pedestrian", "gentleman", "patient", "history", "details", "reports",
    "findings", "status", "hpi", "pmh", "ros", "pacu", "npo",
    "mr", "mrs", "ms", "dr", "hospital course", "history of", "hx",
    "discharge", "admit", "spontaneously", "unchanged", "stable",
    "multiple", "several", "various", "toilet", "soft", "flat", "tender",
    "normal", "abnormal", "states", "denies", "endorses",
    "age", "disease", "condition", "illness", "problem", "issue",
    "procedure", "operation", "surgery", "surgical",
    "woman", "man", "male", "female", "person", "individual",
    "examination", "observation", "course", "period", "level",
    "plan", "note", "team", "floor", "ward", "clinic", "office",
    "placement", "therapy", "treatment", "response", "assessment",
}

_NONE_SENTINELS = {
    "none", "(no change)", "no change", "n/a", "-", "",
    "unchanged", "same", "keep", "as is", "no synonym", "not applicable",
}

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

Terms:
{term_lines}

Generalisations (one per line):"""

PROMPT_LEVEL_2 = """\
You are a clinical NLP expert building an adversarial ICD-coding test set.

TASK: For each numbered medical term, output ONE plausible differential
diagnosis or clinically similar alternative from a different ICD-10 subcategory.

RULES:
- Different ICD-10 3-character prefix than the original.
- 1-4 words. Medically plausible in a similar clinical context.
- Choose the closest differential (overlapping symptoms).
- Write NONE if no plausible differential exists.

EXAMPLES:
1. pulmonary embolism      -> pneumonia
2. appendicitis            -> mesenteric adenitis
3. myocardial infarction   -> aortic dissection
4. ischemic stroke         -> hemorrhagic stroke

Terms:
{term_lines}

Differentials (one per line):"""

PROMPT_LEVEL_3 = """\
You are a clinical adversarial NLP researcher generating a cross-category
ICD confusion test set.

TASK: For each numbered medical term, output ONE term from a DIFFERENT
ICD-10 chapter that shares symptom overlap that could mislead a coding model.

RULES:
- Output ONLY the replacement. No arrows, no original, no explanation.
- 1-4 words. Different ICD-10 major category.
- Terms sharing presenting symptoms but different root cause.
- Write NONE only if impossible.
- The replacement MUST have a different ICD-10 first letter than the original term. For example: I (circulatory) -> J (respiratory) is acceptable. I -> I is not acceptable."


EXAMPLES:
1. heart failure           -> COPD exacerbation
2. bacterial pneumonia     -> lung adenocarcinoma
3. acute kidney injury     -> diabetic ketoacidosis
4. septic shock            -> pulmonary embolism

Terms:
{term_lines}

Cross-category alternatives (one per line):"""

ATTACK_PROMPTS = {0: PROMPT_LEVEL_0, 1: PROMPT_LEVEL_1,
                  2: PROMPT_LEVEL_2, 3: PROMPT_LEVEL_3}


def _clean_synonym_echo(synonym, original):
    synonym = synonym.replace('\u2192', '->').replace('\u27A1', '->')
    if " -> " in synonym:
        synonym = [p.strip() for p in synonym.split("->")][-1].strip()
    orig_escaped = re.escape(original.strip())
    prefix_echo  = re.match(rf'^{orig_escaped}[\s]*[.;:\-]+[\s]+(.+)$',
                             synonym, re.IGNORECASE)
    if prefix_echo:
        synonym = prefix_echo.group(1).strip()
    else:
        orig_tok = original.strip().lower().split()
        syn_tok  = synonym.lower().split()
        n = len(orig_tok)
        if len(syn_tok) > n and syn_tok[:n] == orig_tok:
            synonym = " ".join(synonym.split()[n:]).strip()
    synonym = synonym.strip('"\'\u2018\u2019\u201c\u201d')
    synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
    return synonym


def _assign_heuristic_label(ent):
    t = ent.text.lower().strip()
    anatomy_kw  = {"artery","vein","lobe","spine","abdomen","chest","hip","knee",
                   "shoulder","femur","tibia","fibula","pelvis","thorax","lung",
                   "liver","kidney","bowel","colon","rectum","bladder","uterus",
                   "ovary","prostate","brain","cortex","ventricle","aorta"}
    procedure_kw = {"resection","biopsy","catheterization","angioplasty","bypass",
                    "arthroplasty","cholecystectomy","appendectomy","colectomy",
                    "lobectomy","nephrectomy","transplant","intubation","extubation",
                    "dialysis","chemotherapy","radiation","stenting","thrombectomy",
                    "imaging","scan","mri","ct","echocardiogram","ultrasound",
                    "endoscopy","colonoscopy","bronchoscopy"}
    if ent[0].pos_ == "PROPN" or (ent.text.isupper() and len(ent.text) <= 6):
        return "CHEMICAL"
    for w in anatomy_kw:
        if w in t: return "ANATOMY"
    for w in procedure_kw:
        if w in t: return "PROCEDURE"
    return "OBSERVATION"


def entity_is_medical(ent, accepted_labels, generic_stop_set):
    if accepted_labels is not None and ent.label_ not in accepted_labels:
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
        return False, "generic word"
    if ent[0].pos_ in {"VERB","ADP","DET","PRON","CCONJ","SCONJ","PUNCT","PART","NUM"}:
        return False, f"non-content POS: {ent[0].pos_}"
    if ent.root.pos_ in {"ADJ", "ADV"}:
        return False, "standalone modifier"
    label_str = (_assign_heuristic_label(ent)
                 if accepted_labels is None
                 else (ent.label_ or "ENTITY"))
    return True, label_str


def find_synonyms_batch(terms, attack_level=0):
    if not terms:
        return {}
    term_lines = "\n".join(
        f"{i+1}. [{item['label']}] \"{item['text']}\"  "
        f"(context: \"{item['context'][:80]}...\")"
        for i, item in enumerate(terms)
    )
    prompt   = ATTACK_PROMPTS[attack_level].format(term_lines=term_lines)
    response = query_rwth_server(prompt, max_tokens=len(terms)*14+80, model_to_use=ORACLE_MODEL)
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


def replacement_is_structurally_valid(original, replacement):
    MEANING_INVERTERS = {'no','not','absence','without','lack','negative'}
    if " -> " in replacement:
        return False, "contains arrow"
    if replacement.lower() in _NONE_SENTINELS:
        return False, "no-change placeholder"
    orig_w = original.strip().split()
    repl_w = replacement.strip().split()
    n_orig, n_repl = len(orig_w), len(repl_w)
    is_abbrev  = original.strip().isupper() and len(original.strip()) <= 5
    word_limit = 6 if is_abbrev else max(n_orig*2, n_orig+3)
    if (set(replacement.lower().split()) - set(original.lower().split())) & MEANING_INVERTERS:
        return False, "introduces negation"
    if n_repl > word_limit:
        return False, f"too long ({n_orig}->{n_repl})"
    if n_orig == 1 and n_repl > 3:
        return False, f"single-word expanded to {n_repl}"
    if not replacement.strip():
        return False, "empty"
    if replacement.strip().lower() == original.strip().lower():
        return False, "identical"
    if len(replacement) / max(len(original), 1) > 5.0:
        return False, "ratio too high"
    if re.search(r'[.!?](?!\s*$)', replacement):
        return False, "mid-sentence punct"
    if len(repl_w) < len(orig_w) * 0.4:
        return False, "too sparse"
    return True, "ok"


def validate_replacement_with_llm(original, replacement, context):
    prompt = (
        f"Context: \"{context}\"\n"
        f"Original: '{original}' | Replacement: '{replacement}'\n\n"
        f"Answer YES unless the replacement changes clinical meaning, "
        f"polarity, or diagnosis.\nAnswer YES or NO only."
    )
    resp    = query_rwth_server(prompt, max_tokens=5, model_to_use=ORACLE_MODEL)
    is_safe = resp.strip().upper().startswith("YES")
    return is_safe, ("approved" if is_safe else "rejected")


class StructuredMedicalPerturber:
    _SAFETY_GATE_LEVELS = {0, 1}
    MIN_SUBST_SIM = {0: 0.94, 1: 0.88, 2: 0.75, 3: 0.00}

    def __init__(self):
        try:
            self.nlp_bc = spacy.load("en_ner_bc5cdr_md")
            debug_print("PERTURBER", "Loaded Primary: en_ner_bc5cdr_md")
        except OSError:
            raise RuntimeError(
                "No BC5CDR model found. Run: "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

        self.nlp_sci = None
        for model_name in ["en_core_sci_lg", "en_core_sci_md"]:
            try:
                self.nlp_sci = spacy.load(model_name)
                debug_print("PERTURBER", f"Loaded Secondary: {model_name}")
                break
            except OSError:
                continue
        
        if not self.nlp_sci:
            debug_print("WARNING", "No sci fallback found. Only Diseases/Chemicals will be perturbed.")

    def _is_negated(self, token):
        neg    = {'no','not','denies','negative','without','none','denied'}
        window = [t.text.lower() for t in token.doc[max(0, token.i-3):token.i]]
        return any(t in window for t in neg)

    def extract_medical_entities(self, text):
        entities = []
        seen_spans = set()  
        seen_texts = set()

        doc_bc = self.nlp_bc(text)
        for ent in doc_bc.ents:
            valid, label_str = entity_is_medical(
                ent, {"DISEASE", "CHEMICAL"}, GENERIC_MODEL_STOP_ENTITIES)
            
            if not valid or self._is_negated(ent[0]):
                continue
            
            key = ent.text.strip().lower()
            if key in seen_texts: 
                continue
            
            seen_texts.add(key)
            for i in range(ent.start_char, ent.end_char):
                seen_spans.add(i)

            sent = (ent.sent.text if ent.sent 
                    else text[max(0, ent.start_char-60):ent.end_char+60])
            entities.append({
                'text': ent.text, 'label': label_str,
                'start_char': ent.start_char, 'end_char': ent.end_char,
                'context': sent,
            })

        if self.nlp_sci:
            doc_sci = self.nlp_sci(text)
            for ent in doc_sci.ents:
                overlap = any(i in seen_spans for i in range(ent.start_char, ent.end_char))
                if overlap:
                    continue

                valid, label_str = entity_is_medical(
                    ent, None, GENERIC_MODEL_STOP_ENTITIES)
                
                if not valid or label_str not in {"ANATOMY", "PROCEDURE"} or self._is_negated(ent[0]):
                    continue

                key = ent.text.strip().lower()
                if key in seen_texts: 
                    continue
                
                seen_texts.add(key)
                for i in range(ent.start_char, ent.end_char):
                    seen_spans.add(i)

                sent = (ent.sent.text if ent.sent 
                        else text[max(0, ent.start_char-60):ent.end_char+60])
                entities.append({
                    'text': ent.text, 'label': label_str,
                    'start_char': ent.start_char, 'end_char': ent.end_char,
                    'context': sent,
                })

        return entities

    def validate_and_apply(self, text, entities, synonym_map,
                           attack_level=0, scorer=None):
        accepted           = []
        apply_safety_gate  = attack_level in self._SAFETY_GATE_LEVELS
        sim_threshold      = self.MIN_SUBST_SIM.get(attack_level, 0.0)

        for ent in entities:
            original  = ent['text']
            candidate = synonym_map.get(original)
            if not candidate:
                continue

            struct_ok, reason = replacement_is_structurally_valid(original, candidate)
            if not struct_ok:
                print(f"        LAYER2 BLOCKED ({reason}): '{original}'->'{candidate}'")
                continue

            if scorer is not None and sim_threshold > 0:
                pair_sim = scorer._bert_cosine(original, candidate)
                if pair_sim < sim_threshold:
                    print(f"        LAYER2.5 BLOCKED (sim={pair_sim:.3f}): "
                          f"'{original}'->'{candidate}'")
                    continue

            if apply_safety_gate:
                llm_ok, _ = validate_replacement_with_llm(
                    original, candidate, ent['context'])
                if not llm_ok:
                    print(f"        LAYER3 BLOCKED: '{original}'->'{candidate}'")
                    continue

            accepted.append({
                'original'   : original,
                'replacement': candidate,
                'label'      : ent['label'],
                'start_char' : ent['start_char'],
                'end_char'   : ent['end_char'],
            })
            print(f"        ACCEPTED [{ent['label']:20s}]: '{original}'->'{candidate}'")

        if not accepted:
            return text, []

        perturbed = text
        for item in sorted(accepted, key=lambda x: x['start_char'], reverse=True):
            perturbed = (perturbed[:item['start_char']]
                         + item['replacement']
                         + perturbed[item['end_char']:])

        offset = 0
        pos_map = {}
        for item in sorted(accepted, key=lambda x: x['start_char']):
            orig_len  = item['end_char'] - item['start_char']
            repl_len  = len(item['replacement'])
            final_s   = item['start_char'] + offset
            pos_map[item['start_char']] = (final_s, final_s + repl_len)
            offset   += repl_len - orig_len

        changes_log = [(i['original'], i['replacement'], i['label'])
                       for i in accepted]
        return perturbed, changes_log

    def perturb(self, text, perturbation_rate=None, attack_level=0, scorer=None):
        if perturbation_rate is None:
            perturbation_rate = ATTACK_PERTURBATION_RATES.get(attack_level, 0.30)

        all_entities = self.extract_medical_entities(text)
        if not all_entities:
            return text, []

        n_target     = max(1, int(len(all_entities) * perturbation_rate))
        target_ents  = random.sample(all_entities, min(n_target, len(all_entities)))
        synonym_map  = find_synonyms_batch(target_ents, attack_level=attack_level)

        return self.validate_and_apply(text, target_ents, synonym_map,
                                       attack_level=attack_level,
                                       scorer=scorer)


# ============================================================
# SECTION 4: SEMANTIC EVALUATOR
# ============================================================
def _normalized_edit_distance(s1, s2):
    w1, w2 = s1.lower().split(), s2.lower().split()
    m, n   = len(w1), len(w2)
    if m == 0 and n == 0: return 1.0
    if m == 0 or n == 0:  return 0.0
    prev = list(range(n+1))
    curr = [0]*(n+1)
    for i in range(1, m+1):
        curr[0] = i
        for j in range(1, n+1):
            curr[j] = (prev[j-1] if w1[i-1] == w2[j-1]
                       else 1 + min(prev[j], curr[j-1], prev[j-1]))
        prev, curr = curr, [0]*(n+1)
    return 1.0 - prev[n]/max(m, n)


class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL", "Loading ClinicalBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model     = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def _bert_cosine(self, t1, t2):
        with torch.no_grad():
            inp  = self.tokenizer([t1, t2], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(DEVICE)
            out  = self.model(**inp)
            mask = inp['attention_mask'].unsqueeze(-1).float()
            e1   = (out.last_hidden_state[0]*mask[0]).sum(0)/mask[0].sum()
            e2   = (out.last_hidden_state[1]*mask[1]).sum(0)/mask[1].sum()
            return float(cosine_similarity(
                e1.unsqueeze(0).numpy(), e2.unsqueeze(0).numpy())[0][0])

    def get_sim(self, t1, t2):
        return self._bert_cosine(t1, t2), _normalized_edit_distance(t1, t2)

    def descriptions_match(self, d1, d2, threshold=BASELINE_SIM_THRESHOLD):
        sim = self._bert_cosine(d1, d2)
        return sim >= threshold, sim

    def substitution_semantic_similarity(self, changes_log, weights=None):
        if not changes_log:
            return 1.0
        w             = weights or ENTITY_WEIGHTS
        total_sim     = 0.0
        total_weight  = 0.0
        for entry in changes_log:
            original, replacement, label = entry[0], entry[1], entry[2]
            sim            = self._bert_cosine(original, replacement)
            wt             = w.get(label, 0.7)
            total_sim    += sim * wt
            total_weight += wt
        return total_sim / total_weight if total_weight > 0 else 1.0


# ============================================================
# SECTION 5: BASELINE CORRECTNESS
# ============================================================
def check_clinical_equivalence(pred_icd, gt_icd, gt_desc):
    if pred_icd in ("UNKNOWN", ""):
        return False
    prompt = (
        f"Ground truth: '{gt_desc}' (ICD-10: {gt_icd})\n"
        f"Predicted: {pred_icd}\n\n"
        f"Same primary condition? Minor specificity differences OK.\n"
        f"Answer YES or NO only."
    )
    resp = query_rwth_server(prompt, max_tokens=5, model_to_use=ORACLE_MODEL)
    return resp.strip().upper().startswith("YES")


def _icd_string_match(pred_icd, gt_icd):
    if pred_icd in ("UNKNOWN","") or gt_icd in ("UNKNOWN",""):
        return False, False
    pn = re.sub(r'[^A-Z0-9]', '', pred_icd.strip().upper())
    gn = re.sub(r'[^A-Z0-9]', '', gt_icd.strip().upper())
    exact  = pn == gn
    prefix = len(pn) >= 3 and len(gn) >= 3 and pn[:3] == gn[:3]
    return exact, prefix


def is_baseline_correct(scorer, orig_desc, gt_desc, orig_icd, gt_icd):
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    exact_match, prefix_match = _icd_string_match(orig_icd, gt_icd)

    if exact_match or prefix_match:
        return True, baseline_sim, None, (
            f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'} | "
            f"exact={'YES' if exact_match else 'NO'} | "
            f"prefix={'YES' if prefix_match else 'NO'} | llm=SKIPPED")

    llm_ok      = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)
    high_sim_ok = baseline_sim >= 0.80
    passed      = llm_ok and high_sim_ok

    return passed, baseline_sim, llm_ok, (
        f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'} | "
        f"exact=NO | prefix=NO | "
        f"llm={'YES' if llm_ok else 'NO'} | "
        f"high_sim={'YES' if high_sim_ok else 'NO'} | "
        f"passed={'YES' if passed else 'NO'}")


# ============================================================
# SECTION 6: ICD PREDICTION
# ============================================================
def get_icd_prediction(text, model_to_use=ORACLE_MODEL):
    primary_prompt = (
        f"Read the clinical note and return the single most likely ICD-10-CM code.\n"
        f"Format: one uppercase letter + 2 digits + optional dot + 1-2 digits.\n"
        f"Examples: J18.9  I21.0  K56.9  N39.0\n"
        f"Output the ICD-10 code ONLY.\n\n"
        f"Note:\n{text[:2500]}\n\nICD-10 code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code. "
        f"Format: letter+2digits+optional dot+digits. "
        f"Examples: J18.9, I21.0\n\nNote:\n{text[:800]}\n\nCode:"
    )
    for prompt in [primary_prompt, fallback_prompt]:
        resp = query_rwth_server(prompt, model_to_use=model_to_use)
        if resp:
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.S).strip()
            m    = re.search(r'[A-Z]\d{2}\.\d+', resp.upper())
            if m: return m.group(0)
            m    = re.search(r'[A-Z]\d{2,4}', resp.upper())
            if m: return m.group(0)
    return "UNKNOWN"


def get_diagnosis_description(icd_code):
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    resp = query_rwth_server(
        f"What condition does ICD-10 code '{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words.", max_tokens=20, model_to_use=ORACLE_MODEL)
    return resp or "Unknown"


def get_icd_chapter(icd_code):
    if not icd_code or icd_code == "UNKNOWN":
        return "Unknown"
    code   = re.sub(r'[^A-Z0-9]', '', icd_code.strip().upper())
    letter = code[0] if code else "X"
    chapter_map = {
        'A':'Infectious','B':'Infectious','C':'Neoplasms','D':'Blood',
        'E':'Endocrine','F':'Mental','G':'Nervous','H':'Eye/Ear',
        'I':'Circulatory','J':'Respiratory','K':'Digestive','L':'Skin',
        'M':'Musculoskeletal','N':'Genitourinary','O':'Pregnancy',
        'P':'Perinatal','Q':'Congenital','R':'Symptoms',
        'S':'Injury','T':'Poisoning','Z':'Health Status',
    }
    return chapter_map.get(letter, f"Ch {letter}")


# ============================================================
# SECTION 7: DRIFT CLASSIFICATION
# ============================================================
def classify_drift(orig_icd, pert_icd):
    o = re.sub(r'[^A-Z0-9]', '', orig_icd.upper())
    p = re.sub(r'[^A-Z0-9]', '', pert_icd.upper())
    if "UNKNOWN" in (o, p):
        return {"icd_codes_differ": True, "drift_level": "unknown",
                "categorical_drift": False, "subcategory_drift": False}
    exact  = o == p
    prefix = len(o) >= 3 and len(p) >= 3 and o[:3] == p[:3]
    level  = "none" if exact else ("subcategory" if prefix else "categorical")
    return {
        "icd_codes_differ" : not exact,
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 8: STABILITY CLASSIFICATION
# ============================================================
def classify_stability(df):
    results = []

    for hid, group in df.groupby('hadm_id'):
        group = group.sort_values('attack_level') if 'attack_level' in group.columns else group
        gt_icd  = group['ground_truth_icd'].iloc[0] if 'ground_truth_icd' in group.columns else None
        chapter = group['orig_chapter'].iloc[0] if 'orig_chapter' in group.columns else 'X'
        valid_rows = group[group['baseline_correct'] == True].copy()

        if len(valid_rows) == 0:
            results.append({
                'hadm_id': hid, 'stability_class': 'Baseline Failed',
                'n_valid': 0, 'n_same_as_baseline': 0, 'n_drifted': 0,
                'drift_type': 'N/A', 'chapter': chapter, 'ground_truth_icd': gt_icd,
            })
            continue

        orig_pred = valid_rows['original_icd_pred'].iloc[0]
        same_mask = valid_rows['perturbed_icd_pred'] == orig_pred

        n_valid = len(valid_rows)
        n_same  = int(same_mask.sum())
        n_drift = int((~same_mask).sum())

        drifted_preds = valid_rows.loc[~same_mask, 'perturbed_icd_pred']
        unique_wrong_preds = drifted_preds.nunique() if n_drift > 0 else 0

        if n_same == n_valid:
            cls = 'Stable Correct'
            drift_type = 'No Drift'
        elif n_same == 0:
            cls = 'Unstable'
            drift_type = 'Consistent Systematic Drift' if unique_wrong_preds == 1 else 'Chaotic Random Drift'
        else:
            cls = 'Partially Stable'
            if n_drift > 1 and unique_wrong_preds == 1:
                drift_type = 'Consistent Systematic Drift'
            elif n_drift > 1:
                drift_type = 'Chaotic Random Drift'
            else:
                drift_type = 'Single Drift' 

        results.append({
            'hadm_id': hid, 'stability_class': cls, 'n_valid': n_valid,
            'n_same_as_baseline': n_same, 'n_drifted': n_drift,
            'drift_type': drift_type,
            'chapter': chapter, 'ground_truth_icd': gt_icd,
        })

    return pd.DataFrame(results)


# ============================================================
# SECTION 8b: CONFIDENCE INTERVAL HELPER
# ============================================================
def proportion_ci(p, n, z=1.96):
    """Wilson score CI for a proportion p with n observations."""
    if n == 0:
        return 0.0, 0.0
    centre  = (p + z**2/(2*n)) / (1 + z**2/n)
    margin  = z * ((p*(1-p)/n + z**2/(4*n**2)) ** 0.5) / (1 + z**2/n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ============================================================
# SECTION 8c: PLOTS
# ============================================================
def plot_results(df):
    debug_print("PLOT", "Generating plots...")
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return

    sim_col   = ('substitution_sim' if 'substitution_sim' in valid_df.columns
                 else 'semantic_similarity')
    sim_label = ('Substitution Cosine Sim' if sim_col == 'substitution_sim'
                 else 'BERT Cosine Sim')

    DRIFT_COLORS = {'none':'#90EE90','subcategory':'#FFD700',
                    'categorical':'#FF6B6B','unknown':'#CCCCCC'}
    LEVEL_LABELS = {0:'L0-Synonyms',1:'L1-Broadening',
                    2:'L2-Differential',3:'L3-Adversarial'}
    LEVEL_COLORS = {0:'#4ECDC4',1:'#45B7D1',2:'#FFD700',3:'#FF6B6B'}

    levels = (sorted(valid_df['attack_level'].unique())
              if 'attack_level' in valid_df.columns else [None])

    for level in levels:
        lvl_df   = (valid_df[valid_df['attack_level'] == level]
                    if level is not None else valid_df)
        lvl_tag  = f"L{level}" if level is not None else "all"
        lvl_name = LEVEL_LABELS.get(level, lvl_tag)

        if len(lvl_df) == 0:
            continue

        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        fig.suptitle(f"Robustness — {lvl_name}  (n={len(lvl_df)})",
                     fontsize=13, y=1.02)

        ax = axes[0]
        sv = lvl_df[sim_col].dropna()
        ax.hist(sv, bins=15, color=LEVEL_COLORS.get(level,'teal'),
                edgecolor='black', alpha=0.85)
        ax.set_title(f"{sim_label}\nmean={sv.mean():.4f}")
        ax.set_xlabel(sim_label); ax.set_ylabel("Count")

        ax = axes[1]
        dc = (lvl_df['drift_level'].value_counts()
              .reindex(['none','subcategory','categorical','unknown']).dropna())
        dc.plot(kind='bar', color=[DRIFT_COLORS.get(k,'grey') for k in dc.index],
                ax=ax, edgecolor='black', alpha=0.85)
        cp = lvl_df['categorical_drift'].mean()*100
        sp = lvl_df['subcategory_drift'].mean()*100
        ax.set_title(f"Drift — {lvl_tag}\nCat:{cp:.1f}% Sub:{sp:.1f}%")
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0: ax.text(bar.get_x()+bar.get_width()/2, h+0.2,
                               str(int(h)), ha='center', va='bottom', fontsize=9)

        ax = axes[2]
        for dl in ['none','subcategory','categorical']:
            mask = lvl_df['drift_level'] == dl
            if mask.sum() == 0: continue
            ax.scatter(lvl_df[mask]['changes_count'], lvl_df[mask][sim_col],
                       label=dl, color={'none':'green','subcategory':'orange',
                                        'categorical':'red'}[dl],
                       alpha=0.7, s=60)
        ax.set_title(f"Changes vs Sim\n{lvl_tag}")
        ax.set_xlabel("Terms Changed"); ax.set_ylabel(sim_label); ax.legend(fontsize=8)

        ax = axes[3]
        type_cols = [c for c in ['disease_changes','chemical_changes',
                                  'anatomy_changes','procedure_changes']
                     if c in lvl_df.columns and lvl_df[c].sum() > 0]
        if type_cols:
            totals = lvl_df[type_cols].sum()
            totals.index = [c.replace('_changes','').capitalize() for c in totals.index]
            totals.plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'],
                        ax=ax, edgecolor='black', alpha=0.85)
            ax.set_title(f"Entity Types\n{lvl_tag}")
        else:
            ax.text(0.5, 0.5, 'No entity data', ha='center', va='center',
                    transform=ax.transAxes)

        ax = axes[4]
        groups = [lvl_df[lvl_df['drift_level']==d][sim_col].dropna().values
                  for d in ['none','subcategory','categorical']
                  if d in lvl_df['drift_level'].values]
        labels = [d for d in ['none','subcategory','categorical']
                  if d in lvl_df['drift_level'].values]
        if groups:
            bp = ax.boxplot(groups, labels=labels, patch_artist=True)
            for patch, d in zip(bp['boxes'], labels):
                patch.set_facecolor(DRIFT_COLORS.get(d,'grey')); patch.set_alpha(0.8)
        ax.set_title(f"Sim by Drift Level\n{lvl_tag}")

        plt.tight_layout()
        fig.savefig(PLOT_FILE.replace('.png', f'_{lvl_tag}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {PLOT_FILE.replace('.png', f'_{lvl_tag}.png')}")

    if 'orig_chapter' in valid_df.columns:
        for level in levels:
            lvl_df  = (valid_df[valid_df['attack_level'] == level]
                       if level is not None else valid_df)
            lvl_tag = f"L{level}" if level is not None else "all"
            if len(lvl_df) == 0: continue

            chap = (lvl_df.groupby('orig_chapter')['categorical_drift']
                    .agg(['mean','count'])
                    .rename(columns={'mean':'drift_rate','count':'n'})
                    .sort_values('drift_rate', ascending=False))
            chap = chap[chap['n'] >= MIN_CHAPTER_N]
            if len(chap) == 0: continue

            ci_lo, ci_hi = [], []
            for _, r in chap.iterrows():
                lo, hi = proportion_ci(r['drift_rate'], r['n'])
                ci_lo.append(r['drift_rate']*100 - lo*100)
                ci_hi.append(hi*100 - r['drift_rate']*100)

            fig2, ax2 = plt.subplots(figsize=(max(10, len(chap)*0.9), 5))
            bars  = ax2.bar(chap.index, chap['drift_rate']*100,
                             color='#FF6B6B', edgecolor='black', alpha=0.8,
                             yerr=[ci_lo, ci_hi], capsize=4, ecolor='#333')
            for bar, (idx, r) in zip(bars, chap.iterrows()):
                ax2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+1.5,
                         f"n={int(r['n'])}", ha='center', va='bottom', fontsize=8)
            ax2.set_title(
                f"Categorical Drift by ICD-10 Chapter — {lvl_tag}\n"
                f"(n>={MIN_CHAPTER_N}, error bars = 95% Wilson CI)")
            ax2.set_xlabel("ICD-10 Chapter")
            ax2.set_ylabel("Categorical Drift Rate (%)")
            ax2.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            path = PLOT_FILE.replace('.png', f'_chapter_{lvl_tag}.png')
            fig2.savefig(path, dpi=150); plt.close(fig2)
            print(f"  Saved: {path}")

    if 'ground_truth_icd' in valid_df.columns:
        for level in levels:
            lvl_df  = (valid_df[valid_df['attack_level'] == level]
                       if level is not None else valid_df)
            lvl_tag = f"L{level}" if level is not None else "all"
            if len(lvl_df) == 0: continue

            icd = (lvl_df.groupby('ground_truth_icd')['categorical_drift']
                   .agg(['mean','count'])
                   .rename(columns={'mean':'drift_rate','count':'n'})
                   .sort_values('drift_rate', ascending=False))
            icd = icd[icd['n'] >= MIN_ICD_N].head(MAX_ICD_PLOT_COUNT)
            if len(icd) == 0: continue

            ci_lo, ci_hi = [], []
            for _, r in icd.iterrows():
                lo, hi = proportion_ci(r['drift_rate'], r['n'])
                ci_lo.append(r['drift_rate']*100 - lo*100)
                ci_hi.append(hi*100 - r['drift_rate']*100)

            fig3, ax3 = plt.subplots(figsize=(max(10, len(icd)*0.8), 5))
            clrs = ['#FF6B6B' if r > 0.5 else '#FFD700' if r > 0.2
                    else '#90EE90' for r in icd['drift_rate']]
            ax3.barh(icd.index, icd['drift_rate']*100, color=clrs,
                     edgecolor='black', alpha=0.85,
                     xerr=[ci_lo, ci_hi], capsize=3, ecolor='#333')
            for _, r in icd.iterrows():
                ax3.text(r['drift_rate']*100+1.5, icd.index.get_loc(r.name),
                         f"n={int(r['n'])}", va='center', fontsize=8)
            ax3.set_title(
                f"Drift Rate by ICD-10 Code — {lvl_tag}\n"
                f"(top 20, n>=5, 95% CI)")
            ax3.set_xlabel("Categorical Drift Rate (%)")
            plt.tight_layout()
            path = PLOT_FILE.replace('.png', f'_icd_{lvl_tag}.png')
            fig3.savefig(path, dpi=150); plt.close(fig3)
            print(f"  Saved: {path}")

    if 'attack_level' in valid_df.columns and valid_df['attack_level'].nunique() > 1:
        ls = (valid_df.groupby('attack_level')
              .agg(cat_drift=('categorical_drift','mean'),
                   sub_drift=('subcategory_drift','mean'),
                   avg_sim=(sim_col,'mean'),
                   n=('categorical_drift','count'))
              .reset_index())

        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes4[0]
        ax.bar(ls['attack_level'], ls['cat_drift']*100,
               color=[LEVEL_COLORS.get(l,'grey') for l in ls['attack_level']],
               edgecolor='black', alpha=0.85)
        for _, r in ls.iterrows():
            ax.text(r['attack_level'], r['cat_drift']*100+0.5,
                    f"{r['cat_drift']*100:.1f}%", ha='center', fontsize=9)
        ax.set_title("Categorical Drift by Attack Level")
        ax.set_xlabel("Attack Level"); ax.set_ylabel("Drift Rate (%)")
        ax.set_xticks(ls['attack_level'])
        ax.set_xticklabels([f"L{l}\n{LEVEL_LABELS.get(l,'').split('-')[-1]}\n(n={int(r)})"
                             for l, r in zip(ls['attack_level'], ls['n'])], fontsize=8)

        ax = axes4[1]
        ax.plot(ls['attack_level'], ls['avg_sim'], marker='o', color='teal',
                linewidth=2, markersize=8)
        for _, r in ls.iterrows():
            ax.annotate(f"{r['avg_sim']:.3f}", (r['attack_level'], r['avg_sim']),
                        textcoords="offset points", xytext=(0,8),
                        ha='center', fontsize=8)
        ax.set_title(f"Avg {sim_label} by Attack Level")
        ax.set_xlabel("Attack Level"); ax.set_ylabel(sim_label)

        ax = axes4[2]
        w = 0.35; x = np.array(ls['attack_level'])
        ax.bar(x-w/2, ls['cat_drift']*100, w, label='Categorical',
               color='#FF6B6B', edgecolor='black', alpha=0.85)
        ax.bar(x+w/2, ls['sub_drift']*100, w, label='Subcategory',
               color='#FFD700', edgecolor='black', alpha=0.85)
        ax.set_title("Drift Breakdown by Attack Level")
        ax.set_xlabel("Attack Level"); ax.set_ylabel("Drift Rate (%)")
        ax.set_xticks(x); ax.legend()

        plt.suptitle("Attack Level Comparison", fontsize=13, y=1.02)
        plt.tight_layout()
        path = PLOT_FILE.replace('.png', '_attack_levels.png')
        fig4.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig4); print(f"  Saved: {path}")

    stab_df = classify_stability(df)
    stab_counts = stab_df['stability_class'].value_counts()

    STAB_COLORS = {
        'Stable Correct'  : '#27AE60',
        'Partially Stable': '#F39C12',
        'Unstable'        : '#8E44AD',
        'Baseline Failed' : '#E74C3C',
    }
    STAB_ORDER = ['Stable Correct', 'Partially Stable', 'Unstable', 'Baseline Failed']
    ordered_counts = stab_counts.reindex(STAB_ORDER).dropna()

    fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(20, 5))

    colors = [STAB_COLORS.get(k,'grey') for k in ordered_counts.index]
    bars   = ax5a.bar(ordered_counts.index, ordered_counts.values,
                      color=colors, edgecolor='black', alpha=0.85)
    for bar, (cls, cnt) in zip(bars, ordered_counts.items()):
        pct = 100*cnt/len(stab_df)
        ax5a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                  f"{cnt}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10)
    ax5a.set_title("Patient Stability Classification")
    ax5a.set_ylabel("Number of Patients")
    ax5a.tick_params(axis='x', rotation=15)

    ax5b.pie(ordered_counts.values,
             labels=[f"{k}\n(n={v})" for k, v in ordered_counts.items()],
             colors=[STAB_COLORS.get(k,'grey') for k in ordered_counts.index],
             autopct='%1.1f%%', startangle=90)
    ax5b.set_title(f"Stability Distribution\n(n={len(stab_df)} patients)")

    multi_drift_df = stab_df[stab_df['drift_type'].isin(['Consistent Systematic Drift', 'Chaotic Random Drift'])]
    if len(multi_drift_df) > 0:
        drift_counts = multi_drift_df['drift_type'].value_counts()
        ax5c.pie(drift_counts.values,
                 labels=[f"{k}\n(n={v})" for k, v in drift_counts.items()],
                 colors=['#E67E22' if 'Systematic' in k else '#9B59B6' for k in drift_counts.index],
                 autopct='%1.1f%%', startangle=90)
        ax5c.set_title(f"Nature of the Vulnerability\n(among {len(multi_drift_df)} multi-drifting patients)")
    else:
        ax5c.text(0.5, 0.5, "Not enough drift data yet", ha='center', va='center')
        ax5c.axis('off')

    plt.suptitle("Model Robustness and Systematic Bias Analysis", fontsize=13, y=1.05)
    plt.tight_layout()
    path5 = PLOT_FILE.replace('.png', '_stability.png')
    fig5.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close(fig5); print(f"  Saved: {path5}")

    if 'edit_similarity' in valid_df.columns and 'substitution_sim' in valid_df.columns:
        fig6, axes6 = plt.subplots(1, len(levels), figsize=(6*len(levels), 5))
        if len(levels) == 1:
            axes6 = [axes6]

        for ax6, level in zip(axes6, levels):
            lvl_df  = (valid_df[valid_df['attack_level'] == level]
                       if level is not None else valid_df)
            lvl_tag = f"L{level}" if level is not None else "all"
            if len(lvl_df) == 0: continue

            for dl, color, marker in [
                ('none',        '#27AE60', 'o'),
                ('subcategory', '#F39C12', 's'),
                ('categorical', '#E74C3C', '^'),
            ]:
                mask = lvl_df['drift_level'] == dl
                if mask.sum() == 0: continue
                
                ax6.scatter(lvl_df[mask]['edit_similarity'],
                            lvl_df[mask]['substitution_sim'], 
                            label=dl, color=color, marker=marker,
                            alpha=0.7, s=60, edgecolors='black', linewidths=0.5)

            ax6.set_xlim(0.85, 1.01) 
            ax6.set_ylim(0.50, 1.05) 
            
            ax6.plot([0.5, 1.05], [0.5, 1.05], 'k--', alpha=0.3, linewidth=1, label='edit=subst')
            
            ax6.set_title(f"Edit Sim vs Subst. Sim — {lvl_tag}\n"
                          f"(n={len(lvl_df)})")
            ax6.set_xlabel("Edit Similarity (token-level)")
            ax6.set_ylabel("Substitution Sim (Targeted swaps only)")
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)

        plt.suptitle("Edit Similarity vs Substitution Similarity by Drift Outcome\n"
                     "Points below diagonal: semantic change > surface change",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        
        path6 = PLOT_FILE.replace('.png', '_edit_vs_subst.png')
        fig6.savefig(path6, dpi=150, bbox_inches='tight')
        plt.close(fig6); print(f"  Saved: {path6}")

    if ('orig_chapter' in valid_df.columns and
            'attack_level' in valid_df.columns and
            valid_df['attack_level'].nunique() > 1):

        pivot = (valid_df.groupby(['orig_chapter', 'attack_level'])
                 ['categorical_drift'].mean().unstack(fill_value=np.nan) * 100)

        chapter_n = valid_df.groupby('orig_chapter')['hadm_id'].nunique()
        keep_ch   = chapter_n[chapter_n >= MIN_CHAPTER_N].index
        pivot     = pivot.loc[pivot.index.isin(keep_ch)]

        if len(pivot) > 0:
            col_labels = [LEVEL_LABELS.get(c, f"L{c}")
                          for c in pivot.columns]

            fig7, ax7 = plt.subplots(
                figsize=(max(8, len(pivot.columns)*2),
                         max(5, len(pivot)*0.5)))

            sns.heatmap(
                pivot,
                annot=True, fmt=".0f", cmap="YlOrRd",
                vmin=0, vmax=100,
                linewidths=0.5, linecolor='white',
                ax=ax7,
                cbar_kws={'label': 'Categorical Drift Rate (%)'},
                xticklabels=col_labels,
            )
            ax7.set_title(
                "Categorical Drift Rate (%) — ICD Chapter x Attack Level\n"
                f"(chapters with n>={MIN_CHAPTER_N} valid cases, "
                f"colour scale 0-100%)")
            ax7.set_xlabel("Attack Level")
            ax7.set_ylabel("ICD-10 Chapter")
            ax7.tick_params(axis='x', rotation=15)
            ax7.tick_params(axis='y', rotation=0)
            plt.tight_layout()
            path7 = PLOT_FILE.replace('.png', '_chapter_heatmap.png')
            fig7.savefig(path7, dpi=150, bbox_inches='tight')
            plt.close(fig7); print(f"  Saved: {path7}")

    plt.close('all')
    print("All plots saved.")


# ============================================================
# SECTION 9: HTML REPORT
# ============================================================
def generate_html_report(df, filename=HTML_REPORT):
    DRIFT_COLORS = {
        'none':'#ccffcc','subcategory':'#fff3cd',
        'categorical':'#ffcccc','unknown':'#eeeeee','skipped':'#f9f9f9',
    }
    DRIFT_LABELS = {
        'none':'STABLE','subcategory':'SUBCATEGORY DRIFT',
        'categorical':'CATEGORICAL DRIFT','unknown':'UNKNOWN ICD','skipped':'SKIPPED',
    }
    LEVEL_COLORS_HTML = {
        0:'#2E75B6', 1:'#1F7A4F', 2:'#C55A11', 3:'#C00000',
    }
    LEVEL_NAMES_HTML = {
        0:'L0 - Surface Synonyms', 1:'L1 - Clinical Broadening',
        2:'L2 - Differential Swap', 3:'L3 - Cross-Chapter',
    }

    def highlight_changes(original_text, perturbed_text, changes_str):
        if not changes_str or changes_str in ('None','','SKIPPED'):
            return original_text, perturbed_text
        pairs = []
        for pair in changes_str.split("|"):
            pair = re.sub(r'^\[.*?\]\s*', '', pair.strip())
            if " -> " not in pair: continue
            ow, nw = [x.strip() for x in pair.split(" -> ", 1)]
            pairs.append((ow, nw))
        if not pairs:
            return original_text, perturbed_text

        def _tokenize(text):
            return [(m.group(), m.start(), m.end())
                    for m in re.finditer(r'\S+', text)]

        def _apply(text, targets, bg):
            tokens = _tokenize(text)
            result, pos, used, counts = [], 0, set(), {}
            for token, start, end in tokens:
                result.append(text[pos:start])
                tl = token.lower()
                mt = next((t for t in targets if tl == t.lower()), None)
                if mt is not None:
                    idx = counts.get(mt.lower(), 0)
                    key = (mt.lower(), idx)
                    if key not in used:
                        used.add(key)
                        counts[mt.lower()] = idx+1
                        result.append(
                            f'<span style="background:{bg};color:white;'
                            f'padding:1px 4px;border-radius:3px;'
                            f'font-weight:bold;">{token}</span>')
                    else:
                        counts[mt.lower()] = idx+1
                        result.append(token)
                else:
                    result.append(token)
                pos = end
            result.append(text[pos:])
            return "".join(result)

        orig_h = _apply(original_text,  [p[0] for p in pairs], '#FF6B6B')
        pert_h = _apply(perturbed_text, [p[1] for p in pairs], '#51C878')
        return orig_h, pert_h

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = valid_df['categorical_drift'].mean()*100 if len(valid_df) else 0
    sub_rate    = valid_df['subcategory_drift'].mean()*100 if len(valid_df) else 0
    avg_changes = f"{valid_df['changes_count'].mean():.1f}" if len(valid_df) else 'N/A'
    sim_col     = ('substitution_sim' if 'substitution_sim' in valid_df.columns
                   else 'semantic_similarity')
    sim_lbl     = ('Substitution Cosine Sim' if sim_col == 'substitution_sim'
                   else 'BERT Cosine Sim')
    avg_sim     = f"{valid_df[sim_col].mean():.4f}" if len(valid_df) else 'N/A'

    stab_df = classify_stability(df)
    stab_dict = stab_df.set_index('hadm_id').to_dict('index')

    patient_groups = {}
    for _, row in df.iterrows():
        hid = row['hadm_id']
        lvl = row.get('attack_level', 0)
        patient_groups.setdefault(hid, {})[lvl] = row

    level_summary_html = ""
    if 'attack_level' in valid_df.columns and valid_df['attack_level'].nunique() > 1:
        level_summary_html = (
            "<h2>Per-Level Summary</h2>"
            "<table style='border-collapse:collapse;font-size:0.9em;margin-bottom:20px;'>"
            "<tr style='background:#ddd;'>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>Level</th>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>n</th>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>Cat.Drift%</th>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>Sub.Drift%</th>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>Avg Sim</th>"
            "<th style='padding:6px 12px;border:1px solid #ccc;'>Avg Terms</th>"
            "</tr>")
        for lvl in sorted(valid_df['attack_level'].unique()):
            ld = valid_df[valid_df['attack_level'] == lvl]
            cd = ld['categorical_drift'].mean()*100
            sd = ld['subcategory_drift'].mean()*100
            sm = ld[sim_col].mean() if sim_col in ld.columns else 0
            tc = ld['changes_count'].mean()
            lc = LEVEL_COLORS_HTML.get(lvl,'#333')
            level_summary_html += (
                f"<tr><td style='padding:5px 12px;border:1px solid #ccc;'>"
                f"<b style='color:{lc};'>{LEVEL_NAMES_HTML.get(lvl,f'L{lvl}')}</b></td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{len(ld)}</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;color:#c00;'><b>{cd:.1f}%</b></td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{sd:.1f}%</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{sm:.4f}</td>"
                f"<td style='padding:5px 12px;border:1px solid #ccc;text-align:center;'>{tc:.1f}</td></tr>")
        level_summary_html += "</table>"

    cards_html = ""
    for hid, level_rows in patient_groups.items():
        any_row      = next(iter(level_rows.values()))
        gt_desc      = any_row.get('ground_truth_desc','')
        gt_icd       = any_row.get('ground_truth_icd','')
        orig_icd     = any_row.get('original_icd_pred','')
        orig_desc    = any_row.get('original_pred_desc','')
        gate_d       = any_row.get('baseline_gate','')
        base_correct = any_row.get('baseline_correct', False)
        
        p_stab = stab_dict.get(hid, {})
        stab_cls = p_stab.get('stability_class', 'Unknown')
        drift_type = p_stab.get('drift_type', 'N/A')
        
        worst = 'none'
        for r in level_rows.values():
            dl = r.get('drift_level','none')
            if dl == 'categorical': worst = 'categorical'; break
            elif dl == 'subcategory': worst = 'subcategory'
        hbg = DRIFT_COLORS.get(worst if base_correct else 'skipped','#f9f9f9')

        cards_html += (
            f'\n<div style="border:2px solid #ccc;margin:24px 0;border-radius:10px;overflow:hidden;background:#fff;">'
            f'\n<div style="background:{hbg};padding:12px 16px;border-bottom:1px solid #ccc;">'
            f'<b>Patient {hid}</b>&nbsp;&nbsp;'
            f'<span style="color:#444;font-size:0.9em;">GT: <b>{gt_desc[:70]}</b> ({gt_icd})</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#444;font-size:0.9em;">Pred: <b>{orig_icd}</b> - {orig_desc}</span>'
            f'</div>'
            f'<div style="background:#f8f8f8;padding:5px 16px;border-bottom:1px solid #ddd;'
            f'font-size:0.82em;color:#666;"><b>Gate:</b> {gate_d}</div>')

        if base_correct:
            if drift_type == 'Consistent Systematic Drift':
                cards_html += f'<div style="background:#ffe6e6; border-left:5px solid #d9534f; padding:10px 16px; color:#c9302c; font-size:0.9em; border-bottom:1px solid #ddd;">&#9888; <b>SYSTEMATIC BIAS:</b> The LLM consistently locks onto the same incorrect ICD code when perturbed.</div>'
            elif drift_type == 'Chaotic Random Drift':
                cards_html += f'<div style="background:#f4e6fa; border-left:5px solid #8e44ad; padding:10px 16px; color:#8e44ad; font-size:0.9em; border-bottom:1px solid #ddd;">&#9888; <b>CHAOTIC VULNERABILITY:</b> The LLM is highly sensitive, jumping to different wrong codes across perturbations.</div>'
            elif stab_cls == 'Stable Correct':
                cards_html += f'<div style="background:#e8f5e9; border-left:5px solid #27ae60; padding:10px 16px; color:#2e7d32; font-size:0.9em; border-bottom:1px solid #ddd;">&#10003; <b>HIGHLY ROBUST:</b> The LLM withstood all adversarial attacks.</div>'

        if not base_correct:
            cards_html += ('<div style="padding:14px 16px;color:#999;font-style:italic;">'
                           'Skipped - baseline wrong.</div>\n</div>')
            continue

        levels_present = sorted(level_rows.keys())
        col_w = 100 // len(levels_present)
        col_headers = "".join(
            f'<th style="background:{LEVEL_COLORS_HTML.get(l,"#555")};color:white;'
            f'padding:8px 6px;border:1px solid #ccc;font-size:0.85em;width:{col_w}%;">'
            f'{LEVEL_NAMES_HTML.get(l,f"L{l}")}</th>'
            for l in levels_present)

        def fmt_ch(chg_str):
            if not chg_str or chg_str in ('SKIPPED','','None'):
                return '<i style="color:#aaa;font-size:0.82em;">no changes</i>'
            parts = []
            for part in chg_str.split('|'):
                part = re.sub(r'^\[([^\]]+)\]\s*', r'[\1] ', part.strip())
                if ' -> ' not in part: continue
                lm = re.match(r'\[([^\]]+)\]\s*(.*?)\s*->\s*(.*)', part)
                if lm:
                    parts.append(
                        f'<span style="font-size:0.80em;display:block;margin:1px 0;">'
                        f'<span style="color:#888;">[{lm.group(1).strip()}]</span> '
                        f'<span style="background:#FFE0E0;padding:0 3px;border-radius:2px;">'
                        f'{lm.group(2).strip()}</span>'
                        f' &rarr; '
                        f'<span style="background:#E0FFE0;padding:0 3px;border-radius:2px;">'
                        f'{lm.group(3).strip()}</span></span>')
            return ''.join(parts) or '<i style="color:#aaa;">none</i>'

        chg_cells = "".join(
            f'<td style="padding:8px 6px;border:1px solid #ddd;vertical-align:top;">'
            f'{fmt_ch(level_rows[l].get("changes_made",""))}</td>'
            for l in levels_present)

        def drift_badge(r):
            dl = r.get('drift_level','none')
            bg = DRIFT_COLORS.get(dl,'#eee')
            lb = DRIFT_LABELS.get(dl, dl.upper())
            oi = r.get('original_icd_pred','?')
            pi = r.get('perturbed_icd_pred','?')
            nc = r.get('changes_count',0)
            return (f'<div style="margin-bottom:4px;">'
                    f'<span style="background:{bg};padding:3px 8px;border-radius:4px;'
                    f'font-weight:bold;font-size:0.82em;">{lb}</span></div>'
                    f'<div style="font-size:0.80em;color:#555;">{oi} &rarr; {pi}</div>'
                    f'<div style="font-size:0.76em;color:#888;">{nc} terms changed</div>')

        drift_cells = "".join(
            f'<td style="padding:8px 6px;border:1px solid #ddd;vertical-align:top;'
            f'text-align:center;">{drift_badge(level_rows[l])}</td>'
            for l in levels_present)

        def sim_cell(r):
            s = r.get('substitution_sim', None)
            b = r.get('semantic_similarity', None)
            lines = []
            if s not in (None,''): lines.append(f'<b>Subst:</b> {float(s):.4f}')
            if b not in (None,''): lines.append(f'BERT: {float(b):.4f}')
            return '<br>'.join(f'<span style="font-size:0.80em;color:#555;">{x}</span>'
                               for x in lines) or '<i style="color:#aaa;">N/A</i>'

        sim_cells = "".join(
            f'<td style="padding:8px 6px;border:1px solid #ddd;vertical-align:top;'
            f'text-align:center;">{sim_cell(level_rows[l])}</td>'
            for l in levels_present)

        cards_html += f"""
  <div style="padding:12px 16px;">
    <table style="width:100%;border-collapse:collapse;margin-bottom:10px;">
      <tr>{col_headers}</tr>
      <tr><td colspan="{len(levels_present)}" style="padding:3px 6px;font-size:0.78em;
          color:#888;border:1px solid #eee;background:#fafafa;"><b>Changes</b></td></tr>
      <tr>{chg_cells}</tr>
      <tr><td colspan="{len(levels_present)}" style="padding:3px 6px;font-size:0.78em;
          color:#888;border:1px solid #eee;background:#fafafa;"><b>ICD result</b></td></tr>
      <tr>{drift_cells}</tr>
      <tr><td colspan="{len(levels_present)}" style="padding:3px 6px;font-size:0.78em;
          color:#888;border:1px solid #eee;background:#fafafa;"><b>Similarity</b></td></tr>
      <tr>{sim_cells}</tr>
    </table>
    <details>
      <summary style="cursor:pointer;color:#2E75B6;font-size:0.88em;padding:4px 0;">
        &#9654; Show full note text
      </summary>
      <div style="margin-top:10px;">"""

        orig_text_val = str(any_row.get('original_text',''))
        cards_html += (
            f'<div style="margin-bottom:12px;">'
            f'<div style="background:#e8f4f8;padding:6px 10px;font-weight:bold;'
            f'font-size:0.85em;border-radius:4px 4px 0 0;">Original — ICD: {orig_icd}</div>'
            f'<div style="border:1px solid #ccc;padding:10px;white-space:pre-wrap;'
            f'font-size:0.80em;line-height:1.55em;max-height:250px;overflow-y:auto;'
            f'background:#fff;">{orig_text_val}</div></div>')

        for l in levels_present:
            r        = level_rows[l]
            pert_txt = str(r.get('perturbed_text',''))
            chg_str  = str(r.get('changes_made',''))
            dl       = r.get('drift_level','none')
            pert_icd = r.get('perturbed_icd_pred','?')
            lc       = LEVEL_COLORS_HTML.get(l,'#555')
            dr_bg    = DRIFT_COLORS.get(dl,'#eee')
            dr_label = DRIFT_LABELS.get(dl, dl.upper())
            if pert_txt == 'SKIPPED': continue
            _, ph = highlight_changes(orig_text_val, pert_txt, chg_str)
            cards_html += (
                f'<div style="margin-bottom:12px;">'
                f'<div style="background:{lc};color:white;padding:6px 10px;font-weight:bold;'
                f'font-size:0.85em;border-radius:4px 4px 0 0;'
                f'display:flex;justify-content:space-between;align-items:center;">'
                f'<span>{LEVEL_NAMES_HTML.get(l,f"L{l}")}</span>'
                f'<span>ICD: {pert_icd} '
                f'<span style="background:{dr_bg};color:#333;padding:2px 8px;'
                f'border-radius:3px;font-size:0.85em;margin-left:8px;">{dr_label}</span></span></div>'
                f'<div style="border:1px solid #ccc;padding:10px;white-space:pre-wrap;'
                f'font-size:0.80em;line-height:1.55em;max-height:250px;overflow-y:auto;'
                f'background:#fffef8;">{ph}</div></div>')

        cards_html += "</div></details></div></div>"

    html = f"""<!DOCTYPE html>
<html><head><title>Robustness Report</title><meta charset="utf-8">
<style>
  body{{font-family:Arial,sans-serif;max-width:1500px;margin:0 auto;padding:20px;background:#f0f2f5;}}
  h1,h2{{color:#1F4E79;}} .summary{{background:#fff;padding:15px;border-radius:8px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
  .summary p{{margin:4px 0;}} .method{{background:#e8f4f8;padding:12px;border-radius:6px;margin-bottom:20px;font-size:0.9em;color:#555;}}
  .legend{{display:flex;gap:16px;margin-bottom:16px;font-size:0.85em;flex-wrap:wrap;}}
  .legend span{{padding:3px 10px;border-radius:4px;}}
</style></head><body>
<h1>ICD-10 Robustness Report — All 4 Attack Levels</h1>
<p style="color:#666;"><b>Target Model:</b> {TARGET_MODEL} | <b>Oracle:</b> {ORACLE_MODEL} | Temp: 0.0</p>
<div class="legend">
  <span style="background:#ccffcc;">STABLE</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT</span>
</div>
<div class="summary">
  <h2>Summary</h2>
  <p><b>Unique patients:</b> {df['hadm_id'].nunique()}</p>
  <p><b>Valid for analysis:</b> {valid_df['hadm_id'].nunique()}</p>
  <p><b>Categorical Drift (all levels):</b> {cat_rate:.1f}%</p>
  <p><b>Subcategory Drift (all levels):</b> {sub_rate:.1f}%</p>
  <p><b>Avg {sim_lbl}:</b> {avg_sim}</p>
  <p><b>Avg Terms Changed:</b> {avg_changes}</p>
</div>
{level_summary_html}
{cards_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {filename}")


# ============================================================
# SECTION 10: SANITY CHECK
# ============================================================
def sanity_check(perturber):
    print("\n" + "="*60 + "\nSANITY CHECK")
    sample   = ("Patient had myocardial infarction. "
                "Aspirin and heparin administered.")
    entities = perturber.extract_medical_entities(sample)
    print(f"  Entities found: {len(entities)}")
    for e in entities: print(f"    [{e['label']}] {e['text']}")
    exact, prefix = _icd_string_match("I21.0", "I21.0")
    print(f"  ICD exact  I21.0==I21.0: {'PASS' if exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "I21.9")
    print(f"  ICD prefix I21.0~I21.9 : {'PASS' if prefix and not exact else 'FAIL'}")
    exact, prefix = _icd_string_match("I21.0", "J18.9")
    print(f"  ICD no-match I21.0/J18.9: {'PASS' if not exact and not prefix else 'FAIL'}")
    print("="*60 + "\n")


# ============================================================
# SECTION 11: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Target: {TARGET_MODEL} | Oracle: {ORACLE_MODEL} | 4 attack levels")

    if not query_rwth_server("Say hello", model_to_use=ORACLE_MODEL):
        debug_print("ERROR", "Oracle failed to connect. Check VPN.")
        return
    if not query_rwth_server("Say hello", model_to_use=TARGET_MODEL):
        debug_print("ERROR", "Target model failed to connect.")
        return
    print("Both LLMs OK")

    df = load_data()

    perturber = StructuredMedicalPerturber()
    scorer    = SemanticEvaluator()
    sanity_check(perturber)

    # =========================================================
    # Phase 1: ORACLE (Llama) filters the dataset for solvable notes
    # =========================================================
    debug_print("MAIN", f"Phase 1: Oracle dataset filter for {len(df)} notes...")
    oracle_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Oracle Filter"):
        hadm_id   = row['hadm_id']
        orig_text = row['processed_text']
        
        # Oracle Predicts
        orig_icd = get_icd_prediction(orig_text, model_to_use=ORACLE_MODEL)
        orig_desc = get_diagnosis_description(orig_icd)
        
        # Oracle Grades Itself
        ok, _, _, _ = is_baseline_correct(scorer, orig_desc, row['long_title'], orig_icd, row['icd_code'])
        oracle_cache[hadm_id] = {"baseline_ok": ok}

    n_solvable = sum(1 for v in oracle_cache.values() if v["baseline_ok"])
    debug_print("MAIN", f"Oracle validated {n_solvable} solvable notes out of {len(df)}.")

    # =========================================================
    # Phase 1.5: TARGET (Qwen/MedGemma) takes the Baseline test
    # =========================================================
    debug_print("MAIN", f"Phase 1.5: Target establishing baseline on solvable notes...")
    target_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Target Baseline"):
        hadm_id = row['hadm_id']
        
        # If Llama couldn't solve it, throw it out immediately
        if not oracle_cache.get(hadm_id, {}).get("baseline_ok", False):
            target_cache[hadm_id] = {"baseline_ok": False}
            continue

        orig_text = row['processed_text']
        gt_desc   = row['long_title']
        gt_icd    = row['icd_code']
        
        try:
            # TARGET MODEL Predicts the clean text
            target_icd = get_icd_prediction(orig_text, model_to_use=TARGET_MODEL)
            
            # ORACLE MODEL looks up the definition and grades Qwen's answer
            target_desc = get_diagnosis_description(target_icd)
            ok, sim, llm_ok, gate = is_baseline_correct(
                scorer, target_desc, gt_desc, target_icd, gt_icd)
            
            target_cache[hadm_id] = {
                "orig_icd": target_icd, "orig_desc": target_desc,
                "baseline_ok": ok, "baseline_sim": sim,
                "llm_ok": llm_ok, "gate_str": gate,
                "orig_chapter": get_icd_chapter(target_icd),
            }
        except Exception as e:
            target_cache[hadm_id] = {"baseline_ok": False}

    n_target_pass = sum(1 for v in target_cache.values() if v.get("baseline_ok"))
    debug_print("MAIN", f"Target passed baseline on {n_target_pass} notes.")

    # =========================================================
    # Phase 2: Adversarial Attacks
    # =========================================================
    all_results = []
    for ATTACK_LEVEL in [0, 1, 2, 3]:
        debug_print("MAIN", f"Phase 2: attack level {ATTACK_LEVEL}...")
        results = []
        print("\n" + "="*80 + f"\nATTACK LEVEL {ATTACK_LEVEL}\n" + "="*80)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"L{ATTACK_LEVEL}"):
            try:
                hadm_id  = row['hadm_id']
                orig_text = row['processed_text']
                gt_desc   = row['long_title']
                gt_icd    = row['icd_code']
                icd_ver   = row['icd_version']
                chapter   = row.get('chapter', 'X')

                cache        = target_cache.get(hadm_id, {"baseline_ok": False})
                orig_icd     = cache.get("orig_icd", "UNKNOWN")
                orig_desc    = cache.get("orig_desc", "Unknown")
                baseline_ok  = cache.get("baseline_ok", False)
                baseline_sim = cache.get("baseline_sim", 0.0)
                llm_ok       = cache.get("llm_ok", False)
                gate_str     = cache.get("gate_str", "ERROR")
                orig_chapter = cache.get("orig_chapter", "X")

                print(f"\n[L{ATTACK_LEVEL}|{hadm_id}] GT:{gt_icd} "
                      f"Target Pred:{orig_icd} "
                      f"{'VALID' if baseline_ok else 'SKIP'}")

                if not baseline_ok:
                    results.append({
                        "hadm_id": hadm_id, "ground_truth_icd": gt_icd,
                        "ground_truth_version": icd_ver,
                        "ground_truth_desc": gt_desc,
                        "original_text": orig_text,
                        "perturbed_text": "SKIPPED",
                        "changes_made": "SKIPPED", "changes_count": 0,
                        "disease_changes": 0, "chemical_changes": 0,
                        "anatomy_changes": 0, "procedure_changes": 0,
                        "original_icd_pred": orig_icd,
                        "original_pred_desc": orig_desc,
                        "baseline_sim": baseline_sim,
                        "baseline_llm_ok": llm_ok,
                        "baseline_gate": gate_str,
                        "baseline_correct": False,
                        "perturbed_icd_pred": "SKIPPED",
                        "icd_codes_differ": False, "drift_level": "skipped",
                        "categorical_drift": False, "subcategory_drift": False,
                        "suspicious_drift": False,
                        "semantic_similarity": 0.0, "edit_similarity": 0.0,
                        "substitution_sim": 1.0,
                        "orig_chapter": orig_chapter,
                        "chapter": chapter,
                        "attack_level": ATTACK_LEVEL,
                    })
                    continue

                # The Perturber automatically uses Llama via the defaults we set earlier!
                pert_text, changes_log = perturber.perturb(
                    orig_text, attack_level=ATTACK_LEVEL, scorer=scorer)

                # TARGET MODEL takes the test on the perturbed text
                pert_icd = get_icd_prediction(pert_text, model_to_use=TARGET_MODEL)
                
                # Compare Target's Baseline vs Target's Perturbed
                drift_info = classify_drift(orig_icd, pert_icd)
                
                bert_sim, edit_sim = scorer.get_sim(orig_text, pert_text)
                subst_sim          = scorer.substitution_semantic_similarity(changes_log)

                suspicious = (drift_info['categorical_drift']
                              and edit_sim > 0.995 and len(changes_log) <= 3)

                print(f"   Pert ICD: {pert_icd} | "
                      f"Drift: {drift_info['drift_level']} | "
                      f"Subst-sim: {subst_sim:.4f}")

                changes_str = " | ".join(
                    f"[{l}] {o} -> {n}" for o, n, l in changes_log)

                results.append({
                    "hadm_id": hadm_id, "ground_truth_icd": gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc": gt_desc,
                    "original_text": orig_text,
                    "perturbed_text": pert_text,
                    "changes_made": changes_str,
                    "changes_count": len(changes_log),
                    "disease_changes":   sum(1 for _,_,l in changes_log if l=="DISEASE"),
                    "chemical_changes":  sum(1 for _,_,l in changes_log if l=="CHEMICAL"),
                    "anatomy_changes":   sum(1 for _,_,l in changes_log if l=="ANATOMY"),
                    "procedure_changes": sum(1 for _,_,l in changes_log if l=="PROCEDURE"),
                    "original_icd_pred": orig_icd,
                    "original_pred_desc": orig_desc,
                    "baseline_sim": baseline_sim,
                    "baseline_llm_ok": llm_ok,
                    "baseline_gate": gate_str,
                    "baseline_correct": True,
                    "perturbed_icd_pred": pert_icd,
                    **drift_info,
                    "suspicious_drift": suspicious,
                    "semantic_similarity": bert_sim,
                    "edit_similarity": edit_sim,
                    "substitution_sim": subst_sim,
                    "orig_chapter": orig_chapter,
                    "chapter": chapter,
                    "attack_level": ATTACK_LEVEL,
                })
                print("-"*40)

            except Exception as e:
                import traceback
                debug_print("ERROR", f"{hadm_id} L{ATTACK_LEVEL}: {e}")
                traceback.print_exc()
                continue

        all_results.extend(results)
        debug_print("MAIN", f"Level {ATTACK_LEVEL} done: {len(results)} rows")

    if not all_results:
        print("No results.")
        return

    final_df = pd.DataFrame(all_results)
    csv_df   = final_df.drop(
        columns=['original_text','perturbed_text'], errors='ignore')

    # Save combined CSV
    os.makedirs(os.path.dirname(FINAL_RESULTS), exist_ok=True)
    csv_df.to_csv(FINAL_RESULTS, index=False)
    print(f"CSV saved: {FINAL_RESULTS}")

    # Save per-level CSVs
    for lvl in sorted(csv_df['attack_level'].unique()):
        lvl_path = FINAL_RESULTS.replace('.csv', f'_L{lvl}.csv')
        csv_df[csv_df['attack_level'] == lvl].to_csv(lvl_path, index=False)
        n_v = csv_df[(csv_df['attack_level']==lvl) & csv_df['baseline_correct']].shape[0]
        print(f"  L{lvl} CSV: {lvl_path} ({n_v} valid)")

    plot_results(final_df)
    generate_html_report(final_df)

    valid_df = final_df[final_df['baseline_correct'] == True]
    
    unique_baselines = valid_df.drop_duplicates(subset=['hadm_id'])
    pass_rule = int(unique_baselines['baseline_gate'].str.contains('exact=YES|prefix=YES', na=False).sum())
    pass_llm  = int(unique_baselines['baseline_gate'].str.contains('llm=YES', na=False).sum())

    print("\n" + "="*80 + "\nFINAL SUMMARY")
    print(f"  Unique patients        : {df['hadm_id'].nunique()}")
    print(f"  Total rows (4 levels)  : {len(final_df)}")
    print(f"  Valid rows total       : {len(valid_df)}")
    print(f"  Skipped rows           : {len(final_df)-len(valid_df)}")
    
    print(f"  Baseline Pass (Rule)   : {pass_rule} patients (Exact or Prefix match)")
    print(f"  Baseline Pass (LLM)    : {pass_llm} patients (Semantic + LLM equivalence)")
    
    if len(valid_df):
        print(f"\n  Per-level breakdown:")
        for lvl in sorted(valid_df['attack_level'].unique()):
            ld = valid_df[valid_df['attack_level']==lvl]
            lo, hi = proportion_ci(ld['categorical_drift'].mean(), len(ld))
            print(f"    L{lvl}: n={len(ld):4d} | "
                  f"cat_drift={ld['categorical_drift'].mean()*100:.1f}% "
                  f"[95%CI {lo*100:.1f}-{hi*100:.1f}%] | "
                  f"subst_sim={ld['substitution_sim'].mean():.4f} | "
                  f"terms={ld['changes_count'].mean():.1f}")

        stab = classify_stability(final_df)
        stab_counts = stab['stability_class'].value_counts()
        print(f"\n  Stability classification ({len(stab)} patients):")
        for cls in ['Stable Correct','Partially Stable','Unstable','Baseline Failed']:
            n = stab_counts.get(cls, 0)
            print(f"    {cls:20s}: {n:4d} ({100*n/len(stab):.1f}%)")

        print(f"\n  Overall categorical drift: "
              f"{valid_df['categorical_drift'].mean()*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()