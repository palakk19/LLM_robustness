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

FINAL_RESULTS  = "final_robustness_new_2.csv"
HTML_REPORT    = "robustness_report_new_2.html"
PLOT_FILE      = "robustness_plots_new_2.png"
MAX_SAMPLES    = 100
DEVICE         = "cpu"

BASELINE_SIM_THRESHOLD = 0.78

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
    @staticmethod
    def clean_noise(text):
        text = str(text)
        text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
        text = re.sub(r"_{2,}", " ", text)
        text = text.replace("=", " ").replace("*", " ").replace("#", " ")
        return " ".join(text.split())

    @staticmethod
    def extract_blind_clinical_info(text: str) -> str:
        """
        Extract clinically rich sections for ICD prediction.
        Falls back to the LAST 2000 chars (not first) because MIMIC
        discharge summaries always open with an admin header.
        """
        relevant_parts = []

        cc = re.search(
            r"(?:Chief Complaint)\s*:?\s*(.*?)"
            r"(?:History of Present Illness|HPI|Major Surgical|"
            r"Past Medical History|\Z)",
            text, re.S | re.I)
        if cc:
            content = cc.group(1).strip()
            if len(content) > 10:
                relevant_parts.append(f"Chief Complaint: {content[:300]}")

        hpi = re.search(
            r"(?:History of Present Illness|HPI)\s*:?\s*(.*?)"
            r"(?:Past Medical History|Social History|Review of Systems|"
            r"Physical Exam|Assessment|Plan|\Z)",
            text, re.S | re.I)
        if hpi:
            content = hpi.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"HPI: {content[:1200]}")

        course = re.search(
            r"Brief Hospital Course\s*:?\s*(.*?)"
            r"(?:Medications on Admission|Discharge Medications|"
            r"Discharge Diagnosis|Discharge Condition|\Z)",
            text, re.S | re.I)
        if course:
            content = course.group(1).strip()
            if len(content) > 100:
                relevant_parts.append(f"Hospital Course: {content[:1200]}")

        ap = re.search(
            r"(?:Assessment(?:\s+and\s+Plan)?|A(?:/|&)P)\s*:?\s*(.*?)"
            r"(?:Discharge Condition|Discharge Instructions|"
            r"Discharge Medications|Follow.?up|\Z)",
            text, re.S | re.I)
        if ap:
            content = ap.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"Assessment: {content[:1000]}")

        dd = re.search(
            r"(?:Discharge Diagnosis|Discharge Diagnoses|Primary Diagnosis)"
            r"\s*:?\s*(.*?)"
            r"(?:Discharge Condition|Discharge Instructions|"
            r"Secondary Diagnoses|Follow.?up|\Z)",
            text, re.S | re.I)
        if dd:
            content = dd.group(1).strip()
            if len(content) > 20:
                relevant_parts.append(f"Discharge Diagnosis: {content[:600]}")

        pmh = re.search(
            r"(?:Past Medical History|PMH)\s*:?\s*(.*?)"
            r"(?:Social History|Family History|Review of Systems|\Z)",
            text, re.S | re.I)
        if pmh:
            content = pmh.group(1).strip()
            if len(content) > 50:
                relevant_parts.append(f"PMH: {content[:600]}")

        if relevant_parts:
            return "\n\n".join(relevant_parts)[:3000]

        # Fallback: last 2000 chars — clinical content lives at the end
        return text[-2000:].strip()

    @classmethod
    def load(cls, notes_path, diag_path, dict_path, n_samples):
        debug_print("DATA", "Loading Data Tables...")

        try:
            debug_print("DATA", f"Loading diagnoses from: {diag_path}")
            diags = pd.read_csv(diag_path, compression='gzip', low_memory=False)
            debug_print("DATA", f"Diagnoses loaded: {len(diags)} rows (all versions)")
        except Exception as e:
            debug_print("DATA", f"Failed to load diagnoses: {e}")
            raise

        diags = diags[diags['icd_version'] == 10].copy()
        debug_print("DATA", f"ICD-10 filter applied: {len(diags)} rows remaining")

        try:
            hadm_ids_needed = set(diags['hadm_id'].unique())
            debug_print("DATA",
                        f"Need notes for {len(hadm_ids_needed)} ICD-10 admissions")
        except Exception as e:
            debug_print("DATA", f"Failed to extract hadm_ids: {e}")
            raise

        try:
            debug_print("DATA", f"Loading notes from: {notes_path} (chunked)...")
            chunks = []
            chunk_count = 0
            notes_found = 0
            target_notes = min(n_samples * 100, 50000)

            for chunk in pd.read_csv(notes_path, compression='gzip',
                                     usecols=['subject_id', 'hadm_id', 'text'],
                                     chunksize=5000):
                chunk_count += 1
                chunk = chunk[chunk['hadm_id'].isin(hadm_ids_needed)].copy()
                if len(chunk) > 0:
                    chunks.append(chunk)
                    notes_found += len(chunk)
                    debug_print("DATA",
                                f"  Chunk {chunk_count}: {len(chunk)} notes, "
                                f"total: {notes_found}")
                    if notes_found >= target_notes:
                        debug_print("DATA",
                                    f"  Stopping early at {notes_found} notes")
                        break

            if chunks:
                notes = pd.concat(chunks, ignore_index=True)
                debug_print("DATA", f"Notes loaded: {len(notes)} rows")
            else:
                raise ValueError(
                    "No matching notes found for any ICD-10 admission")
        except Exception as e:
            debug_print("DATA", f"Failed to load notes: {e}")
            raise

        try:
            if os.path.exists(dict_path):
                debug_print("DATA", f"Loading ICD dictionary from: {dict_path}")
                icd_dict = pd.read_csv(dict_path, compression='gzip')
                icd_dict = icd_dict[icd_dict['icd_version'] == 10].copy()
                debug_print("DATA", f"ICD-10 dictionary: {len(icd_dict)} entries")
                diags = pd.merge(
                    diags,
                    icd_dict[['icd_code', 'icd_version', 'long_title']],
                    on=['icd_code', 'icd_version'], how='left')
            else:
                debug_print("DATA", f"Dictionary not found at: {dict_path}")
                diags['long_title'] = "Unknown Description"
        except Exception as e:
            debug_print("DATA", f"Failed to merge ICD dictionary: {e}")
            raise

        try:
            primary_diags = diags[diags['seq_num'] == 1]
            debug_print("DATA",
                        f"Primary diagnoses (seq_num=1): "
                        f"{len(primary_diags)} rows")
        except Exception as e:
            debug_print("DATA", f"Failed to filter primary diagnoses: {e}")
            raise

        try:
            debug_print("DATA", "Merging notes with diagnoses...")
            df = pd.merge(
                notes,
                primary_diags[['hadm_id', 'icd_code', 'icd_version',
                                'long_title']],
                on='hadm_id')
            debug_print("DATA", f"Merge successful: {len(df)} rows")
        except Exception as e:
            debug_print("DATA", f"Failed to merge: {e}")
            raise

        try:
            debug_print("DATA", "Cleaning and extracting clinical info...")
            df['text'] = df['text'].apply(cls.clean_noise)
            df['processed_text'] = df['text'].apply(
                cls.extract_blind_clinical_info)
        except Exception as e:
            debug_print("DATA", f"Failed to clean text: {e}")
            raise

        try:
            result = df.sample(n=min(len(df), n_samples), random_state=42)
            debug_print("DATA", f"Sampled {len(result)} records (all ICD-10)")
            return result
        except Exception as e:
            debug_print("DATA", f"Failed to sample: {e}")
            raise


# ============================================================
# SECTION 3: STRUCTURED MEDICAL PERTURBATION
# ============================================================

# Typed label allowlist — used when the loaded model supports typed NER
ACCEPTED_ENTITY_LABELS = {
    "DISEASE",
    "CHEMICAL",
    "ANATOMY",
    "PROCEDURE",
    "ORGANISM",
    "CELL_TYPE",
    "CELL_LINE",
    "PROTEIN",
    "GENE_OR_GENE_PRODUCT",
}

MIN_ENTITY_CHARS = 3

# Safety patterns — always applied regardless of model type
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

# BUG FIX 2: Generic-model stop set.
# en_core_sci_sm labels almost everything as ENTITY, including common English
# verbs, pronouns, administrative words, and section headers that have no
# place in a medical synonym substitution. This set rejects them at the
# entity-validation stage so they never reach the LLM synonym call.
GENERIC_MODEL_STOP_ENTITIES = {
    # Common verbs with no medical synonym value
    "admitted", "hospitalized", "monitored", "tolerated", "resolved",
    "transitioned", "irrigated", "presented", "transferred", "discharged",
    "ambulated", "voided", "ambulating", "tolerating", "weaned",
    # Generic nouns / section headers / abbreviations that are headers
    "pedestrian", "gentleman", "patient", "history", "details", "reports",
    "services", "intake", "arrival", "recovery", "presentation", "appointment",
    "appointment with", "cardiologist", "outpatient", "findings", "status",
    # Section label abbreviations
    "hpi", "pmh", "ros", "pacu", "npo", "mr", "mrs", "ms", "dr",
    # State adjectives that aren't replaceable medical terms
    "spontaneously", "naturally", "transitioned", "resolved",
}


def _clean_synonym_echo(synonym: str, original: str) -> str:
    """
    BUG FIX 1: Strip the LLM's "original -> synonym" echo pattern.

    Llama frequently ignores the prompt format and returns the original
    term echoed before the synonym, separated by " -> ":
        e.g.  "nausea -> VOMITING"  instead of  "VOMITING"
        e.g.  "LACTOSE INTOLERANCE -> LACTOSE SENSITIVITY"

    When this happens, take only the last element after splitting on " -> ".
    Also handles multi-hop chains like "A -> B -> C" by taking the last part.
    """
    if " -> " in synonym:
        parts   = [p.strip() for p in synonym.split("->")]
        synonym = parts[-1].strip()
        
        synonym = synonym.strip('"\'')
        synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
        return synonym


def entity_is_medical(ent, accepted_labels, generic_stop_set) -> tuple:
    """
    Three-gate filter.
    Gate 1 — label allowlist (typed model only; None = accept all).
    Gate 2 — safety patterns (negation, polarity, numbers, etc.).
    Gate 3 — generic stop set (used only when accepted_labels is None,
              i.e. the sm model that labels everything as ENTITY).
    """
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

    # Gate 3: generic stop set (only when label gate is disabled)
    if accepted_labels is None and lower in generic_stop_set:
        return False, "generic non-medical word"

    # Reject pure verb forms using spacy POS when available
    # (ent[0] is the first token of the entity span)
    if ent[0].pos_ in {"VERB", "ADP", "DET", "PRON", "CCONJ", "SCONJ",
                        "PUNCT", "PART", "NUM", "SPACE"}:
        return False, f"non-content POS: {ent[0].pos_}"

    return True, ent.label_ if ent.label_ else "ENTITY"

PROMPT_LEVEL_1 = (
    f"You are a clinical terminology expert. Below is a numbered list of "
    f"medical terms from a clinical note, each with its type and context.\n\n"
    f"For EACH term, provide ONE clinically related alternative term.\n\n"
    f"RULES:\n"
    f"- Write ONLY the replacement term. Do NOT repeat the original.\n"
    f"- Do NOT use arrows (->) or any other separator.\n"
    f"- 1-4 words maximum.\n"
    f"- The replacement should refer to the same general clinical area "
    f"(same organ system, same drug class, same procedure category) "
    f"but may be more general, less specific, or a related condition "
    f"in the same disease family.\n"
    f"- You MAY use a broader term (e.g. 'STEMI' -> 'acute coronary syndrome'), "
    f"a drug class instead of a specific drug (e.g. 'heparin' -> 'anticoagulant'), "
    f"or a related procedure (e.g. 'echocardiogram' -> 'cardiac imaging').\n"
    f"- Do NOT invert polarity (do not add negation) or switch to a "
    f"completely unrelated condition.\n"
    f"FORMAT (one line per term, nothing else):\n"
    f"1. <related term or NONE>\n"
    f"2. <related term or NONE>\n\n"
    f"Terms:\n{{term_lines}}\n\n"
    f"Replacements:"
)

PROMPT_LEVEL_0 = (
        f"You are a clinical terminology expert. Below is a numbered list of "
        f"medical terms from a clinical note, each with its type and context.\n\n"
        f"For EACH term, provide ONE medically equivalent synonym.\n\n"
        f"STRICT RULES:\n"
        f"- Write ONLY the replacement term. Do NOT repeat the original.\n"
        f"- Do NOT use arrows (->) or any other separator.\n"
        f"- 1-4 words maximum.\n"
        f"- Same clinical meaning — do NOT change polarity, severity, "
        f"laterality, or diagnosis.\n"
        f"FORMAT (one line per term, nothing else):\n"
        f"1. <synonym or NONE>\n"
        f"2. <synonym or NONE>\n\n"
        f"Terms:\n{{term_lines}}\n\n"
        f"Synonyms:"
)

PROMPT_LEVEL_2 = (
    f"You are a clinical terminology expert. Below is a numbered list of "
    f"medical terms from a clinical note, each with its type and context.\n\n"
    f"For EACH term, provide ONE plausible alternative term that could "
    f"realistically appear in a similar clinical note but refers to a "
    f"DIFFERENT specific condition, drug, or procedure.\n\n"
    f"RULES:\n"
    f"- Write ONLY the replacement term. Do NOT repeat the original.\n"
    f"- Do NOT use arrows (->) or any other separator.\n"
    f"- 1-4 words maximum.\n"
    f"- The replacement must be medically plausible and could appear in "
    f"a real clinical note — it should NOT be random or nonsensical.\n"
    f"- Choose alternatives that share symptom overlap or are common "
    f"differential diagnoses (e.g. 'pulmonary embolism' -> 'pneumonia', "
    f"'appendicitis' -> 'mesenteric adenitis', 'heparin' -> 'warfarin').\n"
    f"- The replacement SHOULD refer to a different specific condition "
    f"that could plausibly cause diagnostic confusion.\n"
    f"- Do NOT simply negate (do not add 'no' or 'without').\n"
    f"FORMAT (one line per term, nothing else):\n"
    f"1. <alternative term or NONE>\n"
    f"2. <alternative term or NONE>\n\n"
    f"Terms:\n{{term_lines}}\n\n"
    f"Replacements:"
)

ATTACK_PROMPTS = {
    0: PROMPT_LEVEL_0,
    1: PROMPT_LEVEL_1,
    2: PROMPT_LEVEL_2,
}



def find_synonyms_batch(terms: list, attack_level: int = 0) -> dict:
    if not terms:
        return {}

    term_lines = "\n".join(
        f"{i+1}. [{item['label']}] \"{item['text']}\"  "
        f"(context: \"{item['context'][:80]}...\")"
        for i, item in enumerate(terms)
    )

    prompt_template = ATTACK_PROMPTS.get(attack_level, PROMPT_LEVEL_0)
    prompt = prompt_template.format(term_lines=term_lines)

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

            # BUG FIX 1: strip echo pattern before any other processing
            synonym = _clean_synonym_echo(raw_syn, original)
            synonym = synonym.strip('"\'')
            synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()

            if not synonym or synonym.upper() == 'NONE':
                synonym_map[original] = None
            elif synonym.lower() == original.lower():
                synonym_map[original] = None
            else:
                synonym_map[original] = synonym

    return synonym_map


def replacement_is_structurally_valid(original: str, replacement: str) -> tuple:
    """Layer 2: pure structural rules."""
    # Extra guard: if the replacement still contains " -> " it means the
    # echo-strip failed (e.g. a triple chain). Reject it.
    MEANING_INVERTERS = {'no', 'not', 'absence', 'without', 'lack', 'negative'}

    if " -> " in replacement:
        return False, "contains arrow (echo not fully cleaned)"

    orig_words = original.strip().split()   
    repl_words = replacement.strip().split()
    n_orig, n_repl = len(orig_words), len(repl_words)


    is_abbreviation = original.strip().isupper() and len(original.strip()) <= 5
    word_limit = 6 if is_abbreviation else max(n_orig * 2, n_orig + 3)  

    repl_words = set(replacement.lower().split())
    orig_words_set = set(original.lower().split())

    added_words = repl_words - orig_words_set
    if added_words & MEANING_INVERTERS:
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
    # Add this check — replacement must not be a strict subset of the original meaning
    if len(replacement.split()) < len(original.split()) * 0.4:
        # Replacement lost more than 60% of the token content
        return False, "replacement too sparse relative to original"

    return True, "ok"


def validate_replacement_with_llm(original: str, replacement: str,
                                   context: str) -> tuple:
    """Layer 3: LLM clinical safety check."""
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
    """
    Three-phase structured perturbation pipeline.

    Phase 1 — Extract: scispacy NER filtered by label + POS + safety patterns.
    Phase 2 — Batch synonyms: one LLM call for all terms.
    Phase 3 — Validate (structural + LLM safety) and apply right-to-left.

    Bug fixes integrated:
    - FIX 1: _clean_synonym_echo() strips "original -> synonym" echo from
             Llama responses before any synonym is used.
    - FIX 2: entity_is_medical() rejects generic non-medical words when
             the sm model (which labels everything as ENTITY) is active.
    - FIX 3: Extra " -> " guard in replacement_is_structurally_valid()
             catches any echo that survived fix 1.
    """

    def __init__(self):
        for model_name in ["en_core_sci_md", "en_core_sci_sm"]:
            try:
                self.nlp = spacy.load(model_name)
                debug_print("PERTURBER", f"Loaded {model_name}")
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                "No scispacy model found.\n"
                "Install: pip install scispacy && "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy"
                "/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            )

        # Probe to detect label vocabulary
        test_doc     = self.nlp(
            "The patient had myocardial infarction and was given aspirin "
            "for chest pain. CT scan of the thorax was performed.")
        found_labels = {ent.label_ for ent in test_doc.ents}
        debug_print("PERTURBER", f"Model entity labels detected: {found_labels}")

        if not found_labels or found_labels <= {"ENTITY", ""}:
            self.effective_labels = None
            debug_print("PERTURBER",
                        "Generic NER (sm): label gate disabled, "
                        "POS filter + stop set active")
        else:
            self.effective_labels = ACCEPTED_ENTITY_LABELS
            debug_print("PERTURBER",
                        f"Typed NER: label allowlist active "
                        f"({len(ACCEPTED_ENTITY_LABELS)} types)")

    def _is_negated(self, token) -> bool:
        neg = {'no', 'not', 'denies', 'negative', 'without', 'none', 'denied'}
        window = [t.text.lower()
                  for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(t in window for t in neg)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    def extract_medical_entities(self, text: str) -> list:
        doc      = self.nlp(text)
        entities = []
        seen     = set()

        for ent in doc.ents:
            valid, label_str = entity_is_medical(
                ent, self.effective_labels, GENERIC_MODEL_STOP_ENTITIES)
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

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    def get_synonyms_for_all(self, entities: list) -> dict:
        if not entities:
            return {}

        print(f"\n   > Medical terms found ({len(entities)}):")
        for e in entities:
            print(f"       [{e['label']:25s}] {e['text']}")

        synonym_map = find_synonyms_batch(entities)

        print(f"\n   > Synonym candidates (after echo-strip):")
        for term, syn in synonym_map.items():
            print(f"       {term:35s} -> {syn if syn else 'NONE'}")

        return synonym_map

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    def validate_and_apply(self, text: str, entities: list,
                           synonym_map: dict) -> tuple:
        accepted = []

        for ent in entities:
            original  = ent['text']
            candidate = synonym_map.get(original)

            if not candidate:
                print(f"       SKIP (no synonym)   : '{original}'")
                continue

            struct_ok, struct_reason = replacement_is_structurally_valid(
                original, candidate)
            if not struct_ok:
                print(f"       LAYER2 BLOCKED ({struct_reason}): "
                      f"'{original}' -> '{candidate}'")
                continue

            llm_ok, _ = validate_replacement_with_llm(
                original, candidate, ent['context'])
            if not llm_ok:
                print(f"       LAYER3 BLOCKED (clinical): "
                      f"'{original}' -> '{candidate}'")
                continue

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

        perturbed = text
        for item in sorted(accepted,
                            key=lambda x: x['start_char'], reverse=True):
            perturbed = (perturbed[:item['start_char']]
                         + item['replacement']
                         + perturbed[item['end_char']:])

        changes_log = [(i['original'], i['replacement'], i['label'])
                       for i in accepted]
        return perturbed, changes_log

    # ── Public entry point ────────────────────────────────────────────────────
    def perturb(self, text: str, perturbation_rate: float = 0.30, attack_level: int = 0) -> tuple:
        all_entities = self.extract_medical_entities(text)
        if not all_entities:
            print("   > No accepted medical entities found.")
            return text, []

        n               = max(1, int(len(all_entities) * perturbation_rate))
        target_entities = random.sample(all_entities, min(n, len(all_entities)))
        synonym_map = find_synonyms_batch(target_entities, attack_level=attack_level)

        print(f"\n   > Validation results:")
        return self.validate_and_apply(text, target_entities, synonym_map)


# ============================================================
# SECTION 4: SEMANTIC EVALUATOR (ClinicalBERT)
# ============================================================
class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT (CPU)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)
        self.model.eval()

    def get_sim(self, text1: str, text2: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer([text1, text2], padding=True,
                                    truncation=True, max_length=512,
                                    return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            # Mean pool over non-padding tokens instead of CLS
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            e1 = (outputs.last_hidden_state[0] * mask[0]).sum(0) / mask[0].sum()
            e2 = (outputs.last_hidden_state[1] * mask[1]).sum(0) / mask[1].sum()
            sim = cosine_similarity(e1.unsqueeze(0).numpy(),
                                    e2.unsqueeze(0).numpy())[0][0]
        return float(sim)

    def descriptions_match(self, desc1: str, desc2: str,
                            threshold: float = BASELINE_SIM_THRESHOLD) -> tuple:
        sim = self.get_sim(desc1, desc2)
        return sim >= threshold, sim


# ============================================================
# SECTION 5: BASELINE CORRECTNESS
# ============================================================
def check_clinical_equivalence(pred_icd: str, gt_icd: str,
                                gt_desc: str) -> bool:
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


def is_baseline_correct(scorer, orig_desc, gt_desc,
                         orig_icd, gt_icd) -> tuple:
    """
    Three-tier correctness gate, descending strictness:
      Tier 1 — LLM says YES (exact clinical equivalence)
      Tier 2 — Same ICD-10 chapter (first letter matches)
               e.g. I21.0 vs I48.9: both circulatory system = plausible
      Tier 3 — Semantic similarity > 0.92 (high-confidence embedding match)

    A sample passes if ANY tier passes. This reduces skip rate caused by
    Llama 3.1:8b being a weak ICD coder while still blocking clearly wrong
    predictions (e.g. E47.0 diabetes vs R55 syncope).
    """
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    llm_ok = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)

    
    # Tier 3: high semantic similarity fallback
    high_sim_ok = baseline_sim >= 0.92

    passed = llm_ok or high_sim_ok

    gate_parts = [
        f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'}",
        f"llm={'YES' if llm_ok else 'NO'}",
        f"high_sim={'YES' if high_sim_ok else 'NO'}",
    ]
    return passed, baseline_sim, llm_ok, " | ".join(gate_parts)


# ============================================================
# SECTION 6: ICD-10 PREDICTION & DESCRIPTION
# ============================================================
def get_icd_prediction(text: str) -> str:
    """Two-attempt ICD extraction with fallback regex.
    Returns 'UNKNOWN' instead of blank so downstream comparisons
    don't silently fail on empty strings."""
    primary_prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. "
        f"Output the code only.\n\n"
        f"Note:\n{text}\n\nICD-10 Code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code for this note. "
        f"Format: letter + 2 digits + optional dot + digits. "
        f"Examples: J18.9, I21.0, K56.9\n\n"
        f"Note:\n{text[:800]}\n\nCode:"
    )
    for attempt, prompt in enumerate([primary_prompt, fallback_prompt]):
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
# SECTION 8: PLOTS
# ============================================================
def plot_results(df: pd.DataFrame):
    debug_print("PLOT", "Generating plots...")
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
    plt.title(f"Drift Level Breakdown\nCategorical drift: {cat_rate:.1f}%")
    plt.xlabel("Drift Level")
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
        plt.legend()
    plt.title("Changes vs Similarity\ncoloured by drift level")
    plt.xlabel("Words Changed")
    plt.ylabel("Cosine Similarity")

    plt.subplot(1, 4, 4)
    type_cols = ['disease_changes', 'chemical_changes',
                 'anatomy_changes', 'procedure_changes']
    available = [c for c in type_cols
                 if c in valid_df.columns and valid_df[c].sum() > 0]
    if available:
        totals = valid_df[available].sum()
        totals.index = [c.replace('_changes', '') for c in totals.index]
        totals.plot(kind='bar',
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title("Perturbations by Entity Type")
        plt.xlabel("Entity Type")
        plt.xticks(rotation=0)
    else:
        counts = df['baseline_correct'].value_counts()
        lbl, clr = [], []
        if True  in counts.index:
            lbl.append(f'Correct (n={counts[True]})')
            clr.append('#90EE90')
        if False in counts.index:
            lbl.append(f'Wrong (n={counts[False]})')
            clr.append('#FFB6C1')
        plt.pie(counts, labels=lbl, colors=clr, autopct='%1.1f%%')
        plt.title("Baseline Quality\n(gate: LLM clinical equivalence)")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 9: HTML REPORT
# ============================================================
def generate_html_report(df: pd.DataFrame, filename: str = HTML_REPORT):

    DRIFT_COLORS = {
        'none': '#ccffcc', 'subcategory': '#fff3cd',
        'categorical': '#ffcccc', 'unknown': '#eeeeee', 'skipped': '#f9f9f9',
    }
    DRIFT_LABELS = {
        'none': 'STABLE', 'subcategory': 'SUBCATEGORY DRIFT',
        'categorical': 'CATEGORICAL DRIFT',
        'unknown': 'UNKNOWN ICD', 'skipped': 'SKIPPED',
    }

    def highlight_changes(original_text, perturbed_text, changes_str):
        if not changes_str or changes_str in ('None', '', 'SKIPPED'):
            return original_text, perturbed_text
        orig_h, pert_h = original_text, perturbed_text
        for pair in changes_str.split("|"):
            pair = re.sub(r'^\[.*?\]\s*', '', pair.strip())
            if " -> " not in pair:
                continue
            ow, nw = [x.strip() for x in pair.split(" -> ", 1)]
            orig_h = re.sub(
                rf'\b{re.escape(ow)}\b',
                f'<span style="background:#FF6B6B;color:white;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold;">{ow}</span>',
                orig_h, flags=re.IGNORECASE)
            pert_h = re.sub(
                rf'\b{re.escape(nw)}\b',
                f'<span style="background:#51C878;color:white;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold;">{nw}</span>',
                pert_h, flags=re.IGNORECASE)
        return orig_h, pert_h

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        dc          = DRIFT_COLORS.get(drift_level, '#ccffcc')
        dl          = DRIFT_LABELS.get(drift_level, 'STABLE')
        sim_d       = (f"{float(row['semantic_similarity']):.4f}"
                       if row.get('semantic_similarity') else 'N/A')
        base_d      = f"{float(row.get('baseline_sim', 0)):.4f}"
        pred_desc   = row.get('original_pred_desc', '')
        changes_d   = (row['changes_made']
                       if row.get('changes_made') not in
                       ['SKIPPED', '', None] else 'None')
        gate_d      = row.get('baseline_gate', '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — baseline did not match ground truth.<br>'
                f'<small>Gates: {gate_d}</small></p>')
        else:
            oh, ph = highlight_changes(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                row.get('changes_made', ''))
            text_section = f"""
            <div style="margin-top:10px;">
              <div style="margin-bottom:8px;font-size:0.85em;">
                <span style="background:#FF6B6B;color:white;padding:2px 8px;
                border-radius:3px;">Original terms</span>&nbsp;
                <span style="background:#51C878;color:white;padding:2px 8px;
                border-radius:3px;">Medical synonyms applied</span>
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
                  font-size:0.85em;line-height:1.6em;">{oh}</td>
                  <td style="padding:10px;border:1px solid #ccc;
                  vertical-align:top;white-space:pre-wrap;
                  font-size:0.85em;line-height:1.6em;">{ph}</td>
                </tr>
                <tr>
                  <td style="padding:8px;border:1px solid #ccc;
                  background:#f0f0f0;text-align:center;">
                  <b>ICD-10: {row['original_icd_pred']}</b></td>
                  <td style="padding:8px;border:1px solid #ccc;
                  background:#f0f0f0;text-align:center;">
                  <b>ICD-10: {row['perturbed_icd_pred']}</b></td>
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
                <td>{row['ground_truth_desc']}
                ({row['ground_truth_icd']} ICD-10)</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                <td>{row['original_icd_pred']} - {pred_desc}
                (baseline sim: {base_d})</td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                <td><small>{gate_d}</small></td></tr>
            <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                <td>{changes_d} ({row['changes_count']} terms)</td></tr>
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
<html><head><title>Robustness Report - ICD-10</title>
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
<h1>Clinical Note Robustness Report (ICD-10)</h1>
<p style="color:#666;">Model: {MODEL_NAME} | Perturbation: structured medical
entity extraction + 3-layer validation</p>
<div class="method-note">
  <b>Perturbation method:</b> scispaCy NER extracts typed medical entities.
  All terms are sent in one batch LLM call for synonym generation.
  Responses are echo-stripped (Llama arrow format removed) before validation.
  Each candidate then passes structural rules and an LLM clinical safety check.
  Non-medical English words are rejected by POS tagging and a stop set.
  All diagnoses use ICD-10 exclusively.
</div>
<div class="legend">
  <span style="background:#ccffcc;">STABLE - exact ICD-10 match</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT - same 3-char category</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT - different disease</span>
</div>
<div class="summary">
  <h2>Summary</h2>
  <p><b>Total Samples:</b> {len(df)}</p>
  <p><b>Valid for Analysis:</b> {len(valid_df)}
     <span style="color:#888;">(gate: LLM clinical equivalence = YES)</span></p>
  <p><b>Skipped (wrong baseline):</b> {len(df) - len(valid_df)}</p>
  <p><b>Categorical Drift Rate:</b> {cat_rate:.1f}%
     <span style="color:#888;">(primary robustness metric)</span></p>
  <p><b>Subcategory Drift Rate:</b> {sub_rate:.1f}%</p>
  <p><b>Avg Semantic Similarity:</b> {avg_sim}</p>
  <p><b>Avg Medical Terms Changed Per Note:</b> {avg_changes}</p>
</div>
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {filename}")


# ============================================================
# SECTION 10: SANITY CHECK
# ============================================================
def sanity_check(perturber: StructuredMedicalPerturber):
    """Verify entity extraction and echo-strip are both working."""
    print("\n" + "=" * 60)
    print("SANITY CHECK")

    # Check 1: entity extraction
    sample   = ("The patient had myocardial infarction. "
                "Aspirin and heparin were administered. "
                "Echocardiogram showed reduced ejection fraction.")
    entities = perturber.extract_medical_entities(sample)
    print(f"  Entity extraction: {len(entities)} entities")
    for e in entities:
        print(f"    [{e['label']}] {e['text']}")
    print(f"  Entity extraction: {'PASS' if len(entities) > 0 else 'WARNING: 0 found'}")

    # Check 2: echo-strip
    test_cases = [
        ("nausea",          "nausea -> VOMITING",    "VOMITING"),
        ("LACTOSE INT",     "LACTOSE INT -> LACTOSE SENSITIVITY", "LACTOSE SENSITIVITY"),
        ("vital signs",     "vital signs -> VITALS",  "VITALS"),
        ("clean synonym",   "VOMITING",               "VOMITING"),
        ("triple chain",    "A -> B -> FINAL",        "FINAL"),
    ]
    all_pass = True
    for original, raw, expected in test_cases:
        result = _clean_synonym_echo(raw, original)
        ok     = result == expected
        all_pass = all_pass and ok
        print(f"  Echo-strip [{original}]: '{raw}' -> '{result}' "
              f"({'PASS' if ok else 'FAIL expected: ' + expected})")
    print(f"  Echo-strip: {'PASS' if all_pass else 'SOME FAILED'}")

    # Check 3: text extraction
    fake_note = (
        "Patient: John Doe  MRN: 12345  Date: 2024-01-01\n"
        "Chief Complaint: chest pain\n\n"
        "Assessment and Plan:\n"
        "1. STEMI - treated with primary PCI\n"
        "Discharge Diagnosis: Acute myocardial infarction"
    )
    extracted    = MIMICLoader.extract_blind_clinical_info(fake_note)
    has_clinical = any(kw in extracted
                       for kw in ["STEMI", "MI", "chest", "PCI", "infarction"])
    print(f"\n  Text extraction: {len(extracted)} chars — "
          f"{'PASS' if has_clinical else 'WARNING: may be header'}")
    print(f"  First 100: {extracted[:100]}")

    print("=" * 60 + "\n")


# ============================================================
# SECTION 11: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Starting | LLM: {MODEL_NAME} | ICD-10 only mode")

    debug_print("MAIN", "Testing API connection...")
    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to RWTH server. Check VPN.")
        return
    print("LLM API OK")
    ATTACK_LEVEL = 0

    try:
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH,
                              MAX_SAMPLES)
    except Exception as e:
        import traceback
        debug_print("ERROR", f"Data load failed: {e}")
        traceback.print_exc()
        return

    perturber = StructuredMedicalPerturber()
    scorer    = SemanticEvaluator()
    sanity_check(perturber)

    results = []
    print("\n" + "=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            hadm_id   = row['hadm_id']
            orig_text = row['processed_text']
            gt_desc   = row['long_title']
            gt_icd    = row['icd_code']
            icd_ver   = row['icd_version']

            print(f"\n[ID: {hadm_id}] Processing...")
            print(f"   > Ground Truth  : {gt_desc} ({gt_icd} ICD-10)")

            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd)
            print(f"   > Original ICD  : {orig_icd} - {orig_desc}")

            baseline_ok, baseline_sim, llm_ok, gate_str = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd)
            print(f"   > Baseline Gates: {gate_str}")
            print(f"   > Decision      : "
                  f"{'MATCH - proceeding' if baseline_ok else 'MISMATCH - skipping'}")

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
                    "disease_changes"     : 0,
                    "chemical_changes"    : 0,
                    "anatomy_changes"     : 0,
                    "procedure_changes"   : 0,
                    "original_icd_pred"   : orig_icd,
                    "original_pred_desc"  : orig_desc,
                    "baseline_sim"        : baseline_sim,
                    "baseline_llm_ok"     : llm_ok,
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
                print("-" * 40)
                continue

            print(f"   > Perturbing...")
            pert_text, changes_log = perturber.perturb(orig_text, attack_level=ATTACK_LEVEL)

            if changes_log:
                print(f"   > Attack Status : "
                      f"{len(changes_log)} validated synonyms applied.")
            else:
                print("   > Attack Status : No valid synonyms found.")

            pert_icd   = get_icd_prediction(pert_text)
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995 and len(changes_log) <= 3)

            print(f"   > Perturbed ICD : {pert_icd}")
            print(f"   > Drift Level   : {drift_info['drift_level']}")
            print(f"   > Text Sim      : {sim:.4f}")
            if suspicious:
                print("   > WARNING: suspicious drift (high sim + few changes)")

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
                "disease_changes"     : sum(1 for _, _, l in changes_log
                                            if l == "DISEASE"),
                "chemical_changes"    : sum(1 for _, _, l in changes_log
                                            if l == "CHEMICAL"),
                "anatomy_changes"     : sum(1 for _, _, l in changes_log
                                            if l == "ANATOMY"),
                "procedure_changes"   : sum(1 for _, _, l in changes_log
                                            if l == "PROCEDURE"),
                "original_icd_pred"   : orig_icd,
                "original_pred_desc"  : orig_desc,
                "baseline_sim"        : baseline_sim,
                "baseline_llm_ok"     : llm_ok,
                "baseline_gate"       : gate_str,
                "baseline_correct"    : True,
                "perturbed_icd_pred"  : pert_icd,
                **drift_info,
                "suspicious_drift"    : suspicious,
                "semantic_similarity" : sim,
            })
            print("-" * 40)

        except Exception as e:
            import traceback
            debug_print("ERROR", f"Row {idx} failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No results to save.")
        return

    final_df = pd.DataFrame(results)
    csv_df   = final_df.drop(
        columns=['original_text', 'perturbed_text'], errors='ignore')
    csv_df.to_csv(FINAL_RESULTS, index=False)

    plot_results(final_df)
    generate_html_report(final_df)

    valid_df = final_df[final_df['baseline_correct'] == True]
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print(f"  Total samples             : {len(final_df)}")
    print(f"  Valid (ICD-10, matched)   : {len(valid_df)}")
    print(f"  Skipped (wrong baseline)  : {len(final_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"  Categorical drift         : "
              f"{valid_df['categorical_drift'].mean()*100:.1f}%  <- primary metric")
        print(f"  Subcategory drift         : "
              f"{valid_df['subcategory_drift'].mean()*100:.1f}%")
        print(f"  Avg text similarity       : "
              f"{valid_df['semantic_similarity'].mean():.4f}")
        print(f"  Avg medical terms changed : "
              f"{valid_df['changes_count'].mean():.1f}")
        print(f"  DISEASE changes           : {valid_df['disease_changes'].sum()}")
        print(f"  CHEMICAL changes          : {valid_df['chemical_changes'].sum()}")
        print(f"  ANATOMY changes           : {valid_df['anatomy_changes'].sum()}")
        print(f"  PROCEDURE changes         : {valid_df['procedure_changes'].sum()}")
        print(f"  Suspicious drifts         : {valid_df['suspicious_drift'].sum()}")
    print(f"  CSV  -> {FINAL_RESULTS}")
    print(f"  HTML -> {HTML_REPORT}")
    print(f"  Plot -> {PLOT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()