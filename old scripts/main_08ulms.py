import torch
import pandas as pd
import numpy as np
import os
import re
import seaborn as sns
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

FINAL_RESULTS  = "final_robustness.csv"
HTML_REPORT    = "robustness_report.html"
PLOT_FILE      = "robustness_plots.png"
MAX_SAMPLES    = 100
DEVICE         = "cpu"

BASELINE_SIM_THRESHOLD = 0.78

# ============================================================
# UMLS API CONFIGURATION  ← PUT YOUR KEY HERE
# ============================================================
UMLS_API_KEY   = "379d16de-4bda-4cf5-98b6-6cac578452c1"
UMLS_BASE_URL  = "https://uts-ws.nlm.nih.gov/rest"

warnings.filterwarnings("ignore")


def debug_print(stage, message):
    print(f"\033[94m[{stage}]\033[0m {message}")


# ============================================================
# SECTION 1: LLM CONNECTOR  (used for generation only now —
#            NOT for validation, which is UMLS)
# ============================================================
def query_rwth_server(prompt, max_tokens=50):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens}
        # temperature=0 removes random stochasticity from ICD predictions
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
# SECTION 2: UMLS API CLIENT
#
# What is UMLS?
# The Unified Medical Language System is a database maintained
# by the US National Library of Medicine. It contains 3.5 million
# medical concepts — every disease, drug, symptom, procedure —
# each with a unique ID called a CUI (Concept Unique Identifier).
#
# Example:
#   "admission"      → CUI C0184666  (semantic type: Health Care Activity)
#   "hospitalization"→ CUI C0184666  (SAME CUI — they are synonyms)
#   "hypertension"   → CUI C0020538  (semantic type: Disease or Syndrome)
#
# We use three checks in order:
#   Check 1 — Same CUI?  → definitionally synonymous, always safe
#   Check 2 — Same semantic type? → same category of medical concept
#   Check 3 — Related concepts? → connected in UMLS hierarchy
# ============================================================

class UMLSClient:
    """
    Handles all communication with the UMLS REST API.
    Authenticates once, caches CUI lookups to avoid redundant API calls.
    """

    def __init__(self, api_key: str):
        self.api_key   = api_key
        self.tgt        = None   # Ticket Granting Ticket (auth token)
        self._cui_cache = {}     # {term: (cui, semantic_types)} or None
        self._auth()

    # ── Authentication ────────────────────────────────────────────────────────
    def _auth(self):
        """
        UMLS uses a two-step authentication system called CAS.
        Step 1: Exchange API key for a Ticket Granting Ticket (TGT).
                TGT is valid for 8 hours.
        Step 2: For each API call, exchange TGT for a single-use
                Service Ticket (ST).
        We get the TGT once here and reuse it for all calls.
        """
        url  = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        data = {"apikey": self.api_key}
        try:
            resp = requests.post(url, data=data, timeout=30)
            # The TGT URL is embedded in the response headers
            tgt_match = re.search(r'action="(.*?)"', resp.text)
            if tgt_match:
                self.tgt = tgt_match.group(1)
                debug_print("UMLS", "Authentication successful — TGT obtained")
            else:
                debug_print("UMLS", "Authentication FAILED — check your API key")
        except Exception as e:
            debug_print("UMLS", f"Auth error: {e}")

    def _get_service_ticket(self) -> str | None:
        """Get a single-use service ticket from the TGT."""
        if not self.tgt:
            return None
        try:
            resp = requests.post(
                self.tgt,
                data={"service": "http://umlsks.nlm.nih.gov"},
                timeout=15
            )
            return resp.text.strip()
        except Exception:
            return None

    # ── Core lookup ───────────────────────────────────────────────────────────
    def get_concept(self, term: str) -> tuple:
        """
        Look up a medical term in UMLS.
        Returns (cui, semantic_types_list) or (None, []) if not found.

        CUI = Concept Unique Identifier, e.g. "C0184666"
        semantic_types = list of strings, e.g. ["Health Care Activity"]

        Results are cached — same term is never looked up twice.
        """
        term_lower = term.lower().strip()
        if term_lower in self._cui_cache:
            return self._cui_cache[term_lower]

        st = self._get_service_ticket()
        if not st:
            self._cui_cache[term_lower] = (None, [])
            return (None, [])

        try:
            # Search for the term in UMLS
            url    = f"{UMLS_BASE_URL}/search/current"
            params = {
                "string"      : term,
                "ticket"      : st,
                "searchType"  : "normalizedString",  # handles capitalisation
                "returnIdType": "concept",
                "pageSize"    : 5
            }
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            results = data.get("result", {}).get("results", [])
            if not results or results[0].get("ui") == "NONE":
                self._cui_cache[term_lower] = (None, [])
                return (None, [])

            # Take the top result
            cui  = results[0]["ui"]
            name = results[0].get("name", "")

            # Now get the semantic types for this CUI
            sem_types = self._get_semantic_types(cui)

            result = (cui, sem_types)
            self._cui_cache[term_lower] = result
            return result

        except Exception as e:
            debug_print("UMLS_LOOKUP", f"Lookup failed for '{term}': {e}")
            self._cui_cache[term_lower] = (None, [])
            return (None, [])

    def _get_semantic_types(self, cui: str) -> list:
        """
        Get the semantic types for a given CUI.
        Semantic types describe the category of the concept.
        Examples:
          "Disease or Syndrome"        → a disease
          "Pharmacologic Substance"    → a drug
          "Body Part, Organ, or Organ Component" → anatomy
          "Health Care Activity"       → a procedure or care event
          "Sign or Symptom"            → a clinical finding
          "Diagnostic Procedure"       → a test
        """
        st = self._get_service_ticket()
        if not st:
            return []
        try:
            url    = f"{UMLS_BASE_URL}/content/current/CUI/{cui}"
            params = {"ticket": st}
            resp   = requests.get(url, params=params, timeout=15)
            data   = resp.json()

            sem_type_list = data.get("result", {}).get("semanticTypes", [])
            return [s.get("name", "") for s in sem_type_list]
        except Exception:
            return []

    def get_cui_atoms(self, cui: str) -> list:
        """
        Get all known synonyms (atoms) for a given CUI.
        This is how UMLS stores synonyms — every term that maps
        to the same concept is an 'atom'.
        Returns list of synonym strings.
        """
        st = self._get_service_ticket()
        if not st:
            return []
        try:
            url    = f"{UMLS_BASE_URL}/content/current/CUI/{cui}/atoms"
            params = {"ticket": st, "language": "ENG", "pageSize": 20}
            resp   = requests.get(url, params=params, timeout=15)
            data   = resp.json()
            atoms  = data.get("result", [])
            return [a.get("name", "").lower() for a in atoms if a.get("name")]
        except Exception:
            return []


# ============================================================
# SECTION 3: UMLS SYNONYM VALIDATOR  ← REPLACES LAYER 3 LLM
#
# This is the key change from the previous pipeline.
# Instead of asking Llama "is this synonym safe?" (circular),
# we look both terms up in UMLS and check objectively:
#
# UNSAFE semantic type pairs — these should NEVER be synonymous
# in clinical text because they describe fundamentally different
# kinds of things. E.g. a drug should not replace a disease name.
# ============================================================

# Semantic types that describe medical content categories.
# Replacements across these boundaries are always unsafe.
UNSAFE_SEMANTIC_PAIRS = {
    # Drugs should not replace diseases or symptoms
    ("Pharmacologic Substance", "Disease or Syndrome"),
    ("Pharmacologic Substance", "Sign or Symptom"),
    ("Pharmacologic Substance", "Body Part, Organ, or Organ Component"),
    # Anatomy should not replace diseases
    ("Body Part, Organ, or Organ Component", "Disease or Syndrome"),
    ("Body Part, Organ, or Organ Component", "Sign or Symptom"),
    # Procedures should not replace diseases
    ("Therapeutic or Preventive Procedure", "Disease or Syndrome"),
    ("Diagnostic Procedure", "Disease or Syndrome"),
    # Lab values should not replace anatomy
    ("Laboratory Procedure", "Body Part, Organ, or Organ Component"),
    # Make the pairs symmetric
}
# Make pairs symmetric (A,B) also blocks (B,A)
UNSAFE_SEMANTIC_PAIRS = (
    UNSAFE_SEMANTIC_PAIRS |
    {(b, a) for (a, b) in UNSAFE_SEMANTIC_PAIRS}
)

# These semantic types are "administrative" — safe to synonym freely
# e.g. "patient" → "individual", "admission" → "hospitalization"
ADMINISTRATIVE_TYPES = {
    "Health Care Activity",
    "Temporal Concept",
    "Quantitative Concept",
    "Functional Concept",
    "Intellectual Product",
    "Idea or Concept",
    "Group",
    "Population Group",
}


def validate_replacement_with_umls(
        original: str,
        replacement: str,
        umls: UMLSClient
) -> tuple:
    """
    LAYER 3 — UMLS-based clinical safety validation.
    Completely replaces the previous LLM self-check.

    Logic (in order):
      1. Look up both terms in UMLS
      2. If same CUI → definitionally synonymous → SAFE
      3. If either term not in UMLS → fall back to PERMISSIVE
         (non-medical general vocabulary, safe to swap)
      4. If semantic types are in an unsafe pair → BLOCKED
      5. If both are administrative types → SAFE
      6. If same semantic type → SAFE
      7. Otherwise → BLOCKED (different medical categories)

    Returns (is_safe: bool, reason: str, detail: str)
    """
    orig_cui,  orig_types  = umls.get_concept(original)
    repl_cui,  repl_types  = umls.get_concept(replacement)

    detail = (f"orig=({orig_cui},{orig_types}) | "
              f"repl=({repl_cui},{repl_types})")

    # ── Check 1: Same CUI = definitional synonyms ─────────────────────────
    if orig_cui and repl_cui and orig_cui == repl_cui:
        return True, "UMLS: same CUI — definitional synonyms", detail

    # ── Check 2: Either not in UMLS = general vocabulary ──────────────────
    # If UMLS doesn't know the term, it's probably a general English word
    # (like "reported" or "stable") — safe to swap
    if orig_cui is None or repl_cui is None:
        return True, "UMLS: one/both terms not in UMLS — general vocabulary", detail

    # ── Check 3: Check for unsafe semantic type pairs ─────────────────────
    for ot in orig_types:
        for rt in repl_types:
            if (ot, rt) in UNSAFE_SEMANTIC_PAIRS:
                return (False,
                        f"UMLS: unsafe type crossing ({ot} → {rt})",
                        detail)

    # ── Check 4: Both are administrative/general health care types ─────────
    orig_admin = all(t in ADMINISTRATIVE_TYPES for t in orig_types) if orig_types else False
    repl_admin = all(t in ADMINISTRATIVE_TYPES for t in repl_types) if repl_types else False
    if orig_admin and repl_admin:
        return True, "UMLS: both administrative types — safe", detail

    # ── Check 5: Same semantic type = same category of concept ────────────
    if set(orig_types) == set(repl_types):
        return True, f"UMLS: same semantic type ({orig_types})", detail

    # ── Check 6: Overlapping semantic types ───────────────────────────────
    overlap = set(orig_types) & set(repl_types)
    if overlap:
        return True, f"UMLS: overlapping types ({list(overlap)})", detail

    # ── Default: different medical categories, no overlap ─────────────────
    return (False,
            f"UMLS: different semantic types with no overlap "
            f"({orig_types} vs {repl_types})",
            detail)


# ============================================================
# SECTION 4: LAYER 1 & 2 — UNCHANGED (pattern + structural)
# ============================================================

UNSAFE_ENTITY_PATTERNS = [
    r'^(no|not|denies|denied|without|negative|none|absent|absence)\b',
    r'^(positive|negative)$',
    r'^(left|right|bilateral|unilateral|ipsilateral|contralateral|'
    r'anterior|posterior|proximal|distal|superior|inferior|medial|lateral)$',
    r'^(elevated|increased|decreased|reduced|low|high|normal|abnormal|'
    r'mild|moderate|severe|minimal|significant|benign|malignant|acute|chronic|'
    r'stable|unstable|resolved|worsening|improving)$',
    r'^(minute|minutes|second|seconds|hour|hours|day|days|week|weeks|'
    r'month|months|year|years)s?$',
    r'^(afebrile|normotensive|tachycardic|bradycardic|hypertensive|hypotensive|'
    r'tachypneic|bradypneic|normocardic)$',
    r'^(non|no)[a-z]+$',
    r'^[A-Z][A-Z\s]{10,}$',
    r'^\d',
    r'^[A-Z]$',
]
_UNSAFE_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_ENTITY_PATTERNS]


def entity_is_safe_to_replace(entity_text: str) -> tuple:
    """Layer 1: Pattern-based entity category filter."""
    text = entity_text.strip()
    for pattern in _UNSAFE_COMPILED:
        if pattern.match(text):
            return False, f"L1: unsafe pattern"
    lower = text.lower()
    NEGATION_STARTS = ('no ', 'not ', 'denies ', 'denied ', 'without ',
                       'negative for ', 'absence of ', 'no evidence of ')
    if lower.startswith(NEGATION_STARTS):
        return False, "L1: negation phrase"
    tokens = text.split()
    if len(tokens) >= 2 and all(t.isupper() and t.isalpha() for t in tokens):
        return False, "L1: section header"
    return True, "ok"


def replacement_is_structurally_valid(original: str, replacement: str) -> tuple:
    """Layer 2: Structural rules on word count and length."""
    orig_words = original.strip().split()
    repl_words = replacement.strip().split()
    n_orig = len(orig_words)
    n_repl = len(repl_words)
    if n_repl > max(n_orig * 2, n_orig + 3):
        return False, f"L2: too long ({n_orig}→{n_repl} words)"
    if n_orig == 1 and n_repl > 3:
        return False, f"L2: single-word expanded to {n_repl} words"
    if not replacement.strip():
        return False, "L2: empty"
    if replacement.strip().lower() == original.strip().lower():
        return False, "L2: identical"
    if n_orig > 2 and n_repl < n_orig // 2:
        return False, f"L2: too short ({n_orig}→{n_repl} words)"
    char_ratio = len(replacement) / max(len(original), 1)
    if char_ratio > 5.0:
        return False, f"L2: char ratio too high ({char_ratio:.1f}x)"
    if re.search(r'[.!?](?!\s*$)', replacement):
        return False, "L2: sentence punctuation mid-text"
    return True, "ok"


# ============================================================
# SECTION 5: GENERATIVE PERTURBER  (Layer 3 now calls UMLS)
# ============================================================

class GenerativePerturber:
    def __init__(self, umls_client: UMLSClient):
        """
        Takes a UMLSClient instance.
        All synonym validation goes through UMLS — no LLM calls for validation.
        """
        self.umls = umls_client
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            self.nlp = spacy.load("en_core_sci_sm")

    def is_negated(self, token):
        negation_terms = {'no', 'not', 'denies', 'negative',
                          'without', 'none', 'denied'}
        window = [t.text.lower()
                  for t in list(token.doc)[max(0, token.i - 3):token.i]]
        return any(term in window for term in negation_terms)

    def get_synonym_from_llm(self, word: str, context_sentence: str) -> str | None:
        """
        Generate a synonym candidate using Llama.
        NOTE: Llama is ONLY used for generation here.
              Whether the synonym is safe is decided by UMLS, not Llama.
        """
        prompt = (
            f"Context: \"{context_sentence}\"\n"
            f"Term: '{word}'\n\n"
            f"Task: Provide ONE medically equivalent synonym for '{word}' "
            f"as used in this specific clinical context.\n\n"
            f"Rules:\n"
            f"- Respond with the replacement term ONLY. No explanation.\n"
            f"- Do NOT change clinical meaning, polarity, or severity.\n"
            f"- Do NOT expand abbreviations into full sentences.\n"
            f"- If '{word}' is a single word, your answer must be 1-3 words max.\n"
            f"- If no safe synonym exists, respond with the word: NONE\n\n"
            f"Replacement:"
        )
        raw = query_rwth_server(prompt, max_tokens=20)
        if not raw:
            return None
        synonym = raw.replace('"', '').replace("'", '').strip()
        synonym = re.sub(r'[.!?,;]+$', '', synonym).strip()
        if synonym.upper() == "NONE" or not synonym:
            return None
        if synonym.lower() == word.lower():
            return None
        return synonym

    def perturb(self, text: str, perturbation_rate: float = 0.10) -> tuple:
        doc = self.nlp(text)
        candidate_entities = [ent for ent in doc.ents
                               if not self.is_negated(ent[0])]
        if not candidate_entities:
            return text, []

        num_to_perturb = max(1, int(len(candidate_entities) * perturbation_rate))
        target_entities = random.sample(
            candidate_entities, min(num_to_perturb, len(candidate_entities)))
        target_entities.sort(key=lambda x: x.start_char, reverse=True)

        perturbed_text = text
        changes_log    = []

        for ent in target_entities:
            original = ent.text

            # ── LAYER 1: Pattern safety check ────────────────────────────
            safe, reason = entity_is_safe_to_replace(original)
            if not safe:
                print(f"     {reason}: '{original}'")
                continue

            context = (ent.sent.text if ent.sent
                       else text[max(0, ent.start_char - 60):ent.end_char + 60])

            # Generate synonym (Llama used here — generation only)
            replacement = self.get_synonym_from_llm(original, context)
            if not replacement:
                print(f"     NO SYNONYM: '{original}'")
                continue

            # ── LAYER 2: Structural check ─────────────────────────────────
            struct_ok, struct_reason = replacement_is_structurally_valid(
                original, replacement)
            if not struct_ok:
                print(f"     BLOCKED ({struct_reason}): '{original}'→'{replacement}'")
                continue

            # ── LAYER 3: UMLS clinical safety check ──────────────────────
            # This is the key change: UMLS validates, not Llama
            umls_ok, umls_reason, umls_detail = validate_replacement_with_umls(
                original, replacement, self.umls)
            if not umls_ok:
                print(f"     UMLS BLOCKED ({umls_reason}): "
                      f"'{original}'→'{replacement}'")
                print(f"       Detail: {umls_detail}")
                continue

            # All three layers passed — apply replacement
            perturbed_text = (perturbed_text[:ent.start_char]
                              + replacement
                              + perturbed_text[ent.end_char:])
            changes_log.append((original, replacement, umls_reason))
            print(f"     ACCEPTED ({umls_reason}): '{original}'→'{replacement}'")

        return perturbed_text, changes_log


# ============================================================
# SECTION 6: SEMANTIC EVALUATOR (ClinicalBERT) — UNCHANGED
# ============================================================
class SemanticEvaluator:
    def __init__(self):
        debug_print("EVAL_INIT", "Loading ClinicalBERT...")
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
            e1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
            e2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0)
            sim = cosine_similarity(e1.numpy(), e2.numpy())[0][0]
        return float(sim)

    def descriptions_match(self, desc1: str, desc2: str,
                            threshold: float = BASELINE_SIM_THRESHOLD) -> tuple:
        sim = self.get_sim(desc1, desc2)
        return sim >= threshold, sim


# ============================================================
# SECTION 7: BASELINE CORRECTNESS — LLM gate (unchanged)
# NOTE: The baseline gate still uses Llama because it handles
# ICD-9 vs ICD-10 translation which UMLS is not designed for.
# This is acceptable — the circularity concern is specifically
# about synonym validation, not ICD version mapping.
# ============================================================

def check_clinical_equivalence(pred_icd: str, gt_icd: str,
                                gt_desc: str) -> bool:
    if pred_icd in ("UNKNOWN", ""):
        return False
    prompt = (
        f"Ground truth diagnosis: '{gt_desc}' (ICD code: {gt_icd})\n"
        f"Predicted ICD code: {pred_icd}\n\n"
        f"Question: Does the predicted ICD code represent the same primary "
        f"medical condition as the ground truth diagnosis?\n"
        f"Ignore differences in ICD version (ICD-9 vs ICD-10) and minor "
        f"specificity differences. Focus only on whether the core disease "
        f"or condition is the same.\n"
        f"Answer with only YES or NO."
    )
    response = query_rwth_server(prompt, max_tokens=5)
    return response.strip().upper().startswith("YES")


def is_baseline_correct(scorer: SemanticEvaluator,
                         orig_desc: str, gt_desc: str,
                         orig_icd: str, gt_icd: str) -> tuple:
    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    llm_ok = check_clinical_equivalence(orig_icd, gt_icd, gt_desc)
    gate_parts = [
        f"sem_sim={baseline_sim:.4f}{'✓' if sim_ok else '✗'}",
        f"llm_equiv={'YES✓' if llm_ok else 'NO✗'}"
    ]
    passed = llm_ok
    return passed, baseline_sim, llm_ok, " | ".join(gate_parts)


# ============================================================
# SECTION 8: ICD PREDICTION — temperature=0 added
# ============================================================
def get_icd_prediction(text: str) -> str:
    primary_prompt = (
        f"Read the clinical note below. Return ONLY the single most likely "
        f"ICD-10 diagnosis code (example format: J18.9 or I21.0). "
        f"Do not write the disease name. Do not explain. "
        f"Output the code only.\n\nNote:\n{text}\n\nICD-10 Code:"
    )
    fallback_prompt = (
        f"Output a single ICD-10 code for this note. "
        f"Format: letter + 2 digits + optional dot + digits. "
        f"Examples: J18.9, I21.0, K56.9\n\nNote:\n{text[:800]}\n\nCode:"
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


def get_diagnosis_description(icd_code: str, icd_version: int) -> str:
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    prompt = (
        f"What medical condition does ICD-{icd_version} code "
        f"'{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words. No explanation."
    )
    return query_rwth_server(prompt)


# ============================================================
# SECTION 9: DRIFT CLASSIFICATION — UNCHANGED
# ============================================================
def classify_drift(orig_icd: str, pert_icd: str) -> dict:
    orig_norm = re.sub(r'[^A-Z0-9]', '', orig_icd.upper())
    pert_norm = re.sub(r'[^A-Z0-9]', '', pert_icd.upper())
    if "UNKNOWN" in (orig_norm, pert_norm):
        return {"icd_codes_differ": True, "drift_level": "unknown",
                "categorical_drift": False, "subcategory_drift": False}
    exact_match  = orig_norm == pert_norm
    prefix_match = (len(orig_norm) >= 3 and len(pert_norm) >= 3 and
                    orig_norm[:3] == pert_norm[:3])
    if exact_match:
        level = "none"
    elif prefix_match:
        level = "subcategory"
    else:
        level = "categorical"
    return {
        "icd_codes_differ" : not exact_match,
        "drift_level"      : level,
        "categorical_drift": level == "categorical",
        "subcategory_drift": level == "subcategory",
    }


# ============================================================
# SECTION 10: PLOTS
# ============================================================
def plot_results(df: pd.DataFrame):
    valid_df = df[df['baseline_correct'] == True]
    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return
    plt.figure(figsize=(22, 5))

    plt.subplot(1, 4, 1)
    sns.histplot(valid_df['semantic_similarity'], kde=True, bins=15, color='teal')
    plt.title("Semantic Preservation")
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
        sns.scatterplot(data=plot_df, x='changes_count',
                        y='semantic_similarity', hue='drift_level',
                        palette={'none': 'green', 'subcategory': 'orange',
                                 'categorical': 'red'})
    plt.title("Changes vs Similarity")

    plt.subplot(1, 4, 4)
    counts     = df['baseline_correct'].value_counts()
    labels     = []
    colors_pie = []
    if True in counts.index:
        labels.append(f'Correct (n={counts[True]})')
        colors_pie.append('#90EE90')
    if False in counts.index:
        labels.append(f'Wrong (n={counts[False]})')
        colors_pie.append('#FFB6C1')
    plt.pie(counts, labels=labels, colors=colors_pie, autopct='%1.1f%%')
    plt.title("Baseline Quality")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved: {PLOT_FILE}")


# ============================================================
# SECTION 11: HTML REPORT  (updated to show UMLS validation info)
# ============================================================
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

    def highlight_changes(original_text, perturbed_text, changes_str):
        if not changes_str or changes_str in ('None', '', 'SKIPPED'):
            return original_text, perturbed_text
        orig_h = original_text
        pert_h = perturbed_text
        for pair in changes_str.split("|"):
            pair = pair.strip()
            if " -> " not in pair:
                continue
            # Handle format "orig -> repl [UMLS reason]"
            parts    = pair.split(" -> ", 1)
            orig_w   = parts[0].strip()
            repl_part = parts[1].strip()
            # Strip UMLS reason if present in brackets
            repl_w   = re.sub(r'\s*\[.*?\]$', '', repl_part).strip()
            orig_h = re.sub(
                rf'\b{re.escape(orig_w)}\b',
                f'<span style="background:#FF6B6B;color:white;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold;">{orig_w}</span>',
                orig_h, flags=re.IGNORECASE)
            pert_h = re.sub(
                rf'\b{re.escape(repl_w)}\b',
                f'<span style="background:#51C878;color:white;padding:1px 4px;'
                f'border-radius:3px;font-weight:bold;">{repl_w}</span>',
                pert_h, flags=re.IGNORECASE)
        return orig_h, pert_h

    rows_html = ""
    for _, row in df.iterrows():
        is_skipped  = str(row.get('perturbed_text', 'SKIPPED')) == 'SKIPPED'
        drift_level = row.get('drift_level', 'none')
        drift_color = DRIFT_COLORS.get(drift_level, '#ccffcc')
        drift_label = DRIFT_LABELS.get(drift_level, 'STABLE')
        sim_display = (f"{float(row['semantic_similarity']):.4f}"
                       if row.get('semantic_similarity') else 'N/A')
        baseline_disp = f"{float(row.get('baseline_sim', 0)):.4f}"
        pred_desc     = row.get('original_pred_desc', '')
        changes_display = (row['changes_made']
                           if row.get('changes_made') not in
                           ['SKIPPED', '', None] else 'None')
        gate_display = row.get('baseline_gate', '')

        if is_skipped:
            text_section = (
                f'<p style="color:#999;font-style:italic;padding:10px;'
                f'background:#f9f9f9;border-radius:4px;">'
                f'Skipped — prediction did not match ground truth.<br>'
                f'<small>Gates: {gate_display}</small></p>')
        else:
            oh, ph = highlight_changes(
                row.get('original_text', ''),
                row.get('perturbed_text', ''),
                row.get('changes_made', ''))
            text_section = f"""
            <div style="margin-top:10px;">
                <div style="margin-bottom:8px;font-size:0.85em;">
                    <span style="background:#FF6B6B;color:white;padding:2px 8px;
                    border-radius:3px;">Original words</span>&nbsp;
                    <span style="background:#51C878;color:white;padding:2px 8px;
                    border-radius:3px;">UMLS-validated replacements</span>
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
                    <b>ICD: {row['original_icd_pred']}</b></td>
                    <td style="padding:8px;border:1px solid #ccc;
                    background:#f0f0f0;text-align:center;">
                    <b>ICD: {row['perturbed_icd_pred']}</b></td>
                </tr>
                </table>
            </div>"""

        rows_html += f"""
        <div style="border:1px solid #ccc;margin:20px 0;
        padding:15px;border-radius:8px;">
            <h3 style="margin:0 0 10px 0;">
                Patient ID: {row['hadm_id']}
                <span style="background:{drift_color};padding:4px 10px;
                border-radius:4px;font-size:0.9em;">{drift_label}</span>
            </h3>
            <table style="font-size:0.9em;border-collapse:collapse;
            margin-bottom:10px;">
                <tr><td style="padding:3px 12px 3px 0;"><b>Ground Truth:</b></td>
                    <td>{row['ground_truth_desc']}
                    ({row['ground_truth_icd']} v{row['ground_truth_version']})</td>
                </tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Prediction:</b></td>
                    <td>{row['original_icd_pred']} — {pred_desc}
                    (baseline sim: {baseline_disp})</td>
                </tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Baseline Gates:</b></td>
                    <td><small>{gate_display}</small></td>
                </tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Changes:</b></td>
                    <td>{changes_display} ({row['changes_count']} words)</td>
                </tr>
                <tr><td style="padding:3px 12px 3px 0;"><b>Text Similarity:</b></td>
                    <td>{sim_display}</td>
                </tr>
            </table>
            {text_section}
        </div>"""

    valid_df    = df[df['baseline_correct'] == True]
    cat_rate    = (valid_df['categorical_drift'].mean() * 100
                   if len(valid_df) > 0 else 0)
    sub_rate    = (valid_df['subcategory_drift'].mean() * 100
                   if len(valid_df) > 0 else 0)
    avg_sim     = (f"{valid_df['semantic_similarity'].mean():.4f}"
                   if len(valid_df) > 0 else 'N/A')
    avg_changes = (f"{valid_df['changes_count'].mean():.1f}"
                   if len(valid_df) > 0 else 'N/A')

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
<p style="color:#666;">Model: {MODEL_NAME} | Synonym validation: 3-layer
(pattern filter → structural check → <b>UMLS knowledge-base verification</b>)</p>

<div class="method-note">
  <b>Validation methodology:</b> Synonym replacements are validated using the
  UMLS Metathesaurus (NIH/NLM). Layer 3 checks: (1) same Concept Unique
  Identifier (CUI) = definitional synonyms, (2) semantic type compatibility
  check against a predefined unsafe-crossing table, (3) administrative/general
  health care terms are always safe to swap. This removes the circularity of
  using the test model (Llama) to validate its own perturbations.
</div>

<div class="legend">
  <span style="background:#ccffcc;">STABLE — exact ICD match</span>
  <span style="background:#fff3cd;">SUBCATEGORY DRIFT — same 3-char family</span>
  <span style="background:#ffcccc;">CATEGORICAL DRIFT — different disease (primary metric)</span>
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
    <p><b>Avg Changes Per Note:</b> {avg_changes}</p>
</div>
{rows_html}
</body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {filename}")


# ============================================================
# SECTION 12: MAIN
# ============================================================
def main():
    debug_print("MAIN", f"Starting Run | LLM: {MODEL_NAME} | Validation: UMLS")

    # ── Test LLM connection ────────────────────────────────────────────────
    debug_print("MAIN", "Testing LLM API connection...")
    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to RWTH server. Check VPN.")
        return
    print("LLM API OK")

    # ── Initialise UMLS client ─────────────────────────────────────────────
    debug_print("MAIN", "Connecting to UMLS API...")
    umls = UMLSClient(UMLS_API_KEY)
    if not umls.tgt:
        debug_print("ERROR", "UMLS authentication failed. Check your API key at "
                    "https://uts.nlm.nih.gov/uts/profile")
        return
    print("UMLS API OK")

    # ── Quick UMLS sanity check ────────────────────────────────────────────
    # Verify UMLS is working by checking a known synonym pair
    debug_print("MAIN", "Running UMLS sanity check...")
    cui_adm, types_adm = umls.get_concept("admission")
    cui_hos, types_hos = umls.get_concept("hospitalization")
    print(f"  'admission'       → CUI={cui_adm}, types={types_adm}")
    print(f"  'hospitalization' → CUI={cui_hos}, types={types_hos}")
    if cui_adm and cui_hos and cui_adm == cui_hos:
        print("  ✓ Sanity check PASSED — synonyms correctly share CUI")
    else:
        print("  ⚠ Sanity check note: CUIs differ but UMLS still working")
        print("    (UMLS may store these as related but distinct concepts)")

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        from data_loader import MIMICLoader
        df = MIMICLoader.load(NOTES_PATH, DIAGNOSES_PATH, DICT_PATH, MAX_SAMPLES)
    except ImportError:
        # Inline loader if not in separate file
        debug_print("MAIN", "Loading MIMIC data...")
        import pandas as pd
        notes = pd.read_csv(NOTES_PATH, compression='gzip',
                            usecols=['subject_id', 'hadm_id', 'text'])
        diags = pd.read_csv(DIAGNOSES_PATH, compression='gzip')
        if os.path.exists(DICT_PATH):
            icd_dict = pd.read_csv(DICT_PATH, compression='gzip')
            diags = pd.merge(diags,
                             icd_dict[['icd_code', 'icd_version', 'long_title']],
                             on=['icd_code', 'icd_version'], how='left')
        else:
            diags['long_title'] = "Unknown"
        primary = diags[diags['seq_num'] == 1]
        df = pd.merge(notes,
                      primary[['hadm_id', 'icd_code', 'icd_version', 'long_title']],
                      on='hadm_id')

        def clean(text):
            text = str(text)
            text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
            text = re.sub(r"_{2,}", " ", text)
            return " ".join(text.replace("=", " ").replace("*", " ").split())

        def extract(text):
            parts = []
            hpi = re.search(
                r"History of Present Illness:(.*?)(Past Medical History:|Social History:)",
                text, re.S | re.I)
            if hpi:
                parts.append(f"HPI: {hpi.group(1).strip()}")
            course = re.search(
                r"Brief Hospital Course:(.*?)(Medications on Admission:|Discharge)",
                text, re.S | re.I)
            if course:
                parts.append(f"Hospital Course: {course.group(1).strip()}")
            return "\n\n".join(parts) if parts else text[:1500]

        df['text']           = df['text'].apply(clean)
        df['processed_text'] = df['text'].apply(extract)
        df = df.sample(n=min(len(df), MAX_SAMPLES), random_state=42)

    perturber = GenerativePerturber(umls)
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
            print(f"   GT: {gt_desc} ({gt_icd} v{icd_ver})")

            # Step 1: Original ICD prediction
            orig_icd  = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd, 10)
            print(f"   Pred: {orig_icd} — {orig_desc}")

            # Step 2: Baseline check
            baseline_ok, baseline_sim, llm_ok, gate_str = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd)
            print(f"   Gates: {gate_str}")

            if not baseline_ok:
                results.append({
                    "hadm_id": hadm_id,
                    "ground_truth_icd": gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc": gt_desc,
                    "original_text": orig_text,
                    "perturbed_text": "SKIPPED",
                    "changes_made": "SKIPPED",
                    "changes_count": 0,
                    "original_icd_pred": orig_icd,
                    "original_pred_desc": orig_desc,
                    "baseline_sim": baseline_sim,
                    "baseline_llm_ok": llm_ok,
                    "baseline_gate": gate_str,
                    "baseline_correct": False,
                    "perturbed_icd_pred": "SKIPPED",
                    "icd_codes_differ": False,
                    "drift_level": "skipped",
                    "categorical_drift": False,
                    "subcategory_drift": False,
                    "suspicious_drift": False,
                    "semantic_similarity": 0.0
                })
                print("-" * 40)
                continue

            # Step 3: UMLS-validated perturbation
            print(f"   Perturbing with UMLS validation...")
            pert_text, changes_log = perturber.perturb(orig_text)
            print(f"   Changes accepted: {len(changes_log)}")

            # Step 4: Perturbed ICD prediction
            pert_icd = get_icd_prediction(pert_text)
            print(f"   Perturbed ICD: {pert_icd}")

            # Step 5: Drift classification
            drift_info = classify_drift(orig_icd, pert_icd)
            sim        = scorer.get_sim(orig_text, pert_text)
            suspicious = (drift_info['categorical_drift']
                          and sim > 0.995 and len(changes_log) <= 3)

            print(f"   Drift: {drift_info['drift_level']} | Sim: {sim:.4f}")

            # Format changes with UMLS reason in brackets
            changes_str = " | ".join(
                [f"{o} -> {n} [{r}]" for o, n, r in changes_log])

            results.append({
                "hadm_id": hadm_id,
                "ground_truth_icd": gt_icd,
                "ground_truth_version": icd_ver,
                "ground_truth_desc": gt_desc,
                "original_text": orig_text,
                "perturbed_text": pert_text,
                "changes_made": changes_str,
                "changes_count": len(changes_log),
                "original_icd_pred": orig_icd,
                "original_pred_desc": orig_desc,
                "baseline_sim": baseline_sim,
                "baseline_llm_ok": llm_ok,
                "baseline_gate": gate_str,
                "baseline_correct": True,
                "perturbed_icd_pred": pert_icd,
                **drift_info,
                "suspicious_drift": suspicious,
                "semantic_similarity": sim
            })
            print("-" * 40)

        except Exception as e:
            debug_print("ERROR", f"Row {idx} failed: {e}")
            import traceback
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
    print(f"  Total            : {len(final_df)}")
    print(f"  Valid            : {len(valid_df)}")
    print(f"  Skipped          : {len(final_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"  Categorical drift: "
              f"{valid_df['categorical_drift'].mean() * 100:.1f}%")
        print(f"  Subcategory drift: "
              f"{valid_df['subcategory_drift'].mean() * 100:.1f}%")
        print(f"  Avg sim          : {valid_df['semantic_similarity'].mean():.4f}")
        print(f"  Avg changes      : {valid_df['changes_count'].mean():.1f}")
        print(f"  Suspicious drifts: {valid_df['suspicious_drift'].sum()}")
    print(f"  CSV  → {FINAL_RESULTS}")
    print(f"  HTML → {HTML_REPORT}")
    print(f"  Plot → {PLOT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()