import random
import re
import spacy

from src.config import ATTACK_PERTURBATION_RATES, ORACLE_MODEL
from src.prompts import ATTACK_PROMPTS
from src.llm_connector import query_rwth_server
from src.utils import debug_print

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
        if w in t:
            return "ANATOMY"
    for w in procedure_kw:
        if w in t:
            return "PROCEDURE"
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
        # 1. Load Primary Model (BC5CDR for accurate Disease/Chemical)
        try:
            self.nlp_bc = spacy.load("en_ner_bc5cdr_md")
            debug_print("PERTURBER", "Loaded Primary: en_ner_bc5cdr_md")
        except OSError:
            raise RuntimeError(
                "No BC5CDR model found. Run: "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
            )

        # 2. Load Secondary Model (Core Sci for Anatomy/Procedure fallback)
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
        seen_spans = set()  # Track character indexes to prevent model overlaps
        seen_texts = set()

        # ─── Pass 1: High-Precision Diseases and Chemicals (BC5CDR) ───────────
        doc_bc = self.nlp_bc(text)
        for ent in doc_bc.ents:
            # BC5CDR uses explicit labels, so we pass ACCEPTED_ENTITY_LABELS
            valid, label_str = entity_is_medical(
                ent, {"DISEASE", "CHEMICAL"}, GENERIC_MODEL_STOP_ENTITIES)

            if not valid or self._is_negated(ent[0]):
                continue

            key = ent.text.strip().lower()
            if key in seen_texts:
                continue

            seen_texts.add(key)
            # Record the exact character span occupied by this entity
            for i in range(ent.start_char, ent.end_char):
                seen_spans.add(i)

            sent = (ent.sent.text if ent.sent
                    else text[max(0, ent.start_char-60):ent.end_char+60])
            entities.append({
                'text': ent.text, 'label': label_str,
                'start_char': ent.start_char, 'end_char': ent.end_char,
                'context': sent,
            })

        # ─── Pass 2: Fallback for Anatomy and Procedures (Core Sci) ───────────
        if self.nlp_sci:
            doc_sci = self.nlp_sci(text)
            for ent in doc_sci.ents:
                # Check if this entity overlaps with anything BC5CDR already claimed
                overlap = any(i in seen_spans for i in range(ent.start_char, ent.end_char))
                if overlap:
                    continue

                # Pass None for labels to trigger your _assign_heuristic_label logic
                valid, label_str = entity_is_medical(
                    ent, None, GENERIC_MODEL_STOP_ENTITIES)

                # We only want Anatomy and Procedures from the fallback
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
                print(f"       LAYER2 BLOCKED ({reason}): '{original}'->'{candidate}'")
                continue

            # # FIX: scorer is now actually used (was passed as None before)
            # if scorer is not None and sim_threshold > 0:
            #     pair_sim = scorer._bert_cosine(original, candidate)
            #     if pair_sim < sim_threshold:
            #         print(f"       LAYER2.5 BLOCKED (sim={pair_sim:.3f}): "
            #               f"'{original}'->'{candidate}'")
            #         continue

            if apply_safety_gate:
                llm_ok, _ = validate_replacement_with_llm(
                    original, candidate, ent['context'])
                if not llm_ok:
                    print(f"       LAYER3 BLOCKED: '{original}'->'{candidate}'")
                    continue

            accepted.append({
                'original'   : original,
                'replacement': candidate,
                'label'      : ent['label'],
                'start_char' : ent['start_char'],
                'end_char'   : ent['end_char'],
            })
            print(f"       ACCEPTED [{ent['label']:20s}]: '{original}'->'{candidate}'")

        if not accepted:
            return text, []

        # FIX: replace only the NER-identified span (right-to-left by offset)
        # This means each term is replaced exactly ONCE at the position
        # scispaCy found it, not all occurrences throughout the note.
        perturbed = text
        for item in sorted(accepted, key=lambda x: x['start_char'], reverse=True):
            perturbed = (perturbed[:item['start_char']]
                         + item['replacement']
                         + perturbed[item['end_char']:])


        # Track final positions for accurate changes_count and metrics
        offset = 0
        pos_map = {}
        for item in sorted(accepted, key=lambda x: x['start_char']):
            orig_len  = item['end_char'] - item['start_char']
            repl_len  = len(item['replacement'])
            final_s   = item['start_char'] + offset
            pos_map[item['start_char']] = (final_s, final_s + repl_len)
            offset   += repl_len - orig_len

        # changes_log: 3-tuple (original, replacement, label) for compatibility
        # positions stored separately in pos_map but not needed downstream
        changes_log = [(i['original'], i['replacement'], i['label'])
                       for i in accepted]
        return perturbed, changes_log

    def perturb(self, text, perturbation_rate=None, attack_level=0, scorer=None):
        if perturbation_rate is None:
            perturbation_rate = ATTACK_PERTURBATION_RATES.get(attack_level, 0.30)

        all_entities = self.extract_medical_entities(text)
        if not all_entities:
            return text, []
        
        all_entities = sorted(
            all_entities,
            key=lambda x: (x['start_char'], x['end_char'], x['text'].lower())
        )

        # FIX: cap at MAX_CHANGES to prevent overly aggressive perturbation
        n_target     = max(1, int(len(all_entities) * perturbation_rate))
        # n_target     = min(n_target, MAX_CHANGES)
        target_ents  = random.sample(all_entities, min(n_target, len(all_entities)))
        synonym_map  = find_synonyms_batch(target_ents, attack_level=attack_level)

        # FIX: pass scorer through so Layer 2.5 actually runs
        return self.validate_and_apply(text, target_ents, synonym_map,
                                       attack_level=attack_level,
                                       scorer=None)