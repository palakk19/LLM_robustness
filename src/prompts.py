# ============================================================
# PROMPTS
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

ATTACK_PROMPTS = {
    0: PROMPT_LEVEL_0,
    1: PROMPT_LEVEL_1,
    2: PROMPT_LEVEL_2,
    3: PROMPT_LEVEL_3,
}