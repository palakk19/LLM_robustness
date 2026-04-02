import re

from src.config import TARGET_MODEL, ORACLE_MODEL
from src.llm_connector import query_rwth_server

# ============================================================
# SECTION 6: ICD PREDICTION
# ============================================================
def get_icd_prediction(text):
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
        resp = query_rwth_server(prompt, model_to_use=TARGET_MODEL)
        if resp:
            resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.S).strip()
            m    = re.search(r'[A-Z]\d{2}\.\d+', resp.upper())
            if m:
                return m.group(0)
            m    = re.search(r'[A-Z]\d{2,4}', resp.upper())
            if m:
                return m.group(0)
    return "UNKNOWN"

def get_diagnosis_description(icd_code):
    if icd_code in ("UNKNOWN", ""):
        return "Unknown"
    resp = query_rwth_server(
        f"What condition does ICD-10 code '{icd_code}' represent? "
        f"Reply with ONLY the condition name in 2-5 words.",
        max_tokens=20,
        model_to_use=ORACLE_MODEL
    )
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