import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from src.config import BASELINE_SIM_THRESHOLD, DEVICE, ENTITY_WEIGHTS, ORACLE_MODEL
from src.llm_connector import query_rwth_server
from src.utils import debug_print

# ============================================================
# SECTION 4: SEMANTIC EVALUATOR
# ============================================================
def _normalized_edit_distance(s1, s2):
    w1, w2 = s1.lower().split(), s2.lower().split()
    m, n   = len(w1), len(w2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
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
            # FIX: handle both 3-tuple and 5-tuple changes_log
            original, replacement, label = entry[0], entry[1], entry[2]
            sim           = self._bert_cosine(original, replacement)
            wt            = w.get(label, 0.7)
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
    if pred_icd in ("UNKNOWN", "") or gt_icd in ("UNKNOWN", ""):
        return False, False
    pn = re.sub(r'[^A-Z0-9]', '', pred_icd.strip().upper())
    gn = re.sub(r'[^A-Z0-9]', '', gt_icd.strip().upper())
    exact = pn == gn
    prefix = len(pn) >= 3 and len(gn) >= 3 and pn[:3] == gn[:3]
    return exact, prefix

def is_baseline_correct(scorer, orig_desc, gt_desc, orig_icd, gt_icd):
    gt_n = re.sub(r'[^A-Z0-9]', '', str(gt_icd).upper())
    pred_n = re.sub(r'[^A-Z0-9]', '', str(orig_icd).upper())

    if gt_n == pred_n:
        return True, 1.0, None, (
            "sem_sim=1.0000v | exact=YES | prefix=YES | "
            "same_chapter=YES | llm=SKIPPED | passed=YES"
        )

    if len(gt_n) >= 3 and len(pred_n) >= 3 and gt_n[:3] == pred_n[:3]:
        sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
        same_chapter = (gt_n and pred_n and gt_n[0] == pred_n[0])
        return True, baseline_sim, None, (
            f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'} | "
            f"exact=NO | prefix=YES | "
            f"same_chapter={'YES' if same_chapter else 'NO'} | "
            f"llm=SKIPPED | passed=YES"
        )

    sim_ok, baseline_sim = scorer.descriptions_match(orig_desc, gt_desc)
    same_chapter = (gt_n and pred_n and gt_n[0] == pred_n[0])

    if same_chapter and baseline_sim >= BASELINE_SIM_THRESHOLD:
        return True, baseline_sim, None, (
            f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'} | "
            f"exact=NO | prefix=NO | same_chapter=YES | "
            f"llm=SKIPPED | passed=YES"
        )

    return False, baseline_sim, None, (
        f"sem_sim={baseline_sim:.4f}{'v' if sim_ok else 'x'} | "
        f"exact=NO | prefix=NO | "
        f"same_chapter={'YES' if same_chapter else 'NO'} | "
        f"llm=SKIPPED | passed=NO"
    )