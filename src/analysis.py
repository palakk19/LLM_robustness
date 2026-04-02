import re
import pandas as pd

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

        # --- NEW LOGIC: Check how many UNIQUE wrong codes were predicted ---
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
            # If it drifted multiple times, did it always go to the same wrong place?
            if n_drift > 1 and unique_wrong_preds == 1:
                drift_type = 'Consistent Systematic Drift'
            elif n_drift > 1:
                drift_type = 'Chaotic Random Drift'
            else:
                drift_type = 'Single Drift'  # Only drifted once, so consistency doesn't apply

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