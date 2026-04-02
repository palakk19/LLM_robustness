import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import random
import traceback
import pandas as pd
from tqdm import tqdm

from src.config import (
    TARGET_MODEL,
    ORACLE_MODEL,
    RANDOM_SEED,
    CHECKPOINT_FILE,
    CHECKPOINT_EVERY,
    FINAL_RESULTS,
    PLOT_FILE,
)
from src.utils import debug_print
from src.llm_connector import query_rwth_server
from src.data_loader import load_data
from src.pertubation import StructuredMedicalPerturber
from src.evaluation import SemanticEvaluator, is_baseline_correct
from src.prediction import get_icd_prediction, get_diagnosis_description, get_icd_chapter
from src.analysis import classify_drift, classify_stability, proportion_ci
from src.plotting import plot_results, plot_entity_summary
from src.reporting import generate_html_report
from src.checkpointing import load_checkpoint, save_checkpoint
from src.sanity import sanity_check

def main():
    debug_print(
        "MAIN",
        f"Target model: {TARGET_MODEL} | Helper model: {ORACLE_MODEL} | temp=0.0 | 4 attack levels"
    )
    random.seed(RANDOM_SEED)

    if not query_rwth_server("Say hello"):
        debug_print("ERROR", "Cannot connect to server. Check VPN.")
        return
    print("LLM OK")

    df = load_data()

    perturber = StructuredMedicalPerturber()
    scorer = SemanticEvaluator()
    sanity_check(perturber)

    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    baseline_cache = checkpoint.get("baseline_cache", {})

    debug_print("MAIN", f"Phase 1: baseline for {len(df)} notes...")

    new_baseline_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Baseline"):
        hadm_id = row['hadm_id']
        hadm_key = str(hadm_id)

        if hadm_key in baseline_cache:
            continue

        orig_text = row['processed_text']
        gt_desc = row['long_title']
        gt_icd = row['icd_code']

        try:
            orig_icd = get_icd_prediction(orig_text)
            orig_desc = get_diagnosis_description(orig_icd)
            ok, sim, llm_ok, gate = is_baseline_correct(
                scorer, orig_desc, gt_desc, orig_icd, gt_icd
            )

            baseline_cache[hadm_key] = {
                "orig_icd": orig_icd,
                "orig_desc": orig_desc,
                "baseline_ok": ok,
                "baseline_sim": sim,
                "llm_ok": llm_ok,
                "gate_str": gate,
                "orig_chapter": get_icd_chapter(orig_icd),
            }

            print(f"  [{'PASS' if ok else 'SKIP'}] {hadm_id}: pred={orig_icd} gt={gt_icd}")

        except Exception:
            traceback.print_exc()
            baseline_cache[hadm_key] = {
                "orig_icd": "UNKNOWN",
                "orig_desc": "Unknown",
                "baseline_ok": False,
                "baseline_sim": 0.0,
                "llm_ok": False,
                "gate_str": "ERROR",
                "orig_chapter": "Unknown",
            }

        new_baseline_count += 1
        if new_baseline_count % CHECKPOINT_EVERY == 0:
            checkpoint["baseline_cache"] = baseline_cache
            save_checkpoint(checkpoint)
            debug_print("CKPT", f"Saved baseline checkpoint ({len(baseline_cache)} done)")

    checkpoint["baseline_cache"] = baseline_cache
    save_checkpoint(checkpoint)

    n_v = sum(1 for v in baseline_cache.values() if v["baseline_ok"])
    debug_print("MAIN", f"Phase 1 done: {n_v} valid, {len(baseline_cache)-n_v} skipped")

    valid_hadm_ids = {
        str(hid) for hid, v in baseline_cache.items()
        if v["baseline_ok"]
    }

    valid_notes_df = df[df["hadm_id"].astype(str).isin(valid_hadm_ids)].copy()

    debug_print("MAIN", f"Only {len(valid_notes_df)} baseline-valid notes will be attacked in L0-L3")

    for ATTACK_LEVEL in [0, 1, 2, 3]:
        level_key = str(ATTACK_LEVEL)
        debug_print("MAIN", f"Phase 2: attack level {ATTACK_LEVEL}...")

        existing_results = checkpoint["level_results"].get(level_key, [])
        done_ids = {str(r["hadm_id"]) for r in existing_results}
        results = list(existing_results)

        print("\n" + "="*80 + f"\nATTACK LEVEL {ATTACK_LEVEL}\n" + "="*80)
        debug_print("CKPT", f"L{ATTACK_LEVEL} resume: {len(done_ids)} rows already completed")

        new_level_count = 0

        for _, row in tqdm(valid_notes_df.iterrows(), total=len(valid_notes_df), desc=f"L{ATTACK_LEVEL}"):
            hadm_id = row['hadm_id']
            hadm_key = str(hadm_id)

            if hadm_key in done_ids:
                continue

            try:
                orig_text = row['processed_text']
                gt_desc = row['long_title']
                gt_icd = row['icd_code']
                icd_ver = row['icd_version']
                chapter = row.get('chapter', 'X')

                cache = baseline_cache[hadm_key]
                orig_icd = cache["orig_icd"]
                orig_desc = cache["orig_desc"]
                baseline_ok = cache["baseline_ok"]
                baseline_sim = cache["baseline_sim"]
                llm_ok = cache["llm_ok"]
                gate_str = cache["gate_str"]
                orig_chapter = cache["orig_chapter"]

                print(f"\n[L{ATTACK_LEVEL}|{hadm_id}] GT:{gt_icd} Pred:{orig_icd} {'VALID' if baseline_ok else 'SKIP'}")

                pert_text, changes_log = perturber.perturb(
                    orig_text, attack_level=ATTACK_LEVEL, scorer=scorer
                )

                pert_icd = get_icd_prediction(pert_text)
                drift_info = classify_drift(orig_icd, pert_icd)
                bert_sim, edit_sim = scorer.get_sim(orig_text, pert_text)
                subst_sim = scorer.substitution_semantic_similarity(changes_log)

                suspicious = (
                    drift_info['categorical_drift']
                    and edit_sim > 0.995
                    and len(changes_log) <= 3
                )

                print(f"   Pert ICD: {pert_icd} | Drift: {drift_info['drift_level']} | Subst-sim: {subst_sim:.4f}")

                changes_str = " | ".join(
                    f"[{l}] {o} -> {n}" for o, n, l in changes_log
                )

                row_result = {
                    "hadm_id": hadm_id,
                    "ground_truth_icd": gt_icd,
                    "ground_truth_version": icd_ver,
                    "ground_truth_desc": gt_desc,
                    "original_text": orig_text,
                    "perturbed_text": pert_text,
                    "changes_made": changes_str,
                    "changes_count": len(changes_log),
                    "disease_changes": sum(1 for _, _, l in changes_log if l == "DISEASE"),
                    "chemical_changes": sum(1 for _, _, l in changes_log if l == "CHEMICAL"),
                    "anatomy_changes": sum(1 for _, _, l in changes_log if l == "ANATOMY"),
                    "procedure_changes": sum(1 for _, _, l in changes_log if l == "PROCEDURE"),
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
                    "has_valid_perturbation": len(changes_log) > 0,
                    "attack_level": ATTACK_LEVEL,
                }

                results.append(row_result)
                done_ids.add(hadm_key)
                new_level_count += 1
                print("-"*40)

                if new_level_count % CHECKPOINT_EVERY == 0:
                    checkpoint["level_results"][level_key] = results
                    save_checkpoint(checkpoint)
                    debug_print("CKPT", f"Saved L{ATTACK_LEVEL} checkpoint ({len(results)} rows)")

            except Exception as e:
                debug_print("ERROR", f"{hadm_id} L{ATTACK_LEVEL}: {e}")
                traceback.print_exc()
                continue

        checkpoint["level_results"][level_key] = results
        save_checkpoint(checkpoint)
        debug_print("MAIN", f"Level {ATTACK_LEVEL} done: {len(results)} rows")

    all_results = []
    for lvl in ["0", "1", "2", "3"]:
        all_results.extend(checkpoint["level_results"].get(lvl, []))

    if not all_results:
        print("No results.")
        return

    final_df = pd.DataFrame(all_results)
    csv_df = final_df.drop(columns=['original_text', 'perturbed_text'], errors='ignore')

    os.makedirs(os.path.dirname(FINAL_RESULTS), exist_ok=True)
    csv_df.to_csv(FINAL_RESULTS, index=False)
    print(f"CSV saved: {FINAL_RESULTS}")

    for lvl in sorted(csv_df['attack_level'].unique()):
        lvl_path = FINAL_RESULTS.replace('.csv', f'_L{lvl}.csv')
        csv_df[csv_df['attack_level'] == lvl].to_csv(lvl_path, index=False)
        n_v = csv_df[(csv_df['attack_level'] == lvl) & csv_df['baseline_correct']].shape[0]
        print(f"  L{lvl} CSV: {lvl_path} ({n_v} valid)")

    plot_results(final_df)
    plot_entity_summary(final_df, PLOT_FILE)
    generate_html_report(final_df)

    valid_df = final_df[
        (final_df['baseline_correct'] == True) &
        (final_df['has_valid_perturbation'] == True)
    ]

    unique_baselines = valid_df.drop_duplicates(subset=['hadm_id'])
    pass_rule = int(unique_baselines['baseline_gate'].str.contains(r'exact=YES|prefix=YES', na=False).sum())
    pass_same_chapter_sem = int(
        unique_baselines['baseline_gate'].str.contains(r'same_chapter=YES', na=False).sum()
    )

    print("\n" + "="*80 + "\nFINAL SUMMARY")
    print(f"  Unique patients                 : {df['hadm_id'].nunique()}")
    print(f"  Baseline-valid patients attacked: {len(valid_hadm_ids)}")
    print(f"  Total attacked rows (4 levels)  : {len(final_df)}")
    print(f"  Rows with real perturbation     : {len(valid_df)}")

    print(f"  Baseline Pass (Rule)   : {pass_rule} patients (Exact or Prefix match)")
    print(f"  Baseline Pass (Sem+Chapter) : {pass_same_chapter_sem} patients (Same chapter + semantic threshold)")

    if len(valid_df):
        print(f"\n  Per-level breakdown:")
        for lvl in sorted(valid_df['attack_level'].unique()):
            ld = valid_df[valid_df['attack_level'] == lvl]
            lo, hi = proportion_ci(ld['categorical_drift'].mean(), len(ld))
            print(
                f"    L{lvl}: n={len(ld):4d} | "
                f"cat_drift={ld['categorical_drift'].mean()*100:.1f}% "
                f"[95%CI {lo*100:.1f}-{hi*100:.1f}%] | "
                f"subst_sim={ld['substitution_sim'].mean():.4f} | "
                f"terms={ld['changes_count'].mean():.1f}"
            )

        stab = classify_stability(final_df)
        stab_counts = stab['stability_class'].value_counts()
        print(f"\n  Stability classification ({len(stab)} patients):")
        for cls in ['Stable Correct', 'Partially Stable', 'Unstable', 'Baseline Failed']:
            n = stab_counts.get(cls, 0)
            print(f"    {cls:20s}: {n:4d} ({100*n/len(stab):.1f}%)")

        print(f"\n  Overall categorical drift: {valid_df['categorical_drift'].mean()*100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    main()