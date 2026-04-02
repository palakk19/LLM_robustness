import re

from src.config import HTML_REPORT, TARGET_MODEL, ORACLE_MODEL
from src.analysis import classify_stability

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

    valid_df = df[
    (df['baseline_correct'] == True) &
    (df['has_valid_perturbation'] == True)
]
    cat_rate    = valid_df['categorical_drift'].mean()*100 if len(valid_df) else 0
    sub_rate    = valid_df['subcategory_drift'].mean()*100 if len(valid_df) else 0
    avg_changes = f"{valid_df['changes_count'].mean():.1f}" if len(valid_df) else 'N/A'
    sim_col     = ('substitution_sim' if 'substitution_sim' in valid_df.columns
                   else 'semantic_similarity')
    sim_lbl     = ('Substitution Cosine Sim' if sim_col == 'substitution_sim'
                   else 'BERT Cosine Sim')
    avg_sim     = f"{valid_df[sim_col].mean():.4f}" if len(valid_df) else 'N/A'

    # --- NEW: Get stability data for the entire dataset ---
    stab_df = classify_stability(df)
    # Convert to dictionary for fast lookup by hadm_id
    stab_dict = stab_df.set_index('hadm_id').to_dict('index')

    # Group by patient for side-by-side display
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

        # --- NEW: Retrieve stability insight for this patient ---
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

        # --- NEW: Inject Visual Banners based on drift profile ---
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
