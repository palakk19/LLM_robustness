import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PLOT_FILE,
    MIN_CHAPTER_N,
    MIN_ICD_N,
    MAX_ICD_PLOT_COUNT,
)
from src.analysis import classify_stability, proportion_ci
from src.utils import debug_print

def plot_drift_destination(valid_df, chapters_of_interest, plot_file, level_labels):
    """
    For each chapter with high drift:
    1. Show which chapters the model drifts to (top level)
    2. For each destination chapter, show which specific ICD codes
    """
    CHAPTER_MAP = {
        'A':'Infectious','B':'Infectious','C':'Neoplasms','D':'Blood',
        'E':'Endocrine','F':'Mental','G':'Nervous','H':'Eye/Ear',
        'I':'Circulatory','J':'Respiratory','K':'Digestive','L':'Skin',
        'M':'Musculoskeletal','N':'Genitourinary','O':'Pregnancy',
        'P':'Perinatal','Q':'Congenital','R':'Symptoms',
        'S':'Injury','T':'Poisoning','Z':'Health Status',
    }
    LEVEL_COLORS = {0:'#4ECDC4', 1:'#45B7D1', 2:'#FFD700', 3:'#FF6B6B'}

    for chapter in chapters_of_interest:

        chapter_df = valid_df[valid_df['orig_chapter'] == chapter].copy()
        drift_df   = chapter_df[chapter_df['categorical_drift'] == True].copy()

        if len(drift_df) == 0:
            print(f"  [{chapter}] No categorical drifts, skipping.")
            continue

        # Map destination ICD to chapter name
        drift_df['dest_chapter'] = drift_df['perturbed_icd_pred'].apply(
            lambda x: CHAPTER_MAP.get(
                re.sub(r'[^A-Z]', '', str(x).upper())[:1], 'Unknown'
            )
        )
        # Clean destination ICD to 3-char prefix e.g. G89, J18, K57
        drift_df['dest_icd_3'] = drift_df['perturbed_icd_pred'].apply(
            lambda x: re.sub(r'[^A-Z0-9]', '', str(x).upper())[:3]
        )
        drift_df = drift_df[drift_df['dest_chapter'] != 'Unknown']

        levels   = sorted(drift_df['attack_level'].unique())
        n_levels = len(levels)

        # ── One figure per attack level ───────────────────────────────────────
        for level in levels:
            lvl_df   = drift_df[drift_df['attack_level'] == level]
            lvl_name = level_labels.get(level, f"L{level}")
            lvl_col  = LEVEL_COLORS.get(level, '#999')

            total_at_level = chapter_df[
                chapter_df['attack_level'] == level
            ]['hadm_id'].nunique()
            total_drifts = len(lvl_df)
            drift_rate   = 100 * total_drifts / total_at_level if total_at_level > 0 else 0

            if total_drifts == 0:
                continue

            dest_chapter_counts = lvl_df['dest_chapter'].value_counts()
            n_dest_chapters     = len(dest_chapter_counts)

            # ── Layout: 1 row per destination chapter ────────────────────────
            # Left col = chapter bar, Right col = ICD codes within that chapter
            fig, axes = plt.subplots(
                n_dest_chapters, 2,
                figsize=(16, max(4, 3 * n_dest_chapters)),
                gridspec_kw={'width_ratios': [1, 2]}
            )

            # Ensure axes is always 2D
            if n_dest_chapters == 1:
                axes = [axes]

            fig.suptitle(
                f"{chapter} Chapter — [{lvl_name}] Drift Destination Breakdown\n"
                f"{total_drifts} categorical drifts out of "
                f"{total_at_level} patients = {drift_rate:.1f}% drift rate",
                fontsize=12, y=1.01, color=lvl_col, fontweight='bold'
            )

            for row_idx, (dest_ch, ch_count) in enumerate(dest_chapter_counts.items()):
                ax_ch  = axes[row_idx][0]   # Left: chapter level bar
                ax_icd = axes[row_idx][1]   # Right: ICD codes within chapter

                ch_pct = 100 * ch_count / total_drifts

                # ── Left: Single bar showing this destination chapter ─────────
                bar = ax_ch.barh(
                    [dest_ch],
                    [ch_count],
                    color=lvl_col,
                    edgecolor='black',
                    alpha=0.85,
                    height=0.5
                )
                ax_ch.text(
                    ch_count + 0.05,
                    0,
                    f"n={ch_count}  ({ch_pct:.1f}% of drifts)",
                    va='center', ha='left',
                    fontsize=10, fontweight='bold'
                )
                ax_ch.set_xlim(0, dest_chapter_counts.max() * 1.5)
                ax_ch.set_xlabel("Number of drifts")
                ax_ch.set_title(
                    f"Drifted to: {dest_ch}",
                    fontsize=10, fontweight='bold'
                )
                ax_ch.tick_params(axis='y', labelsize=10)

                # ── Right: ICD codes within this destination chapter ──────────
                icd_subset  = lvl_df[lvl_df['dest_chapter'] == dest_ch]
                icd_counts  = icd_subset['dest_icd_3'].value_counts()

                # Color each ICD bar slightly differently
                icd_colors  = plt.cm.Set2(
                    np.linspace(0, 1, len(icd_counts))
                )

                icd_bars = ax_icd.barh(
                    range(len(icd_counts)),
                    icd_counts.values,
                    color=icd_colors,
                    edgecolor='black',
                    alpha=0.85
                )
                ax_icd.set_yticks(range(len(icd_counts)))
                ax_icd.set_yticklabels(icd_counts.index, fontsize=9)

                # Count + % of chapter drifts on each bar
                for bar, (icd, cnt) in zip(icd_bars, icd_counts.items()):
                    w       = bar.get_width()
                    icd_pct = 100 * cnt / ch_count
                    ax_icd.text(
                        w + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        f"n={int(w)}  ({icd_pct:.1f}% of {dest_ch} drifts)",
                        va='center', ha='left',
                        fontsize=8
                    )

                ax_icd.set_xlim(0, icd_counts.max() * 2.0)
                ax_icd.set_xlabel("Number of patients")
                ax_icd.set_title(
                    f"Specific ICD codes predicted within {dest_ch}",
                    fontsize=10
                )

            plt.tight_layout()
            safe_chapter = chapter.replace('/', '_').replace(' ', '_')
            dest_path    = plot_file.replace(
                '.png',
                f'_drift_dest_{safe_chapter}_L{level}.png'
            )
            fig.savefig(dest_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {dest_path}")


# ============================================================
# SECTION 8c: PLOTS
# ============================================================
def plot_results(df):
    debug_print("PLOT", "Generating plots...")

    valid_df = df[
        (df['baseline_correct'] == True) &
        (df['has_valid_perturbation'] == True)
    ].copy()

    if len(valid_df) == 0:
        print("No valid samples to plot.")
        return

    sim_col = (
        'substitution_sim'
        if 'substitution_sim' in valid_df.columns
        else 'semantic_similarity'
    )
    sim_label = (
        'Substitution Cosine Sim'
        if sim_col == 'substitution_sim'
        else 'BERT Cosine Sim'
    )

    DRIFT_COLORS = {
        'none': '#90EE90',
        'subcategory': '#FFD700',
        'categorical': '#FF6B6B',
        'unknown': '#CCCCCC'
    }
    LEVEL_LABELS = {
        0: 'L0-Synonyms',
        1: 'L1-Broadening',
        2: 'L2-Differential',
        3: 'L3-Adversarial'
    }
    LEVEL_COLORS = {
        0: '#4ECDC4',
        1: '#45B7D1',
        2: '#FFD700',
        3: '#FF6B6B'
    }

    levels = (
        sorted(valid_df['attack_level'].unique())
        if 'attack_level' in valid_df.columns
        else [None]
    )

    print("\nPER-LEVEL COUNTS USED FOR PLOTTING:")
    for lvl in levels:
        temp = valid_df[valid_df['attack_level'] == lvl] if lvl is not None else valid_df
        print(
            f"{('L'+str(lvl)) if lvl is not None else 'ALL'}: "
            f"n={len(temp)}, "
            f"cat={temp['categorical_drift'].mean()*100:.1f}%, "
            f"sub={temp['subcategory_drift'].mean()*100:.1f}%"
        )
    print()

    def _safe_yerr_from_ci(rate, n):
        lo, hi = proportion_ci(rate, n)
        lower = max(0.0, (rate - lo) * 100)
        upper = max(0.0, (hi - rate) * 100)
        return lower, upper

    # ── Per-level 5-plot figures ──────────────────────────────────────────────
    for level in levels:
        lvl_df = valid_df[valid_df['attack_level'] == level] if level is not None else valid_df
        lvl_tag = f"L{level}" if level is not None else "all"
        lvl_name = LEVEL_LABELS.get(level, lvl_tag)

        if len(lvl_df) == 0:
            continue

        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        fig.suptitle(f"Robustness — {lvl_name}  (n={len(lvl_df)})", fontsize=13, y=1.02)

        # [0] Similarity histogram
        ax = axes[0]
        sv = lvl_df[sim_col].dropna()
        ax.hist(
            sv,
            bins=15,
            color=LEVEL_COLORS.get(level, 'teal'),
            edgecolor='black',
            alpha=0.85
        )
        ax.set_title(f"{sim_label}\nmean={sv.mean():.4f}")
        ax.set_xlabel(sim_label)
        ax.set_ylabel("Count")

        # [1] Drift counts
        ax = axes[1]
        dc = (
            lvl_df['drift_level']
            .value_counts()
            .reindex(['none', 'subcategory', 'categorical', 'unknown'])
            .dropna()
        )
        dc.plot(
            kind='bar',
            color=[DRIFT_COLORS.get(k, 'grey') for k in dc.index],
            ax=ax,
            edgecolor='black',
            alpha=0.85
        )
        cp = lvl_df['categorical_drift'].mean() * 100
        sp = lvl_df['subcategory_drift'].mean() * 100
        ax.set_title(f"Drift — {lvl_tag}\nCat:{cp:.1f}% Sub:{sp:.1f}%")

        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.2,
                    str(int(h)),
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        # [2] Changes vs similarity
        ax = axes[2]
        for dl in ['none', 'subcategory', 'categorical']:
            mask = lvl_df['drift_level'] == dl
            if mask.sum() == 0:
                continue

            ax.scatter(
                lvl_df[mask]['changes_count'],
                lvl_df[mask][sim_col],
                label=dl,
                color={'none': 'green', 'subcategory': 'orange', 'categorical': 'red'}[dl],
                alpha=0.7,
                s=60
            )

        ax.set_title(f"Changes vs Sim\n{lvl_tag}")
        ax.set_xlabel("Terms Changed")
        ax.set_ylabel(sim_label)
        ax.legend(fontsize=8)

        # [3] Entity types
        ax = axes[3]
        type_cols = [
            c for c in ['disease_changes', 'chemical_changes', 'anatomy_changes', 'procedure_changes']
            if c in lvl_df.columns and lvl_df[c].sum() > 0
        ]

        if type_cols:
            totals = lvl_df[type_cols].sum()
            totals.index = [c.replace('_changes', '').capitalize() for c in totals.index]

            totals.plot(
                kind='bar',
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(totals)],
                ax=ax,
                edgecolor='black',
                alpha=0.85
            )

            for bar in ax.patches:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.5,
                        str(int(height)),
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )

            ax.tick_params(axis='x', rotation=15)
        else:
            ax.text(
                0.5,
                0.5,
                'No entity data',
                ha='center',
                va='center',
                transform=ax.transAxes
            )

        ax.set_title(f"Entity Types\n{lvl_tag}")

        # [4] Similarity by drift level
        ax = axes[4]
        groups = [
            lvl_df[lvl_df['drift_level'] == d][sim_col].dropna().values
            for d in ['none', 'subcategory', 'categorical']
            if d in lvl_df['drift_level'].values
        ]
        labels = [
            d for d in ['none', 'subcategory', 'categorical']
            if d in lvl_df['drift_level'].values
        ]

        if groups:
            bp = ax.boxplot(groups, labels=labels, patch_artist=True)
            for patch, d in zip(bp['boxes'], labels):
                patch.set_facecolor(DRIFT_COLORS.get(d, 'grey'))
                patch.set_alpha(0.8)
        else:
            ax.text(
                0.5,
                0.5,
                'No drift groups',
                ha='center',
                va='center',
                transform=ax.transAxes
            )

        ax.set_title(f"Sim by Drift Level\n{lvl_tag}")

        plt.tight_layout()
        out_path = PLOT_FILE.replace('.png', f'_{lvl_tag}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Chapter drift with CI ─────────────────────────────────────────────────
    if 'orig_chapter' in valid_df.columns:
        for level in levels:
            lvl_df = valid_df[valid_df['attack_level'] == level] if level is not None else valid_df
            lvl_tag = f"L{level}" if level is not None else "all"

            if len(lvl_df) == 0:
                continue

            chap = (
                lvl_df.groupby('orig_chapter')['categorical_drift']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'drift_rate', 'count': 'n'})
                .sort_values('drift_rate', ascending=False)
            )
            chap = chap[chap['n'] >= MIN_CHAPTER_N]

            if len(chap) == 0:
                continue

            ci_lo, ci_hi = [], []
            for _, r in chap.iterrows():
                lower, upper = _safe_yerr_from_ci(r['drift_rate'], r['n'])
                ci_lo.append(lower)
                ci_hi.append(upper)

            fig2, ax2 = plt.subplots(figsize=(max(10, len(chap) * 0.9), 5))
            bars = ax2.bar(
                chap.index,
                chap['drift_rate'] * 100,
                color='#FF6B6B',
                edgecolor='black',
                alpha=0.8,
                yerr=[ci_lo, ci_hi],
                capsize=4,
                ecolor='#333'
            )

            for bar, (idx, r) in zip(bars, chap.iterrows()):
                pct = r['drift_rate'] * 100
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ci_hi) * 0.05 + 0.8,
                    f"{pct:.1f}%\n(n={int(r['n'])})",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )

            ax2.set_title(
                f"Categorical Drift by ICD-10 Chapter — {lvl_tag}\n"
                f"(n>={MIN_CHAPTER_N}, error bars = 95% Wilson CI)"
            )
            ax2.set_xlabel("ICD-10 Chapter")
            ax2.set_ylabel("Categorical Drift Rate (%)")
            ax2.tick_params(axis='x', rotation=30)
            plt.tight_layout()

            path = PLOT_FILE.replace('.png', f'_chapter_{lvl_tag}.png')
            fig2.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved: {path}")

    # ── ICD code drift with CI ────────────────────────────────────────────────
    if 'ground_truth_icd' in valid_df.columns:
        for level in levels:
            lvl_df = valid_df[valid_df['attack_level'] == level] if level is not None else valid_df
            lvl_tag = f"L{level}" if level is not None else "all"

            if len(lvl_df) == 0:
                continue

            icd = (
                lvl_df.groupby('ground_truth_icd')['categorical_drift']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'drift_rate', 'count': 'n'})
                .sort_values('drift_rate', ascending=False)
            )
            icd = icd[icd['n'] >= MIN_ICD_N].head(MAX_ICD_PLOT_COUNT)

            if len(icd) == 0:
                continue

            ci_lo, ci_hi = [], []
            for _, r in icd.iterrows():
                lower, upper = _safe_yerr_from_ci(r['drift_rate'], r['n'])
                ci_lo.append(lower)
                ci_hi.append(upper)

            fig3, ax3 = plt.subplots(figsize=(max(10, len(icd) * 0.8), 5))
            clrs = [
                '#FF6B6B' if r > 0.5 else '#FFD700' if r > 0.2 else '#90EE90'
                for r in icd['drift_rate']
            ]

            bars = ax3.barh(
                icd.index,
                icd['drift_rate'] * 100,
                color=clrs,
                edgecolor='black',
                alpha=0.85,
                xerr=[ci_lo, ci_hi],
                capsize=3,
                ecolor='#333'
            )

            for bar, (idx, r) in zip(bars, icd.iterrows()):
                pct = r['drift_rate'] * 100
                ax3.text(
                    bar.get_width() + max(ci_hi) * 0.03 + 0.8,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}% (n={int(r['n'])})",
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )

            ax3.set_title(
                f"Drift Rate by ICD-10 Code — {lvl_tag}\n"
                f"(top {MAX_ICD_PLOT_COUNT}, n>={MIN_ICD_N}, 95% CI)"
            )
            ax3.set_xlabel("Categorical Drift Rate (%)")
            plt.tight_layout()

            path = PLOT_FILE.replace('.png', f'_icd_{lvl_tag}.png')
            fig3.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"  Saved: {path}")

    # ── Cross-level comparison ────────────────────────────────────────────────
    if 'attack_level' in valid_df.columns and valid_df['attack_level'].nunique() > 1:
        ls = (
            valid_df.groupby('attack_level')
            .agg(
                cat_drift=('categorical_drift', 'mean'),
                sub_drift=('subcategory_drift', 'mean'),
                avg_sim=(sim_col, 'mean'),
                n=('categorical_drift', 'count')
            )
            .reset_index()
        )

        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes4[0]
        ax.bar(
            ls['attack_level'],
            ls['cat_drift'] * 100,
            color=[LEVEL_COLORS.get(l, 'grey') for l in ls['attack_level']],
            edgecolor='black',
            alpha=0.85
        )
        for _, r in ls.iterrows():
            ax.text(
                r['attack_level'],
                r['cat_drift'] * 100 + 0.5,
                f"{r['cat_drift']*100:.1f}%",
                ha='center',
                fontsize=9
            )
        ax.set_title("Categorical Drift by Attack Level")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel("Drift Rate (%)")
        ax.set_xticks(ls['attack_level'])
        ax.set_xticklabels(
            [
                f"L{l}\n{LEVEL_LABELS.get(l, '').split('-')[-1]}\n(n={int(rn)})"
                for l, rn in zip(ls['attack_level'], ls['n'])
            ],
            fontsize=8
        )

        ax = axes4[1]
        ax.plot(
            ls['attack_level'],
            ls['avg_sim'],
            marker='o',
            color='teal',
            linewidth=2,
            markersize=8
        )
        for _, r in ls.iterrows():
            ax.annotate(
                f"{r['avg_sim']:.3f}",
                (r['attack_level'], r['avg_sim']),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8
            )
        ax.set_title(f"Avg {sim_label} by Attack Level")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel(sim_label)

        ax = axes4[2]
        w = 0.35
        x = np.array(ls['attack_level'])
        ax.bar(
            x - w / 2,
            ls['cat_drift'] * 100,
            w,
            label='Categorical',
            color='#FF6B6B',
            edgecolor='black',
            alpha=0.85
        )
        ax.bar(
            x + w / 2,
            ls['sub_drift'] * 100,
            w,
            label='Subcategory',
            color='#FFD700',
            edgecolor='black',
            alpha=0.85
        )
        ax.set_title("Drift Breakdown by Attack Level")
        ax.set_xlabel("Attack Level")
        ax.set_ylabel("Drift Rate (%)")
        ax.set_xticks(x)
        ax.legend()

        plt.suptitle("Attack Level Comparison", fontsize=13, y=1.02)
        plt.tight_layout()

        path = PLOT_FILE.replace('.png', '_attack_levels.png')
        fig4.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Saved: {path}")

    # ── Stability & Consistency Classification ────────────────────────────────
    stab_df = classify_stability(df)
    stab_counts = stab_df['stability_class'].value_counts()

    STAB_COLORS = {
        'Stable Correct': '#27AE60',
        'Partially Stable': '#F39C12',
        'Unstable': '#8E44AD',
        'Baseline Failed': '#E74C3C',
    }
    STAB_ORDER = ['Stable Correct', 'Partially Stable', 'Unstable', 'Baseline Failed']
    ordered_counts = stab_counts.reindex(STAB_ORDER).dropna()

    fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(20, 5))

    colors = [STAB_COLORS.get(k, 'grey') for k in ordered_counts.index]
    bars = ax5a.bar(
        ordered_counts.index,
        ordered_counts.values,
        color=colors,
        edgecolor='black',
        alpha=0.85
    )
    for bar, (cls, cnt) in zip(bars, ordered_counts.items()):
        pct = 100 * cnt / len(stab_df)
        ax5a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{cnt}\n({pct:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=10
        )
    ax5a.set_title("Patient Stability Classification")
    ax5a.set_ylabel("Number of Patients")
    ax5a.tick_params(axis='x', rotation=15)

    ax5b.pie(
        ordered_counts.values,
        labels=[f"{k}\n(n={v})" for k, v in ordered_counts.items()],
        colors=[STAB_COLORS.get(k, 'grey') for k in ordered_counts.index],
        autopct='%1.1f%%',
        startangle=90
    )
    ax5b.set_title(f"Stability Distribution\n(n={len(stab_df)} patients)")

    multi_drift_df = stab_df[
        stab_df['drift_type'].isin(['Consistent Systematic Drift', 'Chaotic Random Drift'])
    ]
    if len(multi_drift_df) > 0:
        drift_counts = multi_drift_df['drift_type'].value_counts()
        ax5c.pie(
            drift_counts.values,
            labels=[f"{k}\n(n={v})" for k, v in drift_counts.items()],
            colors=['#E67E22' if 'Systematic' in k else '#9B59B6' for k in drift_counts.index],
            autopct='%1.1f%%',
            startangle=90
        )
        ax5c.set_title(
            f"Nature of the Vulnerability\n"
            f"(among {len(multi_drift_df)} multi-drifting patients)"
        )
    else:
        ax5c.text(0.5, 0.5, "Not enough drift data yet", ha='center', va='center')
        ax5c.axis('off')

    plt.suptitle("Model Robustness and Systematic Bias Analysis", fontsize=13, y=1.05)
    plt.tight_layout()

    path5 = PLOT_FILE.replace('.png', '_stability.png')
    fig5.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"  Saved: {path5}")

    # ── Edit Similarity vs Substitution Similarity scatter ────────────────────
    if 'edit_similarity' in valid_df.columns and 'substitution_sim' in valid_df.columns:
        fig6, axes6 = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5))
        if len(levels) == 1:
            axes6 = [axes6]

        for ax6, level in zip(axes6, levels):
            lvl_df = valid_df[valid_df['attack_level'] == level] if level is not None else valid_df
            lvl_tag = f"L{level}" if level is not None else "all"

            if len(lvl_df) == 0:
                continue

            for dl, color, marker in [
                ('none', '#27AE60', 'o'),
                ('subcategory', '#F39C12', 's'),
                ('categorical', '#E74C3C', '^'),
            ]:
                mask = lvl_df['drift_level'] == dl
                if mask.sum() == 0:
                    continue

                ax6.scatter(
                    lvl_df[mask]['edit_similarity'],
                    lvl_df[mask]['substitution_sim'],
                    label=dl,
                    color=color,
                    marker=marker,
                    alpha=0.7,
                    s=60,
                    edgecolors='black',
                    linewidths=0.5
                )

            ax6.set_xlim(0.85, 1.01)
            ax6.set_ylim(0.50, 1.05)
            ax6.plot([0.5, 1.05], [0.5, 1.05], 'k--', alpha=0.3, linewidth=1, label='edit=subst')
            ax6.set_title(f"Edit Sim vs Subst. Sim — {lvl_tag}\n(n={len(lvl_df)})")
            ax6.set_xlabel("Edit Similarity (token-level)")
            ax6.set_ylabel("Substitution Sim (Targeted swaps only)")
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)

        plt.suptitle(
            "Edit Similarity vs Substitution Similarity by Drift Outcome\n"
            "Points below diagonal: semantic change > surface change",
            fontsize=11,
            y=1.02
        )
        plt.tight_layout()

        path6 = PLOT_FILE.replace('.png', '_edit_vs_subst.png')
        fig6.savefig(path6, dpi=150, bbox_inches='tight')
        plt.close(fig6)
        print(f"  Saved: {path6}")

    # ── Chapter vs Attack Level heatmap ───────────────────────────────────────
    if (
        'orig_chapter' in valid_df.columns and
        'attack_level' in valid_df.columns and
        valid_df['attack_level'].nunique() > 1
    ):
        pivot = (
            valid_df.groupby(['orig_chapter', 'attack_level'])['categorical_drift']
            .mean()
            .unstack(fill_value=np.nan) * 100
        )

        chapter_n = valid_df.groupby('orig_chapter')['hadm_id'].nunique()
        keep_ch = chapter_n[chapter_n >= MIN_CHAPTER_N].index
        pivot = pivot.loc[pivot.index.isin(keep_ch)]

        if len(pivot) > 0:
            col_labels = [LEVEL_LABELS.get(c, f"L{c}") for c in pivot.columns]

            fig7, ax7 = plt.subplots(
                figsize=(max(8, len(pivot.columns) * 2), max(5, len(pivot) * 0.5))
            )

            sns.heatmap(
                pivot,
                annot=True,
                fmt=".0f",
                cmap="YlOrRd",
                vmin=0,
                vmax=100,
                linewidths=0.5,
                linecolor='white',
                ax=ax7,
                cbar_kws={'label': 'Categorical Drift Rate (%)'},
                xticklabels=col_labels,
            )
            ax7.set_title(
                "Categorical Drift Rate (%) — ICD Chapter x Attack Level\n"
                f"(chapters with n>={MIN_CHAPTER_N} valid cases, colour scale 0-100%)"
            )
            ax7.set_xlabel("Attack Level")
            ax7.set_ylabel("ICD-10 Chapter")
            ax7.tick_params(axis='x', rotation=15)
            ax7.tick_params(axis='y', rotation=0)
            plt.tight_layout()

            path7 = PLOT_FILE.replace('.png', '_chapter_heatmap.png')
            fig7.savefig(path7, dpi=150, bbox_inches='tight')
            plt.close(fig7)
            print(f"  Saved: {path7}")

    # ── Drift destination — auto select chapters with n >= MIN_CHAPTER_N ─────
    chapter_n = valid_df.groupby('orig_chapter')['hadm_id'].nunique()
    auto_chapters = chapter_n[chapter_n >= MIN_CHAPTER_N].index.tolist()

    print(f"  Chapters qualifying for drift destination (n>={MIN_CHAPTER_N}): {auto_chapters}")

    if auto_chapters:
        plot_drift_destination(
            valid_df,
            chapters_of_interest=auto_chapters,
            plot_file=PLOT_FILE,
            level_labels=LEVEL_LABELS,
        )
    else:
        print("  No chapters met the minimum n threshold for drift destination plots.")

    plt.close('all')
    print("All plots saved.")

def plot_entity_summary(df, plot_file):
    """
    Shows entity distribution counted ONCE per unique note,
    not repeated across attack levels.
    """
    # Take only L0 rows (one row per note) OR deduplicate by hadm_id
    # This ensures each note is counted exactly once
    unique_notes = df.drop_duplicates(subset=['hadm_id']).copy()

    entity_cols = {
        'disease_changes' : 'Disease',
        'chemical_changes': 'Chemical', 
        'anatomy_changes' : 'Anatomy',
        'procedure_changes': 'Procedure'
    }

    # Only keep columns that exist
    available = {k: v for k, v in entity_cols.items() 
                 if k in unique_notes.columns}

    if not available:
        print("No entity columns found.")
        return

    totals = pd.Series({
        label: unique_notes[col].sum() 
        for col, label in available.items()
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Entity Distribution — counted once per unique note\n"
        f"(n={len(unique_notes)} unique notes, "
        f"total entity substitutions = {totals.sum():.0f})",
        fontsize=12
    )

    # ── Left: Total counts bar chart ─────────────────────────────────────────
    ax = axes[0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars   = ax.bar(
        totals.index, totals.values,
        color=colors[:len(totals)],
        edgecolor='black', alpha=0.85
    )
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                str(int(h)),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h / 2,
                f"{100*h/totals.sum():.1f}%",
                ha='center', va='center',
                fontsize=9, color='white', fontweight='bold'
            )
    ax.set_title("Total Entity Substitutions by Type\n(unique notes only)")
    ax.set_ylabel("Total Count")
    ax.set_xlabel("Entity Type")

    # ── Right: Per-level comparison to show how acceptance changes ───────────
    ax2 = axes[1]
    level_data = []
    for level in sorted(df['attack_level'].unique()):
        lvl_df = df[df['attack_level'] == level]
        row = {'Level': f"L{level}"}
        for col, label in available.items():
            row[label] = lvl_df[col].sum()
        level_data.append(row)

    level_df = pd.DataFrame(level_data).set_index('Level')
    x     = np.arange(len(level_df))
    width = 0.2
    for i, (col, color) in enumerate(zip(level_df.columns, colors)):
        bars2 = ax2.bar(
            x + i * width,
            level_df[col],
            width,
            label=col,
            color=color,
            edgecolor='black',
            alpha=0.85
        )
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.3,
                    str(int(h)),
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold'
                )

    ax2.set_xticks(x + width * (len(level_df.columns) - 1) / 2)
    ax2.set_xticklabels(level_df.index)
    ax2.set_title(
        "Entity Substitutions per Attack Level\n"
        "(same notes, different acceptance rates explain the difference)"
    )
    ax2.set_ylabel("Total Count")
    ax2.set_xlabel("Attack Level")
    ax2.legend(title="Entity Type")

    plt.tight_layout()
    path = plot_file.replace('.png', '_entity_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")