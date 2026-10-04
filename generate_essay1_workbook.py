"""
Generate Essay 1 Excel Workbook
================================
Runs the full Essay 1 pipeline (essay1.py + essay1_matched.py) and
writes all results, tables, figures, equations, and interpretations
into a comprehensive Excel workbook.
"""

import warnings
import logging
import sys
from pathlib import Path
from datetime import datetime
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import openpyxl
from openpyxl.styles import (
    Font, Fill, PatternFill, Alignment, Border, Side,
    GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import DataPoint

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# ── Add project root to path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.datastore import DataStore
from model.essay1 import (
    estimate_vix_regimes, select_n_regimes, ff5_by_regime,
    culture_war_by_regime, assemble_macro_controls, sentiment_by_regime,
    _FF5_ALL, benjamini_hochberg
)
from model.essay1_matched import ff5_matched_control_analysis

# ── Style constants ───────────────────────────────────────────────────────
NAVY    = "1F3864"
GOLD    = "C9A84C"
LTBLUE  = "D6E4F0"
LTGOLD  = "FFF3CD"
LTGRAY  = "F2F2F2"
WHITE   = "FFFFFF"
GREEN   = "1E8449"
RED     = "C0392B"
ORANGE  = "E67E22"

REGIME_COLORS = {
    'Low Volatility':  '27AE60',
    'Normal':          '2980B9',
    'High Volatility': 'E74C3C',
}
REGIME_HEX = {
    'Low Volatility':  '#27AE60',
    'Normal':          '#2980B9',
    'High Volatility': '#E74C3C',
}

def _hdr(ws, row, col, value, bold=True, bg=NAVY, fg=WHITE, size=11,
         halign='center', valign='center', wrap=False):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = Font(bold=bold, color=fg, size=size)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal=halign, vertical=valign, wrap_text=wrap)
    return cell

def _subhdr(ws, row, col, value, bg=LTBLUE):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = Font(bold=True, size=10)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    return cell

def _val(ws, row, col, value, fmt=None, bold=False, bg=None, color=None,
         halign='center', border=False):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = Font(bold=bold, size=10, color=color or '000000')
    if bg:
        cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal=halign, vertical='center')
    if fmt:
        cell.number_format = fmt
    if border:
        thin = Side(style='thin')
        cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    return cell

def _sig_color(p, bg_only=False):
    if p < 0.01:
        return ('FF4136' if not bg_only else 'FFD7D4')
    elif p < 0.05:
        return ('E67E22' if not bg_only else 'FDEBD0')
    elif p < 0.10:
        return ('F1C40F' if not bg_only else 'FEF9E7')
    return None

def _thin_border():
    thin = Side(style='thin', color='CCCCCC')
    return Border(left=thin, right=thin, top=thin, bottom=thin)

def set_col_width(ws, col, width):
    ws.column_dimensions[get_column_letter(col)].width = width

def merge_hdr(ws, r, c1, c2, value, bg=NAVY, fg=WHITE, bold=True, size=12):
    ws.merge_cells(start_row=r, start_column=c1, end_row=r, end_column=c2)
    cell = ws.cell(row=r, column=c1, value=value)
    cell.font = Font(bold=bold, color=fg, size=size)
    cell.fill = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    return cell

def fig_to_image(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def insert_image(ws, buf, anchor, width=None, height=None):
    img = XLImage(buf)
    if width:
        img.width = width
    if height:
        img.height = height
    ws.add_image(img, anchor)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def make_vix_regime_chart(regime_result):
    """VIX time series colored by regime."""
    df = regime_result.regime_assignments.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')

    fig, ax = plt.subplots(figsize=(14, 5))
    for label, color in REGIME_HEX.items():
        mask = df['REGIME_LABEL'] == label
        ax.fill_between(df['DATE'], 0, df['VIX'],
                        where=mask, alpha=0.35, color=color, label=label)
    ax.plot(df['DATE'], df['VIX'], color='#1F3864', linewidth=0.8, alpha=0.9)
    ax.set_title('VIX Volatility Index with Markov Regime Assignments (2000–2025)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('VIX', fontsize=10)
    ax.axhline(20, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_image(fig)


def make_regime_bar_chart(regime_result):
    """Bar chart: regime means + std."""
    labels = list(regime_result.regime_means.keys())
    means = [regime_result.regime_means[l] for l in labels]
    stds = [regime_result.regime_summary.set_index('REGIME').loc[l, 'STD_VIX']
            for l in labels]
    colors = [REGIME_HEX[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean VIX
    axes[0].bar(labels, means, color=colors, edgecolor='white', linewidth=1.2)
    axes[0].errorbar(labels, means, yerr=stds, fmt='none', color='black',
                     capsize=5, linewidth=1.5)
    axes[0].set_title('Mean VIX by Regime (±1 SD)', fontweight='bold')
    axes[0].set_ylabel('VIX')
    axes[0].set_ylim(0, max(means) * 1.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.5, f'{m:.1f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    # Days in regime
    n_days = [regime_result.regime_summary.set_index('REGIME').loc[l, 'N_DAYS']
              for l in labels]
    pct = [regime_result.regime_summary.set_index('REGIME').loc[l, 'PCT_DAYS']
           for l in labels]
    bars = axes[1].bar(labels, n_days, color=colors, edgecolor='white', linewidth=1.2)
    axes[1].set_title('Trading Days per Regime', fontweight='bold')
    axes[1].set_ylabel('Days')
    for i, (n, p) in enumerate(zip(n_days, pct)):
        axes[1].text(i, n + 30, f'{n:,}\n({p:.1f}%)', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

    for ax in axes:
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Markov Regime-Switching: Regime Characteristics', fontsize=12,
                 fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig_to_image(fig)


def make_transition_matrix_heatmap(regime_result):
    """Heatmap of regime transition probabilities."""
    labels = ['Low Vol', 'Normal', 'High Vol']
    tm = regime_result.transition_matrix

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(tm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Probability')

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{tm[i, j]:.4f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if tm[i, j] > 0.5 else 'black')

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(['→ ' + l for l in labels])
    ax.set_yticklabels(['From ' + l for l in labels])
    ax.set_title('Regime Transition Probability Matrix', fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_model_selection_chart(selection_df):
    """AIC/BIC vs K chart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ks = selection_df['K'].astype(int).tolist()

    for ax, metric in zip(axes, ['AIC', 'BIC']):
        vals = selection_df[metric].tolist()
        ax.plot(ks, vals, 'o-', color='#1F3864', linewidth=2, markersize=8)
        best_k = ks[vals.index(min(vals))]
        ax.axvline(best_k, color='red', linestyle='--', alpha=0.6,
                   label=f'Best K={best_k}')
        for k, v in zip(ks, vals):
            ax.text(k, v + (max(vals)-min(vals))*0.02, f'{v:,.0f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Number of Regimes (K)', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric} vs. K', fontweight='bold')
        ax.set_xticks(ks)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Markov Model Selection: K=2, 3, 4 Comparison', fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_ff5_coeff_chart(coeff_df):
    """Grouped bar: FF5 betas by regime."""
    factors = ['SMB_BETA', 'HML_BETA', 'RMW_BETA', 'CMA_BETA']
    labels_short = ['SMB', 'HML', 'RMW', 'CMA']
    regimes = coeff_df['REGIME'].tolist()
    colors = [REGIME_HEX[r] for r in regimes]

    x = np.arange(len(labels_short))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (regime, color) in enumerate(zip(regimes, colors)):
        vals = coeff_df[coeff_df['REGIME'] == regime][factors].values[0]
        bars = ax.bar(x + i * width, vals, width, label=regime, color=color,
                      edgecolor='white', linewidth=0.8, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_short, fontsize=12)
    ax.set_ylabel('Beta Coefficient', fontsize=11)
    ax.set_title('FF5 Factor Loadings (MKT_RF Spanning Regression) by Regime',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig_to_image(fig)


def make_factor_premia_chart(premia_df):
    """Annualized factor premia by regime."""
    ann_cols = [c for c in premia_df.columns if 'MEAN_ANN' in c or c in
                ['MKT_RF_MEAN', 'SMB_MEAN', 'HML_MEAN', 'RMW_MEAN', 'CMA_MEAN']]
    # Use annualized where available
    cols_to_plot = {}
    for f in ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']:
        ann = f'{f}_MEAN_ANN'
        plain = f'{f}_MEAN'
        if ann in premia_df.columns:
            cols_to_plot[f] = ann
        elif plain in premia_df.columns:
            cols_to_plot[f] = plain

    regimes = premia_df['REGIME'].tolist()
    factors = list(cols_to_plot.keys())
    x = np.arange(len(factors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (regime, color) in enumerate(zip(regimes, [REGIME_HEX[r] for r in regimes])):
        vals = [premia_df[premia_df['REGIME'] == regime][cols_to_plot[f]].values[0] * 100
                for f in factors]
        ax.bar(x + i * width, vals, width, label=regime, color=color,
               edgecolor='white', linewidth=0.8, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(factors, fontsize=11)
    ax.set_ylabel('Annualized Mean Return (%)', fontsize=11)
    ax.set_title('FF5 Factor Premia (Annualized %) by Volatility Regime',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig_to_image(fig)


def make_cw_alpha_chart(cw_summary):
    """Culture war stock alpha distribution by regime."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    regimes = ['Low Volatility', 'Normal', 'High Volatility']
    for ax, regime in zip(axes, regimes):
        sub = cw_summary[cw_summary['REGIME'] == regime].dropna(subset=['ALPHA'])
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(regime, fontweight='bold')
            continue
        color = REGIME_HEX[regime]
        ax.hist(sub['ALPHA'] * 252, bins=20, color=color, edgecolor='white',
                alpha=0.85, linewidth=0.6)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
        mean_alpha = sub['ALPHA'].mean() * 252
        ax.axvline(mean_alpha, color='gold', linestyle='-', linewidth=1.5,
                   label=f'Mean: {mean_alpha:.3f}')
        ax.set_title(f'{regime}\nn={len(sub)} stocks', fontweight='bold',
                     color=color)
        ax.set_xlabel('Annualized Alpha', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Culture War Stock Alphas (FF5 Pricing, Annualized) by Regime',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_delta_beta_chart(delta_df):
    """Box plots of delta betas by regime."""
    factors = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    regimes = ['Low Volatility', 'Normal', 'High Volatility']

    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    for ax, factor in zip(axes, factors):
        data = [delta_df[delta_df['REGIME'] == r][f'{factor}_DELTA'].dropna().tolist()
                for r in regimes]
        colors = [REGIME_HEX[r] for r in regimes]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'Δ{factor}', fontweight='bold', fontsize=10)
        ax.set_xticklabels(['Low', 'Norm', 'High'], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Treatment – Control Delta Betas by Regime (Matched Pairs)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_ttest_summary_chart(ttest_df):
    """Dot plot of t-statistics from paired t-tests."""
    if ttest_df.empty:
        return None
    regimes = ttest_df['REGIME'].unique()
    variables = ttest_df['VARIABLE'].unique()

    fig, axes = plt.subplots(1, len(regimes), figsize=(14, 5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = ttest_df[ttest_df['REGIME'] == regime]
        vars_sub = sub['VARIABLE'].tolist()
        tstats = sub['T_STAT'].tolist()
        pvals = sub['P_VALUE'].tolist()

        y = range(len(vars_sub))
        colors = []
        for p in pvals:
            if p < 0.01: colors.append('#C0392B')
            elif p < 0.05: colors.append('#E67E22')
            elif p < 0.10: colors.append('#F1C40F')
            else: colors.append('#95A5A6')

        ax.barh(list(y), tstats, color=colors, edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(1.96, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                   label='±1.96')
        ax.axvline(-1.96, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_yticks(list(y))
        ax.set_yticklabels(vars_sub, fontsize=9)
        ax.set_title(regime, fontweight='bold',
                     color=REGIME_HEX.get(regime, 'black'))
        ax.set_xlabel('t-statistic', fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C0392B', label='p<0.01'),
        Patch(facecolor='#E67E22', label='p<0.05'),
        Patch(facecolor='#F1C40F', label='p<0.10'),
        Patch(facecolor='#95A5A6', label='p≥0.10'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Paired t-Test: H₀: Mean Delta Beta = 0', fontsize=12,
                 fontweight='bold')
    fig.tight_layout()
    return fig_to_image(fig)


def make_regime_amplification_chart(amp_df):
    """Bar chart of regime amplification results."""
    if amp_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    variables = amp_df['VARIABLE'].tolist()
    x = np.arange(len(variables))

    low_vals = amp_df['MEAN_DELTA_LOW'].tolist()
    high_vals = amp_df['MEAN_DELTA_HIGH'].tolist()
    diff_vals = amp_df['MEAN_DIFF'].tolist()

    width = 0.25
    ax.bar(x - width, low_vals, width, label='Low Vol Delta', color='#27AE60', alpha=0.8)
    ax.bar(x, high_vals, width, label='High Vol Delta', color='#E74C3C', alpha=0.8)
    ax.bar(x + width, diff_vals, width, label='Amplification (High-Low)', color='#1F3864', alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=10)
    ax.set_ylabel('Mean Delta Beta', fontsize=10)
    ax.set_title('Regime Amplification: High Volatility vs. Low Volatility Delta Betas',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SHEET BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_cover(wb, regime_result, selection_df, ff5, cw, matched, timestamp):
    ws = wb.active
    ws.title = "Cover"

    # Title
    ws.row_dimensions[1].height = 50
    merge_hdr(ws, 1, 1, 8,
              "ESSAY 1: VOLATILITY REGIMES & THE FAMA-FRENCH FIVE-FACTOR MODEL",
              bg=NAVY, fg=WHITE, size=16)
    ws.row_dimensions[2].height = 30
    merge_hdr(ws, 2, 1, 8,
              "Dissertation Essay 1 — Full Results Workbook",
              bg=GOLD, fg=NAVY, size=13)

    r = 4
    _hdr(ws, r, 1, "Analysis Details", bg=LTBLUE, fg=NAVY, size=11, halign='left')
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
    r += 1
    for label, value in [
        ("Generated", timestamp),
        ("Data Source", "SQLite (signals_systems.db) + FRED + Yahoo Finance"),
        ("VIX Period", "Jan 2000 – Present"),
        ("Regimes (K)", str(regime_result.n_regimes if regime_result else "N/A")),
        ("FF5 Factors", "MKT_RF, SMB, HML, RMW, CMA (Fama-French 2015)"),
        ("Culture War Stocks", f"{cw.n_stocks if cw else 'N/A'} firms analyzed"),
        ("Matched Pairs", f"{matched.n_pairs if matched else 'N/A'} pairs ({matched.n_pairs_complete if matched else 'N/A'} complete)"),
        ("Model Selection", "K=2,3,4 Markov Regime-Switching (Hamilton 1989)"),
    ]:
        _val(ws, r, 1, label, bold=True, halign='left')
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        _val(ws, r, 3, value, halign='left')
        ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=8)
        r += 1

    # Table of contents
    r += 1
    merge_hdr(ws, r, 1, 8, "Workbook Contents", bg=NAVY, fg=WHITE, size=12)
    r += 1
    sheets_toc = [
        ("Cover",                "This page — summary of analysis parameters"),
        ("Equations",            "Key mathematical equations and methodology"),
        ("Regime Summary",       "VIX Markov regime statistics & transition matrix"),
        ("Regime Assignments",   "Daily VIX with regime labels (2000–present)"),
        ("Model Selection",      "K=2,3,4 AIC/BIC comparison + best-fit selection"),
        ("FF5 Regime Results",   "Factor loadings & alphas by regime (spanning regression)"),
        ("Factor Premia",        "Annualized factor premia by regime"),
        ("Pooled Regression",    "Full-sample pooled OLS results"),
        ("Interaction Model",    "Regime × Factor interaction model"),
        ("Chow Test",            "Structural break test across regimes"),
        ("CW Stock Summary",     "Culture war firm FF5 results by regime"),
        ("Matched Controls",     "Treatment vs. control delta betas (all pairs)"),
        ("Paired T-Test",        "H₀: mean delta beta = 0 per regime per factor"),
        ("Regime Amplification", "High vol vs. low vol amplification test"),
        ("Sign Consistency",     "Binomial sign consistency test per regime"),
        ("MC Coverage",          "Ticker-regime data coverage table"),
        ("Macro Controls",       "Key macro variables by regime (summary)"),
        ("Figures",              "All charts and visualizations"),
    ]
    _hdr(ws, r, 1, "Sheet", halign='left')
    _hdr(ws, r, 2, "Contents", halign='left')
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=8)
    r += 1
    for i, (sheet, desc) in enumerate(sheets_toc):
        bg = LTGRAY if i % 2 == 0 else WHITE
        _val(ws, r, 1, sheet, bold=True, bg=bg, halign='left')
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=8)
        _val(ws, r, 2, desc, bg=bg, halign='left')
        r += 1

    # Key findings
    r += 1
    merge_hdr(ws, r, 1, 8, "Key Findings", bg=GOLD, fg=NAVY, size=12)
    r += 1
    if regime_result:
        for label, mean in regime_result.regime_means.items():
            n = regime_result.regime_summary.set_index('REGIME').loc[label, 'N_DAYS']
            pct = regime_result.regime_summary.set_index('REGIME').loc[label, 'PCT_DAYS']
            dur = regime_result.expected_durations[label]
            _val(ws, r, 1, f"{label}:", bold=True, halign='left')
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
            _val(ws, r, 3, f"Mean VIX = {mean:.1f} | {n:,} days ({pct:.1f}%) | E[duration] = {dur:.0f} days",
                 halign='left')
            ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=8)
            r += 1
    if ff5:
        chow = ff5.chow_test
        _val(ws, r, 1, "Chow Test:", bold=True, halign='left')
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        sig = "SIGNIFICANT (p<0.001)" if chow['significant_001'] else "Not significant"
        _val(ws, r, 3,
             f"F({chow['df_numerator']},{chow['df_denominator']}) = {chow['f_stat']:.2f}, p = {chow['p_value']:.4f} → {sig}",
             halign='left')
        ws.merge_cells(start_row=r, start_column=3, end_row=r, end_column=8)
        r += 1

    # References
    r += 1
    merge_hdr(ws, r, 1, 8, "References", bg=LTGRAY, fg=NAVY, size=11)
    r += 1
    refs = [
        "Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384.",
        "Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22.",
        "Ang, A. & Bekaert, G. (2002). International asset allocation with regime shifts. Review of Financial Studies, 15(4), 1137–1187.",
        "Guidolin, M. & Timmermann, A. (2008). Size and value anomalies under regime shifts. Journal of Financial Econometrics, 6(1), 1–48.",
        "Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. JRSS-B, 57(1), 289–300.",
        "Barillas, F. & Shanken, J. (2017). Which alpha? Review of Financial Studies, 30(4), 1316–1338.",
    ]
    for ref in refs:
        _val(ws, r, 1, ref, halign='left')
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        ws.row_dimensions[r].height = 20
        r += 1

    for col in range(1, 9):
        ws.column_dimensions[get_column_letter(col)].width = 18
    ws.column_dimensions['C'].width = 80
    ws.sheet_view.showGridLines = False


def build_equations(wb):
    ws = wb.create_sheet("Equations")

    merge_hdr(ws, 1, 1, 6, "Essay 1 — Key Equations and Methodology", size=14)
    ws.row_dimensions[1].height = 35

    r = 3
    sections = [
        ("1. Markov Regime-Switching Model (Hamilton 1989)", [
            ("Model", "Observation equation: y_t = μ_{s_t} + σ_{s_t} ε_t,  ε_t ~ N(0,1)"),
            ("VIX Process", "VIX_t | s_t=j ~ N(μ_j, σ²_j),  j ∈ {Low, Normal, High}"),
            ("Transition Matrix", "P[s_t=j | s_{t-1}=i] = p_{ij}  (K×K matrix of regime-switching probabilities)"),
            ("Regime Persistence", "E[duration in regime j] = 1 / (1 - p_{jj})"),
            ("Filter", "Hamilton filter: P[s_t=j | Ω_t] via forward recursion on Ω_t = {y_1,...,y_t}"),
            ("Smoothed Prob", "P[s_t=j | Ω_T] via Kim (1994) smoother (backward pass)"),
            ("Log-Likelihood", "L = Σ_t log[Σ_j P[s_t=j | Ω_{t-1}] · f(y_t | s_t=j)]"),
            ("AIC", "AIC = -2·L + 2·k,  where k = number of free parameters"),
            ("BIC", "BIC = -2·L + k·ln(T)"),
        ]),
        ("2. FF5 Spanning Regression (MKT_RF dependent; Fama-French 2015)", [
            ("Pooled OLS", "MKT_RF_t = α + β₁·SMB_t + β₂·HML_t + β₃·RMW_t + β₄·CMA_t + ε_t"),
            ("Regime OLS", "MKT_RF_t = α_j + β₁ⱼ·SMB_t + β₂ⱼ·HML_t + β₃ⱼ·RMW_t + β₄ⱼ·CMA_t + ε_t"),
            ("", "  estimated separately for each regime j ∈ {Low Vol, Normal, High Vol}"),
            ("HAC SE", "Newey-West HAC standard errors with maxlags=5"),
            ("Interaction", "MKT_RF_t = α + β·X_t + γ·D_t ⊗ X_t + ε_t"),
            ("", "  D_t = regime dummies; ⊗ = interaction; captures regime-conditional shifts"),
            ("Cond. Number", "κ(X'X) < 30: well-conditioned; warning issued if κ > 100"),
        ]),
        ("3. Chow Structural Break Test", [
            ("H₀", "H₀: No structural break — factor loadings are equal across all regimes"),
            ("H₁", "H₁: At least one loading differs across regimes"),
            ("F-statistic", "F = [(RSS_pool - Σ RSS_j) / (K-1)·m] / [Σ RSS_j / (N - K·m)]"),
            ("", "  RSS_pool = residual SS pooled; RSS_j = regime-j residual SS"),
            ("", "  K = number of regimes, m = number of regressors, N = total obs"),
            ("Decision", "Reject H₀ if F > F_{α, (K-1)m, N-Km}"),
        ]),
        ("4. Individual Stock FF5 Pricing Regression (Culture War Firms)", [
            ("Equation", "(R_{i,t} - RF_t) = α_i + β₁ᵢ·MKT_RF_t + β₂ᵢ·SMB_t + β₃ᵢ·HML_t + β₄ᵢ·RMW_t + β₅ᵢ·CMA_t + ε_{i,t}"),
            ("Per regime", "Estimated separately for each regime j using regime-assigned observations"),
            ("HAC SE", "Newey-West HAC standard errors with maxlags=5"),
            ("Min obs", "Minimum 30 observations per regime required for estimation"),
            ("Alpha H₀", "H₀: α_i = 0 (no abnormal return relative to FF5 benchmark)"),
        ]),
        ("5. Matched Control Analysis", [
            ("Matching", "Treatment (culture war) firms matched to control firms on NAICS industry code"),
            ("Delta Beta", "Δβ_{f,j} = β_{treat,f,j} - β_{ctrl,f,j}  for factor f in regime j"),
            ("Paired t-test", "H₀: E[Δβ_{f,j}] = 0  →  t = (Δ̄β / SE(Δβ)) ~ t_{n-1}"),
            ("Aggregation", "Deltas aggregated to treatment-firm level (avg over controls) to avoid pseudoreplication"),
            ("Regime Amplification", "Δ_amp = Δβ_{High Vol} - Δβ_{Low Vol}"),
            ("", "H₀: E[Δ_amp] = 0  tested via paired t-test across treatment firms"),
            ("Sign Consistency", "Binomial test: H₀: P(Δβ > 0) = 0.50  per factor per regime"),
            ("", "n_pos ~ Binomial(n, 0.5)  under H₀"),
            ("BH Correction", "Benjamini-Hochberg FDR correction at q=0.10 within each test family"),
        ]),
        ("6. FinBERT Sentiment & FOMO Z-Score", [
            ("Model", "FinBERT (Huang et al. 2023): fine-tuned BERT for financial sentiment"),
            ("Labels", "Sentiment ∈ {positive, negative, neutral} with probability scores"),
            ("Net Sentiment", "S_t = P(positive)_t - P(negative)_t  ∈ [-1, 1]"),
            ("FOMO Z-Score", "z_t = (S_t - μ_S) / σ_S  standardized relative to full-sample mean/SD"),
            ("Euphoria", "z_t > 1.0 classified as euphoria (crowd sentiment > 1 SD above mean)"),
            ("Panic", "z_t < -1.0 classified as panic (crowd sentiment > 1 SD below mean)"),
        ]),
        ("7. Macro Control Variables", [
            ("Inflation", "CPI (YoY%), Core CPI, PCE, PPI, 5Y/10Y breakeven inflation expectations"),
            ("Rates", "Fed Funds Rate, 10Y/2Y yield spread (term slope), BAA-AAA credit spread"),
            ("Employment", "Nonfarm payrolls (MoM), unemployment rate, labor force participation"),
            ("GDP", "Real GDP growth (QoQ annualized), PCE, investment components"),
            ("Usage", "Regime-mean macro values computed for each volatility regime"),
        ]),
    ]

    for section_title, rows in sections:
        merge_hdr(ws, r, 1, 6, section_title, bg=LTBLUE, fg=NAVY, size=11)
        r += 1
        for label, formula in rows:
            _val(ws, r, 1, label, bold=bool(label), halign='left', bg=LTGRAY if label else WHITE)
            ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
            _val(ws, r, 2, formula, halign='left',
                 bg=LTGRAY if label else WHITE)
            ws.row_dimensions[r].height = 18
            r += 1
        r += 1

    set_col_width(ws, 1, 22)
    for col in range(2, 7):
        set_col_width(ws, col, 20)
    ws.column_dimensions['B'].width = 90
    ws.sheet_view.showGridLines = False


def build_regime_summary(wb, regime_result):
    ws = wb.create_sheet("Regime Summary")
    merge_hdr(ws, 1, 1, 8, "VIX Markov Regime-Switching: Regime Summary Statistics", size=13)
    ws.row_dimensions[1].height = 30
    r = 3

    # ── Regime statistics table ──────────────────────────────────────────
    merge_hdr(ws, r, 1, 8, "Regime Descriptive Statistics", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    headers = ['Regime', 'Mean VIX', 'Std VIX', 'Min VIX', 'Max VIX',
               'N Days', '% Days', 'E[Duration]']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    summary = regime_result.regime_summary.copy()
    for _, row in summary.iterrows():
        label = row['REGIME']
        bg_color = REGIME_COLORS.get(label, 'FFFFFF')
        _val(ws, r, 1, label, bold=True, halign='left',
             bg=bg_color, color='FFFFFF')
        _val(ws, r, 2, row['MEAN_VIX'], fmt='0.00')
        _val(ws, r, 3, row['STD_VIX'], fmt='0.00')
        _val(ws, r, 4, row['MIN_VIX'], fmt='0.00')
        _val(ws, r, 5, row['MAX_VIX'], fmt='0.00')
        _val(ws, r, 6, int(row['N_DAYS']), fmt='#,##0')
        _val(ws, r, 7, row['PCT_DAYS'] / 100, fmt='0.0%')
        dur = regime_result.expected_durations.get(label, np.nan)
        _val(ws, r, 8, f'{dur:.1f} days')
        r += 1

    r += 1
    # ── Model fit statistics ─────────────────────────────────────────────
    merge_hdr(ws, r, 1, 8, "Model Fit Statistics", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    for label, value in [
        ("Number of Regimes (K)", regime_result.n_regimes),
        ("Log-Likelihood", f"{regime_result.aic / -2 + regime_result.n_regimes:.2f}"),
        ("AIC", f"{regime_result.aic:.2f}"),
        ("BIC", f"{regime_result.bic:.2f}"),
        ("Total Observations", f"{sum(v for v in [s for s in summary['N_DAYS']]):,}"),
    ]:
        _val(ws, r, 1, label, bold=True, halign='left')
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
        _val(ws, r, 4, str(value), halign='left')
        ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=8)
        r += 1

    r += 1
    # ── Transition matrix ────────────────────────────────────────────────
    merge_hdr(ws, r, 1, 8, "Regime Transition Probability Matrix P[s_t = j | s_{t-1} = i]",
              bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    regime_order = ['Low Volatility', 'Normal', 'High Volatility']
    _hdr(ws, r, 1, "From \\ To")
    for c, label in enumerate(regime_order, 2):
        _hdr(ws, r, c, label)
    r += 1

    tm = regime_result.transition_matrix
    for i, from_label in enumerate(regime_order):
        _val(ws, r, 1, from_label, bold=True, halign='left')
        for j, to_label in enumerate(regime_order):
            val = tm[i, j]
            cell = _val(ws, r, j + 2, val, fmt='0.0000')
            # Highlight diagonal (self-transitions)
            if i == j:
                cell.fill = PatternFill("solid", fgColor='D5F5E3')
                cell.font = Font(bold=True, size=10)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 8, "Note: Diagonal elements represent regime persistence probabilities. "
              "E[duration] = 1/(1-p_ii).", bg=LTGOLD, fg=NAVY, size=10)

    # Expected durations note
    r += 1
    _val(ws, r, 1, "Expected Durations:", bold=True, halign='left')
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
    for c, label in enumerate(regime_order, 3):
        dur = regime_result.expected_durations.get(label, np.nan)
        _val(ws, r, c, f"{label}: {dur:.1f} days")

    for col in range(1, 9):
        set_col_width(ws, col, 18)
    ws.column_dimensions['A'].width = 20
    ws.sheet_view.showGridLines = False


def build_regime_assignments(wb, regime_result):
    ws = wb.create_sheet("Regime Assignments")
    merge_hdr(ws, 1, 1, 4, "Daily VIX with Markov Regime Assignments", size=13)
    ws.row_dimensions[1].height = 28

    headers = ['Date', 'VIX', 'Regime #', 'Regime Label']
    r = 2
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    df = regime_result.regime_assignments.copy()
    df = df.sort_values('DATE')
    for _, row in df.iterrows():
        label = row['REGIME_LABEL']
        bg = REGIME_COLORS.get(label, 'FFFFFF')
        date_val = pd.to_datetime(row['DATE'])
        _val(ws, r, 1, date_val, fmt='YYYY-MM-DD', halign='left')
        _val(ws, r, 2, float(row['VIX']), fmt='0.00')
        _val(ws, r, 3, int(row['REGIME']))
        _val(ws, r, 4, label, bold=True, bg=bg, color='FFFFFF')
        r += 1

    for col, width in [(1, 14), (2, 10), (3, 12), (4, 18)]:
        set_col_width(ws, col, width)
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A3'


def build_model_selection(wb, selection_df):
    ws = wb.create_sheet("Model Selection")
    merge_hdr(ws, 1, 1, 6, "Markov Regime-Switching: Model Selection (K=2, 3, 4)", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 6, "AIC / BIC Comparison — Lower is Better", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    headers = ['K (Regimes)', 'AIC', 'BIC', 'Log-Likelihood', '# Parameters', 'Notes']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    best_aic = selection_df['AIC'].min()
    best_bic = selection_df['BIC'].min()

    for _, row in selection_df.iterrows():
        k = int(row['K'])
        is_best_aic = abs(row['AIC'] - best_aic) < 0.01
        is_best_bic = abs(row['BIC'] - best_bic) < 0.01
        bg = '27AE60' if (is_best_aic and is_best_bic) else ('F0B429' if (is_best_aic or is_best_bic) else None)
        note = []
        if is_best_aic: note.append("★ Best AIC")
        if is_best_bic: note.append("★ Best BIC")

        _val(ws, r, 1, k, bold=is_best_aic or is_best_bic,
             bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 2, row['AIC'], fmt='#,##0.00',
             bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 3, row['BIC'], fmt='#,##0.00',
             bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 4, row['LOG_LIKELIHOOD'], fmt='#,##0.00',
             bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 5, int(row['N_PARAMS']),
             bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 6, ' | '.join(note) if note else '—',
             halign='left', bg=bg, color='FFFFFF' if bg else None)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 6,
              "Note: K=3 selected for primary analysis. BIC penalizes complexity; "
              "AIC rewards fit. Both criteria evaluate regime persistence vs. parsimony.",
              bg=LTGOLD, fg=NAVY, size=10)

    r += 2
    merge_hdr(ws, r, 1, 6, "Parameter Count Formula", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    _val(ws, r, 1, "For K regimes:", bold=True, halign='left')
    _val(ws, r, 2, "# params = K means (μ) + K variances (σ²) + K(K-1) transition probs",
         halign='left')
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
    r += 1
    _val(ws, r, 1, "K=2:", bold=True, halign='left')
    _val(ws, r, 2, "2 + 2 + 2(1) = 6 parameters", halign='left')
    r += 1
    _val(ws, r, 1, "K=3:", bold=True, halign='left')
    _val(ws, r, 2, "3 + 3 + 3(2) = 12 parameters", halign='left')
    r += 1
    _val(ws, r, 1, "K=4:", bold=True, halign='left')
    _val(ws, r, 2, "4 + 4 + 4(3) = 20 parameters", halign='left')

    for col, width in [(1, 14), (2, 16), (3, 16), (4, 18), (5, 14), (6, 35)]:
        set_col_width(ws, col, width)
    ws.sheet_view.showGridLines = False


def build_ff5_regime_results(wb, ff5):
    ws = wb.create_sheet("FF5 Regime Results")
    merge_hdr(ws, 1, 1, 14,
              "FF5 Spanning Regression (MKT_RF ~ SMB + HML + RMW + CMA) by Regime",
              size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 14,
              "Regression: MKT_RF_t = α + β₁·SMB_t + β₂·HML_t + β₃·RMW_t + β₄·CMA_t + ε_t  (HAC SEs, maxlags=5)",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    # Coefficient comparison table
    merge_hdr(ws, r, 1, 14, "Coefficient Estimates by Regime", bg=LTBLUE, fg=NAVY, size=11)
    r += 1

    headers = ['Regime', 'N Obs', 'Alpha', 'Alpha t', 'Alpha p',
               'SMB β', 'SMB t', 'SMB p', 'HML β', 'HML t', 'HML p',
               'RMW β', 'RMW t', 'RMW p', 'CMA β', 'CMA t', 'CMA p', 'R²']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    coeff = ff5.coefficient_comparison.copy()
    for _, row in coeff.iterrows():
        label = row['REGIME']
        bg = REGIME_COLORS.get(label, 'FFFFFF')
        _val(ws, r, 1, label, bold=True, halign='left', bg=bg, color='FFFFFF')

        # Get n_obs from regime_regressions
        rr = ff5.regime_regressions.get(label)
        _val(ws, r, 2, rr.n_obs if rr else None, fmt='#,##0')

        # Alpha
        alpha_p = row['ALPHA_P']
        alpha_color = _sig_color(alpha_p)
        _val(ws, r, 3, row['ALPHA'], fmt='0.000000')
        _val(ws, r, 4, row['ALPHA_T'] if 'ALPHA_T' in row.index else np.nan, fmt='0.000')
        cell = _val(ws, r, 5, alpha_p, fmt='0.0000')
        if alpha_color:
            cell.fill = PatternFill("solid", fgColor=_sig_color(alpha_p, bg_only=True))

        c_offset = 6
        for factor in ['SMB', 'HML', 'RMW', 'CMA']:
            beta = row.get(f'{factor}_BETA', np.nan)
            t = row.get(f'{factor}_T', np.nan)
            p = row.get(f'{factor}_P', np.nan)
            _val(ws, r, c_offset, beta, fmt='0.0000')
            _val(ws, r, c_offset + 1, t, fmt='0.000')
            p_cell = _val(ws, r, c_offset + 2, p, fmt='0.0000')
            if isinstance(p, (int, float)) and not np.isnan(p):
                sig_bg = _sig_color(p, bg_only=True)
                if sig_bg:
                    p_cell.fill = PatternFill("solid", fgColor=sig_bg)
            c_offset += 3

        _val(ws, r, c_offset, row['R_SQUARED'], fmt='0.0000')
        r += 1

    r += 1
    # Significance legend
    merge_hdr(ws, r, 1, 5, "Significance Color Legend", bg=LTGRAY, fg=NAVY, size=10)
    r += 1
    for label, color, thresh in [
        ("p < 0.01 (1%)", 'FFD7D4', '<0.01'),
        ("p < 0.05 (5%)", 'FDEBD0', '<0.05'),
        ("p < 0.10 (10%)", 'FEF9E7', '<0.10'),
    ]:
        cell = ws.cell(row=r, column=1, value=label)
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(size=9)
        r += 1

    r += 1
    # Chow test
    build_chow_block(ws, ff5.chow_test, r)

    col_widths = [20, 8, 10, 8, 10, 8, 8, 10, 8, 8, 10, 8, 8, 10, 8, 8, 10, 8]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'B4'


def build_chow_block(ws, chow, r):
    merge_hdr(ws, r, 1, 6, "Chow Structural Break Test", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    merge_hdr(ws, r, 1, 6,
              "H₀: No structural break — factor loadings are equal across all regimes",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    for label, value in [
        ("F-statistic", f"{chow['f_stat']:.4f}"),
        ("df (numerator)", chow['df_numerator']),
        ("df (denominator)", chow['df_denominator']),
        ("p-value", f"{chow['p_value']:.6f}" if chow['p_value'] > 1e-10 else "< 1e-10"),
        ("Significant at 5%?", "YES ✓" if chow['significant_005'] else "NO"),
        ("Significant at 1%?", "YES ✓" if chow['significant_001'] else "NO"),
        ("Regimes tested", chow['n_regimes_tested']),
    ]:
        _val(ws, r, 1, label, bold=True, halign='left')
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
        cell = _val(ws, r, 4, str(value), halign='left')
        if 'YES' in str(value):
            cell.fill = PatternFill("solid", fgColor='D5F5E3')
            cell.font = Font(bold=True, color='1E8449', size=10)
        ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=6)
        r += 1
    return r


def build_chow_sheet(wb, ff5):
    ws = wb.create_sheet("Chow Test")
    merge_hdr(ws, 1, 1, 6, "Chow Structural Break Test — Full Results", size=13)
    ws.row_dimensions[1].height = 28
    build_chow_block(ws, ff5.chow_test, 3)
    for col, width in [(1, 25), (2, 20), (3, 20), (4, 25), (5, 20), (6, 20)]:
        set_col_width(ws, col, width)
    ws.sheet_view.showGridLines = False


def build_factor_premia(wb, ff5):
    ws = wb.create_sheet("Factor Premia")
    merge_hdr(ws, 1, 1, 12, "FF5 Factor Premia by Volatility Regime", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    premia = ff5.factor_premia_comparison.copy()

    merge_hdr(ws, r, 1, 12, "Mean Daily and Annualized Factor Returns (×100 for %)",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    # Build header dynamically from columns
    merge_hdr(ws, r, 1, 12, "Factor Premia Table", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    headers = premia.columns.tolist()
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in premia.iterrows():
        label = row['REGIME']
        bg = REGIME_COLORS.get(label, 'FFFFFF')
        for c, col in enumerate(headers, 1):
            val = row[col]
            if col == 'REGIME':
                _val(ws, r, c, val, bold=True, halign='left', bg=bg, color='FFFFFF')
            elif col == 'N_DAYS':
                _val(ws, r, c, int(val), fmt='#,##0')
            else:
                _val(ws, r, c, float(val) if pd.notna(val) else None, fmt='0.0000%')
        r += 1

    r += 2
    merge_hdr(ws, r, 1, 12, "Individual Regime Factor Premia Details", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    for label, rr in ff5.regime_regressions.items():
        bg = REGIME_COLORS.get(label, 'FFFFFF')
        merge_hdr(ws, r, 1, 12, f"Regime: {label}", bg=bg, fg=WHITE, size=11)
        r += 1
        _hdr(ws, r, 1, "Factor")
        _hdr(ws, r, 2, "Mean Daily Return")
        _hdr(ws, r, 3, "Annualized Return")
        r += 1
        for f, val in rr.factor_premia.items():
            _val(ws, r, 1, f, bold=True, halign='left')
            _val(ws, r, 2, val, fmt='0.00000%')
            _val(ws, r, 3, val * 252, fmt='0.000%')
            r += 1
        r += 1

    for col in range(1, 13):
        set_col_width(ws, col, 15)
    ws.column_dimensions['A'].width = 20
    ws.sheet_view.showGridLines = False


def build_pooled_regression(wb, ff5):
    ws = wb.create_sheet("Pooled Regression")
    merge_hdr(ws, 1, 1, 6,
              "Pooled OLS: MKT_RF ~ SMB + HML + RMW + CMA (Full Sample)", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    model = ff5.pooled_regression
    merge_hdr(ws, r, 1, 6, f"N={int(model.nobs):,}, R²={model.rsquared:.4f}, "
              f"Adj-R²={model.rsquared_adj:.4f}", bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 6, "Coefficient Estimates (HAC Standard Errors, maxlags=5)",
              bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    for c, h in enumerate(['Variable', 'Coefficient', 't-statistic', 'p-value',
                            'Sig. (5%)', 'Interpretation'], 1):
        _hdr(ws, r, c, h)
    r += 1

    interp = {
        'const': 'Unconditional alpha (intercept)',
        'SMB': 'Small-minus-big size premium loading',
        'HML': 'High-minus-low value premium loading',
        'RMW': 'Robust-minus-weak profitability loading',
        'CMA': 'Conservative-minus-aggressive investment loading',
    }
    for var in model.params.index:
        coef = model.params[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        bg = _sig_color(p, bg_only=True)
        _val(ws, r, 1, var, bold=True, halign='left')
        _val(ws, r, 2, coef, fmt='0.000000')
        _val(ws, r, 3, t, fmt='0.000')
        p_cell = _val(ws, r, 4, p, fmt='0.0000')
        if bg: p_cell.fill = PatternFill("solid", fgColor=bg)
        _val(ws, r, 5, sig, halign='center',
             bold=bool(sig), color=GREEN if sig else None)
        _val(ws, r, 6, interp.get(var, ''), halign='left')
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 6, "Model Summary Statistics", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    stats_items = [
        ("R-squared", f"{model.rsquared:.6f}"),
        ("Adj. R-squared", f"{model.rsquared_adj:.6f}"),
        ("F-statistic", f"{model.fvalue:.4f}"),
        ("Prob (F)", f"{model.f_pvalue:.6f}"),
        ("N Observations", f"{int(model.nobs):,}"),
        ("AIC", f"{model.aic:.4f}"),
        ("BIC", f"{model.bic:.4f}"),
    ]
    for label, val in stats_items:
        _val(ws, r, 1, label, bold=True, halign='left')
        _val(ws, r, 2, val, halign='left')
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
        r += 1

    for col, width in [(1, 20), (2, 16), (3, 14), (4, 12), (5, 10), (6, 45)]:
        set_col_width(ws, col, width)
    ws.sheet_view.showGridLines = False


def build_interaction_model(wb, ff5):
    ws = wb.create_sheet("Interaction Model")
    merge_hdr(ws, 1, 1, 6,
              "Interaction Model: MKT_RF ~ Factors × Regime Dummies", size=13)
    ws.row_dimensions[1].height = 28
    r = 3
    model = ff5.interaction_model
    merge_hdr(ws, r, 1, 6,
              f"N={int(model.nobs):,}, R²={model.rsquared:.4f}, Adj-R²={model.rsquared_adj:.4f}",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 6,
              "Reference regime: Low Volatility. D_Normal, D_High_Volatility are regime dummies.",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 6, "Coefficient Estimates (HAC SEs)", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    for c, h in enumerate(['Variable', 'Coefficient', 't-statistic', 'p-value', 'Sig.', 'Type'], 1):
        _hdr(ws, r, c, h)
    r += 1

    for var in model.params.index:
        coef = model.params[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        bg = _sig_color(p, bg_only=True)
        var_type = ('Interaction' if 'x_' in var else
                    ('Dummy' if var.startswith('D_') else 'Base Factor'))
        _val(ws, r, 1, var, bold=True, halign='left')
        _val(ws, r, 2, coef, fmt='0.000000')
        _val(ws, r, 3, t, fmt='0.000')
        p_cell = _val(ws, r, 4, p, fmt='0.0000')
        if bg: p_cell.fill = PatternFill("solid", fgColor=bg)
        _val(ws, r, 5, sig, bold=bool(sig), color=GREEN if sig else None)
        _val(ws, r, 6, var_type, halign='left')
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 6,
              "Interpretation: Interaction terms show how factor loadings shift in Normal and High-Vol "
              "regimes relative to Low-Vol baseline. E.g., D_High_Volatility_x_CMA = −0.726 means "
              "CMA loading is 0.726 lower in High-Vol vs Low-Vol regime.",
              bg=LTGOLD, fg=NAVY, size=10)
    ws.row_dimensions[r].height = 40
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)

    for col, width in [(1, 32), (2, 16), (3, 14), (4, 12), (5, 8), (6, 18)]:
        set_col_width(ws, col, width)
    ws.sheet_view.showGridLines = False


def build_cw_summary(wb, cw):
    ws = wb.create_sheet("CW Stock Summary")
    merge_hdr(ws, 1, 1, 12, "Culture War Stocks: FF5 Pricing Regression by Regime", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    # Summary stats
    merge_hdr(ws, r, 1, 12,
              f"{cw.n_stocks} stocks analyzed | {cw.n_failed} failed | {cw.n_partial} partial",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 12,
              "Equation: (R_{i,t} - RF_t) = α_i + β₁·MKT_RF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA + ε",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1

    summary = cw.summary.copy()
    headers = ['Ticker', 'Regime', 'Alpha', 'Alpha p', 'BH Sig',
               'N Obs', 'R²', 'MKT_RF β', 'SMB β', 'HML β', 'RMW β', 'CMA β']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in summary.sort_values(['TICKER', 'REGIME']).iterrows():
        label = row.get('REGIME', '')
        bg = REGIME_COLORS.get(label, None)
        _val(ws, r, 1, row['TICKER'], bold=True, halign='left')
        _val(ws, r, 2, label, bold=True, bg=bg, color='FFFFFF' if bg else None)
        alpha = row.get('ALPHA')
        _val(ws, r, 3, float(alpha) * 252 if pd.notna(alpha) else None, fmt='0.000%')  # annualized
        alpha_p = row.get('ALPHA_P')
        p_cell = _val(ws, r, 4, float(alpha_p) if pd.notna(alpha_p) else None, fmt='0.0000')
        if pd.notna(alpha_p):
            sig_bg = _sig_color(float(alpha_p), bg_only=True)
            if sig_bg: p_cell.fill = PatternFill("solid", fgColor=sig_bg)
        bh_sig = row.get('ALPHA_SIGNIFICANT_BH', False)
        bh_cell = _val(ws, r, 5, '✓' if bh_sig else '', halign='center',
                       color=GREEN if bh_sig else None, bold=bh_sig)
        _val(ws, r, 6, int(row['N_OBS']) if pd.notna(row.get('N_OBS')) else None, fmt='#,##0')
        _val(ws, r, 7, float(row['R_SQUARED']) if pd.notna(row.get('R_SQUARED')) else None,
             fmt='0.0000')
        for c, factor in enumerate(['MKT_RF_BETA', 'SMB_BETA', 'HML_BETA', 'RMW_BETA', 'CMA_BETA'], 8):
            val = row.get(factor)
            _val(ws, r, c, float(val) if pd.notna(val) else None, fmt='0.0000')
        r += 1

    col_widths = [8, 18, 10, 10, 8, 8, 8, 10, 10, 10, 10, 10]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A5'


def build_matched_controls(wb, matched):
    ws = wb.create_sheet("Matched Controls")
    merge_hdr(ws, 1, 1, 14, "Matched Control Analysis: Treatment vs. Control Delta Betas", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 14,
              f"{matched.n_pairs} pairs | {matched.n_pairs_complete} complete across all regimes",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 14,
              "Δβ = β(Treatment) − β(Control)  |  Matched on NAICS industry code",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1

    df = matched.delta_betas.copy()
    headers = df.columns.tolist()
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in df.sort_values(['TREATMENT_TICKER', 'REGIME']).iterrows():
        label = row.get('REGIME', '')
        bg = REGIME_COLORS.get(label, None)
        for c, col in enumerate(headers, 1):
            val = row[col]
            if col == 'REGIME':
                _val(ws, r, c, val, bold=True, bg=bg, color='FFFFFF' if bg else None)
            elif col in ('TREATMENT_COMPANY', 'CONTROL_FIRM', 'INDUSTRY',
                         'TREATMENT_TICKER', 'CONTROL_TICKER'):
                _val(ws, r, c, str(val) if pd.notna(val) else '', halign='left')
            else:
                try:
                    _val(ws, r, c, float(val) if pd.notna(val) else None, fmt='0.0000')
                except (TypeError, ValueError):
                    _val(ws, r, c, str(val), halign='left')
        r += 1

    for col in range(1, len(headers) + 1):
        set_col_width(ws, col, 12)
    ws.column_dimensions['A'].width = 10
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 25
    ws.column_dimensions['D'].width = 25
    ws.column_dimensions['E'].width = 18
    ws.column_dimensions['F'].width = 18
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A6'


def build_paired_ttest(wb, matched):
    ws = wb.create_sheet("Paired T-Test")
    merge_hdr(ws, 1, 1, 10,
              "Matched Control: Paired t-Test — H₀: Mean Delta Beta = 0", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 10,
              "t = Δ̄β / (SD(Δβ) / √n_firms)  |  Aggregated to treatment-firm level",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 10,
              f"BH FDR correction at q=0.10 | {matched.n_pairs} pairs total",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    df = matched.paired_ttest.copy()
    headers = ['REGIME', 'VARIABLE', 'MEAN_DELTA', 'STD_DELTA',
               'N_TREATMENT_FIRMS', 'N_RAW_PAIRS', 'T_STAT', 'P_VALUE', 'BH_SIGNIFICANT']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in df.iterrows():
        label = row['REGIME']
        bg = REGIME_COLORS.get(label, None)
        _val(ws, r, 1, label, bold=True, bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 2, row['VARIABLE'], bold=True)
        _val(ws, r, 3, row['MEAN_DELTA'], fmt='0.0000')
        _val(ws, r, 4, row['STD_DELTA'], fmt='0.0000')
        _val(ws, r, 5, int(row['N_TREATMENT_FIRMS']), fmt='#,##0')
        _val(ws, r, 6, int(row['N_RAW_PAIRS']), fmt='#,##0')
        _val(ws, r, 7, row['T_STAT'], fmt='0.0000')
        p = row['P_VALUE']
        p_cell = _val(ws, r, 8, p, fmt='0.0000')
        sig_bg = _sig_color(p, bg_only=True)
        if sig_bg: p_cell.fill = PatternFill("solid", fgColor=sig_bg)
        bh = row.get('BH_SIGNIFICANT', False)
        bh_cell = _val(ws, r, 9, '★ BH Sig' if bh else '—',
                       bold=bh, color=GREEN if bh else None)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 10,
              "★ BH Sig = statistically significant after Benjamini-Hochberg FDR correction at q=10%.",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 10,
              "Color coding: Red = p<0.01, Orange = p<0.05, Yellow = p<0.10, White = not significant.",
              bg=LTGRAY, fg=NAVY, size=10)

    col_widths = [20, 10, 12, 12, 18, 14, 10, 10, 14]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False


def build_regime_amplification(wb, matched):
    ws = wb.create_sheet("Regime Amplification")
    merge_hdr(ws, 1, 1, 9,
              "Regime Amplification: High Volatility vs. Low Volatility Delta Betas", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 9,
              "Δ_amp = Δβ(High Vol) − Δβ(Low Vol)  |  H₀: E[Δ_amp] = 0",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1
    merge_hdr(ws, r, 1, 9,
              "Tests whether culture war exposure is amplified in high-volatility regimes",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    if matched.regime_amplification.empty:
        merge_hdr(ws, r, 1, 9, "No amplification results produced (insufficient pairs)",
                  bg=LTGRAY, fg=NAVY, size=11)
        return

    df = matched.regime_amplification.copy()
    headers = ['VARIABLE', 'MEAN_DELTA_LOW', 'MEAN_DELTA_HIGH', 'MEAN_DIFF',
               'N_TREATMENT_FIRMS', 'N_RAW_PAIRS', 'T_STAT', 'P_VALUE', 'BH_SIGNIFICANT']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in df.iterrows():
        _val(ws, r, 1, row['VARIABLE'], bold=True)
        _val(ws, r, 2, row['MEAN_DELTA_LOW'], fmt='0.0000')
        _val(ws, r, 3, row['MEAN_DELTA_HIGH'], fmt='0.0000')
        diff_cell = _val(ws, r, 4, row['MEAN_DIFF'], fmt='0.0000')
        if abs(row['MEAN_DIFF']) > abs(row['MEAN_DELTA_LOW']) * 0.1:
            pass  # amplification present
        _val(ws, r, 5, int(row['N_TREATMENT_FIRMS']), fmt='#,##0')
        _val(ws, r, 6, int(row['N_RAW_PAIRS']), fmt='#,##0')
        _val(ws, r, 7, row['T_STAT'], fmt='0.0000')
        p = row['P_VALUE']
        p_cell = _val(ws, r, 8, p, fmt='0.0000')
        sig_bg = _sig_color(p, bg_only=True)
        if sig_bg: p_cell.fill = PatternFill("solid", fgColor=sig_bg)
        bh = row.get('BH_SIGNIFICANT', False)
        _val(ws, r, 9, '★ BH Sig' if bh else '—',
             bold=bh, color=GREEN if bh else None)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 9,
              "Note: Only Low-Vol and High-Vol (extreme) regimes are compared. "
              "Middle 'Normal' regime is excluded per hypothesis design (amplification at extremes).",
              bg=LTGOLD, fg=NAVY, size=10)
    ws.row_dimensions[r].height = 35

    col_widths = [12, 16, 16, 12, 18, 14, 10, 10, 14]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False


def build_sign_consistency(wb, matched):
    ws = wb.create_sheet("Sign Consistency")
    merge_hdr(ws, 1, 1, 9,
              "Sign Consistency: Binomial Test — H₀: P(Δβ > 0) = 0.50", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 9,
              "n_pos ~ Binomial(n, 0.5) under H₀  |  Aggregated to treatment-firm level",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1

    df = matched.sign_consistency.copy()
    headers = ['REGIME', 'VARIABLE', 'N_POSITIVE', 'N_NEGATIVE',
               'N_TREATMENT_FIRMS', 'PCT_MAJORITY', 'MAJORITY_SIGN', 'BINOMIAL_P', 'BH_SIGNIFICANT']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for _, row in df.iterrows():
        label = row['REGIME']
        bg = REGIME_COLORS.get(label, None)
        _val(ws, r, 1, label, bold=True, bg=bg, color='FFFFFF' if bg else None)
        _val(ws, r, 2, row['VARIABLE'], bold=True)
        _val(ws, r, 3, int(row['N_POSITIVE']), fmt='#,##0')
        _val(ws, r, 4, int(row['N_NEGATIVE']), fmt='#,##0')
        _val(ws, r, 5, int(row['N_TREATMENT_FIRMS']), fmt='#,##0')
        _val(ws, r, 6, row['PCT_MAJORITY'], fmt='0.0%')
        sign_cell = _val(ws, r, 7, row['MAJORITY_SIGN'])
        sign_cell.fill = PatternFill("solid", fgColor='D5F5E3' if row['MAJORITY_SIGN'] == 'positive' else 'FADBD8')
        p = row['BINOMIAL_P']
        p_cell = _val(ws, r, 8, p, fmt='0.0000')
        sig_bg = _sig_color(p, bg_only=True)
        if sig_bg: p_cell.fill = PatternFill("solid", fgColor=sig_bg)
        bh = row.get('BH_SIGNIFICANT', False)
        _val(ws, r, 9, '★ BH Sig' if bh else '—',
             bold=bh, color=GREEN if bh else None)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 9,
              "Interpretation: PCT_MAJORITY > 75% suggests directional consistency across firms. "
              "Binomial p < 0.05 rejects random sign assignment.",
              bg=LTGOLD, fg=NAVY, size=10)
    ws.row_dimensions[r].height = 30

    col_widths = [20, 10, 12, 12, 18, 14, 14, 12, 14]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False


def build_mc_coverage(wb, matched):
    ws = wb.create_sheet("MC Coverage")
    merge_hdr(ws, 1, 1, 5, "Matched Control: Ticker-Regime Data Coverage", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    df = matched.coverage.copy()
    total = len(df)
    covered = df['HAS_RESULT'].sum()
    merge_hdr(ws, r, 1, 5,
              f"Coverage: {covered}/{total} ticker-regime slots ({covered/total:.0%})",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    for c, h in enumerate(['TICKER', 'REGIME', 'HAS_RESULT', 'STATUS', 'N_OBS'], 1):
        _hdr(ws, r, c, h)
    r += 1

    status_colors = {
        'OK': 'D5F5E3',
        'INSUFFICIENT_OBS': 'FDEBD0',
        'NO_DATA': 'FADBD8',
        'REGRESSION_FAILED': 'FADBD8',
    }
    for _, row in df.sort_values(['TICKER', 'REGIME']).iterrows():
        label = row['REGIME']
        _val(ws, r, 1, row['TICKER'], bold=True, halign='left')
        bg = REGIME_COLORS.get(label, None)
        _val(ws, r, 2, label, bg=bg, color='FFFFFF' if bg else None)
        has = row['HAS_RESULT']
        _val(ws, r, 3, '✓' if has else '✗', halign='center',
             color=GREEN if has else RED, bold=True)
        status = row['STATUS']
        status_cell = _val(ws, r, 4, status)
        s_bg = status_colors.get(status, None)
        if s_bg: status_cell.fill = PatternFill("solid", fgColor=s_bg)
        _val(ws, r, 5, int(row['N_OBS']), fmt='#,##0')
        r += 1

    r += 1
    for status, color in status_colors.items():
        cell = ws.cell(row=r, column=1, value=status)
        cell.fill = PatternFill("solid", fgColor=color)
        cell.font = Font(size=9)
        r += 1

    col_widths = [10, 18, 12, 20, 10]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = 'A5'


def build_macro_summary(wb, macro_df, regime_result):
    ws = wb.create_sheet("Macro Controls")
    merge_hdr(ws, 1, 1, 8,
              "Macro Control Variables: Regime-Conditional Summary Statistics", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 8,
              f"Panel: {len(macro_df):,} daily observations × {len(macro_df.columns)-1} macro variables",
              bg=LTGOLD, fg=NAVY, size=10)
    r += 1

    # Select key variables for summary
    key_vars = [
        ('FED_FUNDS_RATE_macro', 'Fed Funds Rate', '%'),
        ('T10Y_RATE_macro', '10Y Treasury Rate', '%'),
        ('SLOPE_10Y_2Y_rates', '10Y-2Y Yield Spread', '%'),
        ('UNEMPLOYMENT_RATE_macro', 'Unemployment Rate', '%'),
        ('LABOR_FORCE_PARTICIPATION_macro', 'Labor Force Participation', '%'),
        ('GDP_GROWTH_macro', 'Real GDP Growth', '%'),
        ('CPI_inflation', 'CPI Level', 'index'),
        ('CPI_YOY_inflation', 'CPI YoY%', '%'),
        ('CORE_CPI_YOY_inflation', 'Core CPI YoY%', '%'),
        ('BAA_10Y_SPREAD_rates', 'BAA-10Y Credit Spread', '%'),
        ('BREAKEVEN_10Y_inflation', '10Y Breakeven Inflation', '%'),
        ('CONSUMER_SENTIMENT_macro', 'Consumer Sentiment', 'index'),
    ]

    # Merge macro with regime labels
    regime_df = regime_result.regime_assignments[['DATE', 'REGIME_LABEL']].copy()
    regime_df['DATE'] = pd.to_datetime(regime_df['DATE'])
    macro_copy = macro_df.copy()
    macro_copy['DATE'] = pd.to_datetime(macro_copy['DATE'])
    merged = macro_copy.merge(regime_df, on='DATE', how='inner')

    regimes = ['Low Volatility', 'Normal', 'High Volatility']

    merge_hdr(ws, r, 1, 8, "Key Macro Variables: Mean by Volatility Regime", bg=LTBLUE, fg=NAVY, size=11)
    r += 1
    headers = ['Variable', 'Unit', 'Full Sample Mean'] + regimes
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    for col, label, unit in key_vars:
        if col not in merged.columns:
            continue
        _val(ws, r, 1, label, halign='left', bold=True)
        _val(ws, r, 2, unit, halign='center')
        full_mean = merged[col].mean()
        _val(ws, r, 3, full_mean, fmt='0.00')
        for c_off, regime in enumerate(regimes, 4):
            regime_mean = merged[merged['REGIME_LABEL'] == regime][col].mean()
            cell = _val(ws, r, c_off, regime_mean, fmt='0.00')
            bg = REGIME_COLORS.get(regime, None)
            if bg:
                cell.fill = PatternFill("solid", fgColor=bg + '44')
        r += 1

    r += 2
    merge_hdr(ws, r, 1, 8,
              f"Full Macro Panel: {len(macro_df.columns)-1} variables across {len(macro_df):,} rows. "
              "Categories: Inflation (39 vars), Rates (43 vars), Employment (34 vars), GDP (26 vars), Macro (13 vars).",
              bg=LTGOLD, fg=NAVY, size=10)
    ws.row_dimensions[r].height = 35

    col_widths = [30, 10, 18, 18, 18, 18]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False


def build_figures_sheet(wb, figures_dict):
    ws = wb.create_sheet("Figures")
    merge_hdr(ws, 1, 1, 6, "Essay 1 — Figures and Visualizations", size=14)
    ws.row_dimensions[1].height = 35

    r = 3
    anchor_row = r
    fig_titles = list(figures_dict.keys())
    bufs = list(figures_dict.values())

    for i, (title, buf) in enumerate(zip(fig_titles, bufs)):
        if buf is None:
            continue
        merge_hdr(ws, anchor_row, 1, 6, f"Figure {i+1}: {title}", bg=LTBLUE, fg=NAVY, size=11)
        anchor_row += 1
        col_letter = 'A'
        anchor = f"{col_letter}{anchor_row}"
        insert_image(ws, buf, anchor, width=720, height=330)
        anchor_row += 22  # approximate row height for image

    ws.sheet_view.showGridLines = False
    for col in range(1, 7):
        set_col_width(ws, col, 20)


def build_bh_sanity(wb):
    ws = wb.create_sheet("BH Sanity Check")
    merge_hdr(ws, 1, 1, 6, "Benjamini-Hochberg FDR Correction — Sanity Check (q=0.10)", size=13)
    ws.row_dimensions[1].height = 28
    r = 3

    merge_hdr(ws, r, 1, 6,
              "Test case: p-values = [0.001, 0.008, 0.039, 0.041, 0.23, 0.45, 0.89], q=0.10",
              bg=LTBLUE, fg=NAVY, size=10)
    r += 1
    test_pvals = [0.001, 0.008, 0.039, 0.041, 0.23, 0.45, 0.89]
    bh_result = benjamini_hochberg(test_pvals, q=0.10)

    headers = ['Rank (i)', 'p-value', 'BH Threshold (i/m × q)', 'Rejected?']
    for c, h in enumerate(headers, 1):
        _hdr(ws, r, c, h)
    r += 1

    m = len(test_pvals)
    sorted_pvals = sorted(enumerate(test_pvals, 1), key=lambda x: x[1])
    for rank, (orig_i, p) in enumerate(sorted_pvals, 1):
        threshold = rank / m * 0.10
        rejected = bh_result[orig_i - 1]
        _val(ws, r, 1, rank)
        p_cell = _val(ws, r, 2, p, fmt='0.0000')
        if rejected:
            p_cell.fill = PatternFill("solid", fgColor='D5F5E3')
        _val(ws, r, 3, threshold, fmt='0.0000')
        _val(ws, r, 4, '✓ Rejected' if rejected else '— Not Rejected',
             color=GREEN if rejected else RED, bold=rejected)
        r += 1

    r += 1
    merge_hdr(ws, r, 1, 6,
              "BH Procedure: Sort p-values ascending. Find largest i where p_(i) ≤ (i/m)×q. "
              "Reject all H_0 with rank ≤ that i.",
              bg=LTGOLD, fg=NAVY, size=10)
    ws.row_dimensions[r].height = 30

    col_widths = [12, 12, 22, 18]
    for c, w in enumerate(col_widths, 1):
        set_col_width(ws, c, w)
    ws.sheet_view.showGridLines = False


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Essay 1 Workbook Generator — {timestamp}")
    print("=" * 60)

    print("Loading data...")
    ds = DataStore()

    print("Step 1: Estimating VIX regimes (K=3)...")
    regime_result = estimate_vix_regimes(ds, n_regimes=3)
    if regime_result is None:
        print("FAILED: Regime estimation failed")
        return

    for label, mean in regime_result.regime_means.items():
        n = regime_result.regime_summary.set_index('REGIME').loc[label, 'N_DAYS']
        print(f"  {label}: VIX={mean:.1f}, n={n:,} days")

    print("Step 2: Model selection (K=2,3,4)...")
    selection_df = select_n_regimes(ds)
    for _, row in selection_df.iterrows():
        print(f"  K={int(row['K'])}: AIC={row['AIC']:.1f}, BIC={row['BIC']:.1f}")

    print("Step 3: FF5 spanning regression by regime...")
    ff5 = ff5_by_regime(ds, regime_result=regime_result)
    if ff5:
        chow = ff5.chow_test
        print(f"  Chow: F={chow['f_stat']:.2f}, p={chow['p_value']:.4f}, "
              f"sig={'YES' if chow['significant_005'] else 'NO'}")

    print("Step 4: Assembling macro controls...")
    macro_df = assemble_macro_controls(ds)
    print(f"  Macro panel: {len(macro_df):,} rows × {len(macro_df.columns)-1} variables")

    print("Step 5: Culture war stocks by regime...")
    cw = culture_war_by_regime(ds, regime_result=regime_result)
    if cw:
        print(f"  {cw.n_stocks} stocks, {cw.n_failed} failed")

    print("Step 6: Matched control FF5 analysis...")
    matched = ff5_matched_control_analysis(ds, regime_result=regime_result)
    if matched:
        print(f"  {matched.n_pairs} pairs, {matched.n_pairs_complete} complete")

    print()
    print("Building Excel workbook...")
    wb = openpyxl.Workbook()

    # ── Build all sheets ──────────────────────────────────────────────────
    print("  Cover page...")
    build_cover(wb, regime_result, selection_df, ff5, cw, matched, timestamp)

    print("  Equations...")
    build_equations(wb)

    print("  Regime summary...")
    build_regime_summary(wb, regime_result)

    print("  Regime assignments...")
    build_regime_assignments(wb, regime_result)

    print("  Model selection...")
    build_model_selection(wb, selection_df)

    if ff5:
        print("  FF5 regime results...")
        build_ff5_regime_results(wb, ff5)

        print("  Factor premia...")
        build_factor_premia(wb, ff5)

        print("  Pooled regression...")
        build_pooled_regression(wb, ff5)

        print("  Interaction model...")
        build_interaction_model(wb, ff5)

        print("  Chow test sheet...")
        build_chow_sheet(wb, ff5)

    if cw:
        print("  Culture war stock summary...")
        build_cw_summary(wb, cw)

    if matched:
        print("  Matched controls...")
        build_matched_controls(wb, matched)

        print("  Paired t-test...")
        build_paired_ttest(wb, matched)

        print("  Regime amplification...")
        build_regime_amplification(wb, matched)

        print("  Sign consistency...")
        build_sign_consistency(wb, matched)

        print("  MC coverage...")
        build_mc_coverage(wb, matched)

    print("  Macro controls summary...")
    build_macro_summary(wb, macro_df, regime_result)

    print("  BH sanity check...")
    build_bh_sanity(wb)

    # ── Figures ───────────────────────────────────────────────────────────
    print("  Generating figures...")
    figures = {}

    try:
        figures["VIX Regime Time Series"] = make_vix_regime_chart(regime_result)
    except Exception as e:
        print(f"    VIX chart failed: {e}")

    try:
        figures["Regime Characteristics (Bar)"] = make_regime_bar_chart(regime_result)
    except Exception as e:
        print(f"    Regime bar chart failed: {e}")

    try:
        figures["Transition Probability Matrix"] = make_transition_matrix_heatmap(regime_result)
    except Exception as e:
        print(f"    Transition heatmap failed: {e}")

    if not selection_df.empty:
        try:
            figures["Model Selection AIC/BIC"] = make_model_selection_chart(selection_df)
        except Exception as e:
            print(f"    Model selection chart failed: {e}")

    if ff5:
        try:
            figures["FF5 Factor Loadings by Regime"] = make_ff5_coeff_chart(ff5.coefficient_comparison)
        except Exception as e:
            print(f"    FF5 coeff chart failed: {e}")
        try:
            figures["FF5 Factor Premia by Regime"] = make_factor_premia_chart(ff5.factor_premia_comparison)
        except Exception as e:
            print(f"    Factor premia chart failed: {e}")

    if cw:
        try:
            figures["Culture War Stock Alpha Distribution"] = make_cw_alpha_chart(cw.summary)
        except Exception as e:
            print(f"    CW alpha chart failed: {e}")

    if matched:
        try:
            figures["Delta Beta Box Plots (Matched)"] = make_delta_beta_chart(matched.delta_betas)
        except Exception as e:
            print(f"    Delta beta chart failed: {e}")
        if not matched.paired_ttest.empty:
            try:
                figures["Paired T-Test t-Statistics"] = make_ttest_summary_chart(matched.paired_ttest)
            except Exception as e:
                print(f"    T-test chart failed: {e}")
        if not matched.regime_amplification.empty:
            try:
                figures["Regime Amplification"] = make_regime_amplification_chart(matched.regime_amplification)
            except Exception as e:
                print(f"    Amplification chart failed: {e}")

    print("  Adding figures sheet...")
    build_figures_sheet(wb, figures)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = ROOT / "essay1_full_results.xlsx"
    print(f"\nSaving workbook to {out_path}...")
    wb.save(out_path)
    print(f"Done! Workbook saved: {out_path}")
    print(f"Sheets: {[ws.title for ws in wb.worksheets]}")

    ds.close()
    return out_path


if __name__ == '__main__':
    import os
    os.chdir(ROOT)
    main()
