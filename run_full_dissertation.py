"""Run the complete dissertation pipeline and export to a master Excel workbook.

Pulls all pre-computed essay results from SQLite, runs the culture factor
pipeline, runs orthogonal integration, and writes everything into one
comprehensive workbook.

Usage:
    python run_full_dissertation.py

Output:
    dissertation_results.xlsx — master workbook with all three essays +
    culture factor + orthogonalization + systematic risk tests
"""

import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Let statsmodels ConvergenceWarning through (Markov regime-switching etc.)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings('default', category=ConvergenceWarning)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Styles ────────────────────────────────────────────────────────────
HDR_FILL = PatternFill('solid', fgColor='1F4E79')
HDR_FONT = Font(bold=True, color='FFFFFF', size=10)
TITLE_FONT = Font(bold=True, size=14)
SUBTITLE_FONT = Font(bold=True, size=12, italic=True, color='1F4E79')
NOTE_FONT = Font(italic=True, size=9, color='555555')
GREEN_FONT = Font(bold=True, color='1E8449')
RED_FONT = Font(bold=True, color='C0392B')
ALT_FILL = PatternFill('solid', fgColor='F2F7FB')


def _write_table(ws, df, start_row=1, title=None, subtitle=None, note=None):
    """Write a DataFrame to worksheet with formatted headers. Returns next free row."""
    row = start_row
    if title:
        ws.cell(row, 1, title).font = TITLE_FONT
        row += 1
    if subtitle:
        ws.cell(row, 1, subtitle).font = SUBTITLE_FONT
        row += 1
    if note:
        ws.cell(row, 1, note).font = NOTE_FONT
        row += 1
    if title or subtitle or note:
        row += 1

    if df is None or df.empty:
        ws.cell(row, 1, '(no data)').font = NOTE_FONT
        return row + 2

    # Headers
    for c, col in enumerate(df.columns, 1):
        cell = ws.cell(row, c, str(col))
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal='center', wrap_text=True)
    row += 1

    # Data
    for i, (_, data_row) in enumerate(df.iterrows()):
        for c, val in enumerate(data_row, 1):
            cell = ws.cell(row + i, c)
            if pd.isna(val):
                cell.value = ''
            elif isinstance(val, (float, np.floating)):
                cell.value = float(val)
                col_name = str(df.columns[c - 1]).upper()
                if 'P_VAL' in col_name or 'PVALUE' in col_name or col_name == 'P':
                    cell.number_format = '0.0000'
                elif 'R_SQ' in col_name or 'R2' in col_name:
                    cell.number_format = '0.0000'
                elif 'BPS' in col_name:
                    cell.number_format = '0.0'
                elif 'T_STAT' in col_name or 'T_' in col_name:
                    cell.number_format = '0.00'
                elif 'BETA' in col_name or 'COEF' in col_name or 'GAMMA' in col_name:
                    cell.number_format = '0.000000'
                elif 'CAR' in col_name or 'RETURN' in col_name:
                    cell.number_format = '0.0000'
                else:
                    cell.number_format = '0.0000'
            else:
                cell.value = val
            if i % 2 == 1:
                cell.fill = ALT_FILL

    end_row = row + len(df)

    # Auto-width
    for c in range(1, len(df.columns) + 1):
        col_letter = get_column_letter(c)
        max_len = max(
            len(str(df.columns[c - 1])),
            df.iloc[:, c - 1].astype(str).str.len().max() if len(df) > 0 else 0
        )
        ws.column_dimensions[col_letter].width = min(max_len + 3, 35)

    return end_row + 2


def _read_table(conn, table_name):
    """Safely read a table from SQLite."""
    try:
        return pd.read_sql(f'SELECT * FROM [{table_name}]', conn)
    except Exception:
        return pd.DataFrame()


def run_full_pipeline(output_path='dissertation_results.xlsx'):
    """Run everything and write master workbook."""
    import sqlite3

    start = datetime.now()
    print("=" * 70)
    print("  DISSERTATION PIPELINE — FULL RUN")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    db_path = 'data/signals_systems.db'
    conn = sqlite3.connect(db_path)

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # ══════════════════════════════════════════════════════════════════
    # COVER SHEET
    # ══════════════════════════════════════════════════════════════════
    ws = wb.create_sheet('Cover')
    ws.cell(1, 1, 'Extending the Fama-French Asset Pricing Model').font = Font(bold=True, size=18)
    ws.cell(2, 1, 'Integrating Behavioral, Structural, and Technological Factors').font = Font(bold=True, size=14, color='1F4E79')
    ws.cell(3, 1, 'Using Econometrics and Machine Learning').font = Font(bold=True, size=14, color='1F4E79')
    ws.cell(5, 1, 'Ashley Roseboro').font = Font(size=12)
    ws.cell(6, 1, f'Generated: {start.strftime("%Y-%m-%d %H:%M")}').font = NOTE_FONT
    ws.cell(8, 1, 'Contents:').font = SUBTITLE_FONT

    toc = [
        'Essay 1 — Volatility Regimes & FF5 Factor Premia',
        'Essay 2 — Sociopolitical Controversies, Sentiment & DiD',
        'Essay 3 — Informed Insider Trading & Political Decisions',
        'Culture Factor — Firm Political Stance (Li et al. 2021 technique)',
        'Orthogonal Integration — Baker-Wurgler (2006) Purging + Pagan (1984) Adjustment',
        'Systematic Risk — CMF Spanning, Fama-MacBeth, Regime Alpha',
    ]
    for i, item in enumerate(toc):
        ws.cell(9 + i, 1, f'  {i+1}. {item}').font = Font(size=11)

    ws.column_dimensions['A'].width = 70

    # ══════════════════════════════════════════════════════════════════
    # ESSAY 1
    # ══════════════════════════════════════════════════════════════════
    print("Essay 1: Volatility Regimes & FF5...")

    essay1_tables = {
        'E1 Regime Summary': ('ESSAY1_REGIME_SUMMARY', 'Markov Regime-Switching Model Summary',
                              '3-state model on VIX (Hamilton 1989)'),
        'E1 Model Selection': ('ESSAY1_MODEL_SELECTION', 'Model Selection (AIC/BIC/LR)',
                               'Comparing 2-state vs 3-state models'),
        'E1 Transition Matrix': ('ESSAY1_TRANSITION_MATRIX', 'Regime Transition Probabilities', None),
        'E1 Factor Premia': ('ESSAY1_FACTOR_PREMIA', 'FF5 Factor Premia by Regime',
                             'Daily premia (HAC standard errors)'),
        'E1 FF5 Coefficients': ('ESSAY1_FF5_COEFFICIENTS', 'FF5 Coefficients by Regime', None),
        'E1 CW Stock Results': ('ESSAY1_CW_STOCK_RESULTS', 'Culture War Stock-Level FF5 Results',
                                'Individual firm alphas and betas by regime'),
        'E1 CW Regime Agg': ('ESSAY1_CW_REGIME_AGGREGATES', 'CW Stock Aggregates by Regime', None),
        'E1 Interaction': ('ESSAY1_INTERACTION', 'Regime × Factor Interaction Tests', None),
        'E1 Chow Test': ('ESSAY1_CHOW_TEST', 'Structural Break (Chow) Test', None),
        'E1 Matched Deltas': ('ESSAY1_MATCHED_DELTAS', 'Matched Control: Treatment - Control Deltas', None),
        'E1 Matched Sign': ('ESSAY1_MATCHED_SIGN', 'Matched Control: Sign Test Results', None),
        'E1 Matched Amplify': ('ESSAY1_MATCHED_AMPLIFICATION', 'Matched Control: Amplification by Regime', None),
    }

    for sheet_name, (table, title, subtitle) in essay1_tables.items():
        df = _read_table(conn, table)
        ws = wb.create_sheet(sheet_name[:31])
        _write_table(ws, df, title=title, subtitle=subtitle)
        if not df.empty:
            print(f"  {sheet_name}: {len(df)} rows")

    # ══════════════════════════════════════════════════════════════════
    # ESSAY 2
    # ══════════════════════════════════════════════════════════════════
    print()
    print("Essay 2: Sociopolitical Controversies & Sentiment...")

    essay2_tables = {
        'E2 Descriptive Stats': ('ESSAY2_DESCRIPTIVE_STATS', 'Descriptive Statistics', None),
        'E2 Event NLP': ('ESSAY2_EVENT_NLP', 'Event-Level NLP Sentiment Scores',
                         'FinBERT sentiment on news + filings'),
        'E2 Multi-Window': ('ESSAY2_MULTI_WINDOW_SUMMARY', 'Multi-Window CAR Analysis',
                            'Treatment vs control CARs across windows'),
        'E2 Treat vs Ctrl': ('ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL', 'Treatment vs Control by Window', None),
        'E2 By Leaning': ('ESSAY2_MULTI_WINDOW_BY_LEAN', 'CARs by Political Leaning',
                          'Liberal vs Conservative vs Mixed firms'),
        'E2 DiD Coefficients': ('ESSAY2_DID_COEFFICIENTS', 'Difference-in-Differences Coefficients',
                                'Dynamic treatment effects (TWFE)'),
        'E2 Cross-Section Reg': ('ESSAY2_CROSS_SECTIONAL_REG', 'Cross-Sectional CAR Regressions', None),
        'E2 Parallel Trends': ('ESSAY2_PARALLEL_TRENDS', 'Parallel Trends Test (Pre-Event)', None),
        'E2 Contagion Summary': ('ESSAY2_CONTAGION_SUMMARY', 'Industry Contagion Effects', None),
        'E2 Contagion By Lean': ('ESSAY2_CONTAGION_BY_LEAN', 'Contagion by Political Leaning', None),
        'E2 Political Align': ('ESSAY2_POLITICAL_ALIGNMENT', 'Political Alignment Classification', None),
        'E2 Filing Sentiment': ('ESSAY2_FILING_SENTIMENT', 'SEC Filing Sentiment (FinBERT)',
                                '10-K/10-Q MD&A and Risk Factors sentiment'),
        'E2 Placebo Tests': ('ESSAY2_DIAG_PLACEBO', 'Placebo Event Tests', None),
        'E2 Bootstrap CI': ('ESSAY2_DIAG_BOOTSTRAP_CI', 'Bootstrap Confidence Intervals', None),
        'E2 Cluster Robust': ('ESSAY2_DIAG_CLUSTER_ROBUST', 'Cluster-Robust Standard Errors', None),
    }

    for sheet_name, (table, title, subtitle) in essay2_tables.items():
        df = _read_table(conn, table)
        ws = wb.create_sheet(sheet_name[:31])
        _write_table(ws, df, title=title, subtitle=subtitle)
        if not df.empty:
            print(f"  {sheet_name}: {len(df)} rows")

    # ══════════════════════════════════════════════════════════════════
    # ESSAY 3
    # ══════════════════════════════════════════════════════════════════
    print()
    print("Essay 3: Informed Insider Trading...")

    essay3_tables = {
        'E3 Informed Trading': ('ESSAY3_INFORMED_TRADING', 'Informed Trading: Political vs Control',
                                'Directional accuracy across all cuts'),
        'E3 Proximity': ('ESSAY3_INFORMED_PROXIMITY', 'Trading by Proximity to Event',
                         '0-30d, 31-60d, 61-90d, 91-180d before event'),
        'E3 Informed Dollars': ('ESSAY3_INFORMED_DOLLARS', 'Informed Trading: Dollar Magnitudes', None),
        'E3 Wilcoxon Family': ('ESSAY3_WILCOXON_FAMILY', 'Family-Level Wilcoxon Tests',
                               'Holm + Benjamini-Hochberg corrections'),
        'E3 Mean vs Distrib': ('ESSAY3_MEAN_VS_DISTRIBUTIONAL', 'Mean vs Distributional Tests',
                               'T-test vs Wilcoxon side-by-side'),
        'E3 Insider Conc': ('ESSAY3_INSIDER_CONCENTRATION', 'Insider Concentration (Gini/HHI)', None),
        'E3 Quantile Reg': ('ESSAY3_QUANTILE_REGRESSION', 'Quantile Regression Results', None),
        'E3 Size Accuracy': ('ESSAY3_SIZE_ACCURACY', 'Trade Size vs Accuracy',
                             'CG attenuation: large trades = less accurate?'),
        'E3 Cluster Inference': ('ESSAY3_CLUSTER_INFERENCE', 'Cluster-Robust Inference', None),
        'E3 Stratification': ('ESSAY3_STRATIFICATION', 'Year × Activity Tercile Strata', None),
        'E3 Price Discovery': ('ESSAY3_PRICE_DISCOVERY', 'Price Discovery Contribution', None),
        'E3 Post-Event Reg': ('ESSAY3_POST_EVENT_REG', 'Post-Event Trading Regression', None),
        'E3 Conn Timing': ('ESSAY3_CONN_TIMING', 'Connection Timing Analysis', None),
        'E3 Reversal Reg': ('ESSAY3_REVERSAL_REGRESSION', 'Reversal Regression', None),
        'E3 Bootstrap': ('ESSAY3_BOOTSTRAP_CI', 'Bootstrap Confidence Intervals', None),
        'E3 Trimmed Robust': ('ESSAY3_TRIMMED_ROBUSTNESS', 'Trimmed Robustness Tests', None),
        'E3 CAR Decomp': ('ESSAY3_CAR_DECOMP', 'CAR Decomposition by Political Channel', None),
        'E3 CRSP Summary': ('ESSAY3_CRSP_SUMMARY', 'CRSP-Validated Profit Summary', None),
    }

    for sheet_name, (table, title, subtitle) in essay3_tables.items():
        df = _read_table(conn, table)
        ws = wb.create_sheet(sheet_name[:31])
        _write_table(ws, df, title=title, subtitle=subtitle)
        if not df.empty:
            print(f"  {sheet_name}: {len(df)} rows")

    conn.close()

    # ══════════════════════════════════════════════════════════════════
    # CULTURE FACTOR PIPELINE (NEW)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("Culture Factor: Running political stance scoring (Li et al. 2021 technique)...")

    from model.datastore import DataStore
    store = DataStore()

    # --- Stance scoring ---
    from clean.word2vec_stance import StanceScorer
    scorer = StanceScorer(
        platform_dir='./party_platforms_data',
        seed_topn=100,
        expand_topn=50,
        expand_threshold=0.3,
    )
    stance_scores = scorer.run()
    seed_summary = scorer.get_seed_summary()
    print(f"  Stance scores: {len(stance_scores)} firm-year obs")

    # --- Alignment ---
    from clean.orthogonalize import build_culture_index
    events_df = store.events.copy()
    culture_index = pd.Series(dtype=float)
    if not events_df.empty:
        events_df.columns = [c.upper() for c in events_df.columns]
        col_map = {}
        for c in events_df.columns:
            if 'EVENT' in c and 'DATE' in c:
                col_map[c] = 'Event Date'
            if 'ESTIMATED' in c and 'POLITICAL' in c:
                col_map[c] = 'Estimated Political Leaning'
            if 'POLITICAL' in c and 'LEAN' in c and 'JUSTIF' not in c and 'ESTIMATED' not in c:
                if 'Estimated Political Leaning' not in col_map.values():
                    col_map[c] = 'Estimated Political Leaning'
        events_df = events_df.rename(columns=col_map)
        if 'Event Date' in events_df.columns:
            events_df['Event Date'] = pd.to_datetime(events_df['Event Date'], errors='coerce')
        try:
            culture_index = build_culture_index(events_df, freq='M', method='net_score')
        except Exception as e:
            logger.warning("Culture index build failed: %s", e)

    from model.culture_factor import (
        compute_alignment, orthogonalize_stance_vs_spending,
        build_cmf, spanning_test, fama_macbeth,
        regime_spanning_test, value_impact_regression,
    )

    if not culture_index.empty:
        alignment_scores = compute_alignment(stance_scores, culture_index)
    else:
        alignment_scores = stance_scores.copy()
        alignment_scores['ALIGNMENT'] = 0.0
        alignment_scores['MISALIGNMENT'] = 0.0
        alignment_scores['DELIBERATE_STANCE'] = alignment_scores['STANCE_SCORE']
        alignment_scores['SURROUNDING_CULTURE'] = 0.0

    # --- Stance vs spending ---
    if not store.political_exposure.empty:
        decomposed = orthogonalize_stance_vs_spending(alignment_scores, store.political_exposure)
    else:
        decomposed = alignment_scores.copy()
        decomposed['STANCE_CULTURE'] = decomposed['STANCE_SCORE']
        decomposed['STANCE_SPENDING'] = 0.0
        decomposed['SPENDING_R2'] = 0.0

    # --- CMF ---
    cmf = build_cmf(decomposed, store.stock_returns, n_quantiles=3)
    print(f"  CMF: {len(cmf)} trading days")

    # --- Spanning ---
    spanning_result = None
    spanning_df = pd.DataFrame()
    if not cmf.empty and not store.ff5.empty:
        spanning_result = spanning_test(cmf, store.ff5)
        spanning_df = spanning_result.factor_loadings

    # --- Fama-MacBeth ---
    fm_result = None
    fm_df = pd.DataFrame()
    if not store.stock_returns.empty and not store.ff5.empty and not cmf.empty:
        fm_result = fama_macbeth(store.stock_returns, store.ff5, cmf, decomposed)
        fm_df = fm_result.all_gammas

    # --- Regime alpha ---
    regime_df = pd.DataFrame()
    if not cmf.empty and not store.ff5.empty and not store.vix.empty:
        regime_df = regime_spanning_test(cmf, store.ff5, store.vix)

    # --- Value impact ---
    value_df = pd.DataFrame()
    if not decomposed.empty and not store.stock_returns.empty:
        value_df = value_impact_regression(decomposed, store.stock_returns)

    # --- Orthogonal integration (all 6 Hu design items) ---
    print()
    print("Orthogonal Integration: Culture vs Politics decomposition...")
    ortho_int = None
    ortho_sheets = {}
    try:
        from model.orthogonal_integration import run_orthogonal_integration
        ortho_int = run_orthogonal_integration(store)
        # Collect results
        if not ortho_int.essay1_results.empty:
            ortho_sheets['Ortho E1 Results'] = ortho_int.essay1_results
        if not ortho_int.essay2_results.empty:
            ortho_sheets['Ortho E2 Results'] = ortho_int.essay2_results
        if not ortho_int.essay3_results.empty:
            ortho_sheets['Ortho E3 Results'] = ortho_int.essay3_results
    except Exception as e:
        logger.warning("Orthogonal integration failed: %s", e)

    store.close()

    # ══════════════════════════════════════════════════════════════════
    # WRITE CULTURE FACTOR SHEETS
    # ══════════════════════════════════════════════════════════════════
    print()
    print("Writing culture factor sheets...")

    # CF Stance Scores
    ws = wb.create_sheet('CF Stance Scores')
    cols = ['TICKER', 'YEAR', 'FILING_DATE', 'L_COUNT', 'C_COUNT',
            'TOTAL_TOKENS', 'L_FREQ', 'C_FREQ', 'STANCE_SCORE', 'STANCE_LABEL']
    _write_table(ws, stance_scores[[c for c in cols if c in stance_scores.columns]],
                 title='Firm Political Stance Scores',
                 subtitle='Li et al. (2021) seed-expand-score technique applied to political stance measurement',
                 note='Li et al. (2021) developed this technique for corporate culture (integrity, teamwork); '
                      'here we apply the same NLP pipeline to political language extracted from party platforms (2004-2024)')

    # CF Seeds
    ws = wb.create_sheet('CF Seed Dictionaries')
    _write_table(ws, seed_summary,
                 title='Seed & Expanded Dictionary Summary',
                 subtitle='TF-IDF log-odds differentiation of Democratic vs Republican platforms')

    # CF Alignment
    ws = wb.create_sheet('CF Alignment')
    align_cols = ['TICKER', 'YEAR', 'STANCE_SCORE', 'SURROUNDING_CULTURE',
                  'ALIGNMENT', 'MISALIGNMENT', 'DELIBERATE_STANCE', 'STANCE_LABEL']
    _write_table(ws, decomposed[[c for c in align_cols if c in decomposed.columns]],
                 title='Firm-Culture Alignment Scores',
                 subtitle='Alignment = stance × surrounding culture; Deliberate = residual from regressing stance on environment')

    # CF Stance vs Spending
    ws = wb.create_sheet('CF Stance vs Spending')
    spend_cols = ['TICKER', 'YEAR', 'STANCE_SCORE', 'STANCE_SPENDING',
                  'STANCE_CULTURE', 'SPENDING_R2']
    _write_table(ws, decomposed[[c for c in spend_cols if c in decomposed.columns]],
                 title='Stance Decomposition: Spending vs Culture',
                 subtitle='STANCE = f(PAC + Lobbying) + culture_residual')

    # CF CMF
    if not cmf.empty:
        ws = wb.create_sheet('CF CMF Returns')
        _write_table(ws, cmf,
                     title='Culture Misalignment Factor (CMF) Daily Returns',
                     subtitle='Long aligned, short misaligned, equal-weighted terciles, annual rebalance')

    # CF Spanning
    if not spanning_df.empty:
        ws = wb.create_sheet('CF Spanning Test')
        _write_table(ws, spanning_df,
                     title='CMF Spanning Test: CMF = α + β\'FF5 + ε',
                     subtitle='H0: α = 0 (CMF is spanned by FF5)')
        # Add summary
        r = ws.max_row + 2
        ws.cell(r, 1, 'Annualized Alpha (bps):').font = Font(bold=True)
        ws.cell(r, 2, spanning_result.annualized_alpha_bps).number_format = '0.0'
        ws.cell(r + 1, 1, 'Alpha t-statistic:').font = Font(bold=True)
        ws.cell(r + 1, 2, spanning_result.alpha_t).number_format = '0.00'
        ws.cell(r + 2, 1, 'Alpha p-value:').font = Font(bold=True)
        ws.cell(r + 2, 2, spanning_result.alpha_p).number_format = '0.0000'
        ws.cell(r + 3, 1, 'R-squared:').font = Font(bold=True)
        ws.cell(r + 3, 2, spanning_result.r_squared).number_format = '0.0000'
        ws.cell(r + 4, 1, 'Conclusion:').font = Font(bold=True)
        if spanning_result.alpha_p < 0.05:
            ws.cell(r + 4, 2, 'SYSTEMATIC: CMF not spanned by FF5').font = GREEN_FONT
        elif spanning_result.alpha_p < 0.10:
            ws.cell(r + 4, 2, 'MARGINAL: Weakly significant alpha').font = Font(bold=True, color='E67E22')
        else:
            ws.cell(r + 4, 2, 'NOT SYSTEMATIC: CMF appears spanned by FF5').font = RED_FONT

    # CF Fama-MacBeth
    if not fm_df.empty:
        ws = wb.create_sheet('CF Fama-MacBeth')
        _write_table(ws, fm_df,
                     title='Fama-MacBeth (1973) Cross-Sectional Pricing Test',
                     subtitle='Two-pass: rolling TS betas → monthly XS regressions, Newey-West (1987) SEs')
        r = ws.max_row + 2
        ws.cell(r, 1, 'Mean γ(stance):').font = Font(bold=True)
        ws.cell(r, 2, fm_result.gamma_cmf).number_format = '0.000000'
        ws.cell(r + 1, 1, 't-statistic:').font = Font(bold=True)
        ws.cell(r + 1, 2, fm_result.gamma_cmf_t).number_format = '0.00'
        ws.cell(r + 2, 1, 'p-value:').font = Font(bold=True)
        ws.cell(r + 2, 2, fm_result.gamma_cmf_p).number_format = '0.0000'
        ws.cell(r + 3, 1, 'N months:').font = Font(bold=True)
        ws.cell(r + 3, 2, fm_result.n_months)
        ws.cell(r + 4, 1, 'Avg stocks/month:').font = Font(bold=True)
        ws.cell(r + 4, 2, fm_result.n_stocks_avg).number_format = '0'
        ws.cell(r + 5, 1, 'Conclusion:').font = Font(bold=True)
        if fm_result.gamma_cmf_p < 0.05:
            ws.cell(r + 5, 2, 'PRICED: Market compensates for political stance').font = GREEN_FONT
        elif fm_result.gamma_cmf_p < 0.10:
            ws.cell(r + 5, 2, 'MARGINAL: Weakly priced (p < 0.10)').font = Font(bold=True, color='E67E22')
        else:
            ws.cell(r + 5, 2, 'NOT PRICED: Stance not compensated cross-sectionally').font = RED_FONT

    # CF Regime Alpha
    if not regime_df.empty:
        ws = wb.create_sheet('CF Regime Alpha')
        _write_table(ws, regime_df,
                     title='CMF Alpha by VIX Regime',
                     subtitle='Does culture factor premium vary with market stress?')

    # CF Value Impact
    if not value_df.empty:
        ws = wb.create_sheet('CF Value Impact')
        _write_table(ws, value_df,
                     title='Value Impact: Forward Return Regressions',
                     subtitle='FWD_RETURN(window) = α + β × STANCE_VARIABLE + ε')

    # ══════════════════════════════════════════════════════════════════
    # ORTHOGONAL INTEGRATION SHEETS
    # ══════════════════════════════════════════════════════════════════
    for sheet_name, df in ortho_sheets.items():
        ws = wb.create_sheet(sheet_name[:31])
        _write_table(ws, df,
                     title=f'Orthogonal Integration: {sheet_name.replace("Ortho ", "")}',
                     subtitle='Baker-Wurgler (2006) purging: does culture survive after controlling for politics?')
        print(f"  {sheet_name}: {len(df)} rows")

    # ══════════════════════════════════════════════════════════════════
    # HU DESIGN ITEMS (6 new sheets)
    # ══════════════════════════════════════════════════════════════════
    if ortho_int is not None:
        print()
        print("Writing Hu design sheets...")

        # --- Item 1: First-stage R² prominently ---
        ws = wb.create_sheet('Ortho First Stage')
        ws.cell(1, 1, 'FIRST-STAGE ORTHOGONALIZATION RESULTS').font = Font(bold=True, size=16)
        ws.cell(2, 1, 'Baker-Wurgler (2006) Purging: Culture = α + β\'Political + ε').font = SUBTITLE_FONT
        ws.cell(3, 1, 'The residual ε is CULTURE_ORTHO — the component orthogonal to politics').font = NOTE_FONT
        ws.column_dimensions['A'].width = 40
        ws.column_dimensions['B'].width = 30
        r = 5
        # Large prominent R² display
        ws.cell(r, 1, 'First-Stage R²').font = Font(bold=True, size=14)
        ws.cell(r, 2, ortho_int.first_stage_r2).font = Font(bold=True, size=14, color='C0392B')
        ws.cell(r, 2).number_format = '0.0000'
        r += 1
        ws.cell(r, 1, 'Adjusted R²').font = Font(bold=True, size=12)
        ws.cell(r, 2, ortho_int.first_stage_adj_r2).number_format = '0.0000'
        r += 1
        ws.cell(r, 1, '% of culture variance explained by politics').font = Font(bold=True)
        ws.cell(r, 2, f'{ortho_int.first_stage_r2*100:.1f}%')
        r += 1
        ws.cell(r, 1, '% surviving orthogonalization').font = Font(bold=True)
        ws.cell(r, 2, f'{(1-ortho_int.first_stage_r2)*100:.1f}%').font = GREEN_FONT
        r += 1
        ws.cell(r, 1, 'Estimation method').font = Font(bold=True)
        ws.cell(r, 2, ortho_int.estimation_method)
        r += 1
        ws.cell(r, 1, 'N political proxies').font = Font(bold=True)
        ws.cell(r, 2, ortho_int.n_political_proxies)
        r += 2

        # Coefficient table
        if not ortho_int.first_stage_coefficients.empty:
            _write_table(ws, ortho_int.first_stage_coefficients, start_row=r,
                        title='First-Stage Coefficients',
                        subtitle='Culture_t = α + β₁PRES_DEM + β₂SENATE_DEM + β₃HOUSE_DEM + β₄UNIFIED_GOV + [β₅EPU] + ε_t',
                        note='HAC (Newey-West) standard errors')
        print(f"  Ortho First Stage: R²={ortho_int.first_stage_r2:.4f}")

        # --- Item 2: Rolling window results ---
        if ortho_int.rolling_result is not None:
            ws = wb.create_sheet('Ortho Rolling Window')
            rolling_r = ortho_int.rolling_result
            ws.cell(1, 1, 'ROLLING-WINDOW ORTHOGONALIZATION (Pagan 1984)').font = Font(bold=True, size=14)
            ws.cell(2, 1, 'Out-of-sample residuals avoid look-ahead bias in predictive tests').font = SUBTITLE_FONT
            ws.cell(3, 1, f'Window: {rolling_r.window_size or 60} months').font = NOTE_FONT
            ws.column_dimensions['A'].width = 40
            ws.column_dimensions['B'].width = 30
            r = 5
            ws.cell(r, 1, 'Full-sample R² (for comparison)').font = Font(bold=True)
            ws.cell(r, 2, rolling_r.first_stage_r2).number_format = '0.0000'
            r += 1
            n_valid = rolling_r.diagnostics.get('valid_residuals', 0)
            n_total = rolling_r.diagnostics.get('total_obs', 0)
            ws.cell(r, 1, 'Valid out-of-sample residuals').font = Font(bold=True)
            ws.cell(r, 2, f'{n_valid}/{n_total}')
            r += 2
            if not rolling_r.first_stage_coefficients.empty:
                _write_table(ws, rolling_r.first_stage_coefficients, start_row=r,
                            title='Full-Sample Coefficients (reference)',
                            note='Rolling estimation uses these coefficients recursively with expanding data')
            # Also write the culture factor time series
            if not ortho_int.culture_factors.empty:
                cf = ortho_int.culture_factors.reset_index()
                ts_row = ws.max_row + 3
                cols = ['DATE', 'CULTURE_RAW', 'CULTURE_ORTHO']
                if 'CULTURE_ORTHO_ROLLING' in cf.columns:
                    cols.append('CULTURE_ORTHO_ROLLING')
                _write_table(ws, cf[cols].head(50), start_row=ts_row,
                            title='Culture Factor Time Series (first 50 months)',
                            note='Full series in Ortho Time Series sheet')
            print(f"  Ortho Rolling Window: {n_valid} valid residuals")

        # Full time series sheet
        if not ortho_int.culture_factors.empty:
            ws = wb.create_sheet('Ortho Time Series')
            cf = ortho_int.culture_factors.reset_index()
            _write_table(ws, cf,
                        title='Culture Factor Time Series',
                        subtitle='Full-sample and rolling-window orthogonalized factors',
                        note='CULTURE_ORTHO = full-sample residual; CULTURE_ORTHO_ROLLING = out-of-sample (Pagan 1984)')

        # --- Item 3: Pagan (1984) adjusted SEs ---
        if not ortho_int.pagan_adjustment.empty:
            ws = wb.create_sheet('Ortho Pagan SEs')
            _write_table(ws, ortho_int.pagan_adjustment,
                        title='Pagan (1984) Generated-Regressor SE Adjustment',
                        subtitle='Bootstrap comparison: OLS vs joint-estimation standard errors',
                        note='SE_RATIO > 1 means OLS understates uncertainty from first-stage estimation. '
                             '500 bootstrap iterations resampling both stages jointly.')
            print(f"  Ortho Pagan SEs: {len(ortho_int.pagan_adjustment)} variables")

        # --- Item 4: Time-series plot embedded in Excel ---
        if ortho_int.plot_path and os.path.exists(ortho_int.plot_path):
            ws = wb.create_sheet('Ortho Plot')
            ws.cell(1, 1, 'RAW vs ORTHOGONALIZED CULTURE FACTOR').font = Font(bold=True, size=14)
            ws.cell(2, 1, 'Panel A: Raw (confounded), Panel B: Orthogonalized, Panel C: Overlay').font = NOTE_FONT
            try:
                img = XlImage(ortho_int.plot_path)
                img.width = 900
                img.height = 640
                ws.add_image(img, 'A4')
                print(f"  Ortho Plot: embedded from {ortho_int.plot_path}")
            except Exception as e:
                ws.cell(4, 1, f'Plot saved to: {ortho_int.plot_path}').font = NOTE_FONT
                logger.warning("Could not embed plot: %s", e)

        # --- Item 5: Predictive regressions ---
        if not ortho_int.predictive_regressions.empty:
            ws = wb.create_sheet('Ortho Predictive Reg')
            _write_table(ws, ortho_int.predictive_regressions,
                        title='Predictive Regressions: Forward Returns on Purged Factor',
                        subtitle='R_{t+h} = α + β × CULTURE_ORTHO_t + ε — Newey-West SEs (lags ≥ h)',
                        note='Tests whether the purged culture factor predicts future market returns. '
                             'Rolling version avoids look-ahead bias per Pagan (1984).')
            print(f"  Ortho Predictive Reg: {len(ortho_int.predictive_regressions)} specs")

        # --- Item 6: Fama-MacBeth on purged factor ---
        if not ortho_int.fm_purged.empty:
            ws = wb.create_sheet('Ortho FM Purged')
            _write_table(ws, ortho_int.fm_purged,
                        title='Fama-MacBeth (1973) on Firm Culture Exposure × Purged Factor',
                        subtitle='Monthly XS: R_i,t = γ₀ + γ₁ × (EVENT_INTENSITY_i × CULTURE_ORTHO_t) + ε — NW SEs on γ₁',
                        note='EVENT_INTENSITY = normalized firm event count (cross-sectional variation). '
                             'CULTURE_ORTHO = aggregate purged factor (time-series variation). '
                             'Last row = time-series mean with Newey-West standard errors (Andrews 1991 lag).')
            # Add summary box
            summary = ortho_int.fm_purged[ortho_int.fm_purged['MONTH'] == 'MEAN (NW)']
            if not summary.empty:
                s = summary.iloc[0]
                r = ws.max_row + 2
                gamma_col = 'GAMMA_EXPOSURE' if 'GAMMA_EXPOSURE' in s.index else 'GAMMA_ORTHO'
                ws.cell(r, 1, f'Mean γ({gamma_col}):').font = Font(bold=True)
                ws.cell(r, 2, s[gamma_col]).number_format = '0.000000'
                if pd.notna(s.get('T_STAT')):
                    ws.cell(r + 1, 1, 't-statistic:').font = Font(bold=True)
                    ws.cell(r + 1, 2, s['T_STAT']).number_format = '0.00'
                    ws.cell(r + 2, 1, 'p-value:').font = Font(bold=True)
                    ws.cell(r + 2, 2, s['P_VALUE']).number_format = '0.0000'
                    ws.cell(r + 3, 1, 'Conclusion:').font = Font(bold=True)
                    if s['P_VALUE'] < 0.05:
                        ws.cell(r + 3, 2, 'PRICED: Purged culture factor compensated cross-sectionally').font = GREEN_FONT
                    elif s['P_VALUE'] < 0.10:
                        ws.cell(r + 3, 2, 'MARGINAL: Weakly priced (p < 0.10)').font = Font(bold=True, color='E67E22')
                    else:
                        ws.cell(r + 3, 2, 'NOT PRICED: Purged factor not significant').font = RED_FONT
            print(f"  Ortho FM Purged: {len(ortho_int.fm_purged) - 1} months")

        # --- Item 7: Decision-level test ---
        if not ortho_int.decision_level.empty:
            ws = wb.create_sheet('Ortho Decision Level')
            _write_table(ws, ortho_int.decision_level,
                        title='Decision-Level Test: Event Direction vs Prior Firm Identity',
                        subtitle='CAR = α + β₁·DECISION_DIR + β₂·FIRM_IDENTITY + β₃·CONGRUENCE + ε  (HC1 SEs)',
                        note='DECISION_DIR: event political direction (Lib=+1, Con=-1). '
                             'FIRM_IDENTITY: pre-event political alignment score (Essay 2 text-based). '
                             'CONGRUENCE: DIR × IDENTITY (positive when event aligns with firm). '
                             'If β₁ significant in full model → market reacts to event politics, not just firm identity.')
            # Highlight the key result
            full = ortho_int.decision_level[ortho_int.decision_level['SPEC'] == 'Full model']
            if not full.empty and 'DECISION_DIR_P' in full.columns:
                r = ws.max_row + 2
                best = full.loc[full['DECISION_DIR_P'].idxmin()]
                ws.cell(r, 1, 'Key question:').font = Font(bold=True)
                ws.cell(r, 2, 'Does event direction explain CARs beyond firm prior identity?')
                r += 1
                ws.cell(r, 1, f'Best window ({best["WINDOW"]}):').font = Font(bold=True)
                p = best['DECISION_DIR_P']
                b = best['DECISION_DIR_BETA']
                if p < 0.05:
                    ws.cell(r, 2, f'YES — β₁={b:+.4f}, p={p:.4f}').font = GREEN_FONT
                elif p < 0.10:
                    ws.cell(r, 2, f'MARGINAL — β₁={b:+.4f}, p={p:.4f}').font = Font(bold=True, color='E67E22')
                else:
                    ws.cell(r, 2, f'NO — β₁={b:+.4f}, p={p:.4f}').font = RED_FONT
            print(f"  Ortho Decision Level: {len(ortho_int.decision_level)} specs")

    # ══════════════════════════════════════════════════════════════════
    # MASTER SUMMARY SHEET
    # ══════════════════════════════════════════════════════════════════
    ws = wb.create_sheet('Master Summary')
    ws.cell(1, 1, 'DISSERTATION RESULTS — MASTER SUMMARY').font = Font(bold=True, size=16)
    ws.cell(2, 1, f'Generated: {start.strftime("%Y-%m-%d %H:%M")}').font = NOTE_FONT
    ws.column_dimensions['A'].width = 45
    ws.column_dimensions['B'].width = 50

    row = 4
    sections = [
        ('ESSAY 1: Volatility Regimes & FF5', []),
        ('ESSAY 2: Sociopolitical Controversies', []),
        ('ESSAY 3: Informed Insider Trading', []),
        ('CULTURE FACTOR: Firm Political Stance', [
            ('Firm-year observations', len(stance_scores)),
            ('Unique tickers scored', stance_scores['TICKER'].nunique()),
            ('Mean stance score', f"{stance_scores['STANCE_SCORE'].mean():.4f}"),
            ('Liberal / Mixed / Conservative',
             f"{(stance_scores['STANCE_LABEL']=='Liberal').sum()} / "
             f"{(stance_scores['STANCE_LABEL']=='Mixed').sum()} / "
             f"{(stance_scores['STANCE_LABEL']=='Conservative').sum()}"),
            ('Deliberate stance variance',
             f"{alignment_scores['DELIBERATE_STANCE'].var():.4f}" if 'DELIBERATE_STANCE' in alignment_scores.columns else 'N/A'),
            ('Stance vs Spending R²',
             f"{decomposed['SPENDING_R2'].iloc[0]:.4f}" if 'SPENDING_R2' in decomposed and len(decomposed) > 0 else 'N/A'),
        ]),
        ('SYSTEMATIC RISK TESTS', [
            ('CMF trading days', len(cmf) if not cmf.empty else 0),
            ('CMF mean daily return (bps)',
             f"{cmf['CMF'].mean() * 10000:.2f}" if not cmf.empty else 'N/A'),
            ('Spanning alpha (ann bps)',
             f"{spanning_result.annualized_alpha_bps:.1f}" if spanning_result else 'N/A'),
            ('Spanning alpha p-value',
             f"{spanning_result.alpha_p:.4f}" if spanning_result else 'N/A'),
            ('Fama-MacBeth γ(stance)',
             f"{fm_result.gamma_cmf:.6f}" if fm_result else 'N/A'),
            ('Fama-MacBeth p-value',
             f"{fm_result.gamma_cmf_p:.4f}" if fm_result else 'N/A'),
        ]),
        ('ORTHOGONAL INTEGRATION (Hu Design)', [
            ('First-stage R²',
             f"{ortho_int.first_stage_r2:.4f} ({ortho_int.first_stage_r2*100:.1f}% political)"
             if ortho_int else 'N/A'),
            ('Estimation methods', 'Full-sample + Rolling (60-month)' if ortho_int else 'N/A'),
            ('Rolling valid residuals',
             f"{ortho_int.culture_factors['CULTURE_ORTHO_ROLLING'].notna().sum()}"
             if ortho_int and 'CULTURE_ORTHO_ROLLING' in getattr(ortho_int, 'culture_factors', pd.DataFrame()).columns else 'N/A'),
            ('Pagan (1984) SE adjustment',
             'Yes — bootstrap (500 iterations)' if ortho_int and not ortho_int.pagan_adjustment.empty else 'N/A'),
            ('Predictive regressions',
             f"{len(ortho_int.predictive_regressions)} specs" if ortho_int and not ortho_int.predictive_regressions.empty else 'N/A'),
            ('FM on purged factor',
             f"{len(ortho_int.fm_purged) - 1} months" if ortho_int and not ortho_int.fm_purged.empty else 'N/A'),
        ]),
        ('VALUE IMPACT (3-month forward returns)', []),
    ]

    # Add value impact summary
    if not value_df.empty:
        v3 = value_df[value_df['WINDOW_DAYS'] == 63]
        for _, r in v3.iterrows():
            sig = '***' if r['P_VALUE'] < 0.01 else '**' if r['P_VALUE'] < 0.05 else '*' if r['P_VALUE'] < 0.1 else ''
            sections[-1][1].append((
                r['VARIABLE'],
                f"β={r['BETA']:+.4f} (t={r['T_STAT']:+.2f}, p={r['P_VALUE']:.4f}) {sig}"
            ))

    for section_title, items in sections:
        ws.cell(row, 1, section_title).font = SUBTITLE_FONT
        ws.cell(row, 1).fill = PatternFill('solid', fgColor='D6E4F0')
        ws.cell(row, 2).fill = PatternFill('solid', fgColor='D6E4F0')
        row += 1
        for label, value in items:
            ws.cell(row, 1, label)
            ws.cell(row, 2, str(value))
            row += 1
        row += 1

    # Move Master Summary and Cover to front
    wb.move_sheet('Cover', offset=-len(wb.sheetnames) + 1)
    wb.move_sheet('Master Summary', offset=-len(wb.sheetnames) + 2)

    # Save
    print()
    print(f"Writing workbook to {output_path}...")
    wb.save(output_path)

    elapsed = datetime.now() - start
    n_sheets = len(wb.sheetnames)
    print()
    print("=" * 70)
    print(f"  COMPLETE — {elapsed.total_seconds():.0f}s")
    print(f"  {n_sheets} sheets written to {os.path.abspath(output_path)}")
    print("=" * 70)

    return output_path


if __name__ == '__main__':
    run_full_pipeline()
