"""Run the full Culture Misalignment Factor pipeline and export to Excel.

Usage:
    python run_culture_factor.py

Outputs:
    culture_factor_results.xlsx — multi-sheet workbook with:
      1. Stance Scores (firm-year)
      2. Seed Dictionaries
      3. Alignment Scores
      4. Stance vs Spending decomposition
      5. CMF Returns (daily)
      6. Spanning Test (FF5 alpha)
      7. Fama-MacBeth (cross-sectional pricing)
      8. Regime Alpha (conditional)
      9. Value Impact (forward returns)
     10. Summary
"""

import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def run_pipeline(output_path: str = 'culture_factor_results.xlsx'):
    """Run the full culture factor pipeline."""
    start_time = datetime.now()
    print("=" * 70)
    print("  CULTURE MISALIGNMENT FACTOR — FULL PIPELINE")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # ── Step 1: Load DataStore ────────────────────────────────────────
    print("Step 1: Loading research data...")
    from model.datastore import DataStore
    store = DataStore()
    print(f"  Backend: {store.backend}")
    print(f"  Events: {len(store.events)}")
    print(f"  Tickers: {store.stock_returns['TICKER'].nunique() if not store.stock_returns.empty else 0}")
    print()

    # ── Step 2: Score firm political stance ───────────────────────────
    print("Step 2: Scoring firm political stance from SEC filings...")
    from clean.word2vec_stance import StanceScorer

    scorer = StanceScorer(
        platform_dir='./party_platforms_data',
        seed_topn=100,
        expand_topn=50,
        expand_threshold=0.3,
    )
    stance_scores = scorer.run()
    seed_summary = scorer.get_seed_summary()
    print(f"  Scored {len(stance_scores)} firm-year observations")
    print()

    # ── Step 3: Compute alignment with surrounding culture ───────────
    print("Step 3: Computing alignment with surrounding culture...")
    from clean.orthogonalize import build_political_proxies, build_culture_index

    # Build surrounding culture from events
    events_df = store.events.copy()
    if not events_df.empty:
        events_df.columns = [c.upper() for c in events_df.columns]
        # Map columns for build_culture_index
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
            events_df['Event Date'] = pd.to_datetime(
                events_df['Event Date'], errors='coerce')

        try:
            culture_index = build_culture_index(events_df, freq='M', method='net_score')
        except Exception as e:
            logger.warning("Could not build culture index: %s", e)
            culture_index = pd.Series(dtype=float)
    else:
        culture_index = pd.Series(dtype=float)

    from model.culture_factor import compute_alignment
    if not culture_index.empty:
        alignment_scores = compute_alignment(stance_scores, culture_index)
    else:
        alignment_scores = stance_scores.copy()
        alignment_scores['ALIGNMENT'] = 0.0
        alignment_scores['MISALIGNMENT'] = 0.0
        alignment_scores['DELIBERATE_STANCE'] = alignment_scores['STANCE_SCORE']
        alignment_scores['SURROUNDING_CULTURE'] = 0.0
    print(f"  Alignment scores computed for {len(alignment_scores)} observations")
    print()

    # ── Step 4: Orthogonalize stance vs political spending ────────────
    print("Step 4: Decomposing stance vs political spending...")
    from model.culture_factor import orthogonalize_stance_vs_spending

    if not store.political_exposure.empty:
        decomposed = orthogonalize_stance_vs_spending(
            alignment_scores, store.political_exposure,
        )
    else:
        decomposed = alignment_scores.copy()
        decomposed['STANCE_CULTURE'] = decomposed['STANCE_SCORE']
        decomposed['STANCE_SPENDING'] = 0.0
        decomposed['SPENDING_R2'] = 0.0
        logger.info("No political exposure data — skipping spending decomposition")
    print()

    # ── Step 5: Build CMF (long-short portfolio) ──────────────────────
    print("Step 5: Constructing Culture Misalignment Factor...")
    from model.culture_factor import build_cmf

    cmf = build_cmf(
        decomposed, store.stock_returns,
        n_quantiles=3, weighting='equal',
    )
    if cmf.empty:
        print("  WARNING: CMF construction failed (insufficient overlap)")
        print("  Proceeding with available results...")
    else:
        print(f"  CMF: {len(cmf)} trading days")
        print(f"  Mean daily return: {cmf['CMF'].mean() * 10000:.2f} bps")
        ann_sharpe = (cmf['CMF'].mean() / cmf['CMF'].std() * np.sqrt(252)
                      if cmf['CMF'].std() > 0 else 0)
        print(f"  Annualized Sharpe: {ann_sharpe:.2f}")
    print()

    # ── Step 6: Spanning test ─────────────────────────────────────────
    spanning_result = None
    spanning_df = pd.DataFrame()
    if not cmf.empty and not store.ff5.empty:
        print("Step 6: Spanning test (CMF vs FF5)...")
        from model.culture_factor import spanning_test
        spanning_result = spanning_test(cmf, store.ff5)
        spanning_df = spanning_result.factor_loadings
        print(f"  Alpha: {spanning_result.annualized_alpha_bps:.1f} bps/year "
              f"(t={spanning_result.alpha_t:.2f}, p={spanning_result.alpha_p:.4f})")
        print(f"  R²: {spanning_result.r_squared:.4f}")
        if spanning_result.alpha_p < 0.05:
            print("  → SIGNIFICANT: CMF captures systematic risk beyond FF5")
        else:
            print("  → Not significant at 5%: CMF may be spanned by FF5")
    else:
        print("Step 6: Spanning test — SKIPPED (insufficient data)")
    print()

    # ── Step 7: Fama-MacBeth ──────────────────────────────────────────
    fm_result = None
    fm_df = pd.DataFrame()
    if not store.stock_returns.empty and not store.ff5.empty and not cmf.empty:
        print("Step 7: Fama-MacBeth cross-sectional test...")
        from model.culture_factor import fama_macbeth
        fm_result = fama_macbeth(
            store.stock_returns, store.ff5, cmf, decomposed,
        )
        fm_df = fm_result.all_gammas
        print(f"  Gamma(stance): {fm_result.gamma_cmf:.6f} "
              f"(t={fm_result.gamma_cmf_t:.2f}, p={fm_result.gamma_cmf_p:.4f})")
        print(f"  {fm_result.n_months} months, {fm_result.n_stocks_avg:.0f} avg stocks")
        if fm_result.gamma_cmf_p < 0.05:
            print("  → PRICED: Market compensates for culture misalignment")
        else:
            print("  → Not priced at 5%")
    else:
        print("Step 7: Fama-MacBeth — SKIPPED (insufficient data)")
    print()

    # ── Step 8: Regime-conditioned alpha ──────────────────────────────
    regime_df = pd.DataFrame()
    if not cmf.empty and not store.ff5.empty and not store.vix.empty:
        print("Step 8: Regime-conditioned spanning test...")
        from model.culture_factor import regime_spanning_test
        regime_df = regime_spanning_test(cmf, store.ff5, store.vix)
        if not regime_df.empty:
            for _, r in regime_df.iterrows():
                sig = '***' if r['ALPHA_P'] < 0.01 else '**' if r['ALPHA_P'] < 0.05 else '*' if r['ALPHA_P'] < 0.1 else ''
                print(f"  {r['REGIME']}: alpha={r['ALPHA_ANN_BPS']:.1f} bps/yr "
                      f"(t={r['ALPHA_T']:.2f}) {sig}")
    else:
        print("Step 8: Regime alpha — SKIPPED")
    print()

    # ── Step 9: Value impact regression ───────────────────────────────
    value_df = pd.DataFrame()
    if not decomposed.empty and not store.stock_returns.empty:
        print("Step 9: Value impact regression (forward returns)...")
        from model.culture_factor import value_impact_regression
        value_df = value_impact_regression(decomposed, store.stock_returns)
        if not value_df.empty:
            for _, r in value_df.iterrows():
                sig = '***' if r['P_VALUE'] < 0.01 else '**' if r['P_VALUE'] < 0.05 else '*' if r['P_VALUE'] < 0.1 else ''
                print(f"  {r['WINDOW_LABEL']:>10} | {r['VARIABLE']:>20}: "
                      f"beta={r['BETA']:+.4f} (t={r['T_STAT']:+.2f}) {sig}")
    else:
        print("Step 9: Value impact — SKIPPED")
    print()

    # ── Step 10: Build summary ────────────────────────────────────────
    print("Step 10: Building summary...")
    summary_rows = [
        {'METRIC': 'Pipeline Run Date', 'VALUE': start_time.strftime('%Y-%m-%d %H:%M')},
        {'METRIC': 'N Firm-Year Stance Scores', 'VALUE': len(stance_scores)},
        {'METRIC': 'N Unique Tickers', 'VALUE': stance_scores['TICKER'].nunique()},
        {'METRIC': 'Liberal Seeds', 'VALUE': len(scorer.liberal_seeds)},
        {'METRIC': 'Conservative Seeds', 'VALUE': len(scorer.conservative_seeds)},
        {'METRIC': 'Liberal Expanded', 'VALUE': len(scorer.liberal_expanded)},
        {'METRIC': 'Conservative Expanded', 'VALUE': len(scorer.conservative_expanded)},
        {'METRIC': 'Mean Stance Score', 'VALUE': f"{stance_scores['STANCE_SCORE'].mean():.4f}"},
        {'METRIC': 'Std Stance Score', 'VALUE': f"{stance_scores['STANCE_SCORE'].std():.4f}"},
        {'METRIC': 'N Liberal', 'VALUE': (stance_scores['STANCE_LABEL'] == 'Liberal').sum()},
        {'METRIC': 'N Mixed', 'VALUE': (stance_scores['STANCE_LABEL'] == 'Mixed').sum()},
        {'METRIC': 'N Conservative', 'VALUE': (stance_scores['STANCE_LABEL'] == 'Conservative').sum()},
    ]

    if 'SPENDING_R2' in decomposed.columns:
        r2 = decomposed['SPENDING_R2'].iloc[0] if len(decomposed) > 0 else 0
        summary_rows.append({
            'METRIC': 'Stance vs Spending R²',
            'VALUE': f"{r2:.4f}",
        })
        summary_rows.append({
            'METRIC': '% Language Explained by Money',
            'VALUE': f"{r2 * 100:.1f}%",
        })

    if not cmf.empty:
        summary_rows.extend([
            {'METRIC': 'CMF Trading Days', 'VALUE': len(cmf)},
            {'METRIC': 'CMF Mean Daily Return (bps)', 'VALUE': f"{cmf['CMF'].mean() * 10000:.2f}"},
            {'METRIC': 'CMF Annualized Sharpe', 'VALUE': f"{cmf['CMF'].mean() / cmf['CMF'].std() * np.sqrt(252):.2f}" if cmf['CMF'].std() > 0 else 'N/A'},
        ])

    if spanning_result:
        summary_rows.extend([
            {'METRIC': 'Spanning Alpha (ann bps)', 'VALUE': f"{spanning_result.annualized_alpha_bps:.1f}"},
            {'METRIC': 'Spanning Alpha t-stat', 'VALUE': f"{spanning_result.alpha_t:.2f}"},
            {'METRIC': 'Spanning Alpha p-value', 'VALUE': f"{spanning_result.alpha_p:.4f}"},
            {'METRIC': 'Spanning R²', 'VALUE': f"{spanning_result.r_squared:.4f}"},
            {'METRIC': 'SYSTEMATIC RISK?',
             'VALUE': 'YES — Not spanned by FF5' if spanning_result.alpha_p < 0.05
             else 'INCONCLUSIVE — May be spanned by FF5'},
        ])

    if fm_result:
        summary_rows.extend([
            {'METRIC': 'Fama-MacBeth Gamma (stance)', 'VALUE': f"{fm_result.gamma_cmf:.6f}"},
            {'METRIC': 'Fama-MacBeth t-stat', 'VALUE': f"{fm_result.gamma_cmf_t:.2f}"},
            {'METRIC': 'Fama-MacBeth p-value', 'VALUE': f"{fm_result.gamma_cmf_p:.4f}"},
            {'METRIC': 'PRICED RISK?',
             'VALUE': 'YES — Market compensates' if fm_result.gamma_cmf_p < 0.05
             else 'NO — Not priced cross-sectionally'},
        ])

    summary_df = pd.DataFrame(summary_rows)

    # ── Write Excel workbook ──────────────────────────────────────────
    print()
    print(f"Writing results to {output_path}...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Stance Scores
        cols_out = ['TICKER', 'YEAR', 'FILING_DATE', 'L_COUNT', 'C_COUNT',
                    'TOTAL_TOKENS', 'L_FREQ', 'C_FREQ', 'STANCE_SCORE',
                    'STANCE_LABEL']
        stance_out = stance_scores[[c for c in cols_out if c in stance_scores.columns]]
        stance_out.to_excel(writer, sheet_name='Stance Scores', index=False)

        # Sheet 3: Seed Dictionaries
        seed_summary.to_excel(writer, sheet_name='Seed Dictionaries', index=False)

        # Sheet 4: Alignment
        align_cols = ['TICKER', 'YEAR', 'STANCE_SCORE', 'SURROUNDING_CULTURE',
                      'ALIGNMENT', 'MISALIGNMENT', 'DELIBERATE_STANCE',
                      'STANCE_LABEL']
        align_out = decomposed[[c for c in align_cols if c in decomposed.columns]]
        align_out.to_excel(writer, sheet_name='Alignment', index=False)

        # Sheet 5: Stance vs Spending
        spend_cols = ['TICKER', 'YEAR', 'STANCE_SCORE', 'STANCE_SPENDING',
                      'STANCE_CULTURE', 'SPENDING_R2']
        spend_out = decomposed[[c for c in spend_cols if c in decomposed.columns]]
        spend_out.to_excel(writer, sheet_name='Stance vs Spending', index=False)

        # Sheet 6: CMF Returns
        if not cmf.empty:
            cmf.to_excel(writer, sheet_name='CMF Returns', index=False)

        # Sheet 7: Spanning Test
        if not spanning_df.empty:
            spanning_df.to_excel(writer, sheet_name='Spanning Test', index=False)

        # Sheet 8: Fama-MacBeth
        if not fm_df.empty:
            fm_df.to_excel(writer, sheet_name='Fama-MacBeth', index=False)

        # Sheet 9: Regime Alpha
        if not regime_df.empty:
            regime_df.to_excel(writer, sheet_name='Regime Alpha', index=False)

        # Sheet 10: Value Impact
        if not value_df.empty:
            value_df.to_excel(writer, sheet_name='Value Impact', index=False)

    elapsed = datetime.now() - start_time
    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETE — {elapsed.total_seconds():.0f}s")
    print(f"  Output: {os.path.abspath(output_path)}")
    print("=" * 70)

    # Close store
    store.close()

    return output_path


if __name__ == '__main__':
    output = run_pipeline()
