#!/usr/bin/env python3
"""
Dissertation essay monitor — detailed change analysis with APNs notifications.

Runs all three essays, extracts key statistical metrics, compares to previous
run, identifies what changed and how it affects downstream results, and sends
a rich APNs notification.

Usage:
    python3 monitor_essays.py           # run all three essays
    python3 monitor_essays.py --essay 1 # run only essay 1
    python3 monitor_essays.py --dry-run # compare without sending notifications
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dissertation-monitor")

# Where we store snapshots between runs
SNAPSHOT_FILE = PROJECT_ROOT / "data" / "essay_monitor_snapshot.json"

# APNs sender
APNS_SCRIPT = Path("/Users/administrator/Services/roseboro-backend/send_apns.py")

# Full log of change reports
CHANGE_LOG = PROJECT_ROOT / "data" / "essay_change_log.jsonl"


# ══════════════════════════════════════════════════════════════════════
# Metric extraction — pull specific numbers from each essay's results
# ══════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    """Convert to float, handling NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _df_shape(df) -> Optional[str]:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return f"{df.shape[0]}x{df.shape[1]}"


def extract_essay1_metrics(store) -> dict:
    """Run Essay 1 and extract key statistical metrics."""
    from model.essay1 import (
        estimate_vix_regimes,
        ff5_by_regime,
        culture_war_by_regime,
        sentiment_by_regime,
        assemble_macro_controls,
    )
    from model.essay1_matched import ff5_matched_control_analysis

    metrics = {}

    # ── Regime estimation ────────────────────────────────────────────
    logger.info("Essay 1: Estimating VIX regimes...")
    regime_result = estimate_vix_regimes(store, n_regimes=3)
    if regime_result is None:
        metrics["essay1_status"] = "regime_estimation_failed"
        return metrics

    # Regime means and durations
    for label, mean_vix in regime_result.regime_means.items():
        safe_label = label.replace(" ", "_").lower()
        metrics[f"e1_regime_{safe_label}_mean_vix"] = _safe_float(mean_vix)
        metrics[f"e1_regime_{safe_label}_variance"] = _safe_float(
            regime_result.regime_variances.get(label))
        metrics[f"e1_regime_{safe_label}_expected_duration"] = _safe_float(
            regime_result.expected_durations.get(label))

    # Day counts per regime
    if regime_result.regime_summary is not None and not regime_result.regime_summary.empty:
        for _, row in regime_result.regime_summary.iterrows():
            safe_label = str(row.get("REGIME", "")).replace(" ", "_").lower()
            metrics[f"e1_regime_{safe_label}_n_days"] = int(row.get("N_DAYS", 0))

    # Transition matrix (flatten)
    if regime_result.transition_matrix is not None:
        tm = regime_result.transition_matrix
        for i in range(tm.shape[0]):
            for j in range(tm.shape[1]):
                metrics[f"e1_transition_{i}_{j}"] = _safe_float(tm[i, j])

    # Model fit
    metrics["e1_regime_aic"] = _safe_float(regime_result.aic)
    metrics["e1_regime_bic"] = _safe_float(regime_result.bic)
    metrics["e1_regime_loglik"] = _safe_float(regime_result.log_likelihood)

    # ── FF5 by regime ────────────────────────────────────────────────
    macro_controls = assemble_macro_controls(store)

    logger.info("Essay 1: FF5 by regime...")
    ff5 = ff5_by_regime(store, regime_result=regime_result, macro_controls=macro_controls)
    if ff5 is not None:
        # Chow test (structural break)
        metrics["e1_chow_f_stat"] = _safe_float(ff5.chow_test.get("f_stat"))
        metrics["e1_chow_p_value"] = _safe_float(ff5.chow_test.get("p_value"))
        metrics["e1_chow_significant"] = bool(ff5.chow_test.get("significant_005", False))

        # Per-regime factor coefficients
        for regime_label, rfr in ff5.regime_regressions.items():
            safe_label = regime_label.replace(" ", "_").lower()
            metrics[f"e1_ff5_{safe_label}_alpha"] = _safe_float(rfr.alpha)
            metrics[f"e1_ff5_{safe_label}_alpha_p"] = _safe_float(rfr.alpha_p)
            metrics[f"e1_ff5_{safe_label}_r_squared"] = _safe_float(rfr.r_squared)
            metrics[f"e1_ff5_{safe_label}_n_obs"] = rfr.n_obs
            for factor, beta in rfr.betas.items():
                metrics[f"e1_ff5_{safe_label}_{factor.lower()}_beta"] = _safe_float(beta)
                p = rfr.beta_pvalues.get(factor)
                metrics[f"e1_ff5_{safe_label}_{factor.lower()}_p"] = _safe_float(p)

        # Factor premia comparison
        if ff5.factor_premia_comparison is not None and not ff5.factor_premia_comparison.empty:
            for _, row in ff5.factor_premia_comparison.iterrows():
                factor = str(row.get("FACTOR", "")).lower()
                for col in row.index:
                    if col != "FACTOR" and "MEAN" in col.upper():
                        metrics[f"e1_premia_{factor}_{col.lower()}"] = _safe_float(row[col])

    # ── Culture war stocks ───────────────────────────────────────────
    logger.info("Essay 1: Culture war by regime...")
    cw = culture_war_by_regime(
        store, regime_result=regime_result, ff5_analysis=ff5,
        macro_controls=macro_controls,
    )
    if cw is not None:
        metrics["e1_cw_n_stocks"] = cw.n_stocks
        metrics["e1_cw_n_failed"] = cw.n_failed

        # Aggregate: how many stocks have significant alphas per regime
        if cw.summary is not None and not cw.summary.empty:
            for regime_label in cw.summary["REGIME"].unique():
                safe_label = str(regime_label).replace(" ", "_").lower()
                regime_df = cw.summary[cw.summary["REGIME"] == regime_label]
                n_sig = int((regime_df["ALPHA_P"] < 0.05).sum())
                mean_alpha = _safe_float(regime_df["ALPHA"].mean())
                metrics[f"e1_cw_{safe_label}_n_sig_alpha"] = n_sig
                metrics[f"e1_cw_{safe_label}_mean_alpha"] = mean_alpha

    # ── Matched control ──────────────────────────────────────────────
    logger.info("Essay 1: Matched control analysis...")
    matched = ff5_matched_control_analysis(store, regime_result=regime_result)
    if matched is not None:
        metrics["e1_matched_n_pairs"] = matched.n_pairs
        metrics["e1_matched_n_complete"] = matched.n_pairs_complete

        # Paired t-test results per regime × factor
        if matched.paired_ttest is not None and not matched.paired_ttest.empty:
            for _, row in matched.paired_ttest.iterrows():
                regime = str(row.get("REGIME", "")).replace(" ", "_").lower()
                var = str(row.get("VARIABLE", "")).lower()
                metrics[f"e1_matched_{regime}_{var}_delta"] = _safe_float(row.get("MEAN_DELTA"))
                metrics[f"e1_matched_{regime}_{var}_t"] = _safe_float(row.get("T_STAT"))
                metrics[f"e1_matched_{regime}_{var}_p"] = _safe_float(row.get("P_VALUE"))
                metrics[f"e1_matched_{regime}_{var}_bh_sig"] = bool(row.get("BH_SIGNIFICANT", False))

        # Regime amplification
        if matched.regime_amplification is not None and not matched.regime_amplification.empty:
            for _, row in matched.regime_amplification.iterrows():
                var = str(row.get("VARIABLE", "")).lower()
                metrics[f"e1_amplification_{var}_high_low_diff"] = _safe_float(row.get("MEAN_DIFF"))
                metrics[f"e1_amplification_{var}_p"] = _safe_float(row.get("P_VALUE"))
                metrics[f"e1_amplification_{var}_bh_sig"] = bool(row.get("BH_SIGNIFICANT", False))

    # ── Sentiment ────────────────────────────────────────────────────
    logger.info("Essay 1: Sentiment analysis...")
    sentiment = sentiment_by_regime(store, regime_result=regime_result)
    if sentiment is not None:
        metrics["e1_sentiment_n_articles"] = sentiment.n_articles
        metrics["e1_sentiment_n_scored"] = sentiment.n_scored

        if sentiment.fomo_by_regime is not None and not sentiment.fomo_by_regime.empty:
            for _, row in sentiment.fomo_by_regime.iterrows():
                regime = str(row.get("REGIME", "")).replace(" ", "_").lower()
                metrics[f"e1_fomo_{regime}_mean_z"] = _safe_float(row.get("MEAN_FOMO_Z"))
                metrics[f"e1_fomo_{regime}_pct_euphoria"] = _safe_float(row.get("PCT_EUPHORIA"))
                metrics[f"e1_fomo_{regime}_pct_panic"] = _safe_float(row.get("PCT_PANIC"))

    return metrics


def extract_essay2_metrics(store) -> dict:
    """Run Essay 2 and extract key statistical metrics."""
    from model.essay1 import estimate_vix_regimes, sentiment_by_regime
    from model.essay2 import run_nlp_analysis
    from model.essay2_did import run_did

    metrics = {}

    # ── NLP pipeline ─────────────────────────────────────────────────
    logger.info("Essay 2: NLP analysis...")
    nlp = None
    try:
        nlp = run_nlp_analysis(store)
        if nlp is not None:
            metrics["e2_nlp_n_articles"] = nlp.n_articles_scored
            metrics["e2_nlp_n_filings"] = nlp.n_filings_scored
            metrics["e2_nlp_n_tickers"] = nlp.n_tickers

            # Aggregate sentiment stats
            if not nlp.news_sentiment.empty and "SENT_SCORE" in nlp.news_sentiment.columns:
                metrics["e2_nlp_news_mean_sent"] = _safe_float(nlp.news_sentiment["SENT_SCORE"].mean())
                metrics["e2_nlp_news_std_sent"] = _safe_float(nlp.news_sentiment["SENT_SCORE"].std())
                for label in ["positive", "negative", "neutral"]:
                    pct = (nlp.news_sentiment["LABEL"] == label).mean() if "LABEL" in nlp.news_sentiment.columns else None
                    metrics[f"e2_nlp_news_pct_{label}"] = _safe_float(pct)

            if not nlp.filing_sentiment.empty and "SENT_MEAN" in nlp.filing_sentiment.columns:
                metrics["e2_nlp_filing_mean_sent"] = _safe_float(nlp.filing_sentiment["SENT_MEAN"].mean())
                for section in ["MDA", "RISK_FACTORS"]:
                    sec_df = nlp.filing_sentiment[nlp.filing_sentiment.get("SECTION", pd.Series()) == section]
                    if not sec_df.empty:
                        metrics[f"e2_nlp_filing_{section.lower()}_mean"] = _safe_float(sec_df["SENT_MEAN"].mean())

            metrics["e2_nlp_event_panel_shape"] = _df_shape(nlp.event_panel)
    except ImportError:
        logger.warning("Essay 2 NLP skipped (FinBERT/transformers not installed)")
        metrics["e2_nlp_status"] = "skipped_no_finbert"
    except Exception as e:
        logger.error("Essay 2 NLP failed: %s", e)
        metrics["e2_nlp_status"] = f"error:{e}"

    # ── DiD analysis ─────────────────────────────────────────────────
    logger.info("Essay 2: DiD analysis...")
    try:
        regime_result = estimate_vix_regimes(store, n_regimes=3)
        sentiment = sentiment_by_regime(store, regime_result=regime_result)
        did = run_did(store, regime_result=regime_result, sentiment_analysis=sentiment)

        if did is not None:
            metrics["e2_did_n_events"] = did.n_events
            metrics["e2_did_n_treatment"] = did.n_treatment_firms
            metrics["e2_did_n_control"] = did.n_control_firms
            metrics["e2_did_n_obs"] = did.n_observations
            metrics["e2_car_panel_shape"] = _df_shape(did.car_panel)

            # CAR summary stats
            if did.car_panel is not None and not did.car_panel.empty:
                for car_col in ["CAR_PRE", "CAR_POST", "CAR_FULL"]:
                    if car_col in did.car_panel.columns:
                        vals = did.car_panel[car_col].dropna()
                        metrics[f"e2_{car_col.lower()}_mean"] = _safe_float(vals.mean())
                        metrics[f"e2_{car_col.lower()}_median"] = _safe_float(vals.median())
                        metrics[f"e2_{car_col.lower()}_std"] = _safe_float(vals.std())

                # Treatment vs control CARs
                if "IS_TREATMENT" in did.car_panel.columns and "CAR_POST" in did.car_panel.columns:
                    treat = did.car_panel[did.car_panel["IS_TREATMENT"] == True]["CAR_POST"].dropna()
                    ctrl = did.car_panel[did.car_panel["IS_TREATMENT"] == False]["CAR_POST"].dropna()
                    metrics["e2_car_post_treatment_mean"] = _safe_float(treat.mean())
                    metrics["e2_car_post_control_mean"] = _safe_float(ctrl.mean())

            # Coefficient table — the key results
            if did.coefficient_table is not None and not did.coefficient_table.empty:
                ct = did.coefficient_table
                for _, row in ct.iterrows():
                    spec = str(row.get("SPECIFICATION", "")).lower()
                    var = str(row.get("VARIABLE", "")).lower().replace("×", "x").replace("*", "x")
                    metrics[f"e2_did_{spec}_{var}_coef"] = _safe_float(row.get("COEFFICIENT"))
                    metrics[f"e2_did_{spec}_{var}_t"] = _safe_float(row.get("T_STAT"))
                    metrics[f"e2_did_{spec}_{var}_p"] = _safe_float(row.get("P_VALUE"))
                    if "BH_SIGNIFICANT" in row.index:
                        metrics[f"e2_did_{spec}_{var}_bh_sig"] = bool(row.get("BH_SIGNIFICANT", False))

            # Parallel trends
            if did.parallel_trends is not None:
                pt = did.parallel_trends
                metrics["e2_parallel_trends_f_stat"] = _safe_float(pt.joint_f_stat)
                metrics["e2_parallel_trends_p_value"] = _safe_float(pt.joint_p_value)
                metrics["e2_parallel_trends_passes"] = bool(pt.passes)

            # Diagnostics
            if did.diagnostics is not None:
                diag = did.diagnostics
                if diag.bootstrap_ci is not None and not diag.bootstrap_ci.empty:
                    for _, row in diag.bootstrap_ci.iterrows():
                        var = str(row.get("VARIABLE", "")).lower()
                        metrics[f"e2_bootstrap_{var}_point"] = _safe_float(row.get("POINT_EST"))
                        metrics[f"e2_bootstrap_{var}_ci_lo"] = _safe_float(row.get("CI_LOWER"))
                        metrics[f"e2_bootstrap_{var}_ci_hi"] = _safe_float(row.get("CI_UPPER"))

                if diag.placebo_tests is not None and not diag.placebo_tests.empty:
                    pct_rejected = diag.placebo_tests["REJECTED"].mean() if "REJECTED" in diag.placebo_tests.columns else None
                    metrics["e2_placebo_pct_rejected"] = _safe_float(pct_rejected)

    except Exception as e:
        logger.error("Essay 2 DiD failed: %s", e)
        metrics["e2_did_status"] = f"error:{e}"

    return metrics


def extract_essay3_metrics(store) -> dict:
    """Run Essay 3 and extract key statistical metrics.

    Reframed: Fundamental vs Cultural political events contrast.
    """
    from model.essay3 import run_essay3 as _run_essay3

    metrics = {}

    logger.info("Essay 3: Insider trading — fundamental vs cultural...")
    try:
        results = _run_essay3(store)

        if not isinstance(results, dict):
            results = {}
            for tbl, key in [
                ("ESSAY3_PANEL", "panel"),
                ("ESSAY3_PRIMARY_CONTRAST", "primary_contrast"),
                ("ESSAY3_WITHIN_CATEGORY", "within_category"),
                ("ESSAY3_SUBGROUPS", "subgroups"),
                ("ESSAY3_TOST", "tost"),
                ("ESSAY3_BOOTSTRAP", "bootstrap"),
                ("ESSAY3_PLACEBO", "placebo"),
                ("ESSAY3_FAMA_MACBETH", "fama_macbeth"),
                ("ESSAY3_CLUSTERING", "clustering"),
            ]:
                try:
                    df = store.read_table(tbl)
                    if not df.empty:
                        results[key] = df
                except Exception:
                    pass

        # ── Panel summary ────────────────────────────────────────────
        panel = results.get("panel")
        if panel is not None and isinstance(panel, pd.DataFrame) and not panel.empty:
            metrics["e3_panel_n_events"] = len(panel)
            if "HAS_SUFFICIENT_DATA" in panel.columns:
                metrics["e3_panel_n_sufficient"] = int(panel["HAS_SUFFICIENT_DATA"].sum())
            if "EVENT_CATEGORY" in panel.columns:
                metrics["e3_panel_n_cultural"] = int((panel["EVENT_CATEGORY"] == "CULTURAL").sum())
                metrics["e3_panel_n_fundamental"] = int((panel["EVENT_CATEGORY"] == "FUNDAMENTAL").sum())
            metrics["e3_panel_shape"] = _df_shape(panel)

        # ── Primary contrast ─────────────────────────────────────────
        pc = results.get("primary_contrast")
        if pc is not None and isinstance(pc, pd.DataFrame) and not pc.empty:
            ttest = pc[pc["TEST"] == "TWO_SAMPLE_TTEST"]
            if not ttest.empty:
                r = ttest.iloc[0]
                metrics["e3_contrast_diff_mean"] = _safe_float(r.get("DIFF_MEAN"))
                metrics["e3_contrast_cohen_d"] = _safe_float(r.get("COHEN_D"))
                metrics["e3_contrast_p_value"] = _safe_float(r.get("P_VALUE"))

            did = pc[(pc["TEST"] == "DID_CONTRAST") & (pc.get("VARIABLE", pd.Series()) == "FUND_X_POST")]
            if not did.empty:
                r = did.iloc[0]
                metrics["e3_did_fund_x_post_coef"] = _safe_float(r.get("COEFFICIENT"))
                metrics["e3_did_fund_x_post_p"] = _safe_float(r.get("P_VALUE"))

        # ── Within-category abnormal trading ──────────────────────────
        wc = results.get("within_category")
        if wc is not None and isinstance(wc, pd.DataFrame) and not wc.empty:
            for _, row in wc.iterrows():
                cat = str(row.get("EVENT_CATEGORY", "")).lower()
                metrics[f"e3_{cat}_mean_abnormal"] = _safe_float(row.get("MEAN_ABNORMAL"))
                metrics[f"e3_{cat}_cohen_d"] = _safe_float(row.get("COHEN_D"))
                metrics[f"e3_{cat}_p_value"] = _safe_float(row.get("T_PVALUE"))

        # ── Subgroups (Holm-corrected) ────────────────────────────────
        subgroups = results.get("subgroups")
        if subgroups is not None and isinstance(subgroups, pd.DataFrame) and not subgroups.empty:
            for _, row in subgroups.iterrows():
                sg = str(row.get("SUBGROUP", "")).lower()
                metrics[f"e3_subgroup_{sg}_cohen_d"] = _safe_float(row.get("COHEN_D"))
                metrics[f"e3_subgroup_{sg}_p"] = _safe_float(row.get("P_VALUE"))
                metrics[f"e3_subgroup_{sg}_holm_sig"] = bool(row.get("HOLM_SIGNIFICANT", False))

        # ── TOST equivalence ──────────────────────────────────────────
        tost = results.get("tost")
        if tost is not None and isinstance(tost, pd.DataFrame) and not tost.empty:
            for _, row in tost.iterrows():
                cat = str(row.get("EVENT_CATEGORY", "")).lower()
                metrics[f"e3_tost_{cat}_p"] = _safe_float(row.get("P_TOST"))
                metrics[f"e3_tost_{cat}_equivalent"] = bool(row.get("EQUIVALENT", False))
                metrics[f"e3_tost_{cat}_power"] = _safe_float(row.get("POWER_AT_OBSERVED"))

        # ── Bootstrap CIs ────────────────────────────────────────────
        bootstrap = results.get("bootstrap")
        if bootstrap is not None and isinstance(bootstrap, pd.DataFrame) and not bootstrap.empty:
            r = bootstrap.iloc[0]
            metrics["e3_bootstrap_diff"] = _safe_float(r.get("OBSERVED_DIFF"))
            metrics["e3_bootstrap_ci_lo"] = _safe_float(r.get("CI_2_5"))
            metrics["e3_bootstrap_ci_hi"] = _safe_float(r.get("CI_97_5"))

        # ── Placebo ──────────────────────────────────────────────────
        placebo = results.get("placebo")
        if placebo is not None and isinstance(placebo, pd.DataFrame) and not placebo.empty:
            r = placebo.iloc[0]
            metrics["e3_placebo_p_value"] = _safe_float(r.get("P_VALUE"))
            metrics["e3_placebo_n_iterations"] = int(r.get("N_ITERATIONS", 0))

        # ── Fama-MacBeth ─────────────────────────────────────────────
        fm = results.get("fama_macbeth")
        if fm is not None and isinstance(fm, pd.DataFrame) and not fm.empty:
            r = fm.iloc[0]
            metrics["e3_fmb_coef"] = _safe_float(r.get("FM_COEFFICIENT"))
            metrics["e3_fmb_p"] = _safe_float(r.get("FM_P_VALUE"))

    except Exception as e:
        logger.error("Essay 3 failed: %s", e)
        metrics["e3_status"] = f"error:{e}"

    return metrics


# ══════════════════════════════════════════════════════════════════════
# Change analysis — compare snapshots and build human-readable report
# ══════════════════════════════════════════════════════════════════════

# Dependency map: upstream metric patterns → downstream essays affected
DEPENDENCY_CHAIN = {
    # Essay 1 regimes feed into everything
    "e1_regime_": {
        "affects": ["Essay 2 DiD (regime conditioning)", "Essay 3 (regime interaction)"],
        "why": "Regime boundaries shifted — all regime-conditioned analyses will differ",
    },
    "e1_chow_": {
        "affects": ["Essay 1 narrative (structural break significance)"],
        "why": "Structural break test changed — affects whether regime differences are significant",
    },
    "e1_ff5_": {
        "affects": ["Essay 1 factor premia", "Essay 2 (estimation window models)", "Essay 3 (CAR computation)"],
        "why": "Factor loadings changed — CARs and abnormal returns will shift",
    },
    "e1_matched_": {
        "affects": ["Essay 1 matched control narrative"],
        "why": "Treatment-control differences in factor loadings changed",
    },
    "e1_fomo_": {
        "affects": ["Essay 2 DiD (FOMO z-score control variable)"],
        "why": "FOMO z-scores changed — DiD with sentiment controls will differ",
    },
    # Essay 2 NLP feeds into DiD and Essay 3
    "e2_nlp_": {
        "affects": ["Essay 2 DiD (NLP controls)", "Essay 3 (event panel)"],
        "why": "Sentiment scores changed — event-level NLP features differ",
    },
    "e2_did_": {
        "affects": ["Essay 2 treatment effect narrative", "Essay 3 (CAR_POST in insider panel)"],
        "why": "DiD coefficients changed — treatment effects and CARs feeding Essay 3 differ",
    },
    "e2_car_": {
        "affects": ["Essay 3 (CAR-insider regression, post-event reversal)"],
        "why": "CARs changed — all Essay 3 analyses using CAR_POST will shift",
    },
    "e2_parallel_trends_": {
        "affects": ["Essay 2 DiD validity"],
        "why": "Parallel trends test changed — DiD identification may be compromised",
    },
    # Essay 3 is terminal but has internal dependencies
    "e3_abnormal_": {
        "affects": ["Essay 3 tail diagnostics", "Essay 3 subgroup analysis"],
        "why": "Abnormal selling baseline changed — tail threshold and subgroups shift",
    },
    "e3_tost_": {
        "affects": ["Essay 3 null characterization"],
        "why": "Equivalence test changed — affects whether null is characterized as true null vs underpowered",
    },
}

# Human-readable labels for metric keys
METRIC_LABELS = {
    "e1_chow_p_value": "Chow test p-value (regime structural break)",
    "e1_chow_f_stat": "Chow test F-statistic",
    "e1_chow_significant": "Chow test significant at 5%",
    "e2_parallel_trends_passes": "Parallel trends assumption holds",
    "e2_parallel_trends_p_value": "Parallel trends joint F p-value",
}


def _classify_change(key: str, old_val, new_val) -> dict:
    """Classify a single metric change with magnitude and interpretation."""
    change = {
        "metric": key,
        "old": old_val,
        "new": new_val,
        "label": METRIC_LABELS.get(key, key),
    }

    # Numeric diff
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        diff = new_val - old_val
        change["diff"] = round(diff, 6)
        if old_val != 0:
            change["pct_change"] = round(100 * diff / abs(old_val), 2)
        else:
            change["pct_change"] = None

        # Classify significance changes
        if "_p_value" in key or "_p" in key:
            crossed_005 = (old_val >= 0.05 and new_val < 0.05) or (old_val < 0.05 and new_val >= 0.05)
            crossed_010 = (old_val >= 0.10 and new_val < 0.10) or (old_val < 0.10 and new_val >= 0.10)
            if crossed_005:
                direction = "now significant" if new_val < 0.05 else "no longer significant"
                change["significance_change"] = f"Crossed 5% threshold — {direction}"
                change["severity"] = "critical"
            elif crossed_010:
                direction = "now marginal" if new_val < 0.10 else "no longer marginal"
                change["significance_change"] = f"Crossed 10% threshold — {direction}"
                change["severity"] = "important"
            else:
                change["severity"] = "minor"
        elif "_bh_sig" in key:
            if old_val != new_val:
                change["significance_change"] = f"BH significance flipped: {old_val} → {new_val}"
                change["severity"] = "critical"
            else:
                change["severity"] = "minor"
        else:
            # Use pct_change to classify severity
            pct = abs(change.get("pct_change") or 0)
            if pct > 20:
                change["severity"] = "important"
            elif pct > 5:
                change["severity"] = "moderate"
            else:
                change["severity"] = "minor"
    elif isinstance(old_val, bool) and isinstance(new_val, bool):
        if old_val != new_val:
            change["severity"] = "critical"
            change["significance_change"] = f"Flipped: {old_val} → {new_val}"
        else:
            change["severity"] = "minor"
    else:
        change["severity"] = "moderate"

    # Downstream impact
    change["downstream"] = []
    for pattern, impact in DEPENDENCY_CHAIN.items():
        if key.startswith(pattern):
            change["downstream"] = impact["affects"]
            change["downstream_why"] = impact["why"]
            break

    return change


def build_change_report(previous: dict, current: dict, essays_run: list) -> dict:
    """Build a comprehensive change report."""
    changes = []
    essay_prefixes = {1: "e1_", 2: "e2_", 3: "e3_"}

    # Only compare metrics for essays we ran
    relevant_prefixes = [essay_prefixes[e] for e in essays_run]

    for key, new_val in current.items():
        if not any(key.startswith(p) for p in relevant_prefixes):
            continue
        old_val = previous.get(key)
        if old_val is None:
            changes.append({
                "metric": key, "old": None, "new": new_val,
                "severity": "new", "label": METRIC_LABELS.get(key, key),
                "downstream": [],
            })
        elif old_val != new_val:
            changes.append(_classify_change(key, old_val, new_val))

    for key, old_val in previous.items():
        if not any(key.startswith(p) for p in relevant_prefixes):
            continue
        if key not in current:
            changes.append({
                "metric": key, "old": old_val, "new": None,
                "severity": "important", "label": METRIC_LABELS.get(key, key),
                "downstream": [],
            })

    # Sort: critical first, then important, moderate, minor, new
    severity_order = {"critical": 0, "important": 1, "moderate": 2, "new": 3, "minor": 4}
    changes.sort(key=lambda c: severity_order.get(c.get("severity", "minor"), 5))

    # Collect all downstream impacts
    all_downstream = set()
    for c in changes:
        for d in c.get("downstream", []):
            all_downstream.add(d)

    # Summary stats
    n_critical = sum(1 for c in changes if c.get("severity") == "critical")
    n_important = sum(1 for c in changes if c.get("severity") == "important")
    n_moderate = sum(1 for c in changes if c.get("severity") == "moderate")

    return {
        "timestamp": datetime.now().isoformat(),
        "essays_run": essays_run,
        "n_metrics_compared": len(current),
        "n_changes": len(changes),
        "n_critical": n_critical,
        "n_important": n_important,
        "n_moderate": n_moderate,
        "downstream_affected": sorted(all_downstream),
        "changes": changes,
    }


def format_notification(report: dict) -> tuple:
    """Build APNs title + body from change report."""
    n = report["n_changes"]
    if n == 0:
        return None, None

    # Title
    n_crit = report["n_critical"]
    n_imp = report["n_important"]
    if n_crit:
        title = f"Dissertation: {n_crit} Critical Change{'s' if n_crit > 1 else ''}"
    elif n_imp:
        title = f"Dissertation: {n_imp} Important Change{'s' if n_imp > 1 else ''}"
    else:
        title = f"Dissertation: {n} Minor Change{'s' if n > 1 else ''}"

    # Body — show top 3 changes with actual values
    lines = []
    for c in report["changes"][:3]:
        key = c["metric"]
        # Shorten key for display
        short_key = key.replace("essay", "E").replace("e1_", "E1:").replace("e2_", "E2:").replace("e3_", "E3:")

        if c.get("significance_change"):
            lines.append(f"{short_key}: {c['significance_change']}")
        elif "diff" in c and c["diff"] is not None:
            old_str = f"{c['old']:.4f}" if isinstance(c["old"], float) else str(c["old"])
            new_str = f"{c['new']:.4f}" if isinstance(c["new"], float) else str(c["new"])
            pct = c.get("pct_change")
            pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
            lines.append(f"{short_key}: {old_str}→{new_str}{pct_str}")
        else:
            lines.append(f"{short_key}: {c['old']}→{c['new']}")

    if n > 3:
        lines.append(f"+{n - 3} more")

    # Downstream cascade
    if report["downstream_affected"]:
        lines.append(f"Affects: {', '.join(report['downstream_affected'][:2])}")

    body = "\n".join(lines)
    return title, body


def format_log_summary(report: dict) -> str:
    """Build a detailed log summary for the terminal."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  CHANGE REPORT — {report['timestamp']}")
    lines.append(f"  Essays: {report['essays_run']} | "
                 f"Metrics: {report['n_metrics_compared']} | "
                 f"Changes: {report['n_changes']}")
    lines.append(f"  Critical: {report['n_critical']} | "
                 f"Important: {report['n_important']} | "
                 f"Moderate: {report['n_moderate']}")
    lines.append("=" * 70)

    if not report["changes"]:
        lines.append("  No changes detected.")
        return "\n".join(lines)

    # Group by severity
    for severity in ["critical", "important", "moderate", "new", "minor"]:
        group = [c for c in report["changes"] if c.get("severity") == severity]
        if not group:
            continue
        lines.append(f"\n  ── {severity.upper()} ({len(group)}) {'─' * 40}")
        for c in group:
            key = c["metric"]
            if c.get("significance_change"):
                lines.append(f"    {key}")
                lines.append(f"      {c['significance_change']}")
                if isinstance(c.get("old"), (int, float)) and isinstance(c.get("new"), (int, float)):
                    lines.append(f"      {c['old']:.6f} → {c['new']:.6f}")
            elif "diff" in c and c["diff"] is not None:
                pct = c.get("pct_change")
                pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
                lines.append(f"    {key}: {c['old']} → {c['new']} (Δ={c['diff']:.6f}{pct_str})")
            elif c.get("old") is None:
                lines.append(f"    {key}: NEW = {c['new']}")
            elif c.get("new") is None:
                lines.append(f"    {key}: REMOVED (was {c['old']})")
            else:
                lines.append(f"    {key}: {c['old']} → {c['new']}")

            if c.get("downstream"):
                lines.append(f"      ↳ Affects: {', '.join(c['downstream'])}")
            if c.get("downstream_why"):
                lines.append(f"        Why: {c['downstream_why']}")

    if report["downstream_affected"]:
        lines.append(f"\n  ── DOWNSTREAM CASCADE {'─' * 46}")
        for d in report["downstream_affected"]:
            lines.append(f"    → {d}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# APNs
# ══════════════════════════════════════════════════════════════════════

def send_apns_notification(title: str, body: str):
    """Send APNs push notification using the existing send_apns.py."""
    if not APNS_SCRIPT.exists():
        logger.warning("APNs script not found at %s", APNS_SCRIPT)
        return False

    import importlib.util
    spec = importlib.util.spec_from_file_location("send_apns", str(APNS_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        with open(mod.DEVICE_TOKENS_FILE) as f:
            data = json.load(f)
        devices = [d for d in data.get("devices", []) if d.get("active")]
    except Exception as e:
        logger.error("Failed to load device tokens: %s", e)
        return False

    sent = 0
    for device in devices:
        token = device.get("device_token", "")
        if not token:
            continue
        try:
            if mod.send_apns(token, title, body):
                sent += 1
                logger.info("Notification sent to %s", device.get("device_name", token[:8]))
        except Exception as e:
            logger.error("Failed to send to %s: %s", token[:8], e)

    return sent > 0


# ══════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════

def load_previous_snapshot() -> dict:
    if SNAPSHOT_FILE.exists():
        with open(SNAPSHOT_FILE) as f:
            return json.load(f)
    return {}


def save_snapshot(metrics: dict):
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def append_change_log(report: dict):
    """Append change report to JSONL log."""
    CHANGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CHANGE_LOG, "a") as f:
        f.write(json.dumps(report, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dissertation essay monitor")
    parser.add_argument("--essay", type=int, choices=[1, 2, 3],
                        help="Run only this essay (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compare and report but don't send notifications")
    parser.add_argument("--force-notify", action="store_true",
                        help="Send notification even if nothing changed")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Dissertation Essay Monitor (v2 — detailed analysis)")
    logger.info("  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # Load data
    from model.datastore import DataStore
    logger.info("Connecting to DataStore...")
    t0 = time.time()
    store = DataStore()
    logger.info("DataStore ready (%.1fs, backend=%s)", time.time() - t0, store.backend)

    # Run selected essays and extract metrics
    current_metrics = {}
    essays_to_run = [args.essay] if args.essay else [1, 2, 3]
    extractors = {1: extract_essay1_metrics, 2: extract_essay2_metrics, 3: extract_essay3_metrics}

    for essay_num in essays_to_run:
        logger.info("-" * 40)
        logger.info("Running Essay %d...", essay_num)
        t_essay = time.time()

        essay_metrics = extractors[essay_num](store)
        current_metrics.update(essay_metrics)

        logger.info("Essay %d done (%.1fs, %d metrics extracted)",
                     essay_num, time.time() - t_essay, len(essay_metrics))

    store.close()

    # Compare to previous snapshot
    previous_metrics = load_previous_snapshot()
    report = build_change_report(previous_metrics, current_metrics, essays_to_run)

    # Log detailed report
    log_text = format_log_summary(report)
    logger.info("\n%s", log_text)

    # Save updated snapshot (merge for essays we didn't run)
    merged = dict(previous_metrics)
    merged.update(current_metrics)
    save_snapshot(merged)

    # Save change log
    if report["n_changes"] > 0:
        append_change_log(report)

    # Send notification
    if report["n_changes"] > 0 or args.force_notify:
        title, body = format_notification(report)
        if title:
            if args.dry_run:
                logger.info("DRY RUN — would send:\n  Title: %s\n  Body: %s", title, body)
            else:
                logger.info("Sending APNs notification...")
                send_apns_notification(title, body)
    else:
        logger.info("No changes — no notification sent.")

    total_time = time.time() - t0
    logger.info("Total time: %.1fs", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
