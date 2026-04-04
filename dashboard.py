"""
dashboard.py - Executive Briefing Dashboard for Signals & Systems
Presidential-briefing-quality visualization of the dissertation research pipeline.
Displays model results, visualizations, and data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import logging
import os
import subprocess
import sys

from Database import AthenaLoader, SQLiteLoader

# =============================================================================
# CONFIG & CONSTANTS
# =============================================================================

st.set_page_config(
    page_title="Signals & Systems: The Political Economy of Investor Sentiment",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "signals_systems.db"


# =============================================================================
# DATABASE BACKEND (Athena preferred, SQLite fallback)
# =============================================================================

@st.cache_resource
def _detect_backend():
    """Probe AWS credentials once; return 'athena' or 'sqlite'."""
    try:
        with AthenaLoader() as db:
            db.list_tables()
        return "athena"
    except Exception:
        logging.info("Athena unavailable, falling back to SQLite")
        return "sqlite"


DB_BACKEND = _detect_backend()


def _get_loader():
    """Return the appropriate loader (context manager) for the active backend."""
    if DB_BACKEND == "athena":
        return AthenaLoader()
    return SQLiteLoader(db_path=str(DB_PATH))


NAVY = "#1B2A4A"
GOLD = "#C5A55A"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F0"


# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:wght@300;400;600&display=swap');

/* Global typography */
html, body, [class*="css"] {{
    font-family: 'Source Serif 4', 'Georgia', 'Times New Roman', serif;
    color: {NAVY};
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Playfair Display', 'Georgia', serif;
    color: {NAVY};
    font-weight: 600;
}}
h1 {{ font-size: 2rem; letter-spacing: 0.02em; }}
h2 {{ font-size: 1.5rem; }}
h3 {{ font-size: 1.25rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {NAVY};
    padding-top: 1rem;
}}
section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
section[data-testid="stSidebar"] p {{
    font-size: 0.95rem;
    line-height: 1.6;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {GOLD} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
}}
section[data-testid="stSidebar"] h4 {{
    color: {GOLD} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
    font-size: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}
/* Sidebar metric cards */
section[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background-color: rgba(255, 255, 255, 0.08);
    border-left: 3px solid {GOLD};
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.15rem;
}}
section[data-testid="stSidebar"] .stMetric label,
section[data-testid="stSidebar"] [data-testid="stMetric"] label {{
    color: {GOLD} !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {WHITE} !important;
    font-family: 'Source Serif 4', 'Georgia', serif;
    font-size: 1.3rem;
    font-weight: 600;
}}
/* Sidebar button */
section[data-testid="stSidebar"] .stButton > button {{
    background-color: {GOLD};
    color: {NAVY} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #D4B86A;
    color: {NAVY} !important;
}}
/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {{
    color: rgba(255, 255, 255, 0.6) !important;
    font-size: 0.8rem;
}}
/* Sidebar dividers */
section[data-testid="stSidebar"] hr {{
    border-color: rgba(197, 165, 90, 0.3) !important;
    margin: 0.75rem 0;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background-color: {LIGHT_GRAY};
    border-left: 4px solid {GOLD};
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
}}
[data-testid="stMetric"] label {{
    color: {NAVY} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {NAVY} !important;
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid #E0E0E0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {NAVY};
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    border-bottom: 3px solid transparent;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: {NAVY};
    border-bottom: 3px solid {GOLD};
    background-color: transparent;
}}

/* Expanders */
.streamlit-expanderHeader {{
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    color: {NAVY};
    font-size: 1.1rem;
}}

/* DataFrames */
.stDataFrame {{
    border: 1px solid #E0E0E0;
    border-radius: 4px;
}}

/* Hide Streamlit footer */
footer {{visibility: hidden;}}

/* Divider */
hr {{
    border-color: {GOLD};
    opacity: 0.3;
}}

/* ---- Mobile / narrow-viewport overrides ---- */
@media (max-width: 768px) {{
    section[data-testid="stSidebar"] {{
        padding-top: 0.5rem;
    }}
    section[data-testid="stSidebar"] [data-testid="stMetric"] {{
        padding: 0.4rem 0.5rem;
        margin-bottom: 0.1rem;
    }}
    section[data-testid="stSidebar"] .stMetric label,
    section[data-testid="stSidebar"] [data-testid="stMetric"] label {{
        font-size: 0.65rem;
        letter-spacing: 0.02em;
    }}
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"],
    section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 1rem;
    }}
    section[data-testid="stSidebar"] p {{
        font-size: 0.85rem;
        line-height: 1.4;
    }}
    section[data-testid="stSidebar"] h4 {{
        font-size: 0.85rem;
    }}
    section[data-testid="stSidebar"] hr {{
        margin: 0.5rem 0;
    }}
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
    }}
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }}
    h1 {{ font-size: 1.4rem; }}
    h2 {{ font-size: 1.2rem; }}
    h3 {{ font-size: 1.05rem; }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.5rem 0.6rem;
        font-size: 0.8rem;
    }}
    [data-testid="stMetric"] {{
        padding: 0.5rem 0.75rem;
    }}
    [data-testid="stMetric"] label {{
        font-size: 0.65rem;
    }}
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fmt_sig(p):
    """Return significance stars for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_num(val, dec=0):
    """Format a number with commas; return '--' for None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    if dec == 0:
        return f"{int(val):,}"
    return f"{val:,.{dec}f}"


def fmt_pval(p):
    """Format a p-value for display."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def render_figure(png_data, caption="", filename="figure.png", key_prefix="dl_fig"):
    """Display a PNG BLOB via st.image with download button."""
    if png_data:
        st.image(BytesIO(png_data), caption=caption, use_container_width=True)
        st.download_button(
            "Download Figure",
            data=png_data,
            file_name=filename,
            mime="image/png",
            key=f"{key_prefix}_{filename}",
        )


def render_df(df, label, key, height=None):
    """Display a DataFrame with a CSV download button."""
    kwargs = dict(use_container_width=True, hide_index=True)
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(df, **kwargs)
    except TypeError:
        kwargs.pop("hide_index", None)
        st.dataframe(df, **kwargs)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Download {label}",
        data=csv,
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def render_chart_gallery(chart_list, expanded=False, gallery_id="gallery"):
    """Render a 2-column gallery of figures from the FIGURES table."""
    label = f"Chart Gallery ({len(chart_list)} figures)"
    with st.expander(label, expanded=expanded):
        for i in range(0, len(chart_list), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(chart_list):
                    name, description = chart_list[idx]
                    with col:
                        fig_data = load_figure_blob(name)
                        if fig_data:
                            render_figure(fig_data, caption=description,
                                          filename=f"{name}.png",
                                          key_prefix=f"{gallery_id}_{name}")
                        else:
                            st.caption(f"{name}: not generated yet")


# Chart name lists (from visual.py)
E1_CHARTS = [
    ('e1_01_factor_premia_by_regime', 'Factor premia by regime'),
    ('e1_02_alpha_by_regime', 'Alpha by regime'),
    ('e1_03_beta_heatmap', 'Factor beta heatmap'),
    ('e1_04_tstat_heatmap', 't-statistic heatmap'),
    ('e1_05_chow_test', 'Chow structural break test'),
    ('e1_06_rsquared_by_regime', 'R-squared by regime'),
    ('e1_07_factor_premia_dot', 'Factor premia dot plot'),
    ('e1_08_cw_alpha_boxplot', 'CW stock alpha boxplot'),
    ('e1_09_cw_alpha_hist', 'CW stock alpha histogram'),
    ('e1_10_cw_rsquared', 'CW stock R-squared by regime'),
    ('e1_11_cw_mkt_beta', 'CW stock market beta'),
    ('e1_12_cw_hml_beta', 'CW stock HML beta'),
    ('e1_13_cw_alpha_significance', 'CW alpha BH significance'),
    ('e1_14_cw_mean_betas_by_regime', 'Mean betas by regime'),
    ('e1_15_fomo_by_regime', 'FOMO by regime'),
    ('e1_16_euphoria_panic', 'Euphoria vs panic days'),
    ('e1_17_sentiment_by_regime', 'Sentiment by regime'),
    ('e1_18_sentiment_timeseries', 'Sentiment time series'),
    ('e1_19_matched_ttest_forest', 'Matched t-test forest plot'),
    ('e1_20_matched_sign_consistency', 'Sign consistency'),
    ('e1_21_matched_amplification', 'Regime amplification'),
    ('e1_22_matched_delta_dist', 'Delta distribution'),
    ('e1_23_matched_delta_heatmap', 'Delta heatmap'),
    ('e1_24_matched_coverage', 'Matched coverage'),
    ('e1_25_cw_stock_count', 'CW stock count by regime'),
    ('e1_26_alpha_vs_rsq', 'Alpha vs R-squared'),
    ('e1_27_matched_n_firms', 'Matched N firms'),
    ('e1_28_matched_pvalue_dist', 'Matched p-value distribution'),
    ('e1_29_vix_distribution', 'VIX regime distribution'),
    ('e1_30_coefficient_table', 'Coefficient table'),
    ('e1_31_matched_mkt_delta', 'Matched market delta'),
    ('e1_32_matched_factor_deltas', 'Matched factor deltas'),
    ('e1_33_summary_dashboard', 'Essay 1 summary dashboard'),
]
E2_CHARTS = [
    ('e2_01_car_distribution', 'CAR distribution'),
    ('e2_02_car_by_leaning', 'CAR by political leaning'),
    ('e2_03_car_pre_vs_post', 'CAR pre vs post'),
    ('e2_04_car_by_regime', 'CAR by regime'),
    ('e2_05_car_treat_vs_ctrl', 'CAR treatment vs control'),
    ('e2_06_did_coefficients', 'DiD coefficients'),
    ('e2_07_parallel_trends', 'Parallel trends'),
    ('e2_08_peer_parallel_trends', 'Peer parallel trends'),
    ('e2_09_news_sentiment_dist', 'News sentiment distribution'),
    ('e2_10_news_pre_vs_post', 'News pre vs post'),
    ('e2_11_sentiment_change', 'Sentiment change'),
    ('e2_12_filing_sentiment_sections', 'Filing sentiment sections'),
    ('e2_13_filing_pct_breakdown', 'Filing pct breakdown'),
    ('e2_14_alignment_distribution', 'Alignment distribution'),
    ('e2_15_distinctive_phrases', 'Distinctive phrases'),
    ('e2_16_alignment_validation', 'Alignment validation'),
    ('e2_17_event_nlp_heatmap', 'Event NLP heatmap'),
    ('e2_18_multiwindow_by_horizon', 'Multi-window by horizon'),
    ('e2_19_multiwindow_by_lean', 'Multi-window by leaning'),
    ('e2_20_multiwindow_tc', 'Multi-window treat/ctrl'),
    ('e2_21_contagion_summary', 'Contagion summary'),
    ('e2_22_contagion_by_lean', 'Contagion by leaning'),
    ('e2_23_contagion_by_facing', 'Contagion by facing'),
    ('e2_24_contagion_peer_vs_nonpeer', 'Contagion peer vs non-peer'),
    ('e2_25_contagion_consumer_b2b', 'Contagion consumer vs B2B'),
    ('e2_26_contagion_lean_pairwise', 'Contagion lean pairwise'),
    ('e2_27_contagion_tight_diff', 'Contagion tight DiD'),
    ('e2_28_mda_vs_risk_tone', 'MDA vs risk tone'),
    ('e2_29_alignment_components', 'Alignment components'),
    ('e2_30_car_lean_regime', 'CAR by lean and regime'),
    ('e2_31_event_count_by_lean', 'Event count by leaning'),
    ('e2_32_contagion_mechanism', 'Contagion mechanism'),
    ('e2_33_summary_dashboard', 'Essay 2 summary dashboard'),
]
E3_CHARTS = [
    ('e3_01_window_net_dollar', 'Net dollar sold by window'),
    ('e3_02_window_sell_ratio', 'Net sell ratio by window'),
    ('e3_03_window_transactions', 'Transaction volume by window'),
    ('e3_04_window_opportunistic', 'Opportunistic trades by window'),
    ('e3_05_abnormal_selling', 'Abnormal selling pre vs benchmark'),
    ('e3_06_abnormal_diff', 'Abnormal selling effect size'),
    ('e3_07_treat_vs_ctrl', 'Treatment vs control histogram'),
    ('e3_08_treat_ctrl_box', 'Treatment vs control box plot'),
    ('e3_09_leaning_net_dollar', 'Net dollar sold by leaning'),
    ('e3_10_leaning_median', 'Median selling by leaning'),
    ('e3_11_leaning_variability', 'Selling variability by leaning'),
    ('e3_12_regime_net_sell', 'Net sell daily by regime'),
    ('e3_13_regime_significance', 'Regime significance heatmap'),
    ('e3_14_regime_sample_size', 'Sample size by regime'),
    ('e3_15_rvo_comparison', 'Routine vs opportunistic paired'),
    ('e3_16_rvo_effect_size', 'RVO effect size'),
    ('e3_17_placebo_dist', 'Placebo distribution'),
    ('e3_18_placebo_percentile', 'Placebo percentile gauge'),
    ('e3_19_accel_jt_stats', 'Acceleration JT statistics'),
    ('e3_20_accel_monotonic', 'Acceleration monotonic check'),
    ('e3_21_accel_window_gradient', 'Acceleration window gradient'),
    ('e3_22_panel_distribution', 'Insider selling distribution'),
    ('e3_23_panel_abnormal_flag', 'Abnormal selling classification'),
    ('e3_24_panel_coverage', 'Data sufficiency coverage'),
    ('e3_25_car_vs_selling', 'CAR vs insider selling scatter'),
    ('e3_26_panel_lean_box', 'Selling by leaning box plot'),
    ('e3_27_panel_windows', 'Pre-event sub-window comparison'),
    ('e3_28_panel_opp_routine', 'Opportunistic vs routine stacked'),
    ('e3_29_panel_insiders', 'Unique insiders by window'),
    ('e3_30_abnormal_pvalues', 'Abnormal t vs Wilcoxon p-values'),
    ('e3_31_sell_ratio_by_treatment', 'Sell ratio by treatment'),
    ('e3_32_regime_t_stats', 'Regime t-statistics'),
    ('e3_33_post_event_selling', 'Post-event selling distribution'),
    ('e3_34_summary_dashboard', 'Essay 3 summary dashboard'),
]


# =============================================================================
# DATA LOADING (cached)
# =============================================================================

@st.cache_data(ttl=300)
def query_df(sql):
    """Run a SQL query against the active backend (Athena or SQLite)."""
    try:
        with _get_loader() as db:
            return db.run_query(sql)
    except Exception as e:
        logging.warning("Query failed: %s", e)
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_table(table_name):
    """Load a full table from the active backend (Athena or SQLite).

    Model-result tables (written by Model.py, not the ETL pipeline) only exist
    in SQLite, so we fall back automatically when Athena raises an error.

    Returns pd.DataFrame() on failure. Use the sidebar Refresh button to clear
    cached failures.
    """
    try:
        with _get_loader() as db:
            df = db.read_table(table_name)
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()
    except Exception:
        pass
    # Table may only exist in SQLite (model results, figures, etc.)
    if DB_BACKEND == "athena":
        try:
            with SQLiteLoader(db_path=str(DB_PATH)) as db:
                df = db.read_table(table_name)
                if df is not None and not df.empty:
                    return df
        except Exception as e:
            logging.warning("load_table(%s) failed on both backends: %s", table_name, e)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_figure_blob(figure_name):
    """Load a figure's PNG bytes from the FIGURES table."""
    safe_name = figure_name.replace("'", "''")
    try:
        with _get_loader() as db:
            df = db.run_query(
                f"SELECT IMAGE_DATA FROM FIGURES WHERE FIGURE_NAME = '{safe_name}'"
            )
            if not df.empty:
                return df.iloc[0]["IMAGE_DATA"]
    except Exception:
        pass
    # Fallback to SQLite if Athena didn't have it
    if DB_BACKEND == "athena":
        try:
            with SQLiteLoader(db_path=str(DB_PATH)) as db:
                df = db.run_query(
                    f"SELECT IMAGE_DATA FROM FIGURES WHERE FIGURE_NAME = '{safe_name}'"
                )
                if not df.empty:
                    return df.iloc[0]["IMAGE_DATA"]
        except Exception:
            pass
    return None


@st.cache_data(ttl=300)
def load_summary():
    """Load aggregate summary stats from the database."""
    summary = {}
    try:
        run = load_table("MODEL_RUN_SUMMARY")
        if not run.empty:
            r = run.iloc[-1]
            summary["n_events"] = int(r.get("N_EVENTS", 0))
            summary["n_tickers"] = int(r.get("N_TICKERS", 0))
            summary["n_event_studies"] = int(r.get("N_EVENT_STUDIES", 0))
            summary["n_significant"] = int(r.get("N_SIGNIFICANT_CAR_005", 0))
            summary["avg_car"] = float(r.get("AVG_CAR", 0))
            summary["median_car"] = float(r.get("MEDIAN_CAR", 0))
            summary["n_did"] = int(r.get("N_DID", 0))
            summary["model_name"] = r.get("MODEL_NAME", "FF5")
    except Exception:
        pass
    return summary


# =============================================================================
# DATABASE CHECK
# =============================================================================

@st.cache_data(ttl=60)
def database_has_data():
    """Check if model results exist in the active backend."""
    _check_tables = [
        "EVENT_STUDY_RESULTS",
        "ESSAY1_FF5_COEFFICIENTS",
        "ESSAY3_INSIDER_PANEL",
    ]
    for _tbl in _check_tables:
        try:
            with _get_loader() as db:
                result = db.run_query(f"SELECT COUNT(*) as n FROM {_tbl}")
                if not result.empty and result.iloc[0]["n"] > 0:
                    return True
        except Exception:
            if DB_BACKEND == "athena" and DB_PATH.exists():
                try:
                    with SQLiteLoader(db_path=str(DB_PATH)) as db:
                        result = db.run_query(f"SELECT COUNT(*) as n FROM {_tbl}")
                        if not result.empty and result.iloc[0]["n"] > 0:
                            return True
                except Exception:
                    pass
    return False


_DATA_EXISTS = database_has_data()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        f"<h2 style='font-size:1.25rem; line-height:1.4; margin-bottom:0.25rem;'>"
        f"Signals & Systems</h2>",
        unsafe_allow_html=True,
    )
    st.caption("The Political Economy of Investor Sentiment and Financial Innovation")
    st.divider()
    if st.button("\u21ba Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        f"<p style='font-size:0.8rem; line-height:1.5; margin-top:0.25rem; "
        f"color:rgba(255,255,255,0.75);'>"
        f"Ashley D. Roseboro<br>"
        f"<span style='font-size:0.75rem; color:rgba(255,255,255,0.5);'>"
        f"College of Arts and Sciences<br>University of South Alabama</span></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    summary = load_summary()

    col_l, col_r = st.columns(2)
    col_l.metric("Culture War Events", fmt_num(summary.get("n_events", 0)))
    col_r.metric("Unique Tickers", fmt_num(summary.get("n_tickers", 0)))
    col_l2, col_r2 = st.columns(2)
    col_l2.metric("Event Studies", fmt_num(summary.get("n_event_studies", 0)))
    col_r2.metric("Significant (p<.05)", fmt_num(summary.get("n_significant", 0)))

    st.divider()
    st.markdown("#### Three-Essay Structure")
    for num, title in [
        (1, "Breaking the Model"),
        (2, "Culture Wars & Capital Markets"),
        (3, "Insider Trading & Controversies"),
    ]:
        st.markdown(
            f"<span style='color:{GOLD};font-weight:600;'>{num}.</span> {title}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Data Sources")
    for src in [
        "Culture War Companies (160 events)",
        "Yahoo Finance (Stock Data)",
        "Fama-French 5-Factor + Momentum",
        "CBOE VIX",
        "FRED Macro Series (GDP, CPI, Rates)",
        "Guardian, NYT, Reddit (News)",
    ]:
        st.markdown(
            f"<span style='color:{GOLD}; margin-right:0.4rem;'>&#8226;</span> {src}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Methodology")
    for item in [
        "Markov Regime-Switching (VIX)",
        "FF5 Spanning & Pricing Regressions",
        "Matched Control (Paired t-test)",
        "Event Study (CAR, Patell t-test)",
        "Difference-in-Differences (DiD)",
        "FinBERT Sentiment & FOMO Z-Scores",
        "Benjamini-Hochberg FDR Correction",
    ]:
        st.markdown(
            f"<span style='color:{GOLD}; margin-right:0.4rem;'>&#8226;</span> {item}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Pipeline")

    # --- Step 1: Clean (long-running, background) ---
    _clean_log = BASE_DIR / "data" / "clean.log"
    _clean_pid = BASE_DIR / "data" / "clean.pid"

    _clean_running = False
    if _clean_pid.exists():
        try:
            _pid = int(_clean_pid.read_text().strip())
            os.kill(_pid, 0)  # check if alive
            _clean_running = True
        except (OSError, ValueError):
            _clean_pid.unlink(missing_ok=True)

    if _clean_running:
        _progress_file = BASE_DIR / "data" / "clean_progress.txt"
        _pct_label = ""
        if _progress_file.exists():
            try:
                _parts = _progress_file.read_text().strip().split("|")
                _step_info, _pct, _step_name = _parts[0], _parts[1], _parts[2]
                _pct_label = f" — {_pct} ({_step_name})"
                _pct_int = int(_pct.replace("%", ""))
                st.progress(_pct_int / 100, text=f"Step 1: Cleaning{_pct_label}")
            except Exception:
                st.info(f"Step 1: Cleaning in progress (PID {_pid})")
        else:
            st.info(f"Step 1: Cleaning in progress (PID {_pid})")
        if _clean_log.exists():
            _tail = _clean_log.read_text().splitlines()[-5:]
            st.caption("Last 5 lines:")
            st.code("\n".join(_tail), language="text")
    else:
        if _clean_log.exists():
            _last_line = _clean_log.read_text().splitlines()[-1:]
            if _last_line:
                st.caption(f"Last clean run: {_last_line[0][:120]}")

        if st.button("Step 1: Clean Raw Data (~20h)", use_container_width=True):
            _clean_log.parent.mkdir(parents=True, exist_ok=True)
            with open(_clean_log, "w") as _lf:
                _proc = subprocess.Popen(
                    [sys.executable, "-m", "clean.orchestration"],
                    cwd=str(BASE_DIR),
                    stdout=_lf, stderr=subprocess.STDOUT,
                )
            _clean_pid.write_text(str(_proc.pid))
            st.success(f"Clean started in background (PID {_proc.pid}). Check back later.")
            st.rerun()

    # --- Steps 2-9: ETL through charts (foreground) ---
    if st.button("Step 2: Run ETL → Models → Charts", use_container_width=True):
        _steps = [
            ("Running ETL", [sys.executable, "ETL.py"]),
            ("Loading database", [sys.executable, "Database.py"]),
            ("Essay 1: Regimes & FF5", [sys.executable, "-m", "model.essay1"]),
            ("Essay 1: Matched Controls", [sys.executable, "-m", "model.essay1_matched"]),
            ("Essay 2: NLP & Alignment", [sys.executable, "-m", "model.essay2"]),
            ("Essay 2: DiD Analysis", [sys.executable, "-m", "model.essay2_did"]),
            ("Essay 3: Insider Trading", [sys.executable, "-m", "model.essay3"]),
            ("Generating charts", [sys.executable, "visual.py"]),
        ]
        with st.status("Running pipeline (ETL → charts)...", expanded=True) as _status:
            _failed = False
            for _i, (_label, _cmd) in enumerate(_steps):
                st.write(f"Step {_i+1}/{len(_steps)}: {_label}...")
                _result = subprocess.run(
                    _cmd, cwd=str(BASE_DIR), capture_output=True, text=True,
                    timeout=1800,
                )
                if _result.returncode != 0:
                    st.error(f"Failed at: {_label}\n```\n{_result.stderr[-500:]}\n```")
                    _status.update(label=f"Pipeline failed at: {_label}", state="error")
                    _failed = True
                    break
            if not _failed:
                _status.update(label="Pipeline complete!", state="complete")
                st.cache_data.clear()
                st.rerun()

    st.divider()
    _backend_label = "AWS Athena" if DB_BACKEND == "athena" else "Local SQLite"
    st.caption(f"Data: {_backend_label}")
    st.caption("Roseboro | University of South Alabama | 2026")


# =============================================================================
# MAIN TABS
# =============================================================================

tab_overview, tab_a1, tab_a2, tab_a3, tab_enriched, tab_raw = st.tabs([
    "Study Overview",
    "Article 1",
    "Article 2",
    "Article 3",
    "Enriched Data",
    "Raw Data",
])


# =============================================================================
# TAB 0: STUDY OVERVIEW
# =============================================================================

with tab_overview:
    # Title page
    st.markdown(
        f"<div style='text-align:center; padding:2rem 1rem 1rem;'>"
        f"<p style='font-size:0.95rem; text-transform:uppercase; letter-spacing:0.08em; "
        f"color:{NAVY}; margin-bottom:0.5rem;'>University of South Alabama</p>"
        f"<p style='font-size:0.85rem; color:{NAVY}; margin-bottom:2rem;'>"
        f"College of Arts and Sciences</p>"
        f"<h1 style='font-size:1.8rem; line-height:1.4; margin-bottom:0.5rem;'>"
        f"Signals and Systems:<br>The Political Economy of Investor Sentiment "
        f"and Financial Innovation</h1>"
        f"<p style='font-size:1rem; color:{NAVY}; margin-top:1.5rem;'>"
        f"A Dissertation</p>"
        f"<p style='font-size:0.95rem; color:{NAVY}; margin-top:0.5rem;'>"
        f"Submitted in partial fulfillment of the requirements<br>"
        f"for the degree of Doctor of Philosophy</p>"
        f"<p style='font-size:1rem; font-weight:600; color:{NAVY}; margin-top:1.5rem;'>"
        f"Ashley D. Roseboro</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Committee
    st.markdown(
        f"<div style='text-align:center; padding:1rem; margin-bottom:1rem;'>"
        f"<p style='font-size:0.85rem; color:{NAVY};'>"
        f"<strong>Committee Chair:</strong> Ying Johnson, Ph.D.<br>"
        f"<strong>Committee Members:</strong> Ermanno Affuso, Ph.D., "
        f"Joshua Lambert, Ph.D., and Xiankui (Bill) Hu, Ph.D.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Abstract ---
    st.subheader("Abstract")
    st.markdown(
        "This dissertation examines the intersection of political signaling, investor "
        "sentiment, and financial market behavior through three interconnected essays. "
        "Drawing on behavioral finance theory, market microstructure, and political economy, "
        "the research investigates how political signals and culture war events impact "
        "financial markets, asset pricing models, and trading behavior."
    )
    st.markdown(
        "**Essay 1**, *\"Breaking the Model: How Culture Wars Expose Factor Model "
        "Fragility,\"* examines how political events create market conditions that "
        "challenge traditional asset pricing frameworks, specifically the Fama-French "
        "five-factor model. **Essay 2**, *\"Culture Wars and Capital Markets: The Political "
        "Economy of Abnormal Returns,\"* investigates the impact of culture war events on "
        "stock returns using event study methodology and difference-in-differences analysis. "
        "**Essay 3**, *\"Insider Trading and Political Controversies: Evidence from Culture "
        "War Events,\"* explores whether corporate insiders trade on advance knowledge of "
        "upcoming political controversies."
    )
    st.markdown(
        "The study utilizes a comprehensive dataset of 160 culture war events spanning "
        "2015-2025, combined with daily stock returns, Fama-French factor data, and "
        "macroeconomic indicators. Methodologically, the research employs event study "
        "analysis with multiple factor model specifications (FF3, FF5, FF5+MOM), "
        "difference-in-differences estimation with matched control firms, and "
        "cross-sectional regression analysis to identify heterogeneous treatment effects "
        "across firms classified by political leaning (Conservative, Liberal, Mixed)."
    )

    st.divider()

    # --- Chapter 1: Introduction ---
    st.subheader("Chapter 1: Introduction")

    st.markdown("##### 1.1 Background and Motivation")
    st.markdown(
        "The American political landscape has undergone a dramatic transformation over "
        "the past decade, with corporations increasingly finding themselves at the center "
        "of partisan disputes. From Nike's Colin Kaepernick campaign to Disney's opposition "
        "to Florida's Parental Rights in Education Act, companies have become active "
        "participants in what scholars term \"culture wars\" -- public conflicts over "
        "fundamental social values and norms."
    )
    st.markdown(
        "These culture war events represent a unique and understudied category of "
        "corporate risk. Unlike traditional operational or financial risks, culture war "
        "events are inherently political, generating responses that split along partisan "
        "lines. When Nike featured Colin Kaepernick in its \"Just Do It\" campaign in "
        "September 2018, the company simultaneously experienced boycott calls from "
        "conservative consumers and increased brand loyalty from liberal consumers. "
        "This duality creates a natural experiment: how do financial markets process "
        "information that is valued differently by ideologically segmented investor groups?"
    )

    st.markdown("##### 1.2 Research Questions")
    rq_col1, rq_col2, rq_col3 = st.columns(3)
    with rq_col1:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:10rem;'>"
            f"<strong style='color:{GOLD};'>Essay 1</strong><br>"
            f"Do culture war events create market conditions that expose "
            f"<strong>factor model fragility</strong> in traditional asset pricing "
            f"frameworks?</div>",
            unsafe_allow_html=True,
        )
    with rq_col2:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:10rem;'>"
            f"<strong style='color:{GOLD};'>Essay 2</strong><br>"
            f"How do culture war events generate <strong>abnormal returns</strong>, "
            f"and do these effects vary by a firm's political alignment?</div>",
            unsafe_allow_html=True,
        )
    with rq_col3:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:10rem;'>"
            f"<strong style='color:{GOLD};'>Essay 3</strong><br>"
            f"Do corporate <strong>insiders trade</strong> on advance knowledge of "
            f"upcoming political controversies?</div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### 1.3 Significance and Contributions")
    st.markdown(
        "This dissertation makes several contributions to the literature:\n\n"
        "1. **To asset pricing theory**: By documenting how political events create "
        "pricing anomalies that traditional factor models cannot explain, extending the "
        "work of Fama and French (2015) and Hong and Kacperczyk (2009).\n\n"
        "2. **To behavioral finance**: By providing evidence of politically motivated "
        "trading behavior and its impact on market efficiency, building on the sentiment "
        "framework of Baker and Wurgler (2006).\n\n"
        "3. **To corporate governance**: By examining whether insiders exploit political "
        "controversies for personal gain, contributing to the insider trading literature "
        "of Cohen, Malloy, and Pomorski (2012).\n\n"
        "4. **To political economy of finance**: By documenting the mechanisms through "
        "which political polarization enters financial markets, extending the work of "
        "Addoum and Kumar (2016) and Cookson, Engelberg, and Mullins (2020)."
    )

    st.divider()

    # --- Chapter 2: Literature Review ---
    st.subheader("Chapter 2: Literature Review")

    st.markdown("##### 2.1 Asset Pricing and Factor Models")
    st.markdown(
        "The Capital Asset Pricing Model (CAPM) of Sharpe (1964) and Lintner (1965) "
        "established the foundation for understanding the relationship between risk and "
        "expected returns. The Fama-French three-factor model (1993) extended this "
        "framework by adding size (SMB) and value (HML) factors, while the five-factor "
        "model (2015) incorporated profitability (RMW) and investment (CMA). Despite "
        "these advances, considerable evidence suggests that factor models exhibit "
        "systematic breakdowns during periods of market stress or when behavioral factors "
        "dominate (Kozak, Nagel, and Santosh, 2018)."
    )

    st.markdown("##### 2.2 Political Economy and Financial Markets")
    st.markdown(
        "A growing body of research examines the relationship between political factors "
        "and financial markets. Hong and Kacperczyk (2009) document that \"sin stocks\" "
        "receive less analyst coverage and institutional ownership due to social norm "
        "constraints. Addoum and Kumar (2016) show that political climate affects local "
        "stock market participation and portfolio allocation. Cookson, Engelberg, and "
        "Mullins (2020) provide direct evidence that partisan disagreement increases "
        "stock market trading volume using data from StockTwits."
    )

    st.markdown("##### 2.3 Event Study Methodology")
    st.markdown(
        "Event study methodology, formalized by Fama, Fisher, Jensen, and Roll (1969) "
        "and refined by Brown and Warner (1985), provides the primary empirical tool for "
        "assessing the market impact of culture war events. The approach estimates \"normal\" "
        "returns using a factor model over an estimation window, then measures abnormal "
        "returns during the event window as the difference between actual and predicted "
        "returns. Cumulative Abnormal Returns (CAR) aggregate these effects across the "
        "event window."
    )

    st.markdown("##### 2.4 Insider Trading and Information Asymmetry")
    st.markdown(
        "The insider trading literature provides the theoretical foundation for Essay 3. "
        "Corporate insiders possess material nonpublic information that may include advance "
        "knowledge of planned political statements or responses to emerging controversies. "
        "Cohen, Malloy, and Pomorski (2012) distinguish between \"routine\" and "
        "\"opportunistic\" insider trades, finding that opportunistic trades predict future "
        "returns and firm news."
    )

    st.divider()

    # --- Chapter 3: Data and Methodology ---
    st.subheader("Chapter 3: Data and Methodology")

    st.markdown("##### 3.1 Data Sources")
    data_col1, data_col2 = st.columns(2)
    with data_col1:
        st.markdown("**Primary Data**")
        st.markdown(
            "- **Culture War Events**: 160 events from 2015-2025, manually curated\n"
            "- **Stock Data**: Daily OHLCV from Yahoo Finance\n"
            "- **Factor Data**: FF3, FF5, Momentum from Ken French's library\n"
            "- **Control Firms**: Matched by industry and size"
        )
    with data_col2:
        st.markdown("**Supplementary Data**")
        st.markdown(
            "- **VIX**: CBOE Volatility Index\n"
            "- **Macro**: GDP, CPI, employment, interest rates from FRED\n"
            "- **News**: Guardian, NYT, Reddit sentiment data\n"
            "- **SEC Form 4**: Insider transaction filings"
        )

    st.markdown("##### 3.2 Methodology Overview")
    meth_data = {
        "Component": [
            "Factor Model Estimation",
            "Event Study (CAR)",
            "Difference-in-Differences",
            "Cross-Sectional Analysis",
        ],
        "Description": [
            "FF3, FF5, FF5+MOM specifications over 252-day estimation window",
            "Cumulative Abnormal Returns over [-30, +30] event window with Patell t-test",
            "Treatment firm vs. matched control with factor controls",
            "Mean CAR grouped by political leaning with t-tests for significance",
        ],
        "Essay": [
            "Essay 1",
            "Essay 2",
            "Essay 2 & 3",
            "Essay 2",
        ],
    }
    try:
        st.dataframe(pd.DataFrame(meth_data), use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(pd.DataFrame(meth_data), use_container_width=True)

    st.markdown("##### 3.3 Event Study Design")
    st.markdown(
        "The event study follows the standard framework:\n\n"
        "1. **Estimation Window**: 252 trading days ending 10 days before the event\n"
        "2. **Gap Period**: 10 trading days (to prevent contamination)\n"
        "3. **Event Window**: [-30, +30] trading days around the event date\n"
        "4. **Normal Returns**: Estimated via OLS regression of firm excess returns "
        "on factor returns\n"
        "5. **Abnormal Returns**: AR(t) = R(t) - E[R(t)] where E[R(t)] comes from "
        "the factor model\n"
        "6. **Significance**: Patell (1976) standardized t-test for CAR"
    )

    st.divider()

    # --- References ---
    with st.expander("References", expanded=False):
        st.markdown(
            "- Addoum, J.M. and Kumar, A. (2016). Political sentiment and predictable returns. "
            "*Review of Financial Studies*, 29(12), 3471-3518.\n"
            "- Baker, M. and Wurgler, J. (2006). Investor sentiment and the cross-section of "
            "stock returns. *Journal of Finance*, 61(4), 1645-1680.\n"
            "- Brown, S.J. and Warner, J.B. (1985). Using daily stock returns: The case of "
            "event studies. *Journal of Financial Economics*, 14(1), 3-31.\n"
            "- Cohen, L., Malloy, C. and Pomorski, L. (2012). Decoding inside information. "
            "*Journal of Finance*, 67(3), 1009-1043.\n"
            "- Cookson, J.A., Engelberg, J.E. and Mullins, W. (2020). Does partisanship shape "
            "investor beliefs? Evidence from the COVID-19 pandemic. *Review of Asset Pricing "
            "Studies*, 10(4), 863-893.\n"
            "- Fama, E.F. and French, K.R. (1993). Common risk factors in the returns on "
            "stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.\n"
            "- Fama, E.F. and French, K.R. (2015). A five-factor asset pricing model. "
            "*Journal of Financial Economics*, 116(1), 1-22.\n"
            "- Fama, E.F., Fisher, L., Jensen, M.C. and Roll, R. (1969). The adjustment of "
            "stock prices to new information. *International Economic Review*, 10(1), 1-21.\n"
            "- Hong, H. and Kacperczyk, M. (2009). The price of sin: The effects of social "
            "norms on markets. *Journal of Financial Economics*, 93(1), 15-36.\n"
            "- Kozak, S., Nagel, S. and Santosh, S. (2018). Interpreting factor models. "
            "*Journal of Finance*, 73(3), 1183-1223.\n"
            "- Patell, J.M. (1976). Corporate forecasts of earnings per share and stock price "
            "behavior: Empirical test. *Journal of Accounting Research*, 14(2), 246-276.\n"
            "- Sharpe, W.F. (1964). Capital asset prices: A theory of market equilibrium "
            "under conditions of risk. *Journal of Finance*, 19(3), 425-442."
        )


# =============================================================================
# TAB 1: ARTICLE 1 - Breaking the Model
# =============================================================================

with tab_a1:
    st.header("Essay 1: Breaking the Model")
    st.markdown("*How Culture Wars Expose Factor Model Fragility*")
    st.markdown(
        f"<p style='font-size:1rem; color:{NAVY}; margin-top:-0.5rem;'>"
        f"<strong>Research Question:</strong> Do culture war events create market conditions "
        f"that challenge the explanatory power of the Fama-French five-factor model?</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load data
    event_df = load_table("EVENT_STUDY_RESULTS")
    cw_df_a1 = load_table("CULTURE_WAR_COMPANIES")
    ok_a1 = event_df[event_df["STATUS"] == "OK"].copy() if not event_df.empty else pd.DataFrame()

    if ok_a1.empty:
        st.info("No event study results available. Run Model.py first.")
    else:
        for col in ["CAR", "CAR_T", "CAR_P", "BHAR", "ALPHA", "MKT_BETA", "R_SQUARED", "N_OBS"]:
            if col in ok_a1.columns:
                ok_a1[col] = pd.to_numeric(ok_a1[col], errors="coerce")

        # Merge industry from culture war companies
        if not cw_df_a1.empty and "TICKER" in cw_df_a1.columns:
            ok_a1 = ok_a1.merge(
                cw_df_a1[["TICKER", "INDUSTRY", "YEAR"]].drop_duplicates(subset=["TICKER"]),
                on="TICKER", how="left",
            )

        # --- Key finding callout ---
        rsq = ok_a1["R_SQUARED"].dropna()
        alpha = ok_a1["ALPHA"].dropna()
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1.25rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
            f"<strong style='color:{GOLD};'>Key Finding</strong><br>"
            f"The FF5 model explains only <strong>{rsq.mean():.1%}</strong> of return variation "
            f"on average during culture war event windows (median R-sq = {rsq.median():.4f}). "
            f"<strong>{(rsq < 0.20).mean():.0%}</strong> of regressions have R-sq below 0.20, "
            f"indicating systematic factor model breakdown when political signals dominate.</div>",
            unsafe_allow_html=True,
        )

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Events Analyzed", fmt_num(len(ok_a1)))
        c2.metric("Mean R-sq", fmt_num(rsq.mean(), 4))
        c3.metric("Median R-sq", fmt_num(rsq.median(), 4))
        c4.metric("Mean Alpha (daily)", fmt_num(alpha.mean(), 6))
        c5.metric("Mean Mkt Beta", fmt_num(ok_a1["MKT_BETA"].mean(), 3))

        st.divider()

        # --- R-squared distribution ---
        st.subheader("R-squared Distribution")
        st.markdown(
            "The distribution of R-squared values from FF5 regressions over 252-day "
            "estimation windows. Low values indicate that the five factors (MKT-RF, "
            "SMB, HML, RMW, CMA) fail to capture the return-generating process for "
            "firms embroiled in culture war events."
        )

        rsq_stats = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std Dev", "25th Percentile", "75th Percentile",
                          "Min", "Max", "R-sq < 0.10", "R-sq < 0.20", "R-sq > 0.50"],
            "Value": [
                f"{rsq.mean():.4f}",
                f"{rsq.median():.4f}",
                f"{rsq.std():.4f}",
                f"{rsq.quantile(0.25):.4f}",
                f"{rsq.quantile(0.75):.4f}",
                f"{rsq.min():.4f}",
                f"{rsq.max():.4f}",
                f"{(rsq < 0.10).sum()} ({(rsq < 0.10).mean():.1%})",
                f"{(rsq < 0.20).sum()} ({(rsq < 0.20).mean():.1%})",
                f"{(rsq > 0.50).sum()} ({(rsq > 0.50).mean():.1%})",
            ],
        })
        render_df(rsq_stats, "R-sq Distribution", "a1_rsq_dist")

        st.divider()

        # --- Alpha distribution ---
        st.subheader("Alpha (Intercept) Analysis")
        st.markdown(
            "Non-zero alphas represent returns that the factor model cannot explain. "
            "Under the efficient markets hypothesis, alphas should be zero on average. "
            "Systematic deviations during culture war windows reveal the limits of "
            "traditional pricing frameworks."
        )

        alpha_col1, alpha_col2 = st.columns(2)
        with alpha_col1:
            alpha_stats = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "25th Percentile", "75th Percentile",
                              "Positive %", "Mean (Positive)", "Mean (Negative)"],
                "Value": [
                    f"{alpha.mean():.6f}",
                    f"{alpha.median():.6f}",
                    f"{alpha.std():.6f}",
                    f"{alpha.quantile(0.25):.6f}",
                    f"{alpha.quantile(0.75):.6f}",
                    f"{(alpha > 0).mean():.1%}",
                    f"{alpha[alpha > 0].mean():.6f}" if (alpha > 0).any() else "--",
                    f"{alpha[alpha <= 0].mean():.6f}" if (alpha <= 0).any() else "--",
                ],
            })
            render_df(alpha_stats, "Alpha Distribution", "a1_alpha_dist")

        with alpha_col2:
            beta = ok_a1["MKT_BETA"].dropna()
            beta_stats = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Std Dev", "25th Percentile", "75th Percentile",
                              "Beta < 0.5", "Beta 0.5-1.5", "Beta > 1.5"],
                "Value": [
                    f"{beta.mean():.4f}",
                    f"{beta.median():.4f}",
                    f"{beta.std():.4f}",
                    f"{beta.quantile(0.25):.4f}",
                    f"{beta.quantile(0.75):.4f}",
                    f"{(beta < 0.5).sum()} ({(beta < 0.5).mean():.1%})",
                    f"{((beta >= 0.5) & (beta <= 1.5)).sum()} ({((beta >= 0.5) & (beta <= 1.5)).mean():.1%})",
                    f"{(beta > 1.5).sum()} ({(beta > 1.5).mean():.1%})",
                ],
            })
            render_df(beta_stats, "Market Beta Distribution", "a1_beta_dist")

        st.divider()

        # --- Model fit by political leaning ---
        st.subheader("Model Fit by Political Leaning")
        st.markdown(
            "Does the factor model break down more for firms of a particular "
            "political alignment? Comparing R-squared and alpha across groups "
            "tests whether politically charged firms are systematically mispriced."
        )

        if "POLITICAL_LEANING" in ok_a1.columns:
            leaning_fit = ok_a1.groupby("POLITICAL_LEANING").agg(
                mean_rsq=("R_SQUARED", "mean"),
                median_rsq=("R_SQUARED", "median"),
                mean_alpha=("ALPHA", "mean"),
                mean_beta=("MKT_BETA", "mean"),
                n=("R_SQUARED", "count"),
            ).reset_index()
            leaning_fit.columns = ["Political Leaning", "Mean R-sq", "Median R-sq",
                                   "Mean Alpha", "Mean Mkt Beta", "N"]
            leaning_fit = leaning_fit.round(4)
            render_df(leaning_fit, "Model Fit by Leaning", "a1_fit_leaning")

        st.divider()

        # --- Worst-fit events (fragility cases) ---
        st.subheader("Most Fragile Cases: Lowest R-squared Events")
        st.markdown(
            "Events where the factor model explains the least variation in returns. "
            "These are the starkest examples of model fragility -- traditional "
            "factors are essentially irrelevant."
        )

        worst = ok_a1.nsmallest(15, "R_SQUARED")[
            ["TICKER", "COMPANY", "EVENT_DATE", "POLITICAL_LEANING",
             "R_SQUARED", "ALPHA", "MKT_BETA", "CAR"]
        ].copy()
        worst = worst.round({"R_SQUARED": 4, "ALPHA": 6, "MKT_BETA": 4, "CAR": 4})
        render_df(worst, "Lowest R-sq Events", "a1_worst_rsq")

        st.divider()

        # --- Model fit by industry ---
        if "INDUSTRY" in ok_a1.columns:
            st.subheader("Model Fit by Industry")
            st.markdown(
                "Industry-level variation in model explanatory power. Sectors with "
                "lower R-squared may be more susceptible to politically-driven "
                "pricing disconnects."
            )

            industry_fit = ok_a1.groupby("INDUSTRY").agg(
                mean_rsq=("R_SQUARED", "mean"),
                mean_alpha=("ALPHA", "mean"),
                mean_car=("CAR", "mean"),
                n=("R_SQUARED", "count"),
            ).reset_index()
            industry_fit.columns = ["Industry", "Mean R-sq", "Mean Alpha", "Mean CAR", "N"]
            industry_fit = industry_fit[industry_fit["N"] >= 2].sort_values("Mean R-sq")
            industry_fit = industry_fit.round(4)
            render_df(industry_fit, "Model Fit by Industry", "a1_fit_industry")

            st.divider()

        # --- Model fit by year ---
        if "YEAR" in ok_a1.columns:
            st.subheader("Model Fit Over Time")
            st.markdown(
                "Has factor model fragility worsened as culture wars intensified? "
                "Tracking mean R-squared by event year reveals temporal trends in "
                "the model's ability to price politically-exposed firms."
            )

            year_fit = ok_a1.groupby("YEAR").agg(
                mean_rsq=("R_SQUARED", "mean"),
                mean_alpha=("ALPHA", "mean"),
                mean_car=("CAR", "mean"),
                n=("R_SQUARED", "count"),
            ).reset_index()
            year_fit.columns = ["Year", "Mean R-sq", "Mean Alpha", "Mean CAR", "N"]
            year_fit = year_fit.sort_values("Year")
            year_fit = year_fit.round(4)
            render_df(year_fit, "Model Fit by Year", "a1_fit_year")

            st.divider()

        # --- Estimation quality ---
        st.subheader("Estimation Quality Summary")
        st.markdown(
            "Overview of the estimation windows used across all event studies. "
            "Adequate sample size in the estimation window is critical for reliable "
            "factor loading estimates."
        )

        nobs = ok_a1["N_OBS"].dropna()
        est_stats = pd.DataFrame({
            "Statistic": ["Mean Estimation N", "Median", "Min", "Max",
                          "N < 200 (short window)", "N >= 250 (full year)"],
            "Value": [
                f"{nobs.mean():.0f}",
                f"{nobs.median():.0f}",
                f"{nobs.min():.0f}",
                f"{nobs.max():.0f}",
                f"{(nobs < 200).sum()} ({(nobs < 200).mean():.1%})",
                f"{(nobs >= 250).sum()} ({(nobs >= 250).mean():.1%})",
            ],
        })
        render_df(est_stats, "Estimation Quality", "a1_est_quality")

        # Full event-level results
        with st.expander("Full Event-Level Factor Model Results", expanded=False):
            display_cols = ["TICKER", "COMPANY", "EVENT_DATE", "POLITICAL_LEANING",
                            "ALPHA", "MKT_BETA", "R_SQUARED", "N_OBS", "CAR", "MODEL_NAME"]
            if "INDUSTRY" in ok_a1.columns:
                display_cols.insert(4, "INDUSTRY")
            display = ok_a1[[c for c in display_cols if c in ok_a1.columns]].copy()
            display = display.round({"ALPHA": 6, "MKT_BETA": 4, "R_SQUARED": 4, "CAR": 4})
            display = display.sort_values("R_SQUARED")
            render_df(display, "Factor Model Results", "a1_full_results", height=500)

    # =====================================================================
    # ESSAY 1 — MARKOV REGIME ANALYSIS (from model/essay1.py)
    # =====================================================================

    st.divider()
    st.header("Volatility Regime Analysis")
    st.markdown(
        "Markov regime-switching on VIX identifies distinct volatility regimes "
        "(Hamilton, 1989). FF5 spanning regressions and culture war stock pricing "
        "regressions are then estimated within each regime."
    )

    # --- Model Selection (K=2,3,4) ---
    model_sel = load_table("ESSAY1_MODEL_SELECTION")
    if not model_sel.empty:
        st.subheader("Regime Model Selection")
        st.markdown(
            "Comparing Markov regime-switching models across K = 2, 3, 4 regimes. "
            "Lower AIC/BIC indicates better fit."
        )

        ms_display = model_sel.copy()
        for col in ms_display.columns:
            ms_display[col] = pd.to_numeric(ms_display[col], errors="coerce")

        # Highlight best
        best_aic_k = ms_display.loc[ms_display["AIC"].idxmin(), "K"] if "AIC" in ms_display.columns else None
        best_bic_k = ms_display.loc[ms_display["BIC"].idxmin(), "K"] if "BIC" in ms_display.columns else None

        if best_aic_k is not None:
            st.markdown(
                f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
                f"padding:1rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
                f"Best AIC: <strong>K = {int(best_aic_k)}</strong> &nbsp;|&nbsp; "
                f"Best BIC: <strong>K = {int(best_bic_k)}</strong></div>",
                unsafe_allow_html=True,
            )

        ms_display = ms_display.round({"AIC": 1, "BIC": 1, "LOG_LIKELIHOOD": 1})
        render_df(ms_display, "Model Selection", "a1_model_selection")

        st.divider()

    # --- Regime coefficients ---
    regime_coeff = load_table("ESSAY1_FF5_COEFFICIENTS")
    factor_premia = load_table("ESSAY1_FACTOR_PREMIA")

    if not regime_coeff.empty:
        st.subheader("FF5 Spanning Regression by Regime")
        st.markdown(
            "MKT-RF regressed on SMB, HML, RMW, CMA within each volatility regime. "
            "The Chow test confirms a statistically significant structural break (p < 0.001)."
        )

        # Key finding callout
        low_alpha = regime_coeff.loc[regime_coeff["REGIME"].str.contains("Low"), "ALPHA"]
        high_alpha = regime_coeff.loc[regime_coeff["REGIME"].str.contains("High"), "ALPHA"]
        low_rsq = regime_coeff.loc[regime_coeff["REGIME"].str.contains("Low"), "R_SQUARED"]
        high_rsq = regime_coeff.loc[regime_coeff["REGIME"].str.contains("High"), "R_SQUARED"]

        if not low_alpha.empty and not high_alpha.empty:
            st.markdown(
                f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
                f"padding:1.25rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
                f"<strong style='color:{GOLD};'>Key Finding</strong><br>"
                f"Market alpha is <strong>{low_alpha.values[0]:.4f}</strong> in Low Volatility "
                f"(p < 0.001) but disappears in High Volatility "
                f"(<strong>{high_alpha.values[0]:.4f}</strong>, p = 0.60). "
                f"R-squared rises from <strong>{low_rsq.values[0]:.3f}</strong> to "
                f"<strong>{high_rsq.values[0]:.3f}</strong> as crisis-driven factors explain more "
                f"of the market.</div>",
                unsafe_allow_html=True,
            )

        # Format for display
        coeff_display = regime_coeff.copy()
        for col in coeff_display.columns:
            if col != "REGIME":
                coeff_display[col] = pd.to_numeric(coeff_display[col], errors="coerce")
        # Add significance stars
        for factor in ["SMB", "HML", "RMW", "CMA"]:
            pcol = f"{factor}_P"
            if pcol in coeff_display.columns:
                coeff_display[f"{factor}_SIG"] = coeff_display[pcol].apply(fmt_sig)

        render_df(coeff_display, "Regime Coefficients", "a1_regime_coeff")

        # Chow test for structural break
        chow_df = load_table("ESSAY1_CHOW_TEST")
        if not chow_df.empty:
            for col in ["f_stat", "p_value"]:
                if col in chow_df.columns:
                    chow_df[col] = pd.to_numeric(chow_df[col], errors="coerce")
            f_stat = chow_df["f_stat"].iloc[0] if "f_stat" in chow_df.columns else None
            p_val = chow_df["p_value"].iloc[0] if "p_value" in chow_df.columns else None
            sig = chow_df.get("significant_005", chow_df.get("SIGNIFICANT_005", pd.Series([None]))).iloc[0]
            if f_stat is not None:
                verdict_color = "#2E7D32" if sig else "#C62828"
                st.markdown(
                    f"**Chow Test for Structural Break:** F = {f_stat:.2f}, p = {p_val:.4f} "
                    f"{'**— significant at 5%** (regimes are structurally different)' if sig else '— not significant'}"
                )

        st.divider()

    # --- Factor premia ---
    if not factor_premia.empty:
        st.subheader("Factor Premia by Regime")
        st.markdown(
            "Annualized mean factor returns within each volatility regime. "
            "The market premium swings sharply: positive in calm markets, "
            "negative in crisis."
        )

        premia_display = factor_premia.copy()
        for col in premia_display.columns:
            if col not in ("REGIME", "N_DAYS"):
                premia_display[col] = pd.to_numeric(premia_display[col], errors="coerce")
        # Show annualized columns as percentages
        ann_cols = [c for c in premia_display.columns if "ANN" in c]
        for c in ann_cols:
            premia_display[c] = premia_display[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "--")

        render_df(premia_display, "Factor Premia", "a1_factor_premia")

        st.divider()

    # --- Culture war stock results ---
    cw_regime = load_table("ESSAY1_CW_STOCK_RESULTS")
    if not cw_regime.empty:
        st.subheader("Culture War Stocks: FF5 by Regime")
        st.markdown(
            "Individual stock pricing regressions (R_i - RF ~ MKT_RF + SMB + HML + RMW + CMA) "
            "within each regime. Benjamini-Hochberg FDR correction applied at q = 0.05."
        )

        for col in cw_regime.columns:
            if col not in ("TICKER", "REGIME", "ALPHA_SIGNIFICANT_BH"):
                cw_regime[col] = pd.to_numeric(cw_regime[col], errors="coerce")

        # Summary metrics
        valid_cw = cw_regime[cw_regime["ALPHA"].notna()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks Analyzed", fmt_num(cw_regime["TICKER"].nunique()))
        c2.metric("Stock-Regime Obs", fmt_num(len(cw_regime)))
        c3.metric("Valid Regressions", fmt_num(len(valid_cw)))
        c4.metric("BH Significant Alphas",
                   fmt_num(int(cw_regime["ALPHA_SIGNIFICANT_BH"].sum()))
                   if "ALPHA_SIGNIFICANT_BH" in cw_regime.columns else "--")

        # Aggregate by regime
        if not valid_cw.empty:
            cw_agg = valid_cw.groupby("REGIME").agg(
                n=("TICKER", "count"),
                mean_alpha=("ALPHA", "mean"),
                mean_rsq=("R_SQUARED", "mean"),
                mean_mkt=("MKT_RF_BETA", "mean"),
            ).reset_index()
            cw_agg.columns = ["Regime", "N", "Mean Alpha", "Mean R-sq", "Mean Mkt Beta"]
            cw_agg = cw_agg.round(4)
            render_df(cw_agg, "CW Stocks by Regime", "a1_cw_regime_agg")

        with st.expander("Full Stock-Regime Results", expanded=False):
            render_df(cw_regime, "CW Stock Results", "a1_cw_regime_full", height=500)

        st.divider()

    # =====================================================================
    # ESSAY 1 — MATCHED CONTROL ANALYSIS (from model/essay1_matched.py)
    # =====================================================================

    st.header("Matched Control Analysis")
    st.markdown(
        "Treatment (culture war) firms vs. industry-matched control firms. "
        "Deltas isolate culture war exposure from sector effects."
    )

    matched_ttest = load_table("ESSAY1_MATCHED_TTEST")
    matched_amp = load_table("ESSAY1_MATCHED_AMPLIFICATION")
    matched_sign = load_table("ESSAY1_MATCHED_SIGN")
    matched_deltas = load_table("ESSAY1_MATCHED_DELTAS")
    matched_coverage = load_table("ESSAY1_MATCHED_COVERAGE")

    # --- Matched t-test ---
    if not matched_ttest.empty:
        st.subheader("Paired t-test: Treatment - Control Deltas")
        st.markdown(
            "Tests whether mean delta (treatment beta - control beta) differs "
            "from zero within each regime. Aggregated to one observation per "
            "treatment firm before testing."
        )

        ttest_display = matched_ttest.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
        for col in ttest_display.columns:
            if col not in ("REGIME", "VARIABLE", "BH_SIGNIFICANT"):
                ttest_display[col] = pd.to_numeric(ttest_display[col], errors="coerce")
        ttest_display["Sig."] = ttest_display["P_VALUE"].apply(fmt_sig)
        ttest_display = ttest_display.round(4)
        render_df(ttest_display, "Paired t-test", "a1_matched_ttest")

        st.divider()

    # --- Regime amplification ---
    if not matched_amp.empty:
        st.subheader("Regime Amplification: High Vol - Low Vol")
        st.markdown(
            "Tests whether culture war effects *amplify* from low to high volatility. "
            "A significant result means the treatment-control gap widens in crisis."
        )

        # Key finding callout
        amp_display = matched_amp.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
        for col in amp_display.columns:
            if col not in ("VARIABLE", "BH_SIGNIFICANT"):
                amp_display[col] = pd.to_numeric(amp_display[col], errors="coerce")

        sig_amp = amp_display[amp_display["BH_SIGNIFICANT"] == 1] if "BH_SIGNIFICANT" in amp_display.columns else pd.DataFrame()
        if not sig_amp.empty:
            sig_vars = ", ".join(sig_amp["VARIABLE"].tolist())
            st.markdown(
                f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
                f"padding:1.25rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
                f"<strong style='color:{GOLD};'>Key Finding</strong><br>"
                f"Regime amplification is BH-significant for <strong>{sig_vars}</strong>. "
                f"Culture war firms' loadings on these factors shift more than matched controls "
                f"when moving from low to high volatility.</div>",
                unsafe_allow_html=True,
            )

        amp_display["Sig."] = amp_display["P_VALUE"].apply(fmt_sig)
        amp_display = amp_display.round(4)
        render_df(amp_display, "Regime Amplification", "a1_matched_amp")

        st.divider()

    # --- Sign consistency ---
    if not matched_sign.empty:
        st.subheader("Sign Consistency")
        st.markdown(
            "Percentage of pairs where the treatment-control delta has the same sign. "
            "Binomial test for departure from 50%."
        )

        sign_display = matched_sign.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
        for col in sign_display.columns:
            if col not in ("REGIME", "VARIABLE", "MAJORITY_SIGN", "BH_SIGNIFICANT"):
                sign_display[col] = pd.to_numeric(sign_display[col], errors="coerce")
        sign_display["PCT_MAJORITY"] = sign_display["PCT_MAJORITY"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "--")
        sign_display = sign_display.round(4)
        render_df(sign_display, "Sign Consistency", "a1_matched_sign")

        st.divider()

    # --- Delta betas (expandable) ---
    if not matched_deltas.empty:
        with st.expander(f"Full Pair-Level Delta Betas ({len(matched_deltas)} rows)", expanded=False):
            delta_display = matched_deltas.drop(columns=["RUN_TIMESTAMP"], errors="ignore")
            render_df(delta_display, "Matched Deltas", "a1_matched_deltas", height=500)

    # --- Coverage ---
    if not matched_coverage.empty:
        cov = matched_coverage.drop(columns=["RUN_TIMESTAMP"], errors="ignore")
        cov_summary = cov.groupby("STATUS")["TICKER"].count().reset_index()
        cov_summary.columns = ["Status", "Count"]

        with st.expander("Ticker-Regime Coverage", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                render_df(cov_summary, "Coverage Summary", "a1_cov_summary")
            with c2:
                total = len(cov)
                ok_count = int(cov["HAS_RESULT"].astype(str).str.lower().isin(["true", "1"]).sum())
                st.metric("Coverage Rate", f"{ok_count}/{total} ({ok_count/total:.0%})")
            render_df(cov, "Full Coverage", "a1_matched_coverage", height=400)


    # =====================================================================
    # ESSAY 1 — FINBERT SENTIMENT & FOMO Z-SCORES (from model/essay1.py)
    # =====================================================================

    st.header("FinBERT Sentiment & FOMO Z-Scores")
    st.markdown(
        "FinBERT (ProsusAI/finbert) scores 57K+ culture war news articles. "
        "FOMO z-scores measure how extreme each day's sentiment is relative "
        "to its volatility regime norm — euphoria (Z>2) and panic (Z<-2)."
    )

    sent_daily = load_table("ESSAY1_SENTIMENT_DAILY")
    fomo_regime = load_table("ESSAY1_FOMO_BY_REGIME")

    if not fomo_regime.empty:
        st.subheader("Sentiment by Volatility Regime")

        fomo_display = fomo_regime.copy()
        for col in fomo_display.columns:
            if col != "REGIME":
                fomo_display[col] = pd.to_numeric(fomo_display[col], errors="coerce")

        # Format percentage columns
        for pct_col in ["PCT_EUPHORIA", "PCT_PANIC"]:
            if pct_col in fomo_display.columns:
                fomo_display[pct_col] = fomo_display[pct_col].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--"
                )
        fomo_display = fomo_display.round(4)
        render_df(fomo_display, "FOMO by Regime", "a1_fomo_regime")

        st.divider()

    if not sent_daily.empty:
        for col in sent_daily.columns:
            if col not in ("DATE", "TICKER", "REGIME"):
                sent_daily[col] = pd.to_numeric(sent_daily[col], errors="coerce")

        # --- Sentiment distribution by regime ---
        st.subheader("Daily Sentiment Distribution by Regime")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        regimes = sorted(sent_daily["REGIME"].dropna().unique())
        colors = {"Low Volatility": "#2E7D32", "Normal": "#1565C0", "High Volatility": "#C62828"}

        for i, regime in enumerate(regimes):
            ax = axes[i] if len(regimes) > 1 else axes
            subset = sent_daily[sent_daily["REGIME"] == regime]["SENT_MEAN"].dropna()
            color = colors.get(regime, NAVY)
            ax.hist(subset, bins=50, color=color, alpha=0.7, edgecolor="white")
            ax.set_title(regime, fontsize=11, fontweight="bold")
            ax.set_xlabel("Daily Mean Sentiment")
            if i == 0:
                ax.set_ylabel("Frequency")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        fig.suptitle("FinBERT Daily Sentiment by Regime", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # --- FOMO Z-score distribution ---
        st.subheader("FOMO Z-Score Distribution")

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for i, regime in enumerate(regimes):
            ax = axes2[i] if len(regimes) > 1 else axes2
            subset = sent_daily[sent_daily["REGIME"] == regime]["FOMO_Z"].dropna()
            color = colors.get(regime, NAVY)
            ax.hist(subset, bins=50, color=color, alpha=0.7, edgecolor="white")
            ax.set_title(regime, fontsize=11, fontweight="bold")
            ax.set_xlabel("FOMO Z-Score")
            if i == 0:
                ax.set_ylabel("Frequency")
            ax.axvline(2, color="#C62828", linestyle="--", linewidth=1, label="Euphoria (Z>2)")
            ax.axvline(-2, color="#1565C0", linestyle="--", linewidth=1, label="Panic (Z<-2)")
            if i == len(regimes) - 1:
                ax.legend(fontsize=8)

        fig2.suptitle("FOMO Z-Scores by Regime (Euphoria vs Panic Thresholds)", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        st.divider()

        # --- Top euphoria & panic days ---
        st.subheader("Extreme Sentiment Days")
        c1, c2 = st.columns(2)

        euphoria = sent_daily[sent_daily["FOMO_Z"] > 2].sort_values("FOMO_Z", ascending=False).head(15)
        panic = sent_daily[sent_daily["FOMO_Z"] < -2].sort_values("FOMO_Z").head(15)

        with c1:
            st.markdown(f"**Top Euphoria Days** (Z > 2)")
            if not euphoria.empty:
                render_df(
                    euphoria[["DATE", "TICKER", "REGIME", "SENT_MEAN", "FOMO_Z", "N_ARTICLES"]].round(3),
                    "Euphoria Days", "a1_euphoria"
                )
            else:
                st.info("No euphoria days detected.")

        with c2:
            st.markdown(f"**Top Panic Days** (Z < -2)")
            if not panic.empty:
                render_df(
                    panic[["DATE", "TICKER", "REGIME", "SENT_MEAN", "FOMO_Z", "N_ARTICLES"]].round(3),
                    "Panic Days", "a1_panic"
                )
            else:
                st.info("No panic days detected.")

        # --- Full daily data (expandable) ---
        with st.expander(f"Full Daily Sentiment Data ({len(sent_daily)} rows)", expanded=False):
            render_df(sent_daily.round(4), "Daily Sentiment", "a1_sent_daily", height=500)

    # --- Chart Gallery ---
    render_chart_gallery(E1_CHARTS, gallery_id="e1")


# =============================================================================
# TAB 2: ARTICLE 2 - Culture Wars and Capital Markets
# =============================================================================

with tab_a2:
    st.header("Essay 2: Culture Wars and Capital Markets")
    st.markdown("*The Political Economy of Abnormal Returns*")
    st.markdown(
        f"<p style='font-size:1rem; color:{NAVY}; margin-top:-0.5rem;'>"
        f"<strong>Research Question:</strong> How do culture war events generate abnormal "
        f"returns, and do these effects vary by a firm's political alignment?</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load all data
    event_df = load_table("EVENT_STUDY_RESULTS")
    xs_df = load_table("CROSS_SECTIONAL_CAR")
    did_df = load_table("DID_RESULTS")
    cw_df_a2 = load_table("CULTURE_WAR_COMPANIES")
    # Essay 2 DiD tables (from essay2_did.py)
    e2_car_panel = load_table("ESSAY2_CAR_PANEL")
    e2_did_coeff = load_table("ESSAY2_DID_COEFFICIENTS")
    e2_pt = load_table("ESSAY2_PARALLEL_TRENDS")
    ok_a2 = event_df[event_df["STATUS"] == "OK"].copy() if not event_df.empty else pd.DataFrame()

    if ok_a2.empty:
        st.info("No event study results available. Run Model.py first.")
    else:
        for col in ["CAR", "CAR_T", "CAR_P", "BHAR", "R_SQUARED"]:
            if col in ok_a2.columns:
                ok_a2[col] = pd.to_numeric(ok_a2[col], errors="coerce")

        # Merge industry + year
        if not cw_df_a2.empty and "TICKER" in cw_df_a2.columns:
            ok_a2 = ok_a2.merge(
                cw_df_a2[["TICKER", "INDUSTRY", "YEAR"]].drop_duplicates(subset=["TICKER"]),
                on="TICKER", how="left",
            )

        # --- Key finding callout ---
        n_sig = (ok_a2["CAR_P"] < 0.05).sum()
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1.25rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
            f"<strong style='color:{GOLD};'>Key Finding</strong><br>"
            f"Culture war events generate a mean CAR of <strong>{ok_a2['CAR'].mean():.2%}</strong> "
            f"across the [-30, +30] event window. <strong>{n_sig} of {len(ok_a2)}</strong> events "
            f"({n_sig/len(ok_a2):.0%}) produce statistically significant abnormal returns at the "
            f"5% level. All three political leaning groups show significant negative CARs "
            f"(p &lt; 0.01), confirming that culture wars destroy shareholder value regardless "
            f"of partisan alignment.</div>",
            unsafe_allow_html=True,
        )

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Completed Studies", fmt_num(len(ok_a2)))
        c2.metric("Significant (p<.05)", fmt_num(n_sig))
        c3.metric("Mean CAR", f"{ok_a2['CAR'].mean():.2%}")
        c4.metric("Median CAR", f"{ok_a2['CAR'].median():.2%}")
        c5.metric("Mean BHAR", f"{ok_a2['BHAR'].mean():.2%}" if "BHAR" in ok_a2.columns else "--")

        st.divider()

        # ===== SECTION 1: CAR by Political Leaning =====
        st.subheader("CAR by Political Leaning")
        st.markdown(
            "The central visualization: mean Cumulative Abnormal Returns grouped by "
            "political leaning with 95% confidence interval error bars."
        )

        # Show the figure from the database
        fig_data = load_figure_blob("car_by_political_leaning")
        if fig_data:
            render_figure(fig_data, caption="Mean CAR by Political Leaning (95% CI)",
                          filename="car_by_political_leaning.png")
        else:
            fig_path = BASE_DIR / "figures" / "car_by_political_leaning.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Mean CAR by Political Leaning (95% CI)",
                         use_container_width=True)

        st.divider()

        # ===== SECTION 2: Cross-Sectional CAR =====
        st.subheader("Cross-Sectional CAR Test Results")
        st.markdown(
            "One-sample t-tests of whether mean CAR differs significantly from zero "
            "within each political leaning group."
        )

        if not xs_df.empty:
            xs_display = xs_df.copy()
            for col in ["MEAN_CAR", "STD_CAR", "T_STAT", "P_VALUE"]:
                if col in xs_display.columns:
                    xs_display[col] = pd.to_numeric(xs_display[col], errors="coerce")
            if "P_VALUE" in xs_display.columns:
                xs_display["Sig."] = xs_display["P_VALUE"].apply(fmt_sig)

            xs_renamed = xs_display.rename(columns={
                "POLITICAL_LEANING": "Group", "MEAN_CAR": "Mean CAR",
                "STD_CAR": "Std Dev", "N": "N", "T_STAT": "t-stat",
                "P_VALUE": "p-value",
            })
            num_cols = xs_renamed.select_dtypes(include=[np.number]).columns
            xs_renamed[num_cols] = xs_renamed[num_cols].round(4)
            render_df(xs_renamed, "Cross-Sectional CAR", "a2_xs_car")

        st.divider()

        # ===== SECTION 3: CAR Distribution =====
        st.subheader("CAR Distribution by Political Leaning")
        if "POLITICAL_LEANING" in ok_a2.columns:
            car_dist = ok_a2.groupby("POLITICAL_LEANING")["CAR"].agg(
                ["mean", "median", "std", "min", "max", "count"]
            ).reset_index()
            car_dist.columns = ["Political Leaning", "Mean", "Median", "Std Dev", "Min", "Max", "N"]
            car_dist = car_dist.round(4)
            render_df(car_dist, "CAR Distribution", "a2_car_dist")

        st.divider()

        # ===== SECTION 4: Significance Breakdown =====
        st.subheader("Significance Breakdown")
        st.markdown(
            "How many events produce statistically significant abnormal returns at "
            "various conventional thresholds?"
        )

        if "POLITICAL_LEANING" in ok_a2.columns:
            sig_data = []
            for leaning in sorted(ok_a2["POLITICAL_LEANING"].unique()):
                subset = ok_a2[ok_a2["POLITICAL_LEANING"] == leaning]
                n = len(subset)
                sig_01 = (subset["CAR_P"] < 0.01).sum()
                sig_05 = (subset["CAR_P"] < 0.05).sum()
                sig_10 = (subset["CAR_P"] < 0.10).sum()
                neg_car = (subset["CAR"] < 0).sum()
                sig_data.append({
                    "Political Leaning": leaning,
                    "N": n,
                    "Sig. at 1%": f"{sig_01} ({sig_01/n:.0%})",
                    "Sig. at 5%": f"{sig_05} ({sig_05/n:.0%})",
                    "Sig. at 10%": f"{sig_10} ({sig_10/n:.0%})",
                    "Negative CAR": f"{neg_car} ({neg_car/n:.0%})",
                })
            # Add "All" row
            n_all = len(ok_a2)
            sig_data.append({
                "Political Leaning": "All",
                "N": n_all,
                "Sig. at 1%": f"{(ok_a2['CAR_P'] < 0.01).sum()} ({(ok_a2['CAR_P'] < 0.01).mean():.0%})",
                "Sig. at 5%": f"{(ok_a2['CAR_P'] < 0.05).sum()} ({(ok_a2['CAR_P'] < 0.05).mean():.0%})",
                "Sig. at 10%": f"{(ok_a2['CAR_P'] < 0.10).sum()} ({(ok_a2['CAR_P'] < 0.10).mean():.0%})",
                "Negative CAR": f"{(ok_a2['CAR'] < 0).sum()} ({(ok_a2['CAR'] < 0).mean():.0%})",
            })
            render_df(pd.DataFrame(sig_data), "Significance", "a2_significance")

        st.divider()

        # ===== SECTION 5: Most Impactful Events =====
        st.subheader("Most Impactful Culture War Events")

        impact_col1, impact_col2 = st.columns(2)
        with impact_col1:
            st.markdown("**Largest Negative CARs (Shareholder Destruction)**")
            worst_car = ok_a2.nsmallest(10, "CAR")[
                ["TICKER", "COMPANY", "EVENT_DATE", "POLITICAL_LEANING", "CAR", "CAR_P"]
            ].copy().round(4)
            worst_car["Sig."] = worst_car["CAR_P"].apply(fmt_sig)
            render_df(worst_car, "Worst CARs", "a2_worst_car")

        with impact_col2:
            st.markdown("**Largest Positive CARs (Shareholder Gain)**")
            best_car = ok_a2.nlargest(10, "CAR")[
                ["TICKER", "COMPANY", "EVENT_DATE", "POLITICAL_LEANING", "CAR", "CAR_P"]
            ].copy().round(4)
            best_car["Sig."] = best_car["CAR_P"].apply(fmt_sig)
            render_df(best_car, "Best CARs", "a2_best_car")

        st.divider()

        # ===== SECTION 6: Event Timeline =====
        if "YEAR" in ok_a2.columns:
            st.subheader("Events and Returns Over Time")
            st.markdown(
                "How event frequency and average abnormal returns have evolved "
                "as corporate culture wars intensified after 2015."
            )

            year_agg = ok_a2.groupby("YEAR").agg(
                n_events=("CAR", "count"),
                mean_car=("CAR", "mean"),
                median_car=("CAR", "median"),
                pct_sig=("CAR_P", lambda x: (x < 0.05).mean()),
                pct_negative=("CAR", lambda x: (x < 0).mean()),
            ).reset_index()
            year_agg.columns = ["Year", "Events", "Mean CAR", "Median CAR",
                                "% Significant (p<.05)", "% Negative"]
            year_agg["Mean CAR"] = year_agg["Mean CAR"].round(4)
            year_agg["Median CAR"] = year_agg["Median CAR"].round(4)
            year_agg["% Significant (p<.05)"] = year_agg["% Significant (p<.05)"].apply(lambda x: f"{x:.0%}")
            year_agg["% Negative"] = year_agg["% Negative"].apply(lambda x: f"{x:.0%}")
            render_df(year_agg, "Events by Year", "a2_events_year")

            st.divider()

        # ===== SECTION 7: Industry Analysis =====
        if "INDUSTRY" in ok_a2.columns:
            st.subheader("CAR by Industry")
            st.markdown(
                "Cross-industry comparison of culture war impacts. Which sectors "
                "are most vulnerable to politically-motivated sell-offs?"
            )

            industry_car = ok_a2.groupby("INDUSTRY").agg(
                mean_car=("CAR", "mean"),
                median_car=("CAR", "median"),
                pct_sig=("CAR_P", lambda x: (x < 0.05).mean()),
                n=("CAR", "count"),
            ).reset_index()
            industry_car.columns = ["Industry", "Mean CAR", "Median CAR", "% Significant", "N"]
            industry_car = industry_car[industry_car["N"] >= 2].sort_values("Mean CAR")
            industry_car["Mean CAR"] = industry_car["Mean CAR"].round(4)
            industry_car["Median CAR"] = industry_car["Median CAR"].round(4)
            industry_car["% Significant"] = industry_car["% Significant"].apply(lambda x: f"{x:.0%}")
            render_df(industry_car, "CAR by Industry", "a2_car_industry")

            st.divider()

        # ===== SECTION 8: Difference-in-Differences =====
        st.subheader("Difference-in-Differences Analysis")
        st.markdown(
            "DiD estimation isolates the treatment effect by comparing culture war "
            "firms to matched controls. A negative DiD coefficient indicates that the "
            "treatment firm's returns diverged downward relative to its control during "
            "the event window, beyond what common factors predict."
        )

        if not did_df.empty:
            ok_did = did_df[did_df["STATUS"] == "OK"].copy()
            for col in ["DID_COEF", "DID_T", "DID_P", "CI_LOWER", "CI_UPPER", "R_SQUARED"]:
                if col in ok_did.columns:
                    ok_did[col] = pd.to_numeric(ok_did[col], errors="coerce")

            if not ok_did.empty:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("DiD Pairs Completed", fmt_num(len(ok_did)))
                c2.metric("Significant (p<.05)", fmt_num((ok_did["DID_P"] < 0.05).sum()))
                c3.metric("Mean DiD Coef", f"{ok_did['DID_COEF'].mean():.4f}")
                c4.metric("Median DiD Coef", f"{ok_did['DID_COEF'].median():.4f}")
                c5.metric("Negative Coef %", f"{(ok_did['DID_COEF'] < 0).mean():.0%}")

                # DiD by leaning
                if "POLITICAL_LEANING" in ok_did.columns:
                    did_by_leaning = ok_did.groupby("POLITICAL_LEANING").agg(
                        mean_coef=("DID_COEF", "mean"),
                        median_coef=("DID_COEF", "median"),
                        std=("DID_COEF", "std"),
                        pct_negative=("DID_COEF", lambda x: (x < 0).mean()),
                        pct_sig=("DID_P", lambda x: (x < 0.05).mean()),
                        n=("DID_COEF", "count"),
                    ).reset_index()
                    did_by_leaning.columns = ["Political Leaning", "Mean Coef", "Median Coef",
                                              "Std Dev", "% Negative", "% Sig. (p<.05)", "N"]
                    did_by_leaning["% Negative"] = did_by_leaning["% Negative"].apply(lambda x: f"{x:.0%}")
                    did_by_leaning["% Sig. (p<.05)"] = did_by_leaning["% Sig. (p<.05)"].apply(lambda x: f"{x:.0%}")
                    did_by_leaning = did_by_leaning.round(4)
                    render_df(did_by_leaning, "DiD by Leaning", "a2_did_leaning")

                # Most significant DiD pairs
                st.markdown("**Most Significant DiD Pairs**")
                top_did = ok_did.nsmallest(10, "DID_P")[
                    ["TICKER", "CONTROL_TICKER", "COMPANY", "POLITICAL_LEANING",
                     "DID_COEF", "DID_T", "DID_P", "R_SQUARED"]
                ].copy().round(4)
                top_did["Sig."] = top_did["DID_P"].apply(fmt_sig)
                render_df(top_did, "Top DiD Pairs", "a2_did_top")

                with st.expander("Full DiD Results", expanded=False):
                    did_display = ok_did[["TICKER", "CONTROL_TICKER", "COMPANY",
                                          "POLITICAL_LEANING", "DID_COEF", "DID_T",
                                          "DID_P", "R_SQUARED"]].copy()
                    did_display = did_display.round(4)
                    did_display["Sig."] = did_display["DID_P"].apply(fmt_sig)
                    did_display = did_display.sort_values("DID_P")
                    render_df(did_display, "DiD Results", "a2_did_full", height=500)
        else:
            st.info("No DiD results available.")

        st.divider()

        # ===== SECTION 8b: Essay 2 Cross-Sectional DiD (essay2_did.py) =====
        st.subheader("Cross-Sectional DiD — Matched Treatment vs Control")
        st.markdown(
            "Formal DiD estimation from `essay2_did.py`: treatment (culture war) firms "
            "vs industry-matched controls. The model stacks pre- and post-event CARs "
            "into a panel and estimates:"
        )
        st.latex(
            r"\text{CAR}_{i,e} = \alpha + \beta_1 \text{Treat}_i + \beta_2 \text{Post}_e "
            r"+ \beta_3 (\text{Treat} \times \text{Post}) + \gamma \mathbf{X} + \varepsilon"
        )

        # --- Parallel Trends Pre-Test ---
        if not e2_pt.empty:
            st.markdown("**Parallel Trends Pre-Test**")
            for col in ["COEFFICIENT", "STD_ERROR", "T_STAT", "P_VALUE",
                         "JOINT_F_STAT", "JOINT_P_VALUE"]:
                if col in e2_pt.columns:
                    e2_pt[col] = pd.to_numeric(e2_pt[col], errors="coerce")

            joint_f = e2_pt["JOINT_F_STAT"].iloc[0] if "JOINT_F_STAT" in e2_pt.columns else None
            joint_p = e2_pt["JOINT_P_VALUE"].iloc[0] if "JOINT_P_VALUE" in e2_pt.columns else None
            passes = e2_pt["PASSES"].iloc[0] if "PASSES" in e2_pt.columns else None

            if joint_f is not None:
                verdict = "PASS" if passes else "FAIL"
                color = "#2E7D32" if passes else "#C62828"
                st.markdown(
                    f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {color}; "
                    f"padding:1rem; border-radius:0 4px 4px 0;'>"
                    f"<strong>Joint F-test:</strong> F = {joint_f:.2f}, p = {joint_p:.4f} "
                    f"&mdash; <strong style='color:{color};'>{verdict}</strong><br>"
                    f"<small>H₀: all Treat×Day coefficients = 0 in the pre-event window. "
                    f"{'Parallel trends hold — DiD assumptions satisfied.' if passes else 'Parallel trends violated — interpret DiD with caution.'}"
                    f"</small></div>",
                    unsafe_allow_html=True,
                )

            # Day-by-day coefficients
            pt_display = e2_pt[["DAY", "COEFFICIENT", "STD_ERROR", "T_STAT", "P_VALUE"]].copy()
            pt_display = pt_display.round(4)
            pt_display["Sig."] = pt_display["P_VALUE"].apply(fmt_sig)
            with st.expander("Daily Treat × Day Coefficients", expanded=False):
                render_df(pt_display, "Parallel Trends Coefficients", "e2_pt_daily")

            st.divider()

        # --- CAR Panel Summary ---
        if not e2_car_panel.empty:
            st.markdown("**CAR Panel Summary**")
            for col in ["CAR_PRE", "CAR_POST", "CAR_FULL", "EST_R2", "LEAN", "FOMO_Z"]:
                if col in e2_car_panel.columns:
                    e2_car_panel[col] = pd.to_numeric(e2_car_panel[col], errors="coerce")

            if "IS_TREATMENT" in e2_car_panel.columns:
                e2_car_panel["IS_TREATMENT"] = e2_car_panel["IS_TREATMENT"].map(
                    {True: True, False: False, 1: True, 0: False,
                     "true": True, "false": False, "1": True, "0": False,
                     "True": True, "False": False}
                ).fillna(False)

            treat = e2_car_panel[e2_car_panel["IS_TREATMENT"]] if "IS_TREATMENT" in e2_car_panel.columns else pd.DataFrame()
            ctrl = e2_car_panel[~e2_car_panel["IS_TREATMENT"]] if "IS_TREATMENT" in e2_car_panel.columns else pd.DataFrame()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Panel Rows", fmt_num(len(e2_car_panel)))
            c2.metric("Events", fmt_num(e2_car_panel["EVENT_ID"].nunique()) if "EVENT_ID" in e2_car_panel.columns else "--")
            if not treat.empty:
                c3.metric("Mean CAR_POST (Treatment)", f"{treat['CAR_POST'].mean():.4f}")
            if not ctrl.empty:
                c4.metric("Mean CAR_POST (Control)", f"{ctrl['CAR_POST'].mean():.4f}")

            # Treatment vs Control comparison
            if not treat.empty and not ctrl.empty:
                car_compare = pd.DataFrame({
                    "Group": ["Treatment", "Control", "Difference"],
                    "N Firms": [treat["TICKER"].nunique(), ctrl["TICKER"].nunique(), ""],
                    "Mean CAR_PRE": [treat["CAR_PRE"].mean(), ctrl["CAR_PRE"].mean(),
                                     treat["CAR_PRE"].mean() - ctrl["CAR_PRE"].mean()],
                    "Mean CAR_POST": [treat["CAR_POST"].mean(), ctrl["CAR_POST"].mean(),
                                      treat["CAR_POST"].mean() - ctrl["CAR_POST"].mean()],
                    "Mean CAR_FULL": [treat["CAR_FULL"].mean(), ctrl["CAR_FULL"].mean(),
                                      treat["CAR_FULL"].mean() - ctrl["CAR_FULL"].mean()],
                    "Mean Est R²": [treat["EST_R2"].mean(), ctrl["EST_R2"].mean(),
                                    treat["EST_R2"].mean() - ctrl["EST_R2"].mean()],
                }).round(4)
                render_df(car_compare, "Treatment vs Control CARs", "e2_car_compare")

            with st.expander("Full CAR Panel", expanded=False):
                car_cols = [c for c in ["TICKER", "EVENT_ID", "IS_TREATMENT", "REGIME",
                                         "CAR_PRE", "CAR_POST", "CAR_FULL", "LEAN",
                                         "FOMO_Z", "EST_R2"] if c in e2_car_panel.columns]
                render_df(e2_car_panel[car_cols].round(4), "CAR Panel", "e2_car_panel_full", height=500)

            st.divider()

        # --- DiD Coefficient Table ---
        if not e2_did_coeff.empty:
            st.markdown("**DiD Regression Results — Three Specifications**")
            for col in ["COEFFICIENT", "STD_ERROR", "T_STAT", "P_VALUE"]:
                if col in e2_did_coeff.columns:
                    e2_did_coeff[col] = pd.to_numeric(e2_did_coeff[col], errors="coerce")

            specs = e2_did_coeff["SPECIFICATION"].unique() if "SPECIFICATION" in e2_did_coeff.columns else []

            for spec in specs:
                spec_df = e2_did_coeff[e2_did_coeff["SPECIFICATION"] == spec].copy()
                label = {"basic": "Spec 1: Basic DiD",
                         "with_lean": "Spec 2: + Political Lean",
                         "with_fomo": "Spec 3: + FOMO Z-Score"}.get(spec, spec)

                st.markdown(f"**{label}**")
                display_cols = [c for c in ["VARIABLE", "COEFFICIENT", "STD_ERROR",
                                            "T_STAT", "P_VALUE", "BH_SIGNIFICANT"]
                                if c in spec_df.columns]
                spec_display = spec_df[display_cols].copy().round(4)
                if "P_VALUE" in spec_display.columns:
                    spec_display["Sig."] = spec_display["P_VALUE"].apply(fmt_sig)
                render_df(spec_display, f"DiD {spec}", f"e2_did_{spec}")

                # Highlight the treatment effect (Treat x Post)
                treat_row = spec_df[spec_df["VARIABLE"] == "TREAT_x_POST"]
                if not treat_row.empty:
                    coef = treat_row.iloc[0]["COEFFICIENT"]
                    pval = treat_row.iloc[0]["P_VALUE"]
                    sig_marker = fmt_sig(pval) if pval < 0.1 else " (n.s.)"
                    st.caption(
                        f"Treatment effect (Treat × Post): {coef:+.4f}{sig_marker}, "
                        f"p = {pval:.4f}"
                    )

            st.divider()

        # ===== SECTION 9: Failed Studies =====
        failed_a2 = event_df[event_df["STATUS"] != "OK"] if not event_df.empty else pd.DataFrame()
        if not failed_a2.empty:
            with st.expander(f"Failed Event Studies ({len(failed_a2)} events)", expanded=False):
                st.markdown(
                    "Events that could not be analyzed, typically due to private firms "
                    "(no stock data), delistings, or insufficient estimation window data."
                )
                render_df(
                    failed_a2[["TICKER", "COMPANY", "EVENT_DATE", "POLITICAL_LEANING", "STATUS"]],
                    "Failed Events", "a2_failed", height=400,
                )

        # Full event-level results
        with st.expander("Full Event Study Results (All Completed)", expanded=False):
            display = ok_a2[["TICKER", "COMPANY", "EVENT_DATE", "EVENT_DESCRIPTION",
                             "POLITICAL_LEANING", "CAR", "CAR_T", "CAR_P",
                             "BHAR"]].copy()
            display = display.round(4)
            display["Sig."] = display["CAR_P"].apply(fmt_sig)
            display = display.sort_values("CAR_P")
            render_df(display, "Event Study Results", "a2_full_results", height=500)

    # --- Chart Gallery ---
    render_chart_gallery(E2_CHARTS, gallery_id="e2")


# =============================================================================
# TAB 3: ARTICLE 3 - Insider Trading and Political Controversies
# =============================================================================

with tab_a3:
    st.header("Essay 3: Insider Trading and Political Controversies")
    st.markdown("*Evidence from Culture War Events*")
    st.markdown(
        f"<p style='font-size:1rem; color:{NAVY}; margin-top:-0.5rem;'>"
        f"<strong>Research Question:</strong> Do corporate insiders trade on advance "
        f"knowledge of upcoming political controversies?</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Motivation ---
    st.subheader("Motivation")
    st.markdown(
        "Corporate insiders -- executives, directors, and significant shareholders "
        "-- often have advance knowledge of planned political statements, marketing "
        "campaigns, and responses to emerging social controversies. Essay 2 establishes "
        "that culture war events generate significant abnormal returns (mean CAR = "
        f"{summary.get('avg_car', 0):.2%}). "
        "This raises a natural question: do insiders exploit this foreknowledge for "
        "personal gain?"
    )
    st.markdown(
        "Cohen, Malloy, and Pomorski (2012) distinguish between *routine* insider trades "
        "(regular, calendar-driven patterns) and *opportunistic* trades (irregularly timed, "
        "predictive of future returns). This essay applies their framework to culture war "
        "events, testing whether insider selling intensifies in the 60 days before a "
        "politically controversial event becomes public."
    )

    st.divider()

    # --- Hypotheses ---
    st.subheader("Hypotheses")

    hyp_col1, hyp_col2, hyp_col3 = st.columns(3)
    with hyp_col1:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:12rem;'>"
            f"<strong style='color:{GOLD};'>H1: Pre-Event Selling</strong><br><br>"
            f"Insider net selling increases in the [-60, -1] window before culture war "
            f"events relative to a matched benchmark period.</div>",
            unsafe_allow_html=True,
        )
    with hyp_col2:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:12rem;'>"
            f"<strong style='color:{GOLD};'>H2: Opportunistic Timing</strong><br><br>"
            f"The proportion of *opportunistic* (vs. routine) trades increases before "
            f"culture war events, consistent with information-motivated trading.</div>",
            unsafe_allow_html=True,
        )
    with hyp_col3:
        st.markdown(
            f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
            f"padding:1rem; border-radius:0 4px 4px 0; min-height:12rem;'>"
            f"<strong style='color:{GOLD};'>H3: Cross-Sectional Variation</strong><br><br>"
            f"Pre-event insider selling is more pronounced for events with larger "
            f"subsequent negative CARs, and among C-suite insiders vs. directors.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Methodology ---
    st.subheader("Methodology")

    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown("**Data**")
        st.markdown(
            "- **SEC Form 4** filings for all 160 culture war event firms\n"
            "- **EDGAR XBRL** submissions parsed for transaction date, shares, price, "
            "ownership type\n"
            "- **Insider types**: CEO, CFO, COO, Directors, 10%+ Owners\n"
            "- **Benchmark**: Same-firm trades from [-180, -61] window"
        )
    with meth_col2:
        st.markdown("**Analysis Plan**")
        st.markdown(
            "1. Classify trades as routine vs. opportunistic (Cohen et al., 2012)\n"
            "2. Compute net insider selling ratio in [-60, -1] window\n"
            "3. Compare to benchmark period using paired t-tests\n"
            "4. Cross-sectional regression: insider selling ~ f(CAR magnitude, "
            "firm size, leaning)\n"
            "5. Placebo tests using random pseudo-event dates"
        )

    st.divider()

    # --- Sample Construction ---
    st.subheader("Sample Construction")

    event_df_a3 = load_table("EVENT_STUDY_RESULTS")
    ok_a3 = event_df_a3[event_df_a3["STATUS"] == "OK"].copy() if not event_df_a3.empty else pd.DataFrame()

    if not ok_a3.empty:
        for col in ["CAR", "CAR_P"]:
            ok_a3[col] = pd.to_numeric(ok_a3[col], errors="coerce")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event Firms (Public)", fmt_num(ok_a3["TICKER"].nunique()))
        c2.metric("Total Events", fmt_num(len(ok_a3)))
        c3.metric("Sig. Events (p<.05)", fmt_num((ok_a3["CAR_P"] < 0.05).sum()))
        c4.metric("Mean Event CAR", f"{ok_a3['CAR'].mean():.2%}")

        st.markdown(
            "The Essay 3 sample begins with the 141 completed event studies from "
            "Essay 2. For each event, SEC Form 4 filings are collected for the "
            "treatment firm over the [-180, +30] window surrounding the event date."
        )

        # Show the firms that will be in the sample
        st.markdown("**Event Firms by Political Leaning**")
        if "POLITICAL_LEANING" in ok_a3.columns:
            leaning_a3 = ok_a3.groupby("POLITICAL_LEANING").agg(
                unique_tickers=("TICKER", "nunique"),
                n_events=("TICKER", "count"),
                mean_car=("CAR", "mean"),
                n_significant=("CAR_P", lambda x: (x < 0.05).sum()),
            ).reset_index()
            leaning_a3.columns = ["Political Leaning", "Unique Tickers", "Events",
                                  "Mean CAR", "Sig. Events (p<.05)"]
            leaning_a3 = leaning_a3.round(4)
            render_df(leaning_a3, "Essay 3 Sample", "a3_sample_leaning")

    st.divider()

    # =====================================================================
    # ESSAY 3 — MODEL RESULTS
    # =====================================================================

    # Load Essay 3 tables
    e3_panel = load_table("ESSAY3_INSIDER_PANEL")
    e3_window = load_table("ESSAY3_WINDOW_SUMMARY")
    e3_abnormal = load_table("ESSAY3_ABNORMAL_SELLING")
    e3_regression = load_table("ESSAY3_CAR_INSIDER_REGRESSION")
    e3_leaning = load_table("ESSAY3_LEANING_ANALYSIS")
    e3_treat_ctrl = load_table("ESSAY3_TREATMENT_VS_CONTROL")
    e3_rvo = load_table("ESSAY3_ROUTINE_VS_OPPORTUNISTIC")
    e3_regime = load_table("ESSAY3_REGIME_INTERACTION")
    e3_placebo = load_table("ESSAY3_PLACEBO_TEST")
    e3_accel = load_table("ESSAY3_ACCELERATION_TEST")
    e3_gradient = load_table("ESSAY3_INFORMATION_GRADIENT")

    _has_e3 = not e3_panel.empty or not e3_abnormal.empty

    if _has_e3:
        # Coerce numeric columns
        for _df in [e3_abnormal, e3_leaning, e3_rvo, e3_regime, e3_placebo,
                     e3_accel, e3_window, e3_regression, e3_gradient]:
            for col in _df.columns:
                if col not in ("WINDOW", "TEST", "REGIME", "LEAN", "TICKER",
                                "EVENT_ID", "EVENT_DATE", "IS_TREATMENT",
                                "MONOTONIC_INCREASE", "BH_SIGNIFICANT",
                                "ABNORMAL_SELLING", "HAS_SUFFICIENT_DATA",
                                "T_BH_SIGNIFICANT", "WILCOXON_BH_SIGNIFICANT",
                                "RUN_TIMESTAMP"):
                    _df[col] = pd.to_numeric(_df[col], errors="coerce")

        # --- Key finding callout ---
        if not e3_abnormal.empty:
            _pre = e3_abnormal["MEAN_PRE_DAILY"].mean()
            _bench = e3_abnormal["MEAN_BENCH_DAILY"].mean()
            _n_sig = e3_abnormal.get("T_BH_SIGNIFICANT", pd.Series()).sum()
            st.markdown(
                f"<div style='background-color:{LIGHT_GRAY}; border-left:4px solid {GOLD}; "
                f"padding:1.25rem; border-radius:0 4px 4px 0; margin-bottom:1rem;'>"
                f"<strong style='color:{GOLD};'>Key Finding</strong><br>"
                f"Insider selling averages <strong>${_pre:,.0f}/day</strong> in the pre-event window "
                f"vs <strong>${_bench:,.0f}/day</strong> in the benchmark period. "
                f"<strong>{int(_n_sig)}</strong> of {len(e3_abnormal)} window comparisons show "
                f"BH-significant abnormal selling.</div>",
                unsafe_allow_html=True,
            )

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Event-Firm Pairs", fmt_num(len(e3_panel)) if not e3_panel.empty else "--")
        c2.metric("Windows Tested", fmt_num(len(e3_window)) if not e3_window.empty else "--")
        c3.metric("Abnormal Tests", fmt_num(len(e3_abnormal)) if not e3_abnormal.empty else "--")
        if not e3_panel.empty and "IS_TREATMENT" in e3_panel.columns:
            _n_treat = e3_panel["IS_TREATMENT"].astype(str).str.lower().isin(["true", "1"]).sum()
            c4.metric("Treatment Firms", fmt_num(_n_treat))
        if not e3_placebo.empty:
            _emp_p = e3_placebo.iloc[0].get("EMPIRICAL_P", None)
            c5.metric("Placebo p-value", fmt_pval(_emp_p) if _emp_p is not None else "--")

        st.divider()

        # --- Window Summary ---
        if not e3_window.empty:
            st.subheader("Insider Trading by Window")
            st.markdown(
                "Aggregate insider trading metrics across pre-event sub-windows, "
                "the full pre-event window, and the post-event window."
            )
            _win_display = e3_window.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _win_display = _win_display.round(2)
            render_df(_win_display, "Window Summary", "a3_window_summary")
            st.divider()

        # --- Abnormal Selling Tests ---
        if not e3_abnormal.empty:
            st.subheader("Abnormal Selling Tests")
            st.markdown(
                "Paired t-tests and Wilcoxon signed-rank tests comparing pre-event "
                "insider selling to the benchmark period. Both parametric and "
                "non-parametric tests are reported with BH-FDR correction."
            )
            _abn_display = e3_abnormal.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _abn_display["Sig."] = _abn_display.get("T_PVALUE", pd.Series()).apply(fmt_sig)
            _abn_display = _abn_display.round(4)
            render_df(_abn_display, "Abnormal Selling", "a3_abnormal")
            st.divider()

        # --- Political Leaning Analysis ---
        if not e3_leaning.empty:
            st.subheader("Insider Trading by Political Leaning")
            st.markdown(
                "Do insiders at Conservative, Liberal, or Mixed-leaning firms "
                "sell more aggressively before culture war events? Kruskal-Wallis "
                "test for group differences."
            )
            _lean_display = e3_leaning.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _lean_display["Sig."] = _lean_display.get("P_VALUE_VS_ZERO", pd.Series()).apply(fmt_sig)
            _lean_display = _lean_display.round(4)
            render_df(_lean_display, "Leaning Analysis", "a3_leaning")
            st.divider()

        # --- Routine vs Opportunistic ---
        if not e3_rvo.empty:
            st.subheader("Routine vs Opportunistic Trades")
            st.markdown(
                "Cohen, Malloy, and Pomorski (2012) decomposition. Opportunistic "
                "trades — irregular, information-motivated — should increase more "
                "before culture war events if insiders are trading on foreknowledge."
            )
            _rvo_display = e3_rvo.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _rvo_display["Sig."] = _rvo_display.get("P_VALUE", pd.Series()).apply(fmt_sig)
            _rvo_display = _rvo_display.round(4)
            render_df(_rvo_display, "Routine vs Opportunistic", "a3_rvo")
            st.divider()

        # --- VIX Regime Interaction ---
        if not e3_regime.empty:
            st.subheader("Insider Trading by VIX Regime")
            st.markdown(
                "Does insider selling before culture war events intensify during "
                "high-volatility regimes? Cross-tabulation of regime × test."
            )
            _reg_display = e3_regime.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _reg_display["Sig."] = _reg_display.get("P_VALUE", pd.Series()).apply(fmt_sig)
            _reg_display = _reg_display.round(4)
            render_df(_reg_display, "Regime Interaction", "a3_regime")
            st.divider()

        # --- Robustness: Placebo Test ---
        if not e3_placebo.empty:
            st.subheader("Placebo Test")
            st.markdown(
                "Permutation-based placebo test: random pseudo-event dates are assigned "
                "and the test statistic is recomputed. If the observed statistic exceeds "
                "95% of placebo values, the effect is unlikely to be spurious."
            )
            _plac = e3_placebo.iloc[0]
            _plac_cols = ["TEST", "OBSERVED_STAT", "PLACEBO_MEAN", "PLACEBO_STD",
                          "PERCENTILE", "EMPIRICAL_P", "N_ITERATIONS", "N_FIRMS"]
            _plac_data = {c: _plac.get(c, "--") for c in _plac_cols if c in e3_placebo.columns}

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Observed Stat", f"{_plac.get('OBSERVED_STAT', 0):.4f}")
            c2.metric("Placebo Mean", f"{_plac.get('PLACEBO_MEAN', 0):.4f}")
            c3.metric("Percentile", f"{_plac.get('PERCENTILE', 0):.1f}")
            c4.metric("Empirical p", fmt_pval(_plac.get("EMPIRICAL_P")))

            _plac_display = e3_placebo.drop(columns=["RUN_TIMESTAMP"], errors="ignore").round(4)
            render_df(_plac_display, "Placebo Test", "a3_placebo")
            st.divider()

        # --- Robustness: Acceleration Test ---
        if not e3_accel.empty:
            st.subheader("Acceleration Test (Jonckheere-Terpstra)")
            st.markdown(
                "Tests for a monotonic increase in insider selling as the event date "
                "approaches: far window → mid window → near window. The JT test "
                "detects ordered alternatives without assuming normality."
            )
            _acc_display = e3_accel.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _acc_display["Sig."] = _acc_display.get("JT_PVALUE", pd.Series()).apply(fmt_sig)
            _acc_display = _acc_display.round(4)
            render_df(_acc_display, "Acceleration Test", "a3_accel")
            st.divider()

        # --- CAR-Insider Regression ---
        if not e3_regression.empty:
            st.subheader("CAR-Insider Selling Regression")
            st.markdown(
                "Cross-sectional regression: does pre-event insider selling predict "
                "post-event abnormal returns (CAR)?"
            )
            _reg_display = e3_regression.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _reg_display = _reg_display.round(4)
            render_df(_reg_display, "CAR Regression", "a3_car_regression")
            st.divider()

        # --- Treatment vs Control ---
        if not e3_treat_ctrl.empty:
            st.subheader("Treatment vs Control Firms")
            st.markdown(
                "Comparing insider selling in culture war firms (treatment) vs "
                "matched control firms."
            )
            _tc_display = e3_treat_ctrl.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _tc_display = _tc_display.round(4)
            render_df(_tc_display, "Treatment vs Control", "a3_treat_ctrl")
            st.divider()

        # --- Information Gradient ---
        if not e3_gradient.empty:
            st.subheader("Information Gradient")
            st.markdown(
                "Tests whether insider selling is stronger before events "
                "that produce larger subsequent CARs."
            )
            _grad_display = e3_gradient.drop(columns=["RUN_TIMESTAMP"], errors="ignore").copy()
            _grad_display = _grad_display.round(4)
            render_df(_grad_display, "Information Gradient", "a3_gradient")
            st.divider()

        # --- Full Panel (expandable) ---
        if not e3_panel.empty:
            with st.expander(f"Full Insider Trading Panel ({len(e3_panel)} rows)", expanded=False):
                _panel_cols = [c for c in [
                    "TICKER", "EVENT_ID", "EVENT_DATE", "IS_TREATMENT", "LEAN",
                    "ABNORMAL_SELLING", "HAS_SUFFICIENT_DATA", "CAR_POST",
                    "PRE_FULL_NET_DOLLAR_SOLD", "PRE_FULL_NET_SELL_RATIO",
                    "BENCHMARK_NET_DOLLAR_SOLD", "POST_NET_DOLLAR_SOLD",
                ] if c in e3_panel.columns]
                _panel_display = e3_panel[_panel_cols].copy()
                for c in _panel_display.select_dtypes(include=[np.number]).columns:
                    _panel_display[c] = _panel_display[c].round(4)
                render_df(_panel_display, "Insider Panel", "a3_full_panel", height=500)

    else:
        st.info(
            "No Essay 3 model results found. Click **Run Full Pipeline** in the "
            "sidebar or run `python -m model.essay3` to generate results."
        )

    st.divider()

    # --- Expected contribution ---
    st.subheader("Contribution to Literature")
    st.markdown(
        "This essay extends the insider trading literature (Seyhun, 1986; Lakonishok "
        "and Lee, 2001; Cohen et al., 2012) into the political economy domain. If "
        "insiders systematically sell before culture war events that destroy shareholder "
        "value, this has implications for:\n\n"
        "- **SEC enforcement** -- whether politically-motivated corporate actions "
        "constitute material nonpublic information under Rule 10b-5\n"
        "- **Corporate governance** -- whether boards adequately manage the intersection "
        "of political positioning and fiduciary duty\n"
        "- **Market efficiency** -- whether insider trading transmits political "
        "information into prices before public disclosure"
    )

    # --- Chart Gallery ---
    render_chart_gallery(E3_CHARTS, gallery_id="e3")


# =============================================================================
# TAB 4: ENRICHED DATA
# =============================================================================

with tab_enriched:
    st.header("Enriched Dataset")
    st.markdown(
        "Cleaned and enriched culture war event data combined with stock prices, "
        "Fama-French factors, VIX, macroeconomic indicators, and news sentiment."
    )
    st.divider()

    # List all available tables from active backend + SQLite model tables
    table_counts = {}
    with _get_loader() as db:
        for t in db.list_tables():
            name = t["name"]
            if "rows" in t:
                table_counts[name] = t["rows"]
            else:
                try:
                    cnt_df = db.run_query(f'SELECT COUNT(*) as n FROM "{name}"')
                    table_counts[name] = int(cnt_df.iloc[0]["n"]) if not cnt_df.empty else 0
                except Exception:
                    table_counts[name] = 0
    # Also include SQLite-only tables (model results, figures) when using Athena
    if DB_BACKEND == "athena" and DB_PATH.exists():
        with SQLiteLoader(db_path=str(DB_PATH)) as db:
            for t in db.list_tables():
                if t["name"] not in table_counts:
                    table_counts[t["name"]] = t["rows"]

    total_tables = len(table_counts)
    total_rows = sum(table_counts.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tables", fmt_num(total_tables))
    c2.metric("Total Rows", fmt_num(total_rows))
    c3.metric("Database Size",
              f"{DB_PATH.stat().st_size / 1024 / 1024:.1f} MB" if DB_PATH.exists() else "--")

    st.divider()

    # Table overview
    st.subheader("Database Tables")
    table_overview = pd.DataFrame([
        {"Table": k, "Rows": f"{v:,}"} for k, v in sorted(table_counts.items())
    ])
    render_df(table_overview, "Table Overview", "enriched_table_overview")

    st.divider()

    # Culture War Companies explorer
    st.subheader("Culture War Companies")
    cw_df = load_table("CULTURE_WAR_COMPANIES")

    if not cw_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", fmt_num(len(cw_df)))
        c2.metric("Unique Tickers", fmt_num(cw_df["TICKER"].nunique()) if "TICKER" in cw_df.columns else "--")
        c3.metric("Industries", fmt_num(cw_df["INDUSTRY"].nunique()) if "INDUSTRY" in cw_df.columns else "--")
        c4.metric(
            "Political Leanings",
            fmt_num(cw_df["ESTIMATED_POLITICAL_LEANING"].nunique())
            if "ESTIMATED_POLITICAL_LEANING" in cw_df.columns else "--",
        )

        # Filters
        filter_c1, filter_c2, filter_c3 = st.columns(3)
        with filter_c1:
            leaning_options = ["All"]
            if "ESTIMATED_POLITICAL_LEANING" in cw_df.columns:
                leaning_options += sorted(cw_df["ESTIMATED_POLITICAL_LEANING"].dropna().unique().tolist())
            selected_leaning = st.selectbox("Political Leaning", leaning_options, key="enriched_leaning")

        with filter_c2:
            industry_options = ["All"]
            if "INDUSTRY" in cw_df.columns:
                industry_options += sorted(cw_df["INDUSTRY"].dropna().unique().tolist())
            selected_industry = st.selectbox("Industry", industry_options, key="enriched_industry")

        with filter_c3:
            company_search = st.text_input("Search Company", "", key="enriched_company")

        # Apply filters
        filtered = cw_df.copy()
        if selected_leaning != "All" and "ESTIMATED_POLITICAL_LEANING" in filtered.columns:
            filtered = filtered[filtered["ESTIMATED_POLITICAL_LEANING"] == selected_leaning]
        if selected_industry != "All" and "INDUSTRY" in filtered.columns:
            filtered = filtered[filtered["INDUSTRY"] == selected_industry]
        if company_search and "COMPANY" in filtered.columns:
            filtered = filtered[
                filtered["COMPANY"].str.contains(company_search, case=False, na=False)
            ]

        st.markdown(f"**{len(filtered):,} records**")

        display_cols = ["COMPANY", "TICKER", "EVENT_DATE", "CULTURE_WAR_EVENT",
                        "INDUSTRY", "ESTIMATED_POLITICAL_LEANING", "CONTROL_TICKER"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        render_df(filtered[display_cols], "Culture War Companies", "enriched_cw_filtered", height=500)

        with st.expander("View All Columns", expanded=False):
            render_df(filtered, "Full Dataset", "enriched_cw_full", height=500)

    st.divider()

    # Table data explorer
    st.subheader("Table Explorer")
    st.markdown("Select any table from the database to explore its contents.")

    selected_table = st.selectbox(
        "Select table:",
        sorted(table_counts.keys()),
        key="enriched_table_select",
    )

    if selected_table:
        tbl_df = load_table(selected_table)
        if not tbl_df.empty:
            c1, c2 = st.columns(2)
            c1.metric("Rows", fmt_num(len(tbl_df)))
            c2.metric("Columns", fmt_num(len(tbl_df.columns)))

            with st.expander("Column Information", expanded=False):
                col_info = pd.DataFrame({
                    "Column": tbl_df.columns,
                    "Type": tbl_df.dtypes.astype(str),
                    "Non-Null": tbl_df.notna().sum().values,
                    "Null %": (tbl_df.isna().sum() / max(len(tbl_df), 1) * 100).round(2).values,
                })
                render_df(col_info, "Column Info", f"enriched_{selected_table}_colinfo")

            render_df(tbl_df, selected_table, f"enriched_{selected_table}", height=500)
        else:
            st.info(f"Table `{selected_table}` is empty.")


# =============================================================================
# TAB 5: RAW DATA
# =============================================================================

with tab_raw:
    st.header("Raw Data Files")
    st.markdown("Original CSV files before processing.")
    st.divider()

    # Collect available CSV files
    raw_files = {}
    for f in sorted(BASE_DIR.glob("*.csv")):
        raw_files[f.name] = f

    # Check subdirectories for CSVs
    for subdir in ["data", "news_data", "fama_french_data", "sec_form4_data"]:
        sub_path = BASE_DIR / subdir
        if sub_path.exists():
            for f in sorted(sub_path.glob("*.csv")):
                raw_files[f"{subdir}/{f.name}"] = f

    if not raw_files:
        st.info("No CSV data files found.")
    else:
        selected_name = st.selectbox("Select file:", list(raw_files.keys()), key="raw_file")
        selected_path = raw_files[selected_name]

        file_size = selected_path.stat().st_size

        try:
            try:
                df_raw = pd.read_csv(selected_path, encoding="utf-8")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(selected_path, encoding="latin-1")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File", selected_name)
            c2.metric(
                "Size",
                f"{file_size / 1024:.1f} KB" if file_size < 1_048_576
                else f"{file_size / 1_048_576:.2f} MB",
            )
            c3.metric("Rows", fmt_num(len(df_raw)))
            c4.metric("Columns", fmt_num(len(df_raw.columns)))

            with st.expander("Column Information", expanded=False):
                col_info = pd.DataFrame({
                    "Column": df_raw.columns,
                    "Type": df_raw.dtypes.astype(str),
                    "Non-Null": df_raw.notna().sum().values,
                    "Null %": (df_raw.isna().sum() / max(len(df_raw), 1) * 100).round(2).values,
                })
                render_df(col_info, "Column Info", "raw_col_info")

            render_df(df_raw, "Raw Data", "raw_data", height=500)

        except Exception as e:
            st.error(f"Error reading file: {e}")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    f"""<div style='text-align: center; font-family: Source Serif 4, Georgia, serif;
    color: {NAVY}; padding: 1rem 0; font-size: 0.85rem;'>
    Signals and Systems: The Political Economy of Investor Sentiment and Financial Innovation<br>
    <span style='font-size:0.8rem;'>Ashley D. Roseboro &middot;
    College of Arts and Sciences, University of South Alabama &middot; 2026</span>
    </div>""",
    unsafe_allow_html=True,
)
