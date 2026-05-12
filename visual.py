"""
Visualization module for Signals & Systems dissertation research.

Generates 86 publication-quality figures across three essays:
  Essay 1 (33): Volatility regimes & Fama-French five-factor model
  Essay 2 (33): Culture war event study with regime conditioning
  Essay 3 (20): Informed insider trading around political decisions

Each figure is saved to disk (PNG) and stored in the database.

Pipeline order: clean -> etl -> database -> essay1 -> essay1_matched ->
               essay2 -> essay2_did -> essay3 -> **visual** -> dashboard

Usage:
    python visual.py                  # generate all 100 figures
    python visual.py --essay 1        # generate Essay 1 figures only
    python visual.py --test           # run chart tests
"""

import io
import logging
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# ═══════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════

STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Color palettes
C_REGIME = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c',
            'Low VIX': '#27ae60', 'Medium VIX': '#f39c12', 'High VIX': '#e74c3c'}
C_LEAN = {'Conservative': '#c0392b', 'Liberal': '#2980b9', 'Mixed': '#7f8c8d'}
C_TC = {'Treatment': '#e74c3c', 'Control': '#3498db'}
C_FACTORS = {'MKT_RF': '#2c3e50', 'SMB': '#e67e22', 'HML': '#27ae60',
             'RMW': '#8e44ad', 'CMA': '#2980b9'}
C_SEQ = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
         '#1abc9c', '#e84393', '#00b894', '#fdcb6e', '#6c5ce7']


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _empty_fig(title='No data available'):
    """Return a blank figure with a centred message."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='#999')
    ax.set_axis_off()
    return fig


def _sig(p):
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.10:
        return '\u2020'  # dagger
    return ''


def _col(df, candidates):
    """Return the first matching column name or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_float(val, default=np.nan):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════
# RESULT STORE
# ═══════════════════════════════════════════════════════════════════════

class ResultStore:
    """Load all essay results from Database.py loaders (Athena or SQLite)."""

    def __init__(self, backend=None, sqlite_path=None):
        self.backend = backend or 'auto'
        self._loader = None
        self._connect(backend, sqlite_path)
        self._load_all()

    # ── connection ─────────────────────────────────────────────────────

    def _connect(self, backend, sqlite_path):
        if backend == 'sqlite':
            self._connect_sqlite(sqlite_path)
            return
        if backend in ('aws', 'athena'):
            self._connect_athena()
            return

        # Auto: try Athena, fall back to SQLite
        try:
            self._connect_athena()
            test = self._loader.read_table('ESSAY1_FF5_COEFFICIENTS', limit=1)
            if test.empty:
                raise RuntimeError('Athena tables empty')
            return
        except Exception as e:
            logger.warning('Athena unavailable (%s), falling back to SQLite', e)
            if self._loader:
                try:
                    self._loader.close()
                except Exception:
                    pass
        self._connect_sqlite(sqlite_path)

    def _connect_athena(self):
        from Database import AthenaLoader
        loader = AthenaLoader()
        loader.connect()
        self._loader = loader
        self.backend = 'athena'

    def _connect_sqlite(self, db_path=None):
        from Database import SQLiteLoader
        loader = SQLiteLoader(db_path=db_path)
        loader.connect()
        self._loader = loader
        self.backend = 'sqlite'

    def _read(self, table_name):
        try:
            df = self._loader.read_table(table_name)
            if df is None or df.empty:
                return pd.DataFrame()
            # Athena returns lowercase columns; normalize to uppercase
            df.columns = [c.upper() for c in df.columns]
            return df
        except Exception:
            return pd.DataFrame()

    # ── load all tables ────────────────────────────────────────────────

    def _load_all(self):
        logger.info('Loading results from %s ...', self.backend)
        # Essay 1
        self.e1_ff5_coefficients = self._read('ESSAY1_FF5_COEFFICIENTS')
        self.e1_factor_premia = self._read('ESSAY1_FACTOR_PREMIA')
        self.e1_chow_test = self._read('ESSAY1_CHOW_TEST')
        self.e1_cw_stock = self._read('ESSAY1_CW_STOCK_RESULTS')
        self.e1_sentiment = self._read('ESSAY1_SENTIMENT_DAILY')
        self.e1_fomo = self._read('ESSAY1_FOMO_BY_REGIME')
        self.e1_matched_deltas = self._read('ESSAY1_MATCHED_DELTAS')
        self.e1_matched_ttest = self._read('ESSAY1_MATCHED_TTEST')
        self.e1_matched_sign = self._read('ESSAY1_MATCHED_SIGN')
        self.e1_matched_amp = self._read('ESSAY1_MATCHED_AMPLIFICATION')
        self.e1_matched_coverage = self._read('ESSAY1_MATCHED_COVERAGE')
        # Raw data for supplementary charts
        self.vix_data = self._read('VIX_DATA')
        self.cw_companies = self._read('CULTURE_WAR_COMPANIES')

        # Essay 2
        self.e2_car_panel = self._read('ESSAY2_CAR_PANEL')
        self.e2_did_coeff = self._read('ESSAY2_DID_COEFFICIENTS')
        self.e2_parallel = self._read('ESSAY2_PARALLEL_TRENDS')
        self.e2_news_sent = self._read('ESSAY2_NEWS_SENTIMENT')
        self.e2_filing_sent = self._read('ESSAY2_FILING_SENTIMENT')
        self.e2_event_nlp = self._read('ESSAY2_EVENT_NLP')
        self.e2_alignment = self._read('ESSAY2_POLITICAL_ALIGNMENT')
        self.e2_phrases = self._read('ESSAY2_DISTINCTIVE_PHRASES')
        self.e2_validation = self._read('ESSAY2_ALIGNMENT_VALIDATION')
        self.e2_mw_panel = self._read('ESSAY2_MULTI_WINDOW_PANEL')
        self.e2_mw_summary = self._read('ESSAY2_MULTI_WINDOW_SUMMARY')
        self.e2_mw_lean = self._read('ESSAY2_MULTI_WINDOW_BY_LEAN')
        self.e2_mw_tc = self._read('ESSAY2_MULTI_WINDOW_TREAT_VS_CTRL')
        self.e2_cont_summary = self._read('ESSAY2_CONTAGION_SUMMARY')
        self.e2_cont_lean = self._read('ESSAY2_CONTAGION_BY_LEAN')
        self.e2_cont_facing = self._read('ESSAY2_CONTAGION_BY_FACING')
        self.e2_cont_peer = self._read('ESSAY2_CONTAGION_PEER_VS_NONPEER')
        self.e2_cont_cb = self._read('ESSAY2_CONTAGION_CONS_VS_B2B')
        self.e2_cont_lp = self._read('ESSAY2_CONTAGION_LEAN_PAIRWISE')
        self.e2_cont_mech = self._read('ESSAY2_CONTAGION_LEAN_MECH')
        self.e2_cont_tight = self._read('ESSAY2_CONTAGION_TIGHT_DIFF')
        self.e2_peer_parallel = self._read('ESSAY2_PEER_PARALLEL_TRENDS')

        # Essay 3 — Informed Insider Trading Around Political Decisions
        self.e3_panel = self._read('ESSAY3_PANEL')
        self.e3_informed_trading = self._read('ESSAY3_INFORMED_TRADING')
        self.e3_informed_proximity = self._read('ESSAY3_INFORMED_PROXIMITY')
        self.e3_informed_dollars = self._read('ESSAY3_INFORMED_DOLLARS')
        self.e3_size_accuracy = self._read('ESSAY3_SIZE_ACCURACY')
        self.e3_size_accuracy_slopes = self._read('ESSAY3_SIZE_ACCURACY_SLOPES')
        self.e3_reversal_regression = self._read('ESSAY3_REVERSAL_REGRESSION')
        self.e3_control_trades = self._read('ESSAY3_CONTROL_TRADES')
        self.e3_crsp_profits = self._read('ESSAY3_CRSP_PROFITS')
        self.e3_crsp_summary = self._read('ESSAY3_CRSP_SUMMARY')
        self.e3_insider_panel = self._read('ESSAY3_INSIDER_PANEL')
        self.e3_wilcoxon_family = self._read('ESSAY3_WILCOXON_FAMILY')
        self.e3_insider_concentration = self._read('ESSAY3_INSIDER_CONCENTRATION')
        self.e3_concentration_cuts = self._read('ESSAY3_CONCENTRATION_CUTS')
        self.e3_tost = self._read('ESSAY3_TOST')
        self.e3_placebo = self._read('ESSAY3_PLACEBO')
        self.e3_bootstrap_ci = self._read('ESSAY3_BOOTSTRAP_CI')
        self.e3_stratification = self._read('ESSAY3_STRATIFICATION')
        self.e3_mean_vs_dist = self._read('ESSAY3_MEAN_VS_DISTRIBUTIONAL')
        self.e3_quantile_reg = self._read('ESSAY3_QUANTILE_REGRESSION')
        self.e3_trimmed = self._read('ESSAY3_TRIMMED_ROBUSTNESS')
        self.e3_bootstrap_wilcoxon = self._read('ESSAY3_BOOTSTRAP_WILCOXON')
        self.e3_active_subset = self._read('ESSAY3_ACTIVE_SUBSET')
        self.e3_repeat_traders = self._read('ESSAY3_REPEAT_TRADERS')

        logger.info('ResultStore ready')

    # ── figure persistence ─────────────────────────────────────────────

    def save_figure(self, name, fig, metadata=None):
        os.makedirs(FIGURE_DIR, exist_ok=True)
        png_path = os.path.join(FIGURE_DIR, f'{name}.png')
        fig.savefig(png_path)
        file_size = os.path.getsize(png_path)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        png_bytes = buf.read()
        buf.close()

        row = {
            'FIGURE_NAME': name,
            'FORMAT': 'png',
            'WIDTH_PX': int(fig.get_size_inches()[0] * fig.dpi),
            'HEIGHT_PX': int(fig.get_size_inches()[1] * fig.dpi),
            'FILE_SIZE_BYTES': len(png_bytes),
            'FILE_PATH': png_path,
            'IMAGE_DATA': png_bytes,
            'CREATED_AT': datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            row.update(metadata)
        df = pd.DataFrame([row])

        try:
            if self.backend == 'sqlite':
                # Drop and recreate table if schema changed (new metadata cols)
                try:
                    self._loader.conn.execute(
                        'DELETE FROM FIGURES WHERE FIGURE_NAME = ?', (name,))
                    self._loader.conn.commit()
                except Exception:
                    # Table may not exist or schema mismatch — drop to recreate
                    try:
                        self._loader.conn.execute('DROP TABLE IF EXISTS FIGURES')
                        self._loader.conn.commit()
                    except Exception:
                        pass
                result = self._loader.write_table(df, 'FIGURES', replace=False)
            else:
                # Athena/S3: drop binary IMAGE_DATA (not Parquet-safe), append only
                athena_df = df.drop(columns=['IMAGE_DATA'], errors='ignore')
                result = self._loader.write_table(athena_df, 'FIGURES', replace=False)
            return result
        except Exception as e:
            logger.error('Failed to save figure %s: %s', name, e)
            return {'status': 'FAILED', 'error': str(e)}

    def close(self):
        if self._loader:
            self._loader.close()
            self._loader = None


# ═══════════════════════════════════════════════════════════════════════
# ██  ESSAY 1 — Volatility Regimes & Fama-French Five-Factor (33)
# ═══════════════════════════════════════════════════════════════════════

# ── 1. FF5 factor premia by regime ────────────────────────────────────

def e1_01_factor_premia_by_regime(store):
    """Annualised FF5 factor premia by VIX regime (grouped bar)."""
    df = store.e1_factor_premia
    if df.empty:
        return _empty_fig('E1-01: No factor premia data')
    factors = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    ann_cols = [f'{f}_MEAN_ANN' for f in factors]
    present = [c for c in ann_cols if c in df.columns]
    if not present:
        return _empty_fig('E1-01: Missing annualised columns')

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        regimes = df['REGIME'].tolist()
        x = np.arange(len(present))
        width = 0.8 / len(regimes)
        for i, regime in enumerate(regimes):
            row = df[df['REGIME'] == regime].iloc[0]
            vals = [_safe_float(row.get(c)) for c in present]
            color = C_REGIME.get(regime, C_SEQ[i % len(C_SEQ)])
            ax.bar(x + i * width, vals, width, label=regime, color=color, edgecolor='white')
        ax.set_xticks(x + width * (len(regimes) - 1) / 2)
        ax.set_xticklabels([c.replace('_MEAN_ANN', '') for c in present])
        ax.set_ylabel('Annualised Mean Return')
        ax.set_title('FF5 Factor Premia by VIX Regime')
        ax.legend(title='Regime')
        ax.axhline(0, color='black', lw=0.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        fig.tight_layout()
    return fig


# ── 2. FF5 alpha by regime ────────────────────────────────────────────

def e1_02_alpha_by_regime(store):
    """Alpha estimates with significance stars across regimes."""
    df = store.e1_ff5_coefficients
    if df.empty or 'ALPHA' not in df.columns:
        return _empty_fig('E1-02: No FF5 coefficients')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = df['REGIME'].tolist()
        alphas = df['ALPHA'].astype(float).tolist()
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        bars = ax.bar(regimes, alphas, color=colors, edgecolor='white', width=0.5)
        for bar, r in zip(bars, df.itertuples()):
            p = _safe_float(getattr(r, 'ALPHA_P', np.nan))
            star = _sig(p)
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f'{y:.4f}{star}', ha='center',
                    va='bottom' if y >= 0 else 'top', fontsize=10)
        ax.set_ylabel('Alpha (daily)')
        ax.set_title('FF5 Intercept (Alpha) by VIX Regime')
        ax.axhline(0, color='black', lw=0.5)
        fig.tight_layout()
    return fig


# ── 3. FF5 beta heatmap ───────────────────────────────────────────────

def e1_03_beta_heatmap(store):
    """Heatmap of FF5 factor loadings across regimes."""
    df = store.e1_ff5_coefficients
    if df.empty or 'REGIME' not in df.columns:
        return _empty_fig('E1-03: No FF5 coefficients')
    beta_cols = [c for c in df.columns if c.endswith('_BETA')]
    if not beta_cols:
        return _empty_fig('E1-03: No beta columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        data = df.set_index('REGIME')[beta_cols].astype(float)
        data.columns = [c.replace('_BETA', '') for c in beta_cols]
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    ax=ax, linewidths=0.5, cbar_kws={'label': 'Beta'})
        ax.set_title('FF5 Factor Loadings by VIX Regime')
        ax.set_ylabel('')
        fig.tight_layout()
    return fig


# ── 4. FF5 t-statistic heatmap ────────────────────────────────────────

def e1_04_tstat_heatmap(store):
    """Heatmap of t-statistics for FF5 betas (highlights significance)."""
    df = store.e1_ff5_coefficients
    if df.empty or 'REGIME' not in df.columns:
        return _empty_fig('E1-04: No FF5 coefficients')
    t_cols = [c for c in df.columns if c.endswith('_T')]
    if not t_cols:
        return _empty_fig('E1-04: No t-stat columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        data = df.set_index('REGIME')[t_cols].astype(float)
        data.columns = [c.replace('_T', '') for c in t_cols]
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax, linewidths=0.5, cbar_kws={'label': 't-statistic'},
                    vmin=-4, vmax=4)
        ax.set_title('FF5 Factor Loading t-Statistics by Regime')
        ax.set_ylabel('')
        fig.tight_layout()
    return fig


# ── 5. Chow structural break test ─────────────────────────────────────

def e1_05_chow_test(store):
    """Visual summary of Chow structural break F-test."""
    df = store.e1_chow_test
    if df.empty:
        return _empty_fig('E1-05: No Chow test results')
    row = df.iloc[0]
    f_stat = _safe_float(row.get('f_stat', row.get('F_STAT')))
    p_val = _safe_float(row.get('p_value', row.get('P_VALUE')))
    df_num = _safe_float(row.get('df_numerator', row.get('DF_NUMERATOR')))
    df_den = _safe_float(row.get('df_denominator', row.get('DF_DENOMINATOR')))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        # Plot F distribution under null
        if not np.isnan(df_num) and not np.isnan(df_den):
            x = np.linspace(0, max(f_stat * 1.5, 5), 500)
            y = sp_stats.f.pdf(x, df_num, df_den)
            ax.fill_between(x, y, alpha=0.2, color='#3498db')
            ax.plot(x, y, color='#3498db', lw=1.5, label='F distribution (null)')
            # Critical value at 5%
            f_crit = sp_stats.f.ppf(0.95, df_num, df_den)
            ax.axvline(f_crit, color='#e67e22', ls='--', lw=1.2, label=f'Critical (5%) = {f_crit:.2f}')
        # Observed F
        ax.axvline(f_stat, color='#e74c3c', lw=2, label=f'Observed F = {f_stat:.2f}')
        sig = 'YES' if p_val < 0.05 else 'NO'
        ax.set_title(f'Chow Structural Break Test\nF = {f_stat:.2f}, p = {p_val:.4f} — Break: {sig}')
        ax.set_xlabel('F statistic')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')
        fig.tight_layout()
    return fig


# ── 6. R-squared by regime ─────────────────────────────────────────────

def e1_06_rsquared_by_regime(store):
    """R-squared of FF5 model across regimes."""
    df = store.e1_ff5_coefficients
    if df.empty or 'R_SQUARED' not in df.columns:
        return _empty_fig('E1-06: No R-squared data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        regimes = df['REGIME'].tolist()
        r2 = df['R_SQUARED'].astype(float).tolist()
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        bars = ax.bar(regimes, r2, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, r2):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('R\u00b2')
        ax.set_title('FF5 Model Fit (R\u00b2) by VIX Regime')
        ax.set_ylim(0, min(max(r2) * 1.3, 1.0) if r2 else 1.0)
        fig.tight_layout()
    return fig


# ── 7. Factor premia dot plot ──────────────────────────────────────────

def e1_07_factor_premia_dot(store):
    """Dot plot of factor premia with regime comparison."""
    df = store.e1_factor_premia
    if df.empty:
        return _empty_fig('E1-07: No factor premia data')
    factors = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    mean_cols = [f'{f}_MEAN' for f in factors]
    present = [c for c in mean_cols if c in df.columns]
    if not present:
        return _empty_fig('E1-07: No mean columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes = df['REGIME'].tolist()
        y_pos = np.arange(len(present))
        for i, regime in enumerate(regimes):
            row = df[df['REGIME'] == regime].iloc[0]
            vals = [_safe_float(row.get(c)) * 100 for c in present]
            color = C_REGIME.get(regime, C_SEQ[i])
            offset = (i - len(regimes) / 2 + 0.5) * 0.15
            ax.scatter(vals, y_pos + offset, s=80, color=color,
                       label=regime, zorder=3, edgecolors='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([c.replace('_MEAN', '') for c in present])
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Daily Mean Return (bps)')
        ax.set_title('Factor Risk Premia by Regime')
        ax.legend(title='Regime')
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
    return fig


# ── 8. CW stock alpha boxplot ──────────────────────────────────────────

def e1_08_cw_alpha_boxplot(store):
    """Box plot of culture-war stock alphas by regime."""
    df = store.e1_cw_stock
    if df.empty or 'ALPHA' not in df.columns or 'REGIME' not in df.columns:
        return _empty_fig('E1-08: No CW stock results')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes_present = [r for r in ['Low', 'Medium', 'High'] if r in df['REGIME'].values]
        if not regimes_present:
            regimes_present = sorted(df['REGIME'].unique())
        palette = {r: C_REGIME.get(r, '#555') for r in regimes_present}
        sns.boxplot(data=df[df['REGIME'].isin(regimes_present)],
                    x='REGIME', y='ALPHA', order=regimes_present,
                    hue='REGIME', palette=palette, ax=ax, fliersize=3, legend=False)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_title('Culture War Stock Alphas by VIX Regime')
        ax.set_ylabel('Alpha (daily)')
        ax.set_xlabel('VIX Regime')
        fig.tight_layout()
    return fig


# ── 9. CW stock alpha distribution ────────────────────────────────────

def e1_09_cw_alpha_hist(store):
    """Histogram of CW stock alphas coloured by regime."""
    df = store.e1_cw_stock
    if df.empty or 'ALPHA' not in df.columns:
        return _empty_fig('E1-09: No CW stock alphas')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for regime in sorted(df['REGIME'].unique()):
            vals = df[df['REGIME'] == regime]['ALPHA'].astype(float).dropna()
            color = C_REGIME.get(regime, '#555')
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=regime, edgecolor='white')
        ax.axvline(0, color='black', lw=0.5, ls='--')
        ax.set_xlabel('Alpha (daily)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Culture War Stock Alphas')
        ax.legend(title='Regime')
        fig.tight_layout()
    return fig


# ── 10. CW R-squared by regime ─────────────────────────────────────────

def e1_10_cw_rsquared(store):
    """Box plot of CW stock R-squared by regime."""
    df = store.e1_cw_stock
    if df.empty or 'R_SQUARED' not in df.columns:
        return _empty_fig('E1-10: No CW R-squared data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes = sorted(df['REGIME'].unique())
        palette = {r: C_REGIME.get(r, '#555') for r in regimes}
        sns.boxplot(data=df, x='REGIME', y='R_SQUARED', order=regimes,
                    hue='REGIME', palette=palette, ax=ax, fliersize=3, legend=False)
        ax.set_title('FF5 Model Fit Across Culture War Stocks')
        ax.set_ylabel('R\u00b2')
        ax.set_xlabel('VIX Regime')
        fig.tight_layout()
    return fig


# ── 11. CW market beta by regime ──────────────────────────────────────

def e1_11_cw_mkt_beta(store):
    """Market beta scatter for CW stocks, coloured by regime."""
    df = store.e1_cw_stock
    col = _col(df, ['MKT_RF_BETA', 'MKT_BETA'])
    if df.empty or not col:
        return _empty_fig('E1-11: No market beta data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for regime in sorted(df['REGIME'].unique()):
            sub = df[df['REGIME'] == regime]
            ax.scatter(sub.index, sub[col].astype(float), s=20, alpha=0.6,
                       color=C_REGIME.get(regime, '#555'), label=regime)
        ax.axhline(1, color='black', lw=0.5, ls='--', label='Beta = 1')
        ax.set_xlabel('Stock Index')
        ax.set_ylabel('Market Beta')
        ax.set_title('Culture War Stock Market Betas by Regime')
        ax.legend(title='Regime')
        fig.tight_layout()
    return fig


# ── 12. CW HML beta by regime ─────────────────────────────────────────

def e1_12_cw_hml_beta(store):
    """HML (value) beta distribution by regime."""
    df = store.e1_cw_stock
    col = _col(df, ['HML_BETA'])
    if df.empty or not col:
        return _empty_fig('E1-12: No HML beta data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes = sorted(df['REGIME'].unique())
        palette = {r: C_REGIME.get(r, '#555') for r in regimes}
        sns.violinplot(data=df, x='REGIME', y=col, order=regimes,
                       hue='REGIME', palette=palette, ax=ax, inner='box', cut=0, legend=False)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_title('HML (Value) Factor Loading by Regime')
        ax.set_ylabel('HML Beta')
        fig.tight_layout()
    return fig


# ── 13. CW alpha significance counts ──────────────────────────────────

def e1_13_cw_alpha_significance(store):
    """Count of significant vs non-significant alphas by regime."""
    df = store.e1_cw_stock
    if df.empty or 'ALPHA_P' not in df.columns:
        return _empty_fig('E1-13: No alpha p-values')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = sorted(df['REGIME'].unique())
        sig_counts = []
        for r in regimes:
            sub = df[df['REGIME'] == r]['ALPHA_P'].astype(float)
            sig_counts.append({
                'Regime': r,
                'p < 0.05': (sub < 0.05).sum(),
                'p < 0.10': ((sub >= 0.05) & (sub < 0.10)).sum(),
                'Not sig.': (sub >= 0.10).sum(),
            })
        sc = pd.DataFrame(sig_counts).set_index('Regime')
        sc.plot(kind='bar', stacked=True, ax=ax,
                color=['#e74c3c', '#f39c12', '#bdc3c7'], edgecolor='white')
        ax.set_title('Alpha Significance Across Regimes')
        ax.set_ylabel('Number of Stocks')
        ax.set_xlabel('')
        ax.legend(title='Significance')
        plt.xticks(rotation=0)
        fig.tight_layout()
    return fig


# ── 14. CW beta radar / parallel coordinates ──────────────────────────

def e1_14_cw_mean_betas_by_regime(store):
    """Mean factor betas for CW stocks by regime (grouped bar)."""
    df = store.e1_cw_stock
    if df.empty:
        return _empty_fig('E1-14: No CW stock results')
    beta_cols = [c for c in df.columns if c.endswith('_BETA')]
    if not beta_cols:
        return _empty_fig('E1-14: No beta columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        regimes = sorted(df['REGIME'].unique())
        means = df.groupby('REGIME')[beta_cols].mean()
        x = np.arange(len(beta_cols))
        width = 0.8 / len(regimes)
        for i, regime in enumerate(regimes):
            if regime in means.index:
                vals = means.loc[regime].astype(float).tolist()
                ax.bar(x + i * width, vals, width, label=regime,
                       color=C_REGIME.get(regime, C_SEQ[i]), edgecolor='white')
        ax.set_xticks(x + width * (len(regimes) - 1) / 2)
        ax.set_xticklabels([c.replace('_BETA', '') for c in beta_cols])
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Beta')
        ax.set_title('Mean Factor Loadings of CW Stocks by Regime')
        ax.legend(title='Regime')
        fig.tight_layout()
    return fig


# ── 15. FOMO Z-scores by regime ────────────────────────────────────────

def e1_15_fomo_by_regime(store):
    """FOMO z-score statistics by regime."""
    df = store.e1_fomo
    if df.empty or 'MEAN_FOMO_Z' not in df.columns:
        return _empty_fig('E1-15: No FOMO data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = df['REGIME'].tolist()
        means = df['MEAN_FOMO_Z'].astype(float).tolist()
        stds = df['STD_FOMO_Z'].astype(float).tolist()
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        ax.bar(regimes, means, yerr=stds, color=colors, edgecolor='white',
               width=0.5, capsize=5)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean FOMO Z-Score')
        ax.set_title('Fear-of-Missing-Out Z-Scores by VIX Regime')
        fig.tight_layout()
    return fig


# ── 16. Euphoria / panic rates ─────────────────────────────────────────

def e1_16_euphoria_panic(store):
    """Stacked bar of euphoria and panic rates by regime."""
    df = store.e1_fomo
    if df.empty:
        return _empty_fig('E1-16: No FOMO data')
    cols = ['PCT_EUPHORIA', 'PCT_PANIC']
    if not all(c in df.columns for c in cols):
        return _empty_fig('E1-16: Missing euphoria/panic columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = df['REGIME'].tolist()
        x = np.arange(len(regimes))
        euph = df['PCT_EUPHORIA'].astype(float).tolist()
        panic = df['PCT_PANIC'].astype(float).tolist()
        ax.bar(x, euph, color='#27ae60', label='Euphoria (Z > 2)', width=0.4)
        ax.bar(x + 0.4, panic, color='#e74c3c', label='Panic (Z < -2)', width=0.4)
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels(regimes)
        ax.set_ylabel('Fraction of Observations')
        ax.set_title('Euphoria and Panic Rates by Regime')
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
    return fig


# ── 17. Sentiment mean by regime ───────────────────────────────────────

def e1_17_sentiment_by_regime(store):
    """Mean sentiment score by regime."""
    df = store.e1_fomo
    if df.empty or 'MEAN_SENTIMENT' not in df.columns:
        return _empty_fig('E1-17: No sentiment data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = df['REGIME'].tolist()
        means = df['MEAN_SENTIMENT'].astype(float).tolist()
        stds = df['STD_SENTIMENT'].astype(float).tolist()
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        ax.bar(regimes, means, yerr=stds, color=colors, edgecolor='white',
               width=0.5, capsize=5)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean Sentiment')
        ax.set_title('News Sentiment by VIX Regime')
        fig.tight_layout()
    return fig


# ── 18. Sentiment daily time series ────────────────────────────────────

def e1_18_sentiment_timeseries(store):
    """Daily sentiment time series (if available), shaded by regime."""
    df = store.e1_sentiment
    if df.empty:
        return _empty_fig('E1-18: No daily sentiment data')
    date_col = _col(df, ['DATE', 'date'])
    sent_col = _col(df, ['SENT_MEAN', 'sent_mean', 'SENTIMENT'])
    regime_col = _col(df, ['REGIME', 'regime'])
    if not date_col or not sent_col:
        return _empty_fig('E1-18: Missing columns')
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, sent_col]).sort_values(date_col)
    # Aggregate by date
    daily = df.groupby(date_col).agg({sent_col: 'mean'}).reset_index()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily[date_col], daily[sent_col].astype(float).rolling(30).mean(),
                color='#2980b9', lw=1, label='30-day MA')
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean Sentiment')
        ax.set_title('Daily News Sentiment (30-Day Moving Average)')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 19. Matched control: paired t-test forest plot ─────────────────────

def e1_19_matched_ttest_forest(store):
    """Forest plot of paired t-test results (delta betas)."""
    df = store.e1_matched_ttest
    if df.empty or 'VARIABLE' not in df.columns:
        return _empty_fig('E1-19: No matched t-test results')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
        y = np.arange(len(df))
        means = df['MEAN_DELTA'].astype(float)
        stds = df['STD_DELTA'].astype(float) if 'STD_DELTA' in df.columns else pd.Series(0, index=df.index)
        labels = df['VARIABLE'].tolist()
        # Color by significance
        colors = []
        for _, r in df.iterrows():
            p = _safe_float(r.get('P_VALUE'))
            bh = r.get('BH_SIGNIFICANT', False)
            if bh:
                colors.append('#e74c3c')
            elif p < 0.05:
                colors.append('#f39c12')
            else:
                colors.append('#bdc3c7')
        ax.barh(y, means, xerr=stds, color=colors, edgecolor='white', height=0.6, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{v} ({r})' for v, r in zip(df['VARIABLE'], df['REGIME'])]
                           if 'REGIME' in df.columns else labels)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Mean Delta (Treatment - Control)')
        ax.set_title('Matched Control: Treatment-Control Differences')
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#e74c3c', label='BH significant'),
            Patch(color='#f39c12', label='p < 0.05'),
            Patch(color='#bdc3c7', label='Not significant'),
        ], loc='lower right', fontsize=8)
        fig.tight_layout()
    return fig


# ── 20. Matched: sign consistency ──────────────────────────────────────

def e1_20_matched_sign_consistency(store):
    """Percent of pairs with same-sign delta, by variable and regime."""
    df = store.e1_matched_sign
    if df.empty or 'PCT_MAJORITY' not in df.columns:
        return _empty_fig('E1-20: No sign consistency data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        if 'REGIME' in df.columns:
            pivot = df.pivot_table(values='PCT_MAJORITY', index='VARIABLE',
                                    columns='REGIME', aggfunc='first')
            pivot.plot(kind='bar', ax=ax, edgecolor='white',
                       color=[C_REGIME.get(c, '#555') for c in pivot.columns])
        else:
            ax.bar(df['VARIABLE'], df['PCT_MAJORITY'].astype(float),
                   color='#3498db', edgecolor='white')
        ax.axhline(0.5, color='red', lw=1, ls='--', label='50% (random)')
        ax.set_ylabel('% Majority Sign')
        ax.set_title('Sign Consistency of Treatment-Control Deltas')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
    return fig


# ── 21. Matched: regime amplification ──────────────────────────────────

def e1_21_matched_amplification(store):
    """High vs Low regime amplification test results."""
    df = store.e1_matched_amp
    if df.empty or 'VARIABLE' not in df.columns:
        return _empty_fig('E1-21: No amplification data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df))
        low = df['MEAN_DELTA_LOW'].astype(float) if 'MEAN_DELTA_LOW' in df.columns else pd.Series(dtype=float)
        high = df['MEAN_DELTA_HIGH'].astype(float) if 'MEAN_DELTA_HIGH' in df.columns else pd.Series(dtype=float)
        w = 0.35
        ax.bar(x - w / 2, low, w, color='#27ae60', label='Low VIX', edgecolor='white')
        ax.bar(x + w / 2, high, w, color='#e74c3c', label='High VIX', edgecolor='white')
        # Add significance stars
        for idx, (_, r) in enumerate(df.iterrows()):
            p = _safe_float(r.get('P_VALUE'))
            star = _sig(p)
            if star:
                ymax = max(_safe_float(r.get('MEAN_DELTA_LOW', 0)),
                           _safe_float(r.get('MEAN_DELTA_HIGH', 0)))
                ax.text(idx, ymax + abs(ymax) * 0.05, star,
                        ha='center', fontsize=12, color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(df['VARIABLE'].tolist(), rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Delta')
        ax.set_title('Regime Amplification: High vs Low VIX')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 22. Matched: delta beta distributions ──────────────────────────────

def e1_22_matched_delta_dist(store):
    """Histogram of delta alphas across matched pairs."""
    df = store.e1_matched_deltas
    if df.empty or 'ALPHA_DELTA' not in df.columns:
        return _empty_fig('E1-22: No matched deltas')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = df['ALPHA_DELTA'].astype(float).dropna()
        ax.hist(vals, bins=40, color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', lw=1, ls='--')
        ax.axvline(vals.mean(), color='#e74c3c', lw=1.5,
                   label=f'Mean = {vals.mean():.4f}')
        ax.set_xlabel('Alpha Delta (Treatment - Control)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Treatment-Control Alpha Differences')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 23. Matched: delta heatmap ─────────────────────────────────────────

def e1_23_matched_delta_heatmap(store):
    """Heatmap of mean deltas by variable and regime."""
    df = store.e1_matched_ttest
    if df.empty or 'VARIABLE' not in df.columns or 'REGIME' not in df.columns:
        return _empty_fig('E1-23: No matched t-test data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot = df.pivot_table(values='MEAN_DELTA', index='VARIABLE',
                                columns='REGIME', aggfunc='first')
        sns.heatmap(pivot.astype(float), annot=True, fmt='.4f', cmap='RdBu_r',
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title('Mean Treatment-Control Deltas by Variable and Regime')
        fig.tight_layout()
    return fig


# ── 24. Matched: coverage heatmap ─────────────────────────────────────

def e1_24_matched_coverage(store):
    """Coverage: which tickers have results in which regimes."""
    df = store.e1_matched_coverage
    if df.empty:
        return _empty_fig('E1-24: No coverage data')
    ticker_col = _col(df, ['TICKER', 'ticker'])
    regime_col = _col(df, ['REGIME', 'regime'])
    has_col = _col(df, ['HAS_RESULT', 'has_result'])
    if not ticker_col or not regime_col or not has_col:
        return _empty_fig('E1-24: Missing columns')
    with plt.rc_context(STYLE):
        pivot = df.pivot_table(values=has_col, index=ticker_col,
                                columns=regime_col, aggfunc='first').fillna(0)
        n_tickers = len(pivot)
        fig_h = max(4, min(n_tickers * 0.2, 20))
        fig, ax = plt.subplots(figsize=(6, fig_h))
        sns.heatmap(pivot.astype(float), cmap='YlGn', ax=ax, cbar=False,
                    linewidths=0.3, yticklabels=True if n_tickers < 60 else False)
        ax.set_title('Matched Control Coverage by Ticker and Regime')
        fig.tight_layout()
    return fig


# ── 25. CW stock count by regime ──────────────────────────────────────

def e1_25_cw_stock_count(store):
    """Number of CW stocks with valid results per regime."""
    df = store.e1_cw_stock
    if df.empty or 'REGIME' not in df.columns:
        return _empty_fig('E1-25: No CW stock data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = df.groupby('REGIME')['TICKER'].nunique()
        regimes = counts.index.tolist()
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        ax.bar(regimes, counts.values, color=colors, edgecolor='white', width=0.5)
        for i, (r, v) in enumerate(zip(regimes, counts.values)):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=11)
        ax.set_ylabel('Unique Tickers')
        ax.set_title('Culture War Stocks with Valid FF5 Results')
        fig.tight_layout()
    return fig


# ── 26. CW alpha vs R-squared scatter ─────────────────────────────────

def e1_26_alpha_vs_rsq(store):
    """Scatter: alpha vs R-squared for CW stocks."""
    df = store.e1_cw_stock
    if df.empty or 'ALPHA' not in df.columns or 'R_SQUARED' not in df.columns:
        return _empty_fig('E1-26: No alpha/R-squared data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        for regime in sorted(df['REGIME'].unique()):
            sub = df[df['REGIME'] == regime]
            ax.scatter(sub['R_SQUARED'].astype(float), sub['ALPHA'].astype(float),
                       s=30, alpha=0.6, color=C_REGIME.get(regime, '#555'),
                       label=regime)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_xlabel('R\u00b2')
        ax.set_ylabel('Alpha (daily)')
        ax.set_title('Alpha vs Model Fit Across CW Stocks')
        ax.legend(title='Regime')
        fig.tight_layout()
    return fig


# ── 27. Matched: N treatment firms by regime ──────────────────────────

def e1_27_matched_n_firms(store):
    """Sample sizes in matched control analysis."""
    df = store.e1_matched_ttest
    if df.empty:
        return _empty_fig('E1-27: No matched data')
    ncol = _col(df, ['N_TREATMENT_FIRMS', 'N_RAW_PAIRS'])
    if not ncol:
        return _empty_fig('E1-27: No sample size columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        if 'REGIME' in df.columns:
            summary = df.groupby('REGIME')[ncol].first()
            ax.bar(summary.index, summary.values.astype(float),
                   color=[C_REGIME.get(r, '#555') for r in summary.index],
                   edgecolor='white', width=0.5)
        else:
            ax.bar(['All'], [df[ncol].iloc[0]], color='#3498db', width=0.3)
        ax.set_ylabel(ncol.replace('_', ' ').title())
        ax.set_title('Matched Control Sample Sizes')
        fig.tight_layout()
    return fig


# ── 28. Matched: p-value distribution ──────────────────────────────────

def e1_28_matched_pvalue_dist(store):
    """Histogram of p-values from matched t-tests."""
    df = store.e1_matched_ttest
    if df.empty or 'P_VALUE' not in df.columns:
        return _empty_fig('E1-28: No p-values')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        pvals = df['P_VALUE'].astype(float).dropna()
        ax.hist(pvals, bins=20, color='#3498db', edgecolor='white', alpha=0.8,
                range=(0, 1))
        ax.axvline(0.05, color='red', lw=1.5, ls='--', label='p = 0.05')
        ax.axvline(0.10, color='orange', lw=1, ls='--', label='p = 0.10')
        ax.set_xlabel('P-Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Matched T-Test P-Values')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 29. VIX distribution by regime ────────────────────────────────────

def e1_29_vix_distribution(store):
    """VIX kernel density by regime (from FOMO data)."""
    df = store.e1_fomo
    if df.empty:
        return _empty_fig('E1-29: No FOMO/regime data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = df['REGIME'].tolist()
        means = df['MEAN_SENTIMENT'].astype(float).tolist() if 'MEAN_SENTIMENT' in df.columns else []
        n_obs = df['N_OBS'].astype(int).tolist() if 'N_OBS' in df.columns else []
        colors = [C_REGIME.get(r, '#555') for r in regimes]
        ax.bar(regimes, n_obs, color=colors, edgecolor='white', width=0.5)
        for i, (r, n) in enumerate(zip(regimes, n_obs)):
            ax.text(i, n + max(n_obs) * 0.02, f'{n:,}', ha='center', fontsize=10)
        ax.set_ylabel('Number of Observations')
        ax.set_title('VIX Regime Distribution (Sample Sizes)')
        fig.tight_layout()
    return fig


# ── 30. FF5 coefficient comparison table ───────────────────────────────

def e1_30_coefficient_table(store):
    """Publication-quality coefficient table as a figure."""
    df = store.e1_ff5_coefficients
    if df.empty:
        return _empty_fig('E1-30: No FF5 coefficients')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, max(3, len(df) + 1)))
        ax.set_axis_off()
        # Format the table
        display_cols = [c for c in df.columns if c != 'RUN_TIMESTAMP']
        table_data = df[display_cols].copy()
        for c in table_data.columns:
            if c != 'REGIME':
                table_data[c] = table_data[c].apply(
                    lambda v: f'{float(v):.4f}' if pd.notna(v) else '')
        tbl = ax.table(cellText=table_data.values,
                       colLabels=table_data.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        # Header styling
        for j in range(len(display_cols)):
            tbl[0, j].set_facecolor('#2c3e50')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        ax.set_title('FF5 Regression Coefficients by Regime', pad=20, fontsize=13)
        fig.tight_layout()
    return fig


# ── 31. Matched: MKT_RF delta by regime ───────────────────────────────

def e1_31_matched_mkt_delta(store):
    """Market beta delta (treatment - control) by regime."""
    df = store.e1_matched_deltas
    col = _col(df, ['MKT_RF_DELTA', 'MKT_DELTA'])
    if df.empty or not col or 'REGIME' not in df.columns:
        return _empty_fig('E1-31: No matched MKT delta data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        regimes = sorted(df['REGIME'].unique())
        palette = {r: C_REGIME.get(r, '#555') for r in regimes}
        sns.boxplot(data=df, x='REGIME', y=col, order=regimes,
                    hue='REGIME', palette=palette, ax=ax, fliersize=3, legend=False)
        ax.axhline(0, color='red', lw=1, ls='--')
        ax.set_ylabel('Market Beta Delta (Treatment - Control)')
        ax.set_title('Matched Control: Market Sensitivity Differences')
        fig.tight_layout()
    return fig


# ── 32. Matched: all factor deltas violin ──────────────────────────────

def e1_32_matched_factor_deltas(store):
    """Violin plot of all factor deltas."""
    df = store.e1_matched_deltas
    if df.empty:
        return _empty_fig('E1-32: No matched deltas')
    delta_cols = [c for c in df.columns if c.endswith('_DELTA') and c != 'ALPHA_DELTA']
    if not delta_cols:
        return _empty_fig('E1-32: No delta columns')
    with plt.rc_context(STYLE):
        melted = df[delta_cols].melt(var_name='Factor', value_name='Delta')
        melted['Factor'] = melted['Factor'].str.replace('_DELTA', '')
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=melted, x='Factor', y='Delta', ax=ax,
                       hue='Factor', palette=C_SEQ[:len(delta_cols)], inner='box', cut=0, legend=False)
        ax.axhline(0, color='red', lw=1, ls='--')
        ax.set_ylabel('Delta (Treatment - Control)')
        ax.set_title('Factor Loading Differences Across Matched Pairs')
        fig.tight_layout()
    return fig


# ── 33. Essay 1 summary dashboard ─────────────────────────────────────

def e1_33_summary_dashboard(store):
    """Multi-panel summary of key Essay 1 findings."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # Panel A: Alpha by regime
        ax1 = fig.add_subplot(gs[0, 0])
        df = store.e1_ff5_coefficients
        if not df.empty and 'ALPHA' in df.columns:
            regimes = df['REGIME'].tolist()
            alphas = df['ALPHA'].astype(float).tolist()
            ax1.bar(regimes, alphas,
                    color=[C_REGIME.get(r, '#555') for r in regimes],
                    edgecolor='white', width=0.5)
            ax1.axhline(0, color='black', lw=0.5)
        ax1.set_title('A. Alpha by VIX Regime')
        ax1.set_ylabel('Alpha')

        # Panel B: FOMO
        ax2 = fig.add_subplot(gs[0, 1])
        fomo = store.e1_fomo
        if not fomo.empty and 'MEAN_FOMO_Z' in fomo.columns:
            regimes = fomo['REGIME'].tolist()
            ax2.bar(regimes, fomo['MEAN_FOMO_Z'].astype(float),
                    color=[C_REGIME.get(r, '#555') for r in regimes],
                    edgecolor='white', width=0.5)
            ax2.axhline(0, color='black', lw=0.5, ls='--')
        ax2.set_title('B. FOMO Z-Score by Regime')
        ax2.set_ylabel('Mean Z')

        # Panel C: Chow test
        ax3 = fig.add_subplot(gs[1, 0])
        chow = store.e1_chow_test
        if not chow.empty:
            r = chow.iloc[0]
            f_stat = _safe_float(r.get('f_stat', r.get('F_STAT')))
            p_val = _safe_float(r.get('p_value', r.get('P_VALUE')))
            ax3.text(0.5, 0.6, f'F = {f_stat:.2f}', ha='center', va='center',
                     fontsize=24, transform=ax3.transAxes)
            ax3.text(0.5, 0.35, f'p = {p_val:.4f} {_sig(p_val)}', ha='center',
                     va='center', fontsize=16, transform=ax3.transAxes,
                     color='#e74c3c' if p_val < 0.05 else '#555')
            ax3.set_axis_off()
        ax3.set_title('C. Chow Structural Break Test')

        # Panel D: Matched control
        ax4 = fig.add_subplot(gs[1, 1])
        mt = store.e1_matched_ttest
        if not mt.empty and 'P_VALUE' in mt.columns:
            n_sig = (mt['P_VALUE'].astype(float) < 0.05).sum()
            n_bh = mt['BH_SIGNIFICANT'].sum() if 'BH_SIGNIFICANT' in mt.columns else 0
            n_total = len(mt)
            ax4.text(0.5, 0.6, f'{n_sig}/{n_total} sig. at 5%',
                     ha='center', va='center', fontsize=18,
                     transform=ax4.transAxes)
            ax4.text(0.5, 0.35, f'{n_bh}/{n_total} BH-corrected',
                     ha='center', va='center', fontsize=14,
                     transform=ax4.transAxes, color='#555')
            ax4.set_axis_off()
        ax4.set_title('D. Matched Control Significance')

        fig.suptitle('Essay 1 — Volatility Regimes & FF5 Summary', fontsize=15, y=1.02)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# CHART REGISTRY
# ═══════════════════════════════════════════════════════════════════════

ESSAY1_CHARTS = [
    ('e1_01_factor_premia_by_regime', e1_01_factor_premia_by_regime, 'FF5 factor premia by regime'),
    ('e1_02_alpha_by_regime', e1_02_alpha_by_regime, 'FF5 alpha by regime'),
    ('e1_03_beta_heatmap', e1_03_beta_heatmap, 'FF5 beta heatmap'),
    ('e1_04_tstat_heatmap', e1_04_tstat_heatmap, 'FF5 t-statistic heatmap'),
    ('e1_05_chow_test', e1_05_chow_test, 'Chow structural break test'),
    ('e1_06_rsquared_by_regime', e1_06_rsquared_by_regime, 'R-squared by regime'),
    ('e1_07_factor_premia_dot', e1_07_factor_premia_dot, 'Factor premia dot plot'),
    ('e1_08_cw_alpha_boxplot', e1_08_cw_alpha_boxplot, 'CW stock alpha boxplot'),
    ('e1_09_cw_alpha_hist', e1_09_cw_alpha_hist, 'CW stock alpha distribution'),
    ('e1_10_cw_rsquared', e1_10_cw_rsquared, 'CW R-squared by regime'),
    ('e1_11_cw_mkt_beta', e1_11_cw_mkt_beta, 'CW market beta scatter'),
    ('e1_12_cw_hml_beta', e1_12_cw_hml_beta, 'CW HML beta violin'),
    ('e1_13_cw_alpha_significance', e1_13_cw_alpha_significance, 'Alpha significance counts'),
    ('e1_14_cw_mean_betas_by_regime', e1_14_cw_mean_betas_by_regime, 'Mean factor betas by regime'),
    ('e1_15_fomo_by_regime', e1_15_fomo_by_regime, 'FOMO z-scores by regime'),
    ('e1_16_euphoria_panic', e1_16_euphoria_panic, 'Euphoria and panic rates'),
    ('e1_17_sentiment_by_regime', e1_17_sentiment_by_regime, 'Sentiment mean by regime'),
    ('e1_18_sentiment_timeseries', e1_18_sentiment_timeseries, 'Daily sentiment time series'),
    ('e1_19_matched_ttest_forest', e1_19_matched_ttest_forest, 'Matched t-test forest plot'),
    ('e1_20_matched_sign_consistency', e1_20_matched_sign_consistency, 'Sign consistency'),
    ('e1_21_matched_amplification', e1_21_matched_amplification, 'Regime amplification'),
    ('e1_22_matched_delta_dist', e1_22_matched_delta_dist, 'Delta alpha distribution'),
    ('e1_23_matched_delta_heatmap', e1_23_matched_delta_heatmap, 'Delta heatmap'),
    ('e1_24_matched_coverage', e1_24_matched_coverage, 'Matched coverage heatmap'),
    ('e1_25_cw_stock_count', e1_25_cw_stock_count, 'CW stock count by regime'),
    ('e1_26_alpha_vs_rsq', e1_26_alpha_vs_rsq, 'Alpha vs R-squared scatter'),
    ('e1_27_matched_n_firms', e1_27_matched_n_firms, 'Matched sample sizes'),
    ('e1_28_matched_pvalue_dist', e1_28_matched_pvalue_dist, 'P-value distribution'),
    ('e1_29_vix_distribution', e1_29_vix_distribution, 'VIX regime distribution'),
    ('e1_30_coefficient_table', e1_30_coefficient_table, 'Coefficient table figure'),
    ('e1_31_matched_mkt_delta', e1_31_matched_mkt_delta, 'Market beta delta boxplot'),
    ('e1_32_matched_factor_deltas', e1_32_matched_factor_deltas, 'Factor delta violins'),
    ('e1_33_summary_dashboard', e1_33_summary_dashboard, 'Essay 1 summary dashboard'),
]

# ═══════════════════════════════════════════════════════════════════════
# ██  ESSAY 2 — Culture War Event Study & DiD (33)
# ═══════════════════════════════════════════════════════════════════════

# ── 1. CAR distribution ────────────────────────────────────────────────

def e2_01_car_distribution(store):
    """Histogram of cumulative abnormal returns (pre, post, full)."""
    df = store.e2_car_panel
    if df.empty:
        return _empty_fig('E2-01: No CAR panel data')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, col, title in zip(axes, ['CAR_PRE', 'CAR_POST', 'CAR_FULL'],
                                   ['Pre-Event', 'Post-Event', 'Full Window']):
            if col in df.columns:
                vals = df[col].astype(float).dropna()
                ax.hist(vals, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
                ax.axvline(0, color='red', lw=1, ls='--')
                ax.axvline(vals.mean(), color='#e74c3c', lw=1.5,
                           label=f'Mean={vals.mean():.3f}')
                ax.set_xlabel('CAR')
                ax.set_title(title)
                ax.legend(fontsize=8)
        axes[0].set_ylabel('Count')
        fig.suptitle('Distribution of Cumulative Abnormal Returns', fontsize=13)
        fig.tight_layout()
    return fig


# ── 2. CAR by political leaning ────────────────────────────────────────

def e2_02_car_by_leaning(store):
    """Mean CAR by political leaning with error bars."""
    df = store.e2_car_panel
    if df.empty or 'LEAN' not in df.columns or 'CAR_POST' not in df.columns:
        return _empty_fig('E2-02: No CAR/leaning data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        grp = df.groupby('LEAN')['CAR_POST'].agg(['mean', 'std', 'count']).reset_index()
        grp['se'] = grp['std'] / np.sqrt(grp['count'])
        grp['ci'] = 1.96 * grp['se']
        order = ['Conservative', 'Liberal', 'Mixed']
        grp['_o'] = grp['LEAN'].map({g: i for i, g in enumerate(order)})
        grp = grp.sort_values('_o').reset_index(drop=True)
        colors = [C_LEAN.get(l, '#555') for l in grp['LEAN']]
        bars = ax.bar(grp['LEAN'], grp['mean'], yerr=grp['ci'], color=colors,
                       edgecolor='white', width=0.5, capsize=5)
        for bar, r in zip(bars, grp.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2, r.mean,
                    f'{r.mean:.3f}', ha='center',
                    va='bottom' if r.mean >= 0 else 'top', fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean CAR (Post-Event)')
        ax.set_title('Culture War Event Impact by Political Leaning')
        ax.set_xticklabels([f'{l}\n(N={n:.0f})' for l, n in zip(grp['LEAN'], grp['count'])])
        fig.tight_layout()
    return fig


# ── 3. CAR pre vs post scatter ─────────────────────────────────────────

def e2_03_car_pre_vs_post(store):
    """Scatter of pre-event vs post-event CARs."""
    df = store.e2_car_panel
    if df.empty or 'CAR_PRE' not in df.columns or 'CAR_POST' not in df.columns:
        return _empty_fig('E2-03: No pre/post CAR data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 7))
        treat = df[df['IS_TREATMENT'] == True] if 'IS_TREATMENT' in df.columns else df
        ctrl = df[df['IS_TREATMENT'] == False] if 'IS_TREATMENT' in df.columns else pd.DataFrame()
        if not treat.empty:
            ax.scatter(treat['CAR_PRE'].astype(float), treat['CAR_POST'].astype(float),
                       s=30, alpha=0.6, color=C_TC['Treatment'], label='Treatment')
        if not ctrl.empty:
            ax.scatter(ctrl['CAR_PRE'].astype(float), ctrl['CAR_POST'].astype(float),
                       s=30, alpha=0.6, color=C_TC['Control'], label='Control')
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color='black', lw=0.4)
        ax.axvline(0, color='black', lw=0.4)
        ax.plot([-lim, lim], [-lim, lim], color='#bdc3c7', lw=1, ls='--', label='45\u00b0')
        ax.set_xlabel('CAR Pre-Event')
        ax.set_ylabel('CAR Post-Event')
        ax.set_title('Pre-Event vs Post-Event Abnormal Returns')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 4. CAR by regime ───────────────────────────────────────────────────

def e2_04_car_by_regime(store):
    """CAR by VIX regime (boxplot)."""
    df = store.e2_car_panel
    if df.empty or 'REGIME' not in df.columns or 'CAR_POST' not in df.columns:
        return _empty_fig('E2-04: No CAR/regime data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        regimes = sorted(df['REGIME'].dropna().unique())
        palette = {r: C_REGIME.get(r, '#555') for r in regimes}
        sns.boxplot(data=df, x='REGIME', y='CAR_POST', order=regimes,
                    hue='REGIME', palette=palette, ax=ax, fliersize=3, legend=False)
        ax.axhline(0, color='red', lw=1, ls='--')
        ax.set_title('Post-Event CAR by VIX Regime')
        ax.set_ylabel('CAR Post-Event')
        fig.tight_layout()
    return fig


# ── 5. CAR treatment vs control ────────────────────────────────────────

def e2_05_car_treat_vs_ctrl(store):
    """Treatment vs control CAR comparison."""
    df = store.e2_car_panel
    if df.empty or 'IS_TREATMENT' not in df.columns or 'CAR_POST' not in df.columns:
        return _empty_fig('E2-05: No treatment/control data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        df['Group'] = df['IS_TREATMENT'].map({True: 'Treatment', False: 'Control', 1: 'Treatment', 0: 'Control'})
        grp = df.groupby('Group')['CAR_POST'].agg(['mean', 'std', 'count']).reset_index()
        grp['ci'] = 1.96 * grp['std'] / np.sqrt(grp['count'])
        colors = [C_TC.get(g, '#555') for g in grp['Group']]
        bars = ax.bar(grp['Group'], grp['mean'], yerr=grp['ci'], color=colors,
                       edgecolor='white', width=0.4, capsize=5)
        for bar, r in zip(bars, grp.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2, r.mean,
                    f'{r.mean:.4f}', ha='center',
                    va='bottom' if r.mean >= 0 else 'top', fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Post-Event CAR')
        ax.set_title('Treatment vs Control: Post-Event Abnormal Returns')
        fig.tight_layout()
    return fig


# ── 6. DiD coefficient forest plot ─────────────────────────────────────

def e2_06_did_coefficients(store):
    """Forest plot of DiD regression coefficients."""
    df = store.e2_did_coeff
    if df.empty or 'VARIABLE' not in df.columns:
        return _empty_fig('E2-06: No DiD coefficients')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.6)))
        y = np.arange(len(df))
        coefs = df['COEFFICIENT'].astype(float)
        se = df['STD_ERROR'].astype(float) if 'STD_ERROR' in df.columns else pd.Series(0.0, index=df.index)
        ci_lo = coefs - 1.96 * se
        ci_hi = coefs + 1.96 * se
        colors = ['#e74c3c' if _safe_float(r.get('P_VALUE')) < 0.05 else '#bdc3c7'
                  for _, r in df.iterrows()]
        ax.barh(y, coefs, xerr=1.96 * se, color=colors, edgecolor='white',
                height=0.5, capsize=3)
        for i, r in df.iterrows():
            star = _sig(_safe_float(r.get('P_VALUE')))
            ax.text(coefs.iloc[i], y[i], f' {star}', va='center', fontsize=11)
        ax.set_yticks(y)
        labels = df['VARIABLE'].tolist()
        if 'SPECIFICATION' in df.columns:
            labels = [f'{v} ({s})' for v, s in zip(df['VARIABLE'], df['SPECIFICATION'])]
        ax.set_yticklabels(labels)
        ax.axvline(0, color='black', lw=0.5)
        ax.set_xlabel('Coefficient')
        ax.set_title('Difference-in-Differences Regression Coefficients')
        fig.tight_layout()
    return fig


# ── 7. Parallel trends ─────────────────────────────────────────────────

def e2_07_parallel_trends(store):
    """Daily pre-event treatment-control coefficients."""
    df = store.e2_parallel
    if df.empty or 'DAY' not in df.columns:
        return _empty_fig('E2-07: No parallel trends data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('DAY')
        days = df['DAY'].astype(int)
        coefs = df['COEFFICIENT'].astype(float)
        se = df['STD_ERROR'].astype(float) if 'STD_ERROR' in df.columns else pd.Series(0.0, index=df.index)
        ax.fill_between(days, coefs - 1.96 * se, coefs + 1.96 * se,
                         alpha=0.2, color='#3498db')
        ax.plot(days, coefs, 'o-', color='#2980b9', markersize=5, lw=1.5)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5, label='Event Day')
        passes = df['PASSES'].iloc[0] if 'PASSES' in df.columns else None
        f_stat = df['JOINT_F_STAT'].iloc[0] if 'JOINT_F_STAT' in df.columns else None
        f_p = df['JOINT_P_VALUE'].iloc[0] if 'JOINT_P_VALUE' in df.columns else None
        subtitle = ''
        if f_stat is not None:
            subtitle = f'Joint F = {float(f_stat):.2f}, p = {float(f_p):.4f}'
            if passes is not None:
                subtitle += f' — Passes: {"YES" if passes else "NO"}'
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Treatment × Day Coefficient')
        ax.set_title(f'Parallel Trends Test\n{subtitle}')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 8. Peer parallel trends ────────────────────────────────────────────

def e2_08_peer_parallel_trends(store):
    """Contagion peer parallel trends coefficients."""
    df = store.e2_peer_parallel
    if df.empty or 'DAY' not in df.columns:
        return _empty_fig('E2-08: No peer parallel trends')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('DAY')
        days = df['DAY'].astype(int)
        coefs = df['COEFFICIENT'].astype(float)
        se = df['STD_ERROR'].astype(float) if 'STD_ERROR' in df.columns else pd.Series(0.0, index=df.index)
        ax.fill_between(days, coefs - 1.96 * se, coefs + 1.96 * se,
                         alpha=0.2, color='#e67e22')
        ax.plot(days, coefs, 'o-', color='#e67e22', markersize=5, lw=1.5)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_xlabel('Days Relative to Event')
        ax.set_ylabel('Peer × Day Coefficient')
        ax.set_title('Peer Contagion: Parallel Trends Test')
        fig.tight_layout()
    return fig


# ── 9. News sentiment distribution ─────────────────────────────────────

def e2_09_news_sentiment_dist(store):
    """Distribution of FinBERT news sentiment labels."""
    df = store.e2_news_sent
    label_col = _col(df, ['FINBERT_LABEL', 'finbert_label'])
    if df.empty or not label_col:
        return _empty_fig('E2-09: No news sentiment data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = df[label_col].value_counts()
        colors = {'positive': '#27ae60', 'negative': '#e74c3c', 'neutral': '#bdc3c7'}
        ax.bar(counts.index, counts.values,
               color=[colors.get(str(l).lower(), '#555') for l in counts.index],
               edgecolor='white', width=0.5)
        for i, (label, cnt) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, cnt, f'{cnt:,}\n({cnt/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Count')
        ax.set_title(f'News Sentiment Distribution (N={len(df):,})')
        fig.tight_layout()
    return fig


# ── 10. News sentiment pre vs post ─────────────────────────────────────

def e2_10_news_pre_vs_post(store):
    """Pre-event vs post-event mean news sentiment by firm."""
    df = store.e2_event_nlp
    if df.empty or 'NEWS_SENT_PRE' not in df.columns:
        return _empty_fig('E2-10: No event NLP data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 7))
        pre = df['NEWS_SENT_PRE'].astype(float)
        post = df['NEWS_SENT_POST'].astype(float)
        valid = pre.notna() & post.notna()
        ax.scatter(pre[valid], post[valid], s=30, alpha=0.6, color='#3498db')
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.plot([-lim, lim], [-lim, lim], color='#bdc3c7', lw=1, ls='--')
        ax.axhline(0, color='black', lw=0.4)
        ax.axvline(0, color='black', lw=0.4)
        ax.set_xlabel('Pre-Event Sentiment')
        ax.set_ylabel('Post-Event Sentiment')
        ax.set_title('News Sentiment Shift Around Events')
        fig.tight_layout()
    return fig


# ── 11. News sentiment change histogram ────────────────────────────────

def e2_11_sentiment_change(store):
    """Distribution of pre-to-post news sentiment change."""
    df = store.e2_event_nlp
    if df.empty or 'NEWS_SENT_CHANGE' not in df.columns:
        return _empty_fig('E2-11: No sentiment change data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        vals = df['NEWS_SENT_CHANGE'].astype(float).dropna()
        ax.hist(vals, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', lw=1, ls='--')
        ax.axvline(vals.mean(), color='#e74c3c', lw=1.5,
                   label=f'Mean = {vals.mean():.3f}')
        ax.set_xlabel('Sentiment Change (Post - Pre)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Sentiment Change Around Events')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 12. Filing sentiment by section ────────────────────────────────────

def e2_12_filing_sentiment_sections(store):
    """Filing sentiment by section (MDA, Risk Factors, etc.)."""
    df = store.e2_filing_sent
    _sent_col = 'SENT_CONF_WEIGHTED_MEAN' if 'SENT_CONF_WEIGHTED_MEAN' in df.columns else 'SENT_MEAN'
    if df.empty or 'SECTION' not in df.columns or _sent_col not in df.columns:
        return _empty_fig('E2-12: No filing sentiment data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        grp = df.groupby('SECTION')[_sent_col].agg(['mean', 'std', 'count']).reset_index()
        grp['ci'] = 1.96 * grp['std'] / np.sqrt(grp['count'])
        ax.bar(grp['SECTION'], grp['mean'], yerr=grp['ci'],
               color=C_SEQ[:len(grp)], edgecolor='white', capsize=5)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean Sentiment')
        ax.set_title('SEC Filing Sentiment by Section')
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout()
    return fig


# ── 13. Filing positive/negative pct ───────────────────────────────────

def e2_13_filing_pct_breakdown(store):
    """Positive vs negative sentiment % in filings by section."""
    df = store.e2_filing_sent
    if df.empty or 'SECTION' not in df.columns:
        return _empty_fig('E2-13: No filing sentiment data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        grp = df.groupby('SECTION')[['PCT_POSITIVE', 'PCT_NEGATIVE', 'PCT_NEUTRAL']].mean()
        grp.plot(kind='bar', stacked=True, ax=ax,
                 color=['#27ae60', '#e74c3c', '#bdc3c7'], edgecolor='white')
        ax.set_ylabel('Mean Fraction')
        ax.set_title('Filing Sentiment Composition by Section')
        ax.legend(['Positive', 'Negative', 'Neutral'])
        plt.xticks(rotation=30, ha='right')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
    return fig


# ── 14. Political alignment score distribution ─────────────────────────

def e2_14_alignment_distribution(store):
    """Histogram of computed political alignment scores."""
    df = store.e2_alignment
    if df.empty or 'ALIGNMENT_SCORE' not in df.columns:
        return _empty_fig('E2-14: No alignment data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = df['ALIGNMENT_SCORE'].astype(float).dropna()
        ax.hist(vals, bins=30, color='#9b59b6', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='black', lw=0.5, ls='--')
        ax.axvline(vals.mean(), color='#e74c3c', lw=1.5,
                   label=f'Mean = {vals.mean():.3f}')
        ax.set_xlabel('Alignment Score (- = Liberal, + = Conservative)')
        ax.set_ylabel('Count')
        ax.set_title('Political Alignment Score Distribution')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 15. Distinctive phrases by party ───────────────────────────────────

def e2_15_distinctive_phrases(store):
    """Top distinctive phrases for each party."""
    df = store.e2_phrases
    if df.empty or 'PHRASE' not in df.columns:
        return _empty_fig('E2-15: No distinctive phrases')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, party, color in zip(axes, ['Republican', 'Democratic'],
                                     ['#c0392b', '#2980b9']):
            if party == 'Republican':
                sub = df[df['PARTY'] == party].nlargest(15, 'TFIDF_DIFF')
            else:
                sub = df[df['PARTY'] == party].nsmallest(15, 'TFIDF_DIFF')
            if sub.empty:
                ax.text(0.5, 0.5, f'No {party} phrases', transform=ax.transAxes,
                        ha='center', va='center')
                continue
            y = np.arange(len(sub))
            ax.barh(y, sub['TFIDF_DIFF'].astype(float).abs(), color=color,
                    edgecolor='white', height=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(sub['PHRASE'].tolist(), fontsize=8)
            ax.set_xlabel('TF-IDF Difference')
            ax.set_title(f'{party} Distinctive Phrases')
            ax.invert_yaxis()
        fig.suptitle('Party-Distinctive Language in Platform Texts', fontsize=13)
        fig.tight_layout()
    return fig


# ── 16. Alignment validation ───────────────────────────────────────────

def e2_16_alignment_validation(store):
    """Computed alignment vs hand-coded leaning."""
    df = store.e2_validation
    if df.empty or 'ALIGNMENT_SCORE' not in df.columns:
        return _empty_fig('E2-16: No validation data')
    # Try to find a leaning column
    lean_col = _col(df, ['ESTIMATED_POLITICAL_LEANING', 'COMPUTED_LEANING', 'LEAN'])
    if not lean_col:
        return _empty_fig('E2-16: No leaning column')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for lean in df[lean_col].unique():
            sub = df[df[lean_col] == lean]
            ax.scatter(sub.index, sub['ALIGNMENT_SCORE'].astype(float),
                       s=40, alpha=0.7, color=C_LEAN.get(lean, '#555'), label=lean)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_xlabel('Company Index')
        ax.set_ylabel('Alignment Score')
        ax.set_title('Alignment Score by Hand-Coded Political Leaning')
        ax.legend(title='Leaning')
        fig.tight_layout()
    return fig


# ── 17. Event NLP heatmap ──────────────────────────────────────────────

def e2_17_event_nlp_heatmap(store):
    """Heatmap of NLP variables across events."""
    df = store.e2_event_nlp
    if df.empty:
        return _empty_fig('E2-17: No event NLP data')
    idx_col = _col(df, ['TICKER', 'FIRM', 'EVENT_ID'])
    if idx_col is None:
        return _empty_fig('E2-17: No identifier column')
    nlp_cols = [c for c in ['NEWS_SENT_PRE', 'NEWS_SENT_POST', 'NEWS_SENT_CHANGE',
                             'FILING_MDA_TONE', 'FILING_RISK_TONE'] if c in df.columns]
    if not nlp_cols:
        return _empty_fig('E2-17: No NLP columns')
    with plt.rc_context(STYLE):
        sub = df.set_index(idx_col)[nlp_cols].astype(float).dropna(how='all')
        n = len(sub)
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.15)))
        sns.heatmap(sub, cmap='RdBu_r', center=0, ax=ax, linewidths=0.3,
                    yticklabels=n < 50, cbar_kws={'label': 'Score'})
        ax.set_title('NLP Sentiment Variables by Firm')
        fig.tight_layout()
    return fig


# ── 18. Multi-window CAR by horizon ────────────────────────────────────

def e2_18_multiwindow_by_horizon(store):
    """Treatment CARs across multiple event windows."""
    df = store.e2_mw_summary
    if df.empty or 'WINDOW' not in df.columns:
        return _empty_fig('E2-18: No multi-window summary')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        means = df['MEAN_CAR_TREAT'].astype(float) if 'MEAN_CAR_TREAT' in df.columns else pd.Series(dtype=float)
        colors = ['#e74c3c' if _safe_float(r.get('P_VALUE_VS_ZERO')) < 0.05 else '#bdc3c7'
                  for _, r in df.iterrows()]
        bars = ax.bar(windows, means, color=colors, edgecolor='white', width=0.6)
        for bar, r in zip(bars, df.itertuples()):
            p = _safe_float(getattr(r, 'P_VALUE_VS_ZERO', np.nan))
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), _sig(p), ha='center',
                    va='bottom' if bar.get_height() >= 0 else 'top', fontsize=11)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean CAR (Treatment)')
        ax.set_title('Cumulative Abnormal Returns Across Event Windows')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
    return fig


# ── 19. Multi-window by leaning ────────────────────────────────────────

def e2_19_multiwindow_by_lean(store):
    """Multi-window CARs split by political leaning."""
    df = store.e2_mw_lean
    if df.empty or 'LEAN' not in df.columns:
        return _empty_fig('E2-19: No multi-window by lean')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        windows = sorted(df['WINDOW'].unique(), key=lambda w: df[df['WINDOW'] == w]['WINDOW_DAYS'].iloc[0]
                         if 'WINDOW_DAYS' in df.columns else w)
        leans = [l for l in ['Conservative', 'Liberal', 'Mixed'] if l in df['LEAN'].values]
        x = np.arange(len(windows))
        width = 0.8 / max(len(leans), 1)
        for i, lean in enumerate(leans):
            sub = df[df['LEAN'] == lean].set_index('WINDOW').reindex(windows)
            vals = sub['MEAN_CAR'].astype(float).fillna(0)
            ax.bar(x + i * width, vals, width, label=lean,
                   color=C_LEAN.get(lean, C_SEQ[i]), edgecolor='white')
        ax.set_xticks(x + width * (len(leans) - 1) / 2)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean CAR')
        ax.set_title('Event Window CARs by Political Leaning')
        ax.legend(title='Leaning')
        fig.tight_layout()
    return fig


# ── 20. Multi-window treatment vs control ──────────────────────────────

def e2_20_multiwindow_tc(store):
    """Treatment-control difference across windows with significance."""
    df = store.e2_mw_tc
    if df.empty or 'WINDOW' not in df.columns:
        return _empty_fig('E2-20: No multi-window T vs C')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        diffs = df['DIFF_TREAT_CTRL'].astype(float)
        colors = ['#e74c3c' if _safe_float(r.get('P_VALUE')) < 0.05 else '#bdc3c7'
                  for _, r in df.iterrows()]
        bars = ax.bar(windows, diffs, color=colors, edgecolor='white', width=0.6)
        for bar, r in zip(bars, df.itertuples()):
            p = _safe_float(getattr(r, 'P_VALUE', np.nan))
            d = _safe_float(getattr(r, 'COHENS_D', np.nan))
            label = f'{_sig(p)}\nd={d:.2f}' if not np.isnan(d) else _sig(p)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=8)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Treatment - Control CAR Difference')
        ax.set_title('DiD: Treatment vs Control Across Windows')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
    return fig


# ── 21. Contagion summary ──────────────────────────────────────────────

def e2_21_contagion_summary(store):
    """Peer contagion CARs across windows."""
    df = store.e2_cont_summary
    if df.empty or 'WINDOW' not in df.columns:
        return _empty_fig('E2-21: No contagion summary')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        means = df['MEAN_PEER_CAR'].astype(float)
        colors = ['#e67e22' if _safe_float(r.get('P_VALUE_VS_ZERO')) < 0.05 else '#bdc3c7'
                  for _, r in df.iterrows()]
        bars = ax.bar(windows, means, color=colors, edgecolor='white', width=0.6)
        for bar, r in zip(bars, df.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    _sig(_safe_float(getattr(r, 'P_VALUE_VS_ZERO', np.nan))),
                    ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=11)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Mean Peer CAR')
        ax.set_title('Contagion: Peer Firm Abnormal Returns')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
    return fig


# ── 22. Contagion by leaning ───────────────────────────────────────────

def e2_22_contagion_by_lean(store):
    """Peer contagion CARs by event political leaning."""
    df = store.e2_cont_lean
    if df.empty or 'EVENT_LEAN' not in df.columns:
        return _empty_fig('E2-22: No contagion by lean')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        windows = sorted(df['WINDOW'].unique(), key=lambda w: df[df['WINDOW'] == w]['WINDOW_DAYS'].iloc[0]
                         if 'WINDOW_DAYS' in df.columns else w)
        leans = sorted(df['EVENT_LEAN'].unique())
        x = np.arange(len(windows))
        width = 0.8 / max(len(leans), 1)
        for i, lean in enumerate(leans):
            sub = df[df['EVENT_LEAN'] == lean].set_index('WINDOW').reindex(windows)
            vals = sub['MEAN_PEER_CAR'].astype(float).fillna(0)
            ax.bar(x + i * width, vals, width, label=lean,
                   color=C_LEAN.get(lean, C_SEQ[i]), edgecolor='white')
        ax.set_xticks(x + width * (len(leans) - 1) / 2)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Peer CAR')
        ax.set_title('Contagion by Event Political Leaning')
        ax.legend(title='Event Lean')
        fig.tight_layout()
    return fig


# ── 23. Contagion by facing ────────────────────────────────────────────

def e2_23_contagion_by_facing(store):
    """Peer contagion: B2C vs B2B facing."""
    df = store.e2_cont_facing
    if df.empty or 'EVENT_FACING' not in df.columns:
        return _empty_fig('E2-23: No contagion by facing')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        windows = sorted(df['WINDOW'].unique(), key=lambda w: df[df['WINDOW'] == w]['WINDOW_DAYS'].iloc[0]
                         if 'WINDOW_DAYS' in df.columns else w)
        facings = sorted(df['EVENT_FACING'].unique())
        x = np.arange(len(windows))
        width = 0.8 / max(len(facings), 1)
        for i, facing in enumerate(facings):
            sub = df[df['EVENT_FACING'] == facing].set_index('WINDOW').reindex(windows)
            vals = sub['MEAN_PEER_CAR'].astype(float).fillna(0)
            ax.bar(x + i * width, vals, width, label=facing,
                   color=C_SEQ[i], edgecolor='white')
        ax.set_xticks(x + width * (len(facings) - 1) / 2)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Peer CAR')
        ax.set_title('Contagion by Market Facing (B2C vs B2B)')
        ax.legend(title='Facing')
        fig.tight_layout()
    return fig


# ── 24. Contagion peer vs non-peer ─────────────────────────────────────

def e2_24_contagion_peer_vs_nonpeer(store):
    """Peer vs non-peer spillover comparison."""
    df = store.e2_cont_peer
    if df.empty or 'MEAN_PEER_CAR' not in df.columns:
        return _empty_fig('E2-24: No peer vs non-peer data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        x = np.arange(len(windows))
        w = 0.35
        ax.bar(x - w / 2, df['MEAN_PEER_CAR'].astype(float), w,
               label='Peer', color='#e67e22', edgecolor='white')
        ax.bar(x + w / 2, df['MEAN_NONPEER_CAR'].astype(float), w,
               label='Non-Peer', color='#95a5a6', edgecolor='white')
        # Significance stars
        for i, r in df.iterrows():
            p = _safe_float(r.get('P_VALUE'))
            star = _sig(p)
            if star:
                ymax = max(_safe_float(r.get('MEAN_PEER_CAR', 0)),
                           _safe_float(r.get('MEAN_NONPEER_CAR', 0)))
                ax.text(i, ymax + abs(ymax) * 0.05, star, ha='center', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean CAR')
        ax.set_title('Contagion: Peer vs Non-Peer Spillover')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 25. Contagion consumer vs B2B ──────────────────────────────────────

def e2_25_contagion_consumer_b2b(store):
    """Consumer vs B2B contagion effect."""
    df = store.e2_cont_cb
    if df.empty or 'MEAN_CONSUMER' not in df.columns:
        return _empty_fig('E2-25: No consumer vs B2B data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        x = np.arange(len(windows))
        w = 0.35
        ax.bar(x - w / 2, df['MEAN_CONSUMER'].astype(float), w,
               label='Consumer-Facing', color='#3498db', edgecolor='white')
        ax.bar(x + w / 2, df['MEAN_B2B'].astype(float), w,
               label='B2B', color='#95a5a6', edgecolor='white')
        for i, r in df.iterrows():
            star = _sig(_safe_float(r.get('P_VALUE')))
            if star:
                ymax = max(_safe_float(r.get('MEAN_CONSUMER', 0)),
                           _safe_float(r.get('MEAN_B2B', 0)))
                ax.text(i, ymax + abs(ymax) * 0.05, star, ha='center', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean CAR')
        ax.set_title('Contagion: Consumer-Facing vs B2B')
        ax.legend()
        fig.tight_layout()
    return fig


# ── 26. Contagion lean pairwise ────────────────────────────────────────

def e2_26_contagion_lean_pairwise(store):
    """Pairwise leaning contagion differences."""
    df = store.e2_cont_lp
    if df.empty or 'COMPARISON' not in df.columns:
        return _empty_fig('E2-26: No lean pairwise data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        comparisons = df['COMPARISON'].unique()
        windows = sorted(df['WINDOW'].unique())
        x = np.arange(len(windows))
        width = 0.8 / max(len(comparisons), 1)
        for i, comp in enumerate(comparisons):
            sub = df[df['COMPARISON'] == comp].set_index('WINDOW').reindex(windows)
            vals = sub['DIFF'].astype(float).fillna(0)
            ax.bar(x + i * width, vals, width, label=comp,
                   color=C_SEQ[i], edgecolor='white')
        ax.set_xticks(x + width * (len(comparisons) - 1) / 2)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Difference')
        ax.set_title('Contagion: Pairwise Leaning Comparisons')
        ax.legend(title='Comparison', fontsize=8)
        fig.tight_layout()
    return fig


# ── 27. Contagion tight diff ───────────────────────────────────────────

def e2_27_contagion_tight_diff(store):
    """Tight DiD: peer vs matched non-peer."""
    df = store.e2_cont_tight
    if df.empty or 'DIFF' not in df.columns:
        return _empty_fig('E2-27: No tight diff data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        df = df.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in df.columns else df
        windows = df['WINDOW'].tolist()
        diffs = df['DIFF'].astype(float)
        colors = ['#e67e22' if _safe_float(r.get('P_VALUE')) < 0.05 else '#bdc3c7'
                  for _, r in df.iterrows()]
        bars = ax.bar(windows, diffs, color=colors, edgecolor='white', width=0.6)
        for bar, r in zip(bars, df.itertuples()):
            star = _sig(_safe_float(getattr(r, 'P_VALUE', np.nan)))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    star, ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                    fontsize=11)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Peer - Non-Peer CAR Difference')
        ax.set_title('Tight Contagion DiD: Peer vs Matched Non-Peer')
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
    return fig


# ── 28. MDA vs Risk factor tone ────────────────────────────────────────

def e2_28_mda_vs_risk_tone(store):
    """Scatter of MDA tone vs Risk Factor tone per firm."""
    df = store.e2_event_nlp
    if df.empty or 'FILING_MDA_TONE' not in df.columns or 'FILING_RISK_TONE' not in df.columns:
        return _empty_fig('E2-28: No MDA/Risk tone data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 7))
        valid = df[['FILING_MDA_TONE', 'FILING_RISK_TONE']].dropna()
        ax.scatter(valid['FILING_MDA_TONE'].astype(float),
                   valid['FILING_RISK_TONE'].astype(float),
                   s=30, alpha=0.6, color='#9b59b6')
        r, p = sp_stats.pearsonr(valid['FILING_MDA_TONE'].astype(float),
                                  valid['FILING_RISK_TONE'].astype(float))
        ax.set_xlabel('MDA Tone')
        ax.set_ylabel('Risk Factor Tone')
        ax.set_title(f'Filing Section Tones (r = {r:.3f}, p = {p:.4f})')
        # Fit line
        z = np.polyfit(valid['FILING_MDA_TONE'].astype(float),
                       valid['FILING_RISK_TONE'].astype(float), 1)
        xline = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.plot(xline, z[0] * xline + z[1], color='#e74c3c', lw=1, ls='--')
        fig.tight_layout()
    return fig


# ── 29. Alignment component comparison ─────────────────────────────────

def e2_29_alignment_components(store):
    """Box plot of alignment sub-scores."""
    df = store.e2_alignment
    if df.empty:
        return _empty_fig('E2-29: No alignment data')
    comp_cols = [c for c in ['DISTINCTIVE_ALIGN', 'STANCE_ALIGN', 'COSINE_ALIGN']
                 if c in df.columns]
    if not comp_cols:
        return _empty_fig('E2-29: No component columns')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        melted = df[comp_cols].melt(var_name='Component', value_name='Score')
        melted['Component'] = melted['Component'].str.replace('_ALIGN', '')
        sns.boxplot(data=melted, x='Component', y='Score', ax=ax,
                    hue='Component', palette=C_SEQ[:len(comp_cols)], legend=False)
        ax.axhline(0, color='black', lw=0.5, ls='--')
        ax.set_ylabel('Alignment Sub-Score')
        ax.set_title('Political Alignment Component Scores')
        fig.tight_layout()
    return fig


# ── 30. CAR by leaning and regime ──────────────────────────────────────

def e2_30_car_lean_regime(store):
    """Heatmap of mean CARs by leaning x regime."""
    df = store.e2_car_panel
    if df.empty or 'LEAN' not in df.columns or 'REGIME' not in df.columns:
        return _empty_fig('E2-30: No CAR/lean/regime data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        pivot = df.pivot_table(values='CAR_POST', index='LEAN', columns='REGIME',
                                aggfunc='mean')
        if pivot.empty or pivot.size == 0:
            plt.close(fig)
            return _empty_fig('E2-30: Pivot table empty (no CAR×lean×regime combos)')
        sns.heatmap(pivot.astype(float), annot=True, fmt='.4f', cmap='RdBu_r',
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title('Mean Post-Event CAR by Leaning and Regime')
        fig.tight_layout()
    return fig


# ── 31. Event count by leaning ─────────────────────────────────────────

def e2_31_event_count_by_lean(store):
    """Event sample sizes by political leaning."""
    df = store.e2_car_panel
    if df.empty or 'LEAN' not in df.columns:
        return _empty_fig('E2-31: No leaning data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        treat = df[df['IS_TREATMENT'] == True] if 'IS_TREATMENT' in df.columns else df
        counts = treat['LEAN'].value_counts()
        colors = [C_LEAN.get(l, '#555') for l in counts.index]
        ax.bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
        for i, (l, v) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=11)
        ax.set_ylabel('Number of Events')
        ax.set_title('Culture War Event Distribution by Political Leaning')
        fig.tight_layout()
    return fig


# ── 32. Contagion mechanism ────────────────────────────────────────────

def e2_32_contagion_mechanism(store):
    """Contagion mechanism: peer CARs by event leaning."""
    df = store.e2_cont_mech
    if df.empty or 'EVENT_LEAN' not in df.columns:
        return _empty_fig('E2-32: No contagion mechanism data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        windows = sorted(df['WINDOW'].unique())
        leans = sorted(df['EVENT_LEAN'].unique())
        x = np.arange(len(windows))
        width = 0.8 / max(len(leans), 1)
        for i, lean in enumerate(leans):
            sub = df[df['EVENT_LEAN'] == lean].set_index('WINDOW').reindex(windows)
            vals = sub['MEAN_PEER_CAR'].astype(float).fillna(0)
            # Add significance markers
            pvals = sub['P_VALUE_VS_ZERO'].astype(float).fillna(1)
            bar_colors = ['#e67e22' if p < 0.05 else '#ddd' for p in pvals]
            ax.bar(x + i * width, vals, width, label=lean,
                   color=C_LEAN.get(lean, C_SEQ[i]), edgecolor='white', alpha=0.8)
        ax.set_xticks(x + width * (len(leans) - 1) / 2)
        ax.set_xticklabels(windows, rotation=45, ha='right')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Mean Peer CAR')
        ax.set_title('Contagion Mechanism: Peer CARs by Event Leaning')
        ax.legend(title='Event Lean')
        fig.tight_layout()
    return fig


# ── 33. Essay 2 summary dashboard ─────────────────────────────────────

def e2_33_summary_dashboard(store):
    """Multi-panel summary of key Essay 2 findings."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # A: CAR by leaning
        ax1 = fig.add_subplot(gs[0, 0])
        df = store.e2_car_panel
        if not df.empty and 'LEAN' in df.columns and 'CAR_POST' in df.columns:
            grp = df.groupby('LEAN')['CAR_POST'].mean()
            colors = [C_LEAN.get(l, '#555') for l in grp.index]
            ax1.bar(grp.index, grp.values.astype(float), color=colors,
                    edgecolor='white', width=0.5)
            ax1.axhline(0, color='black', lw=0.5)
        ax1.set_title('A. Post-Event CAR by Leaning')
        ax1.set_ylabel('Mean CAR')

        # B: DiD key coefficient
        ax2 = fig.add_subplot(gs[0, 1])
        did = store.e2_did_coeff
        if not did.empty and 'VARIABLE' in did.columns:
            y = np.arange(len(did))
            coefs = did['COEFFICIENT'].astype(float)
            colors = ['#e74c3c' if _safe_float(r.get('P_VALUE')) < 0.05 else '#bdc3c7'
                      for _, r in did.iterrows()]
            ax2.barh(y, coefs, color=colors, edgecolor='white', height=0.5)
            ax2.set_yticks(y)
            ax2.set_yticklabels(did['VARIABLE'].tolist(), fontsize=9)
            ax2.axvline(0, color='black', lw=0.5)
        ax2.set_title('B. DiD Coefficients')

        # C: Contagion summary (one window)
        ax3 = fig.add_subplot(gs[1, 0])
        cont = store.e2_cont_summary
        if not cont.empty and 'MEAN_PEER_CAR' in cont.columns:
            cont_s = cont.sort_values('WINDOW_DAYS') if 'WINDOW_DAYS' in cont.columns else cont
            ax3.bar(cont_s['WINDOW'], cont_s['MEAN_PEER_CAR'].astype(float),
                    color='#e67e22', edgecolor='white')
            ax3.axhline(0, color='black', lw=0.5)
        ax3.set_title('C. Peer Contagion CARs')
        ax3.set_ylabel('Mean Peer CAR')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # D: Parallel trends
        ax4 = fig.add_subplot(gs[1, 1])
        pt = store.e2_parallel
        if not pt.empty and 'DAY' in pt.columns:
            pt = pt.sort_values('DAY')
            days = pt['DAY'].astype(int)
            coefs = pt['COEFFICIENT'].astype(float)
            se = pt['STD_ERROR'].astype(float) if 'STD_ERROR' in pt.columns else pd.Series(0.0, index=pt.index)
            ax4.fill_between(days, coefs - 1.96 * se, coefs + 1.96 * se,
                              alpha=0.2, color='#3498db')
            ax4.plot(days, coefs, 'o-', color='#2980b9', markersize=4, lw=1)
            ax4.axhline(0, color='black', lw=0.5, ls='--')
            ax4.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax4.set_title('D. Parallel Trends')
        ax4.set_xlabel('Days')

        fig.suptitle('Essay 2 — Culture War Event Study Summary', fontsize=15, y=1.02)
    return fig


ESSAY2_CHARTS = [
    ('e2_01_car_distribution', e2_01_car_distribution, 'CAR distribution (pre/post/full)'),
    ('e2_02_car_by_leaning', e2_02_car_by_leaning, 'CAR by political leaning'),
    ('e2_03_car_pre_vs_post', e2_03_car_pre_vs_post, 'Pre vs post event CAR scatter'),
    ('e2_04_car_by_regime', e2_04_car_by_regime, 'CAR by VIX regime'),
    ('e2_05_car_treat_vs_ctrl', e2_05_car_treat_vs_ctrl, 'Treatment vs control CARs'),
    ('e2_06_did_coefficients', e2_06_did_coefficients, 'DiD coefficient forest plot'),
    ('e2_07_parallel_trends', e2_07_parallel_trends, 'Parallel trends test'),
    ('e2_08_peer_parallel_trends', e2_08_peer_parallel_trends, 'Peer contagion parallel trends'),
    ('e2_09_news_sentiment_dist', e2_09_news_sentiment_dist, 'News sentiment distribution'),
    ('e2_10_news_pre_vs_post', e2_10_news_pre_vs_post, 'News sentiment pre vs post'),
    ('e2_11_sentiment_change', e2_11_sentiment_change, 'Sentiment change distribution'),
    ('e2_12_filing_sentiment_sections', e2_12_filing_sentiment_sections, 'Filing sentiment by section'),
    ('e2_13_filing_pct_breakdown', e2_13_filing_pct_breakdown, 'Filing sentiment composition'),
    ('e2_14_alignment_distribution', e2_14_alignment_distribution, 'Political alignment distribution'),
    ('e2_15_distinctive_phrases', e2_15_distinctive_phrases, 'Distinctive phrases by party'),
    ('e2_16_alignment_validation', e2_16_alignment_validation, 'Alignment validation scatter'),
    ('e2_17_event_nlp_heatmap', e2_17_event_nlp_heatmap, 'Event NLP heatmap'),
    ('e2_18_multiwindow_by_horizon', e2_18_multiwindow_by_horizon, 'Multi-window CAR by horizon'),
    ('e2_19_multiwindow_by_lean', e2_19_multiwindow_by_lean, 'Multi-window CARs by lean'),
    ('e2_20_multiwindow_tc', e2_20_multiwindow_tc, 'Multi-window treatment vs control'),
    ('e2_21_contagion_summary', e2_21_contagion_summary, 'Contagion peer CARs'),
    ('e2_22_contagion_by_lean', e2_22_contagion_by_lean, 'Contagion by event leaning'),
    ('e2_23_contagion_by_facing', e2_23_contagion_by_facing, 'Contagion B2C vs B2B'),
    ('e2_24_contagion_peer_vs_nonpeer', e2_24_contagion_peer_vs_nonpeer, 'Peer vs non-peer spillover'),
    ('e2_25_contagion_consumer_b2b', e2_25_contagion_consumer_b2b, 'Consumer vs B2B contagion'),
    ('e2_26_contagion_lean_pairwise', e2_26_contagion_lean_pairwise, 'Pairwise leaning comparisons'),
    ('e2_27_contagion_tight_diff', e2_27_contagion_tight_diff, 'Tight DiD contagion'),
    ('e2_28_mda_vs_risk_tone', e2_28_mda_vs_risk_tone, 'MDA vs Risk factor tone'),
    ('e2_29_alignment_components', e2_29_alignment_components, 'Alignment component scores'),
    ('e2_30_car_lean_regime', e2_30_car_lean_regime, 'CAR by leaning and regime heatmap'),
    ('e2_31_event_count_by_lean', e2_31_event_count_by_lean, 'Event count by leaning'),
    ('e2_32_contagion_mechanism', e2_32_contagion_mechanism, 'Contagion mechanism by lean'),
    ('e2_33_summary_dashboard', e2_33_summary_dashboard, 'Essay 2 summary dashboard'),
]
# ═══════════════════════════════════════════════════════════════════════
# ESSAY 3 — Informed Insider Trading Around Political Decisions (20 charts)
# ═══════════════════════════════════════════════════════════════════════

C_POL_CTRL = {'POLITICAL': '#e74c3c', 'CONTROL': '#3498db'}
C_PROXIMITY = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261']

def e3_01_directional_accuracy(store):
    """Headline: directional accuracy by sample (political vs control)."""
    df = store.e3_informed_trading
    if df.empty:
        return _empty_fig('E3-01: No informed trading data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        # Filter to main cuts
        main = df[df['CUT'].isin(['ALL_TRADES', 'ALL_SELLS', 'ALL_BUYS',
                                   'CONTROL', 'CONTROL_SELLS', 'CONTROL_BUYS'])]
        if main.empty:
            main = df.head(10)
        for col in ['METRIC_VALUE', 'METRIC_PVAL']:
            if col in main.columns:
                main[col] = pd.to_numeric(main[col], errors='coerce')
        labels = main['CUT'].values
        vals = main['METRIC_VALUE'].apply(_safe_float).values
        colors = [C_POL_CTRL.get(str(main.iloc[i].get('SAMPLE', '')), '#888')
                  for i in range(len(main))]
        bars = ax.barh(range(len(labels)), vals, color=colors)
        ax.axvline(0.5, color='grey', lw=1.2, ls='--', label='Null: 50%')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        for bar, p in zip(bars, main['METRIC_PVAL'].values):
            x = bar.get_width()
            ax.text(x + 0.003, bar.get_y() + bar.get_height() / 2,
                    f'{x:.1%} {_sig(p)}', va='center', fontsize=9)
        ax.set_xlabel('Directional Accuracy')
        ax.set_title('Informed Trading: Directional Accuracy\n(Political vs Matched Control)')
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig



def e3_02_sells_premium(store):
    """Sells premium: political vs control (event-CAR and trade-CAR variants)."""
    df = store.e3_informed_trading
    if df.empty:
        return _empty_fig('E3-02: No informed trading data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        premium_rows = df[df['CUT'].str.contains('PREMIUM', na=False)].copy()
        if premium_rows.empty:
            return _empty_fig('E3-02: No premium rows')
        for col in ['METRIC_VALUE', 'METRIC_PVAL']:
            premium_rows[col] = pd.to_numeric(premium_rows[col], errors='coerce')
        labels = premium_rows['CUT'].values
        vals = premium_rows['METRIC_VALUE'].apply(_safe_float).values
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
        bars = ax.bar(range(len(labels)), vals, color=colors)
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        for bar, p in zip(bars, premium_rows['METRIC_PVAL'].values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.001,
                    f'{y:.1%} {_sig(p)}', ha='center', fontsize=9)
        ax.set_ylabel('Accuracy Difference (pp)')
        ax.set_title('Sells Premium: Political − Control')
        fig.tight_layout()
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
    return fig




def e3_03_proximity_accuracy(store):
    """Directional accuracy by proximity-to-event window."""
    df = store.e3_informed_proximity
    if df.empty:
        return _empty_fig('E3-03: No proximity data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        pol = df[df['SAMPLE'] == 'POLITICAL'].copy()
        ctrl = df[df['SAMPLE'] == 'CONTROL'].copy()
        for col in ['METRIC_VALUE', 'METRIC_PVAL']:
            pol[col] = pd.to_numeric(pol[col], errors='coerce')
            if not ctrl.empty:
                ctrl[col] = pd.to_numeric(ctrl[col], errors='coerce')
        # Political sells by window
        sells = pol[pol['CUT'].str.endswith('_SELLS')].copy()
        if sells.empty:
            sells = pol
        x = range(len(sells))
        vals = sells['METRIC_VALUE'].apply(_safe_float).values
        bars = ax.bar(x, vals, color=C_PROXIMITY[:len(sells)], edgecolor='black', lw=0.5)
        ax.axhline(0.5, color='grey', lw=1.2, ls='--', label='Null: 50%')
        # Control reference line
        ctrl_sells = ctrl[ctrl['CUT'] == 'CONTROL_ALL_SELLS']
        if not ctrl_sells.empty:
            cv = _safe_float(ctrl_sells['METRIC_VALUE'].iloc[0])
            ax.axhline(cv, color='#3498db', lw=1.5, ls=':', label=f'Control sells: {cv:.1%}')
        ax.set_xticks(list(x))
        ax.set_xticklabels(sells['CUT'].values, rotation=30, ha='right', fontsize=9)
        for bar, p in zip(bars, sells['METRIC_PVAL'].values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.003,
                    f'{y:.1%} {_sig(p)}', ha='center', fontsize=9)
        ax.set_ylabel('Directional Accuracy')
        ax.set_title('Sells Accuracy by Proximity to Event\n(Political vs Control Baseline)')
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def e3_04_regulatory_period(store):
    """Accuracy by regulatory period (political vs control)."""
    df = store.e3_informed_trading
    if df.empty:
        return _empty_fig('E3-04: No informed trading data')
    with plt.rc_context(STYLE):
        periods = ['PRE_SOX', 'SOX_ERA', 'DODD_FRANK', 'POST_AMENDMENTS']
        fig, ax = plt.subplots(figsize=(10, 6))
        pol_rows = df[(df['CUT'].isin(periods)) & (df.get('SAMPLE', pd.Series()) == 'POLITICAL')]
        ctrl_rows = df[df['CUT'].str.startswith('CTRL_')]
        if pol_rows.empty:
            pol_rows = df[df['CUT'].isin(periods)]
        if not pol_rows.empty:
            for col in ['METRIC_VALUE']:
                pol_rows[col] = pd.to_numeric(pol_rows[col], errors='coerce')
            x = range(len(pol_rows))
            vals = pol_rows['METRIC_VALUE'].apply(_safe_float).values
            ax.bar(x, vals, color='#e74c3c', alpha=0.8, label='Political')
            ax.set_xticks(list(x))
            ax.set_xticklabels(pol_rows['CUT'].values, rotation=30, ha='right')
        if not ctrl_rows.empty:
            for col in ['METRIC_VALUE']:
                ctrl_rows[col] = pd.to_numeric(ctrl_rows[col], errors='coerce')
            cx = range(len(ctrl_rows))
            cvals = ctrl_rows['METRIC_VALUE'].apply(_safe_float).values
            ax.bar([i + 0.3 for i in cx], cvals, width=0.3, color='#3498db', alpha=0.8, label='Control')
        ax.axhline(0.5, color='grey', lw=1.2, ls='--')
        ax.set_ylabel('Directional Accuracy')
        ax.set_title('Accuracy by Regulatory Period')
        ax.legend()
        fig.tight_layout()
    return fig


def e3_05_dollar_magnitudes(store):
    """Dollar magnitudes by proximity window and CAR severity."""
    df = store.e3_informed_dollars
    if df.empty:
        return _empty_fig('E3-05: No dollar data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT', 'SAMPLE', 'METRIC_TYPE'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Left: by proximity
        prox = df[df['CUT'].str.contains('d$', regex=True, na=False)]
        if not prox.empty and 'TOTAL_EVENT_PROFIT' in prox.columns:
            axes[0].bar(range(len(prox)), prox['TOTAL_EVENT_PROFIT'].apply(_safe_float).values,
                        color=C_PROXIMITY[:len(prox)])
            axes[0].set_xticks(range(len(prox)))
            axes[0].set_xticklabels(prox['CUT'].values, rotation=30, ha='right')
        axes[0].set_title('Total Event Profit by Proximity')
        axes[0].set_ylabel('Total Profit ($)')
        # Right: by severity threshold
        sev = df[df['CUT'].str.contains('CAR_', na=False)]
        if not sev.empty and 'TOTAL_EVENT_PROFIT' in sev.columns:
            axes[1].bar(range(len(sev)), sev['TOTAL_EVENT_PROFIT'].apply(_safe_float).values,
                        color=['#f4a261', '#e76f51', '#e63946'][:len(sev)])
            axes[1].set_xticks(range(len(sev)))
            axes[1].set_xticklabels(sev['CUT'].values, rotation=30, ha='right')
        axes[1].set_title('Total Event Profit by CAR Severity')
        axes[1].set_ylabel('Total Profit ($)')
        fig.suptitle('Informed Trading: Dollar Magnitudes', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def e3_06_size_accuracy(store):
    """Trade-size decile accuracy (political vs control)."""
    df = store.e3_size_accuracy
    if df.empty:
        return _empty_fig('E3-06: No size-accuracy data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('DECILE', 'SIZE_DECILE', 'SAMPLE', 'SPEC'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 6))
        for sample, color in C_POL_CTRL.items():
            sub = df[df['SAMPLE'] == sample].sort_values('SIZE_DECILE')
            if not sub.empty and 'PCT_ACCURATE' in sub.columns:
                ax.plot(sub['SIZE_DECILE'], sub['PCT_ACCURATE'].apply(_safe_float),
                        marker='o', color=color, label=sample, lw=2)
        ax.axhline(0.5, color='grey', lw=1, ls='--')
        ax.set_xlabel('Trade-Size Decile')
        ax.set_ylabel('Directional Accuracy')
        ax.set_title('Accuracy by Trade-Size Decile\n(Political vs Control)')
        ax.legend()
        fig.tight_layout()
    return fig


def e3_07_size_accuracy_slopes(store):
    """LPM slopes: accuracy on log(trade size) by sample."""
    df = store.e3_size_accuracy_slopes
    if df.empty:
        return _empty_fig('E3-07: No slope data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('SAMPLE', 'SPEC'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = df['SAMPLE'].values if 'SAMPLE' in df.columns else df.index.values
        slopes = df['SLOPE'].apply(_safe_float).values if 'SLOPE' in df.columns else []
        if len(slopes):
            colors = [C_POL_CTRL.get(str(l), '#888') for l in labels]
            ax.bar(range(len(labels)), slopes, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_ylabel('Slope')
        ax.set_title('LPM: Accuracy ~ log(Trade Size) Slope by Sample')
        fig.tight_layout()
    return fig


def e3_08_reversal_regression(store):
    """Forest plot of reversal regression coefficients."""
    df = store.e3_reversal_regression
    if df.empty:
        return _empty_fig('E3-08: No regression data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('SPEC', 'VARIABLE', 'SAMPLE'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3 + 1)))
        for i, (_, row) in enumerate(df.iterrows()):
            coef = _safe_float(row.get('COEF', 0))
            se = _safe_float(row.get('SE', 0))
            p = _safe_float(row.get('PVAL', 1))
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            color = '#e74c3c' if p < 0.05 else '#3498db'
            ax.plot([ci_lo, ci_hi], [i, i], color=color, lw=2, solid_capstyle='round')
            ax.plot(coef, i, 'o', color=color, ms=7, zorder=5)
        ax.axvline(0, color='black', lw=0.8)
        labels = [f"{row.get('SPEC', '')}: {row.get('VARIABLE', '')}" for _, row in df.iterrows()]
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Coefficient')
        ax.set_title('Reversal Regression Coefficients\n(LPM: Accuracy ~ log(Size) × Political)')
        fig.tight_layout()
    return fig


def e3_09_crsp_summary(store):
    """CRSP abnormal return summary by cut."""
    df = store.e3_crsp_summary
    if df.empty:
        return _empty_fig('E3-09: No CRSP summary')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT', 'SAMPLE', 'METRIC_TYPE'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'CUT' in df.columns and 'MEAN_CAR_30' in df.columns:
            labels = df['CUT'].values
            vals = df['MEAN_CAR_30'].apply(_safe_float).values
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
            ax.bar(range(len(labels)), vals, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_ylabel('Mean CAR (30-day)')
        ax.set_title('CRSP Abnormal Returns by Cut')
        fig.tight_layout()
    return fig


def e3_10_wilcoxon_family(store):
    """Family-level Wilcoxon correction (Holm + BH)."""
    df = store.e3_wilcoxon_family
    if df.empty:
        return _empty_fig('E3-10: No Wilcoxon family data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT', 'FAMILY', 'TEST'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3 + 1)))
        if 'RAW_P' in df.columns and 'BH_P' in df.columns:
            y = range(len(df))
            labels = df['CUT'].values if 'CUT' in df.columns else df.index.values
            raw = df['RAW_P'].apply(_safe_float).values
            bh = df['BH_P'].apply(_safe_float).values
            ax.scatter(raw, list(y), marker='o', color='#e74c3c', label='Raw p', zorder=5)
            ax.scatter(bh, list(y), marker='s', color='#3498db', label='BH-adjusted p', zorder=5)
            ax.axvline(0.05, color='grey', lw=1, ls='--', label='α=0.05')
            ax.set_yticks(list(y))
            ax.set_yticklabels(labels, fontsize=8)
            ax.legend(fontsize=9)
        ax.set_xlabel('p-value')
        ax.set_title('Wilcoxon Family Correction (Holm + BH)')
        ax.set_xscale('log')
        fig.tight_layout()
    return fig


def e3_11_insider_concentration(store):
    """Insider-level concentration metrics (Gini, HHI, top-K%)."""
    df = store.e3_insider_concentration
    if df.empty:
        return _empty_fig('E3-11: No concentration data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT', 'METRIC'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'METRIC' in df.columns and 'VALUE' in df.columns:
            labels = df.apply(lambda r: f"{r.get('CUT', '')}:{r.get('METRIC', '')}", axis=1).values
            vals = df['VALUE'].apply(_safe_float).values
            ax.barh(range(len(labels)), vals, color=C_SEQ[:len(labels)])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Value')
        ax.set_title('Insider Concentration Metrics')
        fig.tight_layout()
    return fig


def e3_12_tost_equivalence(store):
    """TOST equivalence test results with 90% CIs."""
    df = store.e3_tost
    if df.empty:
        return _empty_fig('E3-12: No TOST data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('TEST', 'MARGIN_NAME'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (_, row) in enumerate(df.iterrows()):
            delta = _safe_float(row.get('DELTA', 0.2))
            ci_lo = _safe_float(row.get('CI90_LOWER', 0))
            ci_hi = _safe_float(row.get('CI90_UPPER', 0))
            diff = _safe_float(row.get('DIFF_MEAN', 0))
            equiv = _safe_float(row.get('EQUIVALENT', 0))
            color = '#27ae60' if equiv == 1 else '#e74c3c'
            ax.plot([ci_lo, ci_hi], [i, i], color=color, lw=3, solid_capstyle='round')
            ax.plot(diff, i, 'o', color=color, ms=8, zorder=5)
            ax.plot([-delta, -delta], [i - 0.3, i + 0.3], color='grey', lw=1.5, ls='--')
            ax.plot([delta, delta], [i - 0.3, i + 0.3], color='grey', lw=1.5, ls='--')
        ax.axvline(0, color='black', lw=0.8)
        labels = df['MARGIN_NAME'].values if 'MARGIN_NAME' in df.columns else [str(i) for i in range(len(df))]
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Difference')
        ax.set_title('TOST Equivalence Tests')
        fig.tight_layout()
    return fig


def e3_13_placebo_test(store):
    """Placebo permutation test distribution."""
    df = store.e3_placebo
    if df.empty:
        return _empty_fig('E3-13: No placebo data')
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        obs = _safe_float(df.iloc[0].get('OBSERVED_STAT', 0))
        mean = _safe_float(df.iloc[0].get('PLACEBO_MEAN', 0))
        std = _safe_float(df.iloc[0].get('PLACEBO_STD', 1))
        p = _safe_float(df.iloc[0].get('EMPIRICAL_P', 1))
        # Draw normal approximation
        x_range = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        y_range = sp_stats.norm.pdf(x_range, mean, std) if std > 0 else np.zeros_like(x_range)
        ax.fill_between(x_range, y_range, alpha=0.3, color='#3498db', label='Placebo dist.')
        ax.axvline(obs, color='#e74c3c', lw=2, ls='-', label=f'Observed ({obs:.4f})')
        ax.axvline(mean, color='#3498db', lw=1, ls='--', label=f'Placebo mean ({mean:.4f})')
        ax.set_xlabel('Test Statistic')
        ax.set_ylabel('Density')
        ax.set_title(f'Placebo Test (p={p:.4f})')
        ax.legend(fontsize=9)
        fig.tight_layout()
    return fig


def e3_14_mean_vs_distributional(store):
    """Side-by-side t-test vs Wilcoxon across cuts."""
    df = store.e3_mean_vs_dist
    if df.empty:
        return _empty_fig('E3-14: No mean vs distributional data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT',):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        if 'T_PVALUE' in df.columns:
            axes[0].barh(range(len(df)), -np.log10(df['T_PVALUE'].apply(_safe_float).values + 1e-30),
                         color='#e74c3c', alpha=0.8)
            axes[0].axvline(-np.log10(0.05), color='grey', ls='--', label='p=0.05')
            axes[0].set_yticks(range(len(df)))
            axes[0].set_yticklabels(df['CUT'].values, fontsize=8)
            axes[0].set_xlabel('-log₁₀(p)')
            axes[0].set_title('t-test p-values')
            axes[0].legend(fontsize=8)
        if 'WILCOXON_P' in df.columns:
            axes[1].barh(range(len(df)), -np.log10(df['WILCOXON_P'].apply(_safe_float).values + 1e-30),
                         color='#3498db', alpha=0.8)
            axes[1].axvline(-np.log10(0.05), color='grey', ls='--', label='p=0.05')
            axes[1].set_yticks(range(len(df)))
            axes[1].set_yticklabels(df['CUT'].values, fontsize=8)
            axes[1].set_xlabel('-log₁₀(p)')
            axes[1].set_title('Wilcoxon p-values')
            axes[1].legend(fontsize=8)
        fig.suptitle('Mean vs Distributional Tests', fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def e3_15_concentration_cuts(store):
    """Concentration by event type, policy area, connection."""
    df = store.e3_concentration_cuts
    if df.empty:
        return _empty_fig('E3-15: No concentration cuts data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT', 'METRIC'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35 + 1)))
        labels = df['CUT'].values if 'CUT' in df.columns else df.index.values
        vals = df['GINI'].apply(_safe_float).values if 'GINI' in df.columns else np.zeros(len(df))
        ax.barh(range(len(labels)), vals, color=C_SEQ[:len(labels)])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Gini Coefficient')
        ax.set_title('Insider Trading Concentration by Cut')
        fig.tight_layout()
    return fig


def e3_16_repeat_traders(store):
    """Repeat trader analysis."""
    df = store.e3_repeat_traders
    if df.empty:
        return _empty_fig('E3-16: No repeat trader data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('CUT',):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = df['CUT'].values if 'CUT' in df.columns else df.index.values
        vals = df['ACCURACY'].apply(_safe_float).values if 'ACCURACY' in df.columns else np.zeros(len(df))
        ax.bar(range(len(labels)), vals, color=C_SEQ[:len(labels)])
        ax.axhline(0.5, color='grey', lw=1, ls='--')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Repeat Trader Directional Accuracy')
        fig.tight_layout()
    return fig


def e3_17_quantile_regression(store):
    """Quantile regression coefficients."""
    df = store.e3_quantile_reg
    if df.empty:
        return _empty_fig('E3-17: No quantile regression data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('QUANTILE', 'VARIABLE'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'QUANTILE' in df.columns and 'COEF' in df.columns:
            q = df['QUANTILE'].apply(_safe_float).values
            c = df['COEF'].apply(_safe_float).values
            ax.plot(q, c, marker='o', color='#e74c3c', lw=2)
            ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Coefficient')
        ax.set_title('Quantile Regression: Political Treatment Effect')
        fig.tight_layout()
    return fig


def e3_18_bootstrap_ci(store):
    """Bootstrap confidence intervals."""
    df = store.e3_bootstrap_ci
    if df.empty:
        return _empty_fig('E3-18: No bootstrap CI data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('MARGIN',):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (_, row) in enumerate(df.iterrows()):
            lo = _safe_float(row.get('CI_LOWER', 0))
            hi = _safe_float(row.get('CI_UPPER', 0))
            pt = _safe_float(row.get('POINT_ESTIMATE', 0))
            color = '#e74c3c' if lo > 0 or hi < 0 else '#3498db'
            ax.plot([lo, hi], [i, i], color=color, lw=3, solid_capstyle='round')
            ax.plot(pt, i, 'o', color=color, ms=8, zorder=5)
        ax.axvline(0, color='black', lw=0.8)
        labels = df['MARGIN'].values if 'MARGIN' in df.columns else [str(i) for i in range(len(df))]
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Value')
        ax.set_title('Bootstrap 95% Confidence Intervals')
        fig.tight_layout()
    return fig


def e3_19_stratification(store):
    """Year × activity-tercile strata heatmap."""
    df = store.e3_stratification
    if df.empty:
        return _empty_fig('E3-19: No stratification data')
    with plt.rc_context(STYLE):
        for col in df.columns:
            if col not in ('YEAR', 'TERCILE', 'STRATUM'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'YEAR' in df.columns and 'N_EVENTS' in df.columns:
            pivot = df.pivot_table(values='N_EVENTS', index='YEAR', columns='TERCILE',
                                   aggfunc='sum', fill_value=0) if 'TERCILE' in df.columns else pd.DataFrame()
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
            else:
                yearly = df.groupby('YEAR')['N_EVENTS'].sum()
                ax.bar(yearly.index, yearly.values, color='#e74c3c')
        ax.set_title('Event Coverage: Year × Activity Tercile')
        fig.tight_layout()
    return fig


def e3_20_summary_dashboard(store):
    """2×2 summary of key Essay 3 informed trading findings."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) Headline: political vs control accuracy
        df = store.e3_informed_trading
        if not df.empty:
            main = df[df['CUT'].isin(['ALL_SELLS', 'CONTROL_SELLS'])].copy()
            for col in ['METRIC_VALUE']:
                main[col] = pd.to_numeric(main[col], errors='coerce')
            if not main.empty:
                vals = main['METRIC_VALUE'].apply(_safe_float).values
                labels = main['CUT'].values
                colors = [C_POL_CTRL.get(str(main.iloc[i].get('SAMPLE', '')), '#888')
                          for i in range(len(main))]
                axes[0, 0].bar(labels, vals, color=colors)
                axes[0, 0].axhline(0.5, color='grey', lw=1, ls='--')
        axes[0, 0].set_title('Sells Accuracy: Political vs Control')

        # (0,1) Proximity gradient
        prox = store.e3_informed_proximity
        if not prox.empty:
            sells = prox[prox['CUT'].str.endswith('_SELLS') & (prox['SAMPLE'] == 'POLITICAL')].copy()
            sells['METRIC_VALUE'] = pd.to_numeric(sells['METRIC_VALUE'], errors='coerce')
            if not sells.empty:
                axes[0, 1].bar(range(len(sells)), sells['METRIC_VALUE'].apply(_safe_float).values,
                               color=C_PROXIMITY[:len(sells)])
                axes[0, 1].set_xticks(range(len(sells)))
                axes[0, 1].set_xticklabels(sells['CUT'].values, rotation=30, ha='right', fontsize=8)
                axes[0, 1].axhline(0.5, color='grey', lw=1, ls='--')
        axes[0, 1].set_title('Proximity Gradient (Sells)')

        # (1,0) Size-accuracy
        sa = store.e3_size_accuracy
        if not sa.empty:
            for col in sa.columns:
                if col not in ('DECILE', 'SIZE_DECILE', 'SAMPLE', 'SPEC'):
                    sa[col] = pd.to_numeric(sa[col], errors='coerce')
            for sample, color in C_POL_CTRL.items():
                sub = sa[sa['SAMPLE'] == sample].sort_values('SIZE_DECILE')
                if not sub.empty and 'PCT_ACCURATE' in sub.columns:
                    axes[1, 0].plot(sub['SIZE_DECILE'], sub['PCT_ACCURATE'].apply(_safe_float),
                                    marker='o', color=color, label=sample, lw=2)
            axes[1, 0].axhline(0.5, color='grey', lw=1, ls='--')
            axes[1, 0].legend(fontsize=8)
        axes[1, 0].set_title('Size-Accuracy by Decile')

        # (1,1) Dollar magnitudes
        dol = store.e3_informed_dollars
        if not dol.empty:
            for col in dol.columns:
                if col not in ('CUT', 'SAMPLE', 'METRIC_TYPE'):
                    dol[col] = pd.to_numeric(dol[col], errors='coerce')
            if 'TOTAL_EVENT_PROFIT' in dol.columns:
                top = dol.head(6)
                axes[1, 1].bar(range(len(top)),
                               top['TOTAL_EVENT_PROFIT'].apply(_safe_float).values,
                               color=C_SEQ[:len(top)])
                axes[1, 1].set_xticks(range(len(top)))
                axes[1, 1].set_xticklabels(top['CUT'].values, rotation=30, ha='right', fontsize=8)
        axes[1, 1].set_title('Dollar Magnitudes')

        fig.suptitle('Essay 3: Informed Insider Trading — Summary',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


ESSAY3_CHARTS = [
    ('e3_01_directional_accuracy', e3_01_directional_accuracy, 'Directional accuracy by sample'),
    ('e3_02_sells_premium', e3_02_sells_premium, 'Sells premium: political - control'),
    ('e3_03_proximity_accuracy', e3_03_proximity_accuracy, 'Accuracy by proximity to event'),
    ('e3_04_regulatory_period', e3_04_regulatory_period, 'Accuracy by regulatory period'),
    ('e3_05_dollar_magnitudes', e3_05_dollar_magnitudes, 'Dollar magnitudes by proximity and severity'),
    ('e3_06_size_accuracy', e3_06_size_accuracy, 'Trade-size decile accuracy'),
    ('e3_07_size_accuracy_slopes', e3_07_size_accuracy_slopes, 'LPM accuracy slopes by sample'),
    ('e3_08_reversal_regression', e3_08_reversal_regression, 'Reversal regression coefficients'),
    ('e3_09_crsp_summary', e3_09_crsp_summary, 'CRSP abnormal returns by cut'),
    ('e3_10_wilcoxon_family', e3_10_wilcoxon_family, 'Wilcoxon family correction'),
    ('e3_11_insider_concentration', e3_11_insider_concentration, 'Insider concentration metrics'),
    ('e3_12_tost_equivalence', e3_12_tost_equivalence, 'TOST equivalence tests'),
    ('e3_13_placebo_test', e3_13_placebo_test, 'Placebo permutation test'),
    ('e3_14_mean_vs_distributional', e3_14_mean_vs_distributional, 'Mean vs distributional tests'),
    ('e3_15_concentration_cuts', e3_15_concentration_cuts, 'Concentration by cut'),
    ('e3_16_repeat_traders', e3_16_repeat_traders, 'Repeat trader accuracy'),
    ('e3_17_quantile_regression', e3_17_quantile_regression, 'Quantile regression'),
    ('e3_18_bootstrap_ci', e3_18_bootstrap_ci, 'Bootstrap confidence intervals'),
    ('e3_19_stratification', e3_19_stratification, 'Year × activity stratification'),
    ('e3_20_summary_dashboard', e3_20_summary_dashboard, 'Essay 3 summary dashboard'),
]

ALL_CHARTS = ESSAY1_CHARTS + ESSAY2_CHARTS + ESSAY3_CHARTS


# ═══════════════════════════════════════════════════════════════════════
# GENERATE & SAVE
# ═══════════════════════════════════════════════════════════════════════

def generate_essay(store, chart_list, essay_label):
    """Generate and save all charts in a chart list."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    results = {}
    for name, func, desc in chart_list:
        t0 = time.time()
        try:
            fig = func(store)
            save_result = store.save_figure(name, fig, metadata={
                'ESSAY': essay_label,
                'DESCRIPTION': desc,
            })
            plt.close(fig)
            elapsed = time.time() - t0
            status = save_result.get('status', 'OK') if isinstance(save_result, dict) else 'OK'
            results[name] = {'status': status, 'time': elapsed}
            print(f'  {name:45s} {status:8s} ({elapsed:.1f}s)')
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = {'status': 'FAILED', 'error': str(e), 'time': elapsed}
            print(f'  {name:45s} FAILED   ({elapsed:.1f}s) — {e}')
            logger.error('Chart %s failed: %s', name, e, exc_info=True)
    return results


def generate_all(store, essays=None):
    """Generate all charts (or specific essays)."""
    all_results = {}
    if essays is None or 1 in essays:
        print('\n' + '=' * 60)
        print('  Essay 1 — Volatility Regimes & FF5 (33 charts)')
        print('=' * 60)
        all_results.update(generate_essay(store, ESSAY1_CHARTS, 'Essay 1'))
    if essays is None or 2 in essays:
        print('\n' + '=' * 60)
        print('  Essay 2 — Culture War Event Study (33 charts)')
        print('=' * 60)
        all_results.update(generate_essay(store, ESSAY2_CHARTS, 'Essay 2'))
    if essays is None or 3 in essays:
        print('\n' + '=' * 60)
        print('  Essay 3 — Informed Insider Trading (20 charts)')
        print('=' * 60)
        all_results.update(generate_essay(store, ESSAY3_CHARTS, 'Essay 3'))
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_visuals():
    """Validate that all chart functions execute without error and
    return valid matplotlib Figure objects — Dr. Lambert test suite."""
    import traceback

    print('\n' + '=' * 60)
    print('  Visual Test Suite')
    print('=' * 60)

    store = ResultStore()
    passed = 0
    failed = 0
    errors = []

    for name, func, desc in ALL_CHARTS:
        try:
            fig = func(store)
            # Test 1: returns a Figure
            assert isinstance(fig, plt.Figure), f'{name} did not return a Figure'
            # Test 2: figure has at least one axes
            assert len(fig.get_axes()) >= 1, f'{name} has no axes'
            # Test 3: figure can be rendered to bytes (no broken artists)
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            assert buf.tell() > 0, f'{name} produced empty PNG'
            buf.close()
            plt.close(fig)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f'  FAIL: {name} — {e}')
            traceback.print_exc()
            try:
                plt.close('all')
            except Exception:
                pass

    print(f'\nResults: {passed} passed, {failed} failed out of {passed + failed}')
    if errors:
        print('\nFailures:')
        for name, err in errors:
            print(f'  {name}: {err}')

    store.close()
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print('=' * 60)
    print('  Signals & Systems — Visual Generation')
    print(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC')
    print('=' * 60)

    # Parse args
    essays = None
    run_tests = False
    backend = None
    for arg in sys.argv[1:]:
        if arg == '--test':
            run_tests = True
        elif arg.startswith('--essay'):
            try:
                essays = [int(sys.argv[sys.argv.index(arg) + 1])]
            except (IndexError, ValueError):
                pass
        elif arg.startswith('--backend'):
            try:
                backend = sys.argv[sys.argv.index(arg) + 1]
            except (IndexError, ValueError):
                pass

    if run_tests:
        success = test_visuals()
        sys.exit(0 if success else 1)

    store = ResultStore(backend=backend)
    print(f'\nBackend: {store.backend}')

    results = generate_all(store, essays=essays)

    n_ok = sum(1 for r in results.values() if r.get('status') != 'FAILED')
    n_fail = sum(1 for r in results.values() if r.get('status') == 'FAILED')
    total_time = sum(r.get('time', 0) for r in results.values())

    print(f'\n{"=" * 60}')
    print(f'  Complete: {n_ok} succeeded, {n_fail} failed ({total_time:.1f}s total)')
    print(f'  Figures saved to: {FIGURE_DIR}/')
    print(f'{"=" * 60}')

    store.close()
