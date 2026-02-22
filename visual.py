"""
Visualization module for Signals & Systems dissertation research.

Loads model results from Snowflake (with SQLite fallback) and produces
publication-quality figures. Each figure is saved to disk as PNG and
stored back in the database as a binary blob for reproducibility.

Usage:
    from visual import ResultStore, car_by_political_leaning

    store = ResultStore()
    fig = car_by_political_leaning(store)

    # Or run all visuals and save to database
    python visual.py
"""

import io
import logging
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Output directory for saved figures
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Dissertation-quality style defaults
STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
}


# =============================================================================
# RESULT STORE — loads model outputs from the database
# =============================================================================
class ResultStore:
    """
    Loads model results from Snowflake (primary) or SQLite (fallback).

    Attributes
    ----------
    event_results : pd.DataFrame
        EVENT_STUDY_RESULTS table.
    did_results : pd.DataFrame
        DID_RESULTS table.
    cross_sectional : pd.DataFrame
        CROSS_SECTIONAL_CAR table.
    run_summary : pd.DataFrame
        MODEL_RUN_SUMMARY table.
    backend : str
        Which database backend was used.
    """

    def __init__(self, backend=None, snowflake_schema=None, sqlite_path=None):
        self.backend = backend
        self._loader = None

        self._connect(backend, snowflake_schema, sqlite_path)
        self._load_results()

    def _connect(self, backend, snowflake_schema, sqlite_path):
        """Establish database connection with Snowflake -> SQLite fallback."""
        if backend == 'sqlite':
            self._connect_sqlite(sqlite_path)
            return

        if backend == 'snowflake':
            self._connect_snowflake(snowflake_schema)
            return

        # Auto-detect: try Snowflake, verify reads work, fall back to SQLite
        try:
            self._connect_snowflake(snowflake_schema)
            test = self._loader.read_table('EVENT_STUDY_RESULTS', limit=1)
            if test.empty:
                raise RuntimeError("Snowflake result tables empty or missing")
            logger.info("Snowflake read verified")
            return
        except Exception as e:
            logger.warning("Snowflake unusable (%s). Falling back to SQLite.", e)
            if self._loader:
                try:
                    self._loader.close()
                except Exception:
                    pass

        self._connect_sqlite(sqlite_path)

    def _connect_snowflake(self, schema):
        from Database import SnowflakeLoader
        loader = SnowflakeLoader(schema=schema)
        loader.connect()
        self._loader = loader
        self.backend = 'snowflake'
        logger.info("ResultStore connected to Snowflake")

    def _connect_sqlite(self, db_path):
        from Database import SQLiteLoader
        loader = SQLiteLoader(db_path=db_path)
        loader.connect()
        self._loader = loader
        self.backend = 'sqlite'
        logger.info("ResultStore connected to SQLite")

    def _read(self, table_name):
        try:
            return self._loader.read_table(table_name)
        except Exception as e:
            logger.warning("Could not load %s: %s", table_name, e)
            return pd.DataFrame()

    def _load_results(self):
        logger.info("Loading model results from %s...", self.backend)

        self.event_results = self._read('EVENT_STUDY_RESULTS')
        self.did_results = self._read('DID_RESULTS')
        self.cross_sectional = self._read('CROSS_SECTIONAL_CAR')
        self.run_summary = self._read('MODEL_RUN_SUMMARY')

        n_events = len(self.event_results)
        n_did = len(self.did_results)
        logger.info("ResultStore ready (%d event results, %d DiD results)", n_events, n_did)

    def save_figure(self, name, fig, metadata=None):
        """
        Save a figure to disk (PNG) and to the database as a binary blob.

        Parameters
        ----------
        name : str
            Figure identifier (e.g. 'car_by_political_leaning').
        fig : matplotlib.figure.Figure
            The figure to save.
        metadata : dict, optional
            Extra metadata columns to store alongside the figure.

        Returns
        -------
        dict with save status.
        """
        os.makedirs(FIGURE_DIR, exist_ok=True)

        # Save to disk
        png_path = os.path.join(FIGURE_DIR, f'{name}.png')
        fig.savefig(png_path)
        file_size = os.path.getsize(png_path)
        logger.info("Saved %s (%d KB)", png_path, file_size // 1024)

        # Serialize figure to PNG bytes for database storage
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        png_bytes = buf.read()
        buf.close()

        # Build row for the FIGURES table
        row = {
            'FIGURE_NAME': name,
            'FORMAT': 'png',
            'WIDTH_PX': int(fig.get_size_inches()[0] * fig.dpi),
            'HEIGHT_PX': int(fig.get_size_inches()[1] * fig.dpi),
            'FILE_SIZE_BYTES': len(png_bytes),
            'FILE_PATH': png_path,
            'IMAGE_DATA': png_bytes,
            'CREATED_AT': datetime.now().isoformat(),
        }
        if metadata:
            row.update(metadata)

        df = pd.DataFrame([row])

        # Write to database (replace any existing figure with same name)
        try:
            self._delete_figure(name)
            result = self._loader.write_table(df, 'FIGURES', replace=False)
            logger.info("Saved figure '%s' to database (%d bytes)", name, len(png_bytes))
            return result
        except Exception as e:
            logger.error("Failed to save figure to database: %s", e)
            return {'status': 'FAILED', 'error': str(e)}

    def _delete_figure(self, name):
        """Remove an existing figure row by name before re-inserting."""
        try:
            if self.backend == 'sqlite':
                self._loader.conn.execute(
                    'DELETE FROM FIGURES WHERE FIGURE_NAME = ?', (name,)
                )
                self._loader.conn.commit()
            else:
                cur = self._loader.conn.cursor()
                try:
                    cur.execute(
                        f"DELETE FROM {self._loader.schema}.FIGURES "
                        f"WHERE FIGURE_NAME = %s", (name,)
                    )
                finally:
                    cur.close()
        except Exception:
            pass  # table may not exist yet on first run

    def close(self):
        if self._loader:
            self._loader.close()
            self._loader = None


# =============================================================================
# VISUALIZATIONS
# =============================================================================
def car_by_political_leaning(store: ResultStore) -> plt.Figure:
    """
    Bar chart of mean Cumulative Abnormal Returns (CAR) grouped by
    political leaning, with 95% confidence interval error bars.

    This is the central figure for Essay 1: do culture war events
    affect stock returns differently based on a firm's political
    alignment?
    """
    df = store.event_results.copy()
    ok = df[df['STATUS'] == 'OK'].copy()

    if ok.empty or 'POLITICAL_LEANING' not in ok.columns:
        logger.warning("No valid event study results to plot")
        return plt.figure()

    ok['CAR'] = pd.to_numeric(ok['CAR'], errors='coerce')
    ok = ok.dropna(subset=['CAR', 'POLITICAL_LEANING'])

    # Compute group statistics
    groups = ok.groupby('POLITICAL_LEANING')['CAR']
    stats = groups.agg(['mean', 'std', 'count']).reset_index()
    stats.columns = ['Group', 'Mean', 'Std', 'N']
    stats['SE'] = stats['Std'] / np.sqrt(stats['N'])
    stats['CI95'] = 1.96 * stats['SE']

    # Sort: Conservative, Liberal, Mixed
    order = ['Conservative', 'Liberal', 'Mixed']
    stats['_order'] = stats['Group'].map({g: i for i, g in enumerate(order)})
    stats = stats.sort_values('_order').reset_index(drop=True)

    # Color palette
    colors = {
        'Conservative': '#c0392b',
        'Liberal': '#2980b9',
        'Mixed': '#7f8c8d',
    }
    bar_colors = [colors.get(g, '#333333') for g in stats['Group']]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        x = np.arange(len(stats))
        bars = ax.bar(
            x, stats['Mean'], yerr=stats['CI95'],
            color=bar_colors, edgecolor='white', linewidth=0.8,
            capsize=5, error_kw={'linewidth': 1.2, 'capthick': 1.2},
            width=0.6, zorder=3,
        )

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, stats.itertuples())):
            y = row.Mean
            label = f'{y:.1%}'
            va = 'top' if y < 0 else 'bottom'
            offset = 0.01 if y < 0 else -0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2, y + offset,
                label, ha='center', va=va, fontweight='bold', fontsize=11,
            )

        # Add N labels below x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}\n(N={n:.0f})' for g, n in zip(stats['Group'], stats['N'])])

        # Reference line at zero
        ax.axhline(0, color='black', linewidth=0.6, zorder=1)

        # Labels
        ax.set_ylabel('Mean Cumulative Abnormal Return (CAR)')
        ax.set_title('Culture War Event Impact on Stock Returns\nby Political Leaning')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # Significance annotations
        xs_df = store.cross_sectional
        if not xs_df.empty:
            for _, row in xs_df.iterrows():
                group = row.get('POLITICAL_LEANING', '')
                p = row.get('P_VALUE', 1.0)
                idx_match = stats.index[stats['Group'] == group]
                if len(idx_match) > 0 and p < 0.05:
                    idx = idx_match[0]
                    mean_val = stats.loc[idx, 'Mean']
                    ci = stats.loc[idx, 'CI95']
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                    y_pos = mean_val - ci - 0.02 if mean_val < 0 else mean_val + ci + 0.02
                    ax.text(idx, y_pos, star, ha='center', fontsize=14, fontweight='bold')

        # Subtitle with model info
        if not store.run_summary.empty:
            row = store.run_summary.iloc[-1]
            subtitle = (
                f"FF5 Factor Model  |  "
                f"{int(row.get('N_EVENT_STUDIES', 0))} events  |  "
                f"Error bars = 95% CI"
            )
            ax.annotate(
                subtitle, xy=(0.5, -0.18), xycoords='axes fraction',
                ha='center', fontsize=9, color='#555555',
            )

        fig.tight_layout()

    return fig


# =============================================================================
# MAIN — run all visuals and save to database
# =============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    print("=" * 60)
    print("  Signals & Systems — Visual Generation")
    print("=" * 60)

    store = ResultStore()

    print(f"\nBackend: {store.backend}")
    print(f"Event results: {len(store.event_results)} rows")
    print(f"DiD results:   {len(store.did_results)} rows")

    # --- Generate and save the placeholder visual ---
    print("\nGenerating: CAR by Political Leaning...")
    fig = car_by_political_leaning(store)
    result = store.save_figure('car_by_political_leaning', fig, metadata={
        'DESCRIPTION': 'Mean CAR by political leaning with 95% CI error bars',
        'MODEL_NAME': 'FF5',
    })
    plt.close(fig)

    if result.get('status') == 'SUCCESS':
        print(f"  Saved to database and {FIGURE_DIR}/car_by_political_leaning.png")
    else:
        print(f"  Save failed: {result.get('error', 'unknown')}")

    # Verify round-trip: read the figure back from the database
    print("\nVerifying database round-trip...")
    figures_df = store._read('FIGURES')
    if not figures_df.empty:
        row = figures_df.iloc[-1]
        print(f"  Figure: {row['FIGURE_NAME']}")
        print(f"  Size:   {row['FILE_SIZE_BYTES']:,} bytes")
        print(f"  Dims:   {row['WIDTH_PX']}x{row['HEIGHT_PX']} px")
        print(f"  Stored: {row['CREATED_AT']}")
    else:
        print("  (no figures found in database)")

    store.close()
    print("\n" + "=" * 60)
