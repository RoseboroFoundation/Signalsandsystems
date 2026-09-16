"""Compute Cohen's kappa between coder1 and coder2 initiation codings.

Usage:
    python compute_kappa.py [path_to_csv]

Reads event_initiation_coding.csv (or path given), computes Cohen's kappa
for the `initiation` column between coder1 and coder2 where both are filled.
Reports kappa, N, and the confusion matrix.
"""

import sys
import pandas as pd
import numpy as np


def cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two categorical arrays."""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return np.nan, np.nan, pd.DataFrame()

    # Confusion matrix
    cm = pd.crosstab(
        pd.Categorical(y1, categories=labels),
        pd.Categorical(y2, categories=labels),
        rownames=['coder1'], colnames=['coder2'],
        dropna=False,
    )

    # Observed agreement
    p_o = np.diag(cm.values).sum() / n

    # Expected agreement (by chance)
    row_marginals = cm.sum(axis=1).values / n
    col_marginals = cm.sum(axis=0).values / n
    p_e = (row_marginals * col_marginals).sum()

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return kappa, p_o, cm


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'event_initiation_coding.csv'
    df = pd.read_csv(path)

    # Filter to rows where both coders have filled in initiation
    mask = df['coder2_initiation'].notna() & (df['coder2_initiation'] != '')
    coded = df[mask]
    print(f"Rows with both codings: {len(coded)} / {len(df)}")

    if coded.empty:
        print("No coder2 data yet. Fill coder2_initiation column and rerun.")
        return

    y1 = coded['initiation'].values
    y2 = coded['coder2_initiation'].values

    kappa, p_o, cm = cohens_kappa(y1, y2)
    print(f"\nCohen's kappa: {kappa:.4f}")
    print(f"Observed agreement: {p_o:.4f}")
    print(f"\nConfusion matrix:")
    print(cm.to_string())

    # Also compute kappa for lead_time if both filled
    lt_mask = coded['coder2_lead_time'].notna() & (coded['coder2_lead_time'] != '')
    if lt_mask.any():
        lt_coded = coded[lt_mask]
        lt_kappa, lt_po, lt_cm = cohens_kappa(
            lt_coded['lead_time'].values,
            lt_coded['coder2_lead_time'].values,
        )
        print(f"\nLead-time kappa: {lt_kappa:.4f} (N={len(lt_coded)})")
        print(lt_cm.to_string())


if __name__ == '__main__':
    main()
