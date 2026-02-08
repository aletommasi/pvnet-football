from __future__ import annotations
import numpy as np
import pandas as pd

def split_by_match(df: pd.DataFrame, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split per match_id (anti leakage).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    match_ids = df["match_id"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(match_ids)

    n = len(match_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(match_ids[:n_train])
    val_ids = set(match_ids[n_train:n_train+n_val])
    test_ids = set(match_ids[n_train+n_val:])

    train_df = df[df["match_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["match_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["match_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df
