from __future__ import annotations
import numpy as np
import pandas as pd

def add_future_labels(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Per ogni evento in un possesso (match_id, possession):
      - shot_within_k = 1 se entro i prossimi k eventi del possesso compare uno Shot
      - goal_within_k = 1 se entro i prossimi k eventi compare uno Shot con shot_outcome == "Goal"
    """
    out = df.copy()
    out["is_shot_event"] = (out["type_name"] == "Shot").astype(int)
    out["is_goal_event"] = ((out["type_name"] == "Shot") & (out["shot_outcome"].astype(str) == "Goal")).astype(int)

    shot_within = np.zeros(len(out), dtype=np.int64)
    goal_within = np.zeros(len(out), dtype=np.int64)

    grouped = out.groupby(["match_id", "possession"], sort=False)

    for _, idx in grouped.indices.items():
        idx = np.array(idx, dtype=int)
        shots = out.loc[idx, "is_shot_event"].values
        goals = out.loc[idx, "is_goal_event"].values
        n = len(idx)

        for t in range(n):
            end = min(n, t + k + 1)
            window_shot = shots[t+1:end]
            window_goal = goals[t+1:end]
            shot_within[idx[t]] = int(window_shot.sum() > 0)
            goal_within[idx[t]] = int(window_goal.sum() > 0)

    out["shot_within_k"] = shot_within
    out["goal_within_k"] = goal_within
    return out
