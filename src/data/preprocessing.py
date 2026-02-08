from __future__ import annotations
import numpy as np
import pandas as pd

PITCH_LENGTH = 120.0  # StatsBomb
PITCH_WIDTH = 80.0

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def basic_clean(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce e crea colonne minime, ordina eventi per match/period/index.
    """
    df = events.copy()

    needed = [
        "competition_id","season_id",
        "match_id", "period", "minute", "second", "timestamp",
        "possession", "possession_team", "team", "player",
        "type", "location", "pass_end_location", "carry_end_location",
        "shot_end_location", "pass_outcome", "dribble_outcome",
        "shot_outcome", "index"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    if df["index"].notna().any():
        df = df.sort_values(["match_id", "period", "index"])
    else:
        df = df.sort_values(["match_id", "period", "minute", "second"])

    df["type_name"] = df["type"].astype(str)
    df["team_name"] = df["team"].astype(str)
    df["possession_team_name"] = df["possession_team"].astype(str)
    df["player_name"] = df["player"].astype(str)

    return df.reset_index(drop=True)

def extract_xy(loc) -> tuple[float, float]:
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute"] = _to_float(out["minute"]).fillna(0).astype(int)
    out["second"] = _to_float(out["second"]).fillna(0).astype(int)
    out["time_seconds"] = out["minute"] * 60 + out["second"]
    return out

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    sx, sy = zip(*out["location"].apply(extract_xy))
    out["start_x"] = sx
    out["start_y"] = sy

    end_loc = out["pass_end_location"].where(out["pass_end_location"].notna(), out["carry_end_location"])
    end_loc = end_loc.where(end_loc.notna(), out["shot_end_location"])

    ex, ey = zip(*end_loc.apply(extract_xy))
    out["end_x"] = ex
    out["end_y"] = ey

    goal_x, goal_y = PITCH_LENGTH, PITCH_WIDTH / 2.0
    out["dist_to_goal"] = np.sqrt((goal_x - out["start_x"])**2 + (goal_y - out["start_y"])**2)
    out["angle_to_goal_center"] = np.arctan2(goal_y - out["start_y"], goal_x - out["start_x"])

    out["dx"] = out["end_x"] - out["start_x"]
    out["dy"] = out["end_y"] - out["start_y"]
    out["progress_x"] = out["dx"].fillna(0.0)

    return out

def add_event_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_pass"] = (out["type_name"] == "Pass").astype(int)
    out["is_carry"] = (out["type_name"] == "Carry").astype(int)
    out["is_dribble"] = (out["type_name"] == "Dribble").astype(int)
    out["is_shot"] = (out["type_name"] == "Shot").astype(int)

    # pass_outcome NaN => completato
    out["pass_success"] = ((out["is_pass"] == 1) & (out["pass_outcome"].isna())).astype(int)
    # dribble_outcome NaN => completato
    out["dribble_success"] = ((out["is_dribble"] == 1) & (out["dribble_outcome"].isna())).astype(int)
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_temporal_features(out)
    out = add_spatial_features(out)
    out = add_event_outcome_features(out)

    num_cols = [
        "start_x","start_y","end_x","end_y","dist_to_goal","angle_to_goal_center",
        "dx","dy","progress_x","time_seconds"
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out
