from __future__ import annotations
import pandas as pd
from statsbombpy import sb

def load_competition_events(
    competition_id: int,
    season_id: int,
    include_freeze_frame: bool = False
) -> pd.DataFrame:
    """
    Loads StatsBomb events for a competition and season into a single dataframe.
    Adds match_id.
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()

    all_events = []
    for mid in match_ids:
        ev = sb.events(match_id=mid, fmt="dataframe", flatten_attrs=True)
        ev["match_id"] = mid
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)

    # freeze_frame is huge: we remove it by default to save memory and speed up processing.
    if not include_freeze_frame:
        for col in list(events.columns):
            if "freeze_frame" in col:
                events.drop(columns=[col], inplace=True, errors="ignore")

    return events
