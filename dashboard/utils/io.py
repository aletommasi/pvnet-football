import json
from pathlib import Path
import pandas as pd
import streamlit as st

def get_artifacts_dir() -> Path:
    p = st.session_state.get("ARTIFACTS_DIR")
    if not p:
        p = str(Path(__file__).resolve().parents[2] / "artifacts")
    return Path(p)

@st.cache_data(show_spinner=False)
def load_json(filename: str) -> dict:
    path = get_artifacts_dir() / filename
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_parquet(filename: str) -> pd.DataFrame:
    path = get_artifacts_dir() / filename
    return pd.read_parquet(path)

def assert_artifacts_exist(required: list[str]) -> list[str]:
    base = get_artifacts_dir()
    return [f for f in required if not (base / f).exists()]
