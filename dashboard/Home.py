import streamlit as st
from pathlib import Path

st.set_page_config(page_title="PVNet Dashboard", page_icon="⚽", layout="wide")

st.title("⚽ PVNet — Possession Value Network")
st.caption("Dashboard for match analysts: overview, ranking, match analysis and pitch maps.")

default_artifacts = Path(__file__).resolve().parents[1] / "artifacts"

st.sidebar.header("Settings")
artifacts_dir = st.sidebar.text_input("Artifacts path/", value=str(default_artifacts))
st.session_state["ARTIFACTS_DIR"] = artifacts_dir

st.sidebar.markdown("---")
st.sidebar.caption("Go to pages from the Streamlit menu.")
