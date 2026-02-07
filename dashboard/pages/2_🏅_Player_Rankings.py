import streamlit as st
from utils.io import load_parquet, assert_artifacts_exist
from components.filters import min_actions_slider

REQUIRED = [
    "player_ranking_outfield.parquet",
    "player_ranking_goalkeepers.parquet",
]

st.header("🏅 Player Rankings")

missing = assert_artifacts_exist(REQUIRED)
if missing:
    st.error(f"These files are missing from artifacts/: {missing}")
    st.stop()

outfield = load_parquet("player_ranking_outfield.parquet")
gk = load_parquet("player_ranking_goalkeepers.parquet")

min_actions = min_actions_slider(default=50)

tab1, tab2 = st.tabs(["Outfield", "Goalkeepers (proxy)"])

with tab1:
    d = outfield[outfield["actions"] >= min_actions].copy()
    d = d.sort_values("value_per_100_actions", ascending=False)
    st.caption("Ranking by value_per_100_actions (robust).")
    st.dataframe(d.head(50), use_container_width=True)

with tab2:
    d = gk[gk["actions"] >= min_actions].copy()
    d = d.sort_values("value_per_100_actions", ascending=False)
    st.caption("Separated via deep_action_rate (proxy).")
    st.dataframe(d.head(50), use_container_width=True)
