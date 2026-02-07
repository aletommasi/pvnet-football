import streamlit as st
from utils.io import load_parquet, assert_artifacts_exist
from components.filters import select_team
from components.charts import plot_timeline

REQUIRED = ["test_predictions.parquet", "team_match_ranking.parquet"]

st.header("🎥 Match Analysis")

missing = assert_artifacts_exist(REQUIRED)
if missing:
    st.error(f"These files are missing from artifacts/: {missing}")
    st.stop()

events = load_parquet("test_predictions.parquet")
team_match = load_parquet("team_match_ranking.parquet")

# =========================
# NEW FILTER FLOW:
# Team -> Match (of that team)
# =========================
team = select_team(events, "Team")
if team == "All":
    st.warning("Select a team to see available matches.")
    st.stop()

team_events = events[events["team_name"] == team].copy()

matches = sorted(team_events["match_id"].dropna().unique().tolist())
if not matches:
    st.error("No matches found for the selected team.")
    st.stop()

match_id = st.selectbox("Match", matches)
match_events = team_events[team_events["match_id"] == match_id].copy()

# =========================
# SUMMARY TEAM-MATCH
# =========================
st.subheader("Team summary (match)")
tm = (
    team_match[team_match["match_id"] == match_id]
    .sort_values("team_value_total", ascending=False)
)
st.dataframe(tm, use_container_width=True)

# =========================
# TIMELINE
# =========================
st.subheader("Timeline (5-min buckets)")

if "minute_bucket" not in match_events.columns:
    st.error("The minute_bucket column does not exist. Regenerate test_predictions.parquet (export cell).")
    st.stop()

timeline = (
    match_events.groupby(["minute_bucket"])["action_value_dashboard"]
    .sum()
    .reset_index()
    .sort_values("minute_bucket")
)
timeline["cum_value"] = timeline["action_value_dashboard"].cumsum()

c1, c2 = st.columns(2)
with c1:
    st.pyplot(
        plot_timeline(
            timeline,
            "minute_bucket",
            "action_value_dashboard",
            "Value by 5-min bucket",
            "minute",
            "value",
        )
    )

with c2:
    st.pyplot(
        plot_timeline(
            timeline,
            "minute_bucket",
            "cum_value",
            "Cumulative value",
            "minute",
            "cum_value",
        )
    )

# =========================
# TOP ACTIONS
# =========================
st.subheader("Top actions (match)")

top_actions = match_events.sort_values("action_value_dashboard", ascending=False).head(30)

cols = [
    "minute", "second",
    "team_name", "player_name", "type_name",
    "start_x", "start_y", "end_x", "end_y",
    "p_shot", "p_goal",
    "action_value_dashboard"
]

# alcune colonne potrebbero mancare a seconda della pipeline: filtriamo solo quelle presenti
cols = [c for c in cols if c in top_actions.columns]

st.dataframe(top_actions[cols], use_container_width=True)

st.caption(
    "Tip: Use this table as a list of clips to review in video (most impactful actions)."
)
