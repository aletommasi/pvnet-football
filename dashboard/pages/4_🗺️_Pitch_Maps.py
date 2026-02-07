import streamlit as st
import matplotlib.pyplot as plt
from utils.io import load_parquet, assert_artifacts_exist
from components.filters import select_team
from components.pitch import draw_pitch, plot_action_arrows, heatmap_value

REQUIRED = ["test_predictions.parquet"]

st.header("🗺️ Pitch Maps")

missing = assert_artifacts_exist(REQUIRED)
if missing:
    st.error(f"These files are missing from artifacts/: {missing}")
    st.stop()

events = load_parquet("test_predictions.parquet")

# =========================
# NEW FILTER FLOW:
# Team -> Match (of that team) -> Player (optional)
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

# Keep only actions that contribute to the dashboard
match_actions = match_events[match_events["action_value_dashboard"] != 0.0].copy()

# =========================
# PITCH MAPS
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Heatmap value (start location)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_pitch(ax)
    im = heatmap_value(ax, match_actions, value_col="action_value_dashboard", bins=(12, 8))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

with c2:
    st.subheader("Top arrows (highest value actions)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_pitch(ax)
    plot_action_arrows(ax, match_actions, value_col="action_value_dashboard", max_arrows=250)
    st.pyplot(fig)

# =========================
# PLAYER FILTER (optional)
# =========================
st.subheader("Filter by player (optional)")

players = sorted(match_actions["player_name"].dropna().unique().tolist())
player = st.selectbox("Player", ["All"] + players)

player_actions = match_actions.copy()
if player != "All":
    player_actions = player_actions[player_actions["player_name"] == player]

fig = plt.figure()
ax = fig.add_subplot(111)
draw_pitch(ax)
plot_action_arrows(ax, player_actions, value_col="action_value_dashboard", max_arrows=300)
st.pyplot(fig)

st.caption(
    "Tip: Use these pitch maps to identify areas of strength and weakness for the team and individual players. For example, you might find that a player creates high value actions from the right wing, suggesting a potential area to exploit in video analysis."
)
