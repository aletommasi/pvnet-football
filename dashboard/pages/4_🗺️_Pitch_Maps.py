import streamlit as st
import matplotlib.pyplot as plt

from utils.io import load_parquet, assert_artifacts_exist
from components.filters import select_team
from components.pitch import draw_pitch, plot_action_arrows, heatmap_value

REQUIRED = ["test_predictions.parquet"]
VALUE_COL = "action_value_dashboard"

st.header("🗺️ Pitch Maps")

missing = assert_artifacts_exist(REQUIRED)
if missing:
    st.error(f"These files are missing from artifacts/: {missing}")
    st.stop()

events = load_parquet("test_predictions.parquet")

# =========================
# FILTER FLOW:
# Team -> Match -> Player (optional)
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

# Keep only actions that contribute to the dashboard (can be negative too)
match_actions = match_events[match_events[VALUE_COL] != 0.0].copy()

if len(match_actions) == 0:
    st.warning("No non-zero action values found for this team/match selection.")
    st.stop()

# =========================
# CONTROLS (NEW)
# =========================
st.subheader("Controls")

cA, cB, cC, cD = st.columns([1, 1, 1, 1])

with cA:
    show_mode_ui = st.selectbox("Show actions", ["Positive only", "Negative only", "Both"], index=0)

with cB:
    min_abs_value = st.slider("Minimum |value|", 0.0, 0.30, 0.05, 0.01)

with cC:
    top_n = st.slider("Top N arrows", 5, 60, 20, 5)

with cD:
    heat_stat = st.selectbox("Heatmap statistic", ["mean", "sum"], index=0)

mode_map = {
    "Positive only": "positive",
    "Negative only": "negative",
    "Both": "both",
}
mode = mode_map[show_mode_ui]

# Data for arrows (need end_x/end_y)
df_arrows = match_actions.copy()

# Filter by sign + threshold
if mode == "positive":
    df_arrows = df_arrows[df_arrows[VALUE_COL] > 0]
elif mode == "negative":
    df_arrows = df_arrows[df_arrows[VALUE_COL] < 0]

df_arrows = df_arrows[df_arrows[VALUE_COL].abs() >= float(min_abs_value)]
df_arrows = df_arrows.reindex(df_arrows[VALUE_COL].abs().sort_values(ascending=False).index).head(int(top_n))

# Data for heatmap (start location only, can ignore end_x/end_y)
# Make it "not banal": default is positive-only and thresholded
min_abs_value_heat = st.slider("Heatmap min |value|", 0.0, 0.30, 0.03, 0.01)

df_heat = match_actions.copy()
if mode == "positive":
    df_heat = df_heat[df_heat[VALUE_COL] > 0]
elif mode == "negative":
    df_heat = df_heat[df_heat[VALUE_COL] < 0]

df_heat = df_heat[df_heat[VALUE_COL].abs() >= float(min_abs_value_heat)]

# =========================
# PITCH MAPS
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Heatmap value (start location)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_pitch(ax)

    im = heatmap_value(
        ax,
        df_heat,
        value_col=VALUE_COL,
        bins=(12, 8),
        mode="both" if mode == "both" else mode,
        min_abs_value=0.0,  # already applied
        statistic=heat_stat,
    )

    if im is None:
        st.info("No events available for heatmap with the current filters.")
    else:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

with c2:
    st.subheader("Top arrows (highest value actions)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_pitch(ax)

    if len(df_arrows) == 0:
        st.info("No actions available for arrows with the current filters.")
    else:
        plot_action_arrows(
            ax,
            df_arrows,
            value_col=VALUE_COL,
            max_arrows=int(top_n),
            mode="both",  # df_arrows already filtered
            min_abs_value=0.0,  # already filtered
        )
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

# Apply same filtering logic for player view
df_player = player_actions.copy()
if mode == "positive":
    df_player = df_player[df_player[VALUE_COL] > 0]
elif mode == "negative":
    df_player = df_player[df_player[VALUE_COL] < 0]
df_player = df_player[df_player[VALUE_COL].abs() >= float(min_abs_value)]
df_player = df_player.reindex(df_player[VALUE_COL].abs().sort_values(ascending=False).index).head(int(top_n))

fig = plt.figure()
ax = fig.add_subplot(111)
draw_pitch(ax)

if len(df_player) == 0:
    st.info("No player actions match the current filters.")
else:
    plot_action_arrows(
        ax,
        df_player,
        value_col=VALUE_COL,
        max_arrows=int(top_n),
        mode="both",
        min_abs_value=0.0,
    )
    st.pyplot(fig)

st.caption(
    "Tip: reduce clutter using the sliders (Minimum |value| and Top N). "
    "Green arrows create attacking value; red arrows destroy it. "
    "Use the heatmap to see WHERE value is generated (origin of danger), not just where attacks end."
)
