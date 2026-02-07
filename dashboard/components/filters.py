import streamlit as st

def select_team(df, label="Team"):
    teams = sorted([t for t in df["team_name"].dropna().unique().tolist()])
    return st.selectbox(label, ["All"] + teams)

def select_match(df, label="Match ID"):
    matches = sorted(df["match_id"].dropna().unique().tolist())
    return st.selectbox(label, matches)

def min_actions_slider(default=50):
    return st.slider("Min actions", min_value=0, max_value=300, value=default, step=10)
