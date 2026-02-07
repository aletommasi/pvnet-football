import streamlit as st
import pandas as pd
from utils.io import load_json, assert_artifacts_exist
from components.charts import plot_calibration

REQUIRED = ["metrics.json"]

st.header("📈 Overview")

missing = assert_artifacts_exist(REQUIRED)
if missing:
    st.error(f"These files are missing from artifacts/: {missing}")
    st.stop()

m = load_json("metrics.json")

shot = m["metrics"]["shot"]
goal = m["metrics"]["goal"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Shot task")
    st.dataframe(pd.DataFrame([shot]), use_container_width=True)
    st.pyplot(plot_calibration(m["calibration"]["shot"], "Calibration — Shot"))

with col2:
    st.subheader("Goal task")
    st.dataframe(pd.DataFrame([goal]), use_container_width=True)
    st.pyplot(plot_calibration(m["calibration"]["goal"], "Calibration — Goal"))

st.markdown("---")
st.subheader("Config & feature columns")
st.json({
    "config": m["config"],
    "feature_cols": m.get("feature_cols", [])
})
