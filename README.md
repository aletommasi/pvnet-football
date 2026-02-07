# ⚽ PVNet — Possession Value Network for Football Match Analysis

PVNet (Possession Value Network) is a football analytics project that estimates how each action in a match changes the probability of creating a shot or scoring a goal.

The project combines:

- event data processing
- machine learning modeling
- action value estimation
- player and team evaluation
- interactive visual analytics dashboard

PVNet aims to demonstrate a full end-to-end football analytics pipeline suitable for match analysis and portfolio use.

---

## Project Motivation

Traditional football statistics focus on final outcomes:

- goals
- assists
- shots
- possession %
- passes completed

However, most match impact happens **before** these final events.

A pass breaking defensive lines, a carry progressing play, or a recovery in a dangerous area can drastically change the probability of scoring without appearing in classic statistics.

PVNet estimates the **value added (or lost)** by every action in possession sequences.

---

## What PVNet Produces

For each event in a match, the model estimates:

- probability of a shot occurring soon
- probability of a goal occurring soon
- change in attacking value caused by the action

From these predictions, PVNet generates:

- action value metrics
- player rankings
- team contribution metrics
- match timelines
- pitch value maps

---

## Dataset

The model is trained using **StatsBomb Open Data**, which provides event-level football match data.

Each match contains sequences of events such as:

- passes
- carries
- shots
- recoveries
- duels
- defensive actions

Each event includes:

- event type
- player and team
- location on the pitch
- event outcome
- timestamps
- possession sequences

The training dataset used in this project includes multiple international competitions such as:

- FIFA World Cup
- Copa América
- Africa Cup of Nations
- UEFA Champions League

These competitions provide high-quality matches and diverse tactical styles.

---

## Feature Engineering

For each event, the system extracts features such as:

- start and end location
- distance to goal
- angle to goal
- movement direction and progression
- event type indicators
- success/failure of actions
- match time context

These features describe the state of possession at each action.

---

## Model Objective

The neural network predicts:

P(shot in next K events)
P(goal in next K events)


where K is a configurable future window.

Thus, the model learns patterns indicating when possession sequences become dangerous.

---

## Action Value Formula

For each action, PVNet computes:

Action Value = ΔP(shot) + w_goal * ΔP(goal)

where:

ΔP(shot) = P_after_shot − P_before_shot
ΔP(goal) = P_after_goal − P_before_goal


and `w_goal` weights goal probability higher than shot probability.

Positive values mean the action increases attacking potential.

Negative values mean possession becomes less dangerous.

---

**Configuration used in this project**
- Future window: **K = 10** events  
- Goal weighting: **w_goal = 5.0**

---

## Dashboard Usage

The Streamlit dashboard allows analysts to explore:

### Match Analysis
- value creation timeline
- key momentum changes
- impactful actions

### Pitch Maps
- zones generating attacking value
- trajectories of high-value actions

### Player Rankings
- contribution per action
- positive vs negative contributions

### Overview
- model performance and calibration

---

## Current Limitation: Test Set Dashboard

Currently, the dashboard displays predictions generated on the **test dataset** used during model evaluation.

This means the dashboard is static and does not yet process new matches automatically.

However, the trained model can be applied to new match data to generate new predictions and update the dashboard.

---

## Example Real Usage Scenario

A practical scenario:

> Tomorrow we play against Milan. Let's analyze Milan vs Roma from their previous match.

The analyst could:

- see which Milan players created most value
- identify dangerous zones
- detect build-up patterns
- study possession weaknesses

PVNet helps understand **how danger is created**, not just final outcomes.

---

## Project Structure

```
pvnet-football/
├── artifacts/ # Generated artifacts consumed by the dashboard
│ ├── model.pth # Trained PyTorch model weights + metadata
│ ├── scaler.joblib # StandardScaler fitted on training set
│ ├── metrics.json # Test metrics + calibration + training config
│ ├── test_predictions.parquet # Event-level predictions and action values (dashboard input)
│ ├── player_ranking_outfield.parquet # Player ranking for outfield players
│ ├── player_ranking_goalkeepers.parquet # Goalkeeper-like ranking (proxy split)
│ └── team_match_ranking.parquet # Team value aggregated per match
│
├── dashboard/ # Streamlit dashboard
│ ├── app.py # Dashboard entry point + artifacts path selector
│ ├── pages/
│ │ ├── 1_Overview.py # Metrics & calibration plots from metrics.json
│ │ ├── 2_Player_Rankings.py # Outfield/GK rankings tables
│ │ ├── 3_Match_Analysis.py # Team→Match filter, timeline, top actions
│ │ └── 4_Pitch_Maps.py # Team→Match filter, heatmaps and arrows
│ ├── components/
│ │ ├── charts.py # Matplotlib charts (histograms, calibration, timelines)
│ │ ├── pitch.py # Pitch drawing + arrows + heatmap utilities
│ │ └── filters.py # Streamlit filter widgets (team selection, sliders)
│ └── utils/
│ ├── io.py # Parquet/JSON loading + artifacts checks (cached)
│ └── formatting.py # Small formatting helpers
│
├── notebooks/
│ └── PVNet_training.ipynb # End-to-end training in Google Colab
│
├── src/ # Training and data processing code
│ ├── config.py # Central configuration (K, w_goal concept, training params)
│ ├── data/
│ │ ├── load_statsbomb.py # Download StatsBomb open-data events
│ │ ├── preprocessing.py # Cleaning + feature engineering
│ │ ├── labeling.py # Future-event labels (shot/goal within K events)
│ │ └── split.py # Train/val/test split by match (anti-leakage)
│ ├── models/
│ │ └── pvnet.py # PVNet MLP architecture (multi-task)
│ └── training/
│ ├── train_loop.py # PyTorch training/eval loops
│ ├── evaluate.py # Metrics + calibration utilities
│ └── utils.py # Seed + device helpers
│
├── README.md
└── LICENSE
```

Artifacts store trained models and generated metrics used by the dashboard.

---

## Limitations

PVNet has several limitations:

- uses only event data (no tracking data)
- no tactical formation modeling
- simplified possession modeling
- limited defensive value modeling
- no sequential deep architectures
- dashboard currently limited to test dataset

Therefore, PVNet supports analysis but does not replace expert judgement.

---

### Applying PVNet to New Matches (future extension)

PVNet can be applied to unseen matches without retraining. The required steps are:

1. Load event data for the new match (same schema as the training source).
2. Run the **same preprocessing + feature engineering** pipeline (`basic_clean`, `build_features`).
3. Load the saved `scaler.joblib` and standardize features.
4. Load `model.pth` and compute `p_shot` and `p_goal`.
5. Compute action values using the same formula and export to `artifacts/test_predictions.parquet`.
6. Restart the dashboard to visualize the new match.

This pipeline is not automated yet in the current version, but it can be implemented as a `predict_match.py` script.

---

## License

MIT License.

---

## Author

Football analytics portfolio project combining machine learning and match analysis.
