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

## Input Data Type

PVNet operates on **event data**, not video footage.

Each match is represented as a sequence of events such as passes, carries, shots, recoveries, and defensive actions, including spatial and temporal information.

The model does not process video directly; instead, video analysis tools or data providers are used to generate event datasets.

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

For each event in a possession sequence, the model predicts:

- probability that a **shot occurs within the next K events**
- probability that a **goal occurs within the next K events**

where **K = 10 events** in the current configuration.

This allows the model to estimate how dangerous the current possession state is.

Labels are generated automatically by scanning future events within the same possession and checking whether a shot or goal occurs.

---

## Action Value Formula

The value of an action is defined as the change in attacking potential:

Action Value = (P_shot_after − P_shot_before)
+ w_goal × (P_goal_after − P_goal_before)

where:

- probabilities are predicted by the model
- w_goal = 5.0 weights goal probability more heavily than shot probability

Positive values indicate actions that increase attacking danger, while negative values indicate loss of attacking potential.

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

## Current Limitation: Dashboard Uses Test Predictions

The dashboard currently visualizes predictions computed on the held-out test dataset used for model evaluation.

This allows consistent visualization and benchmarking but does not yet include automatic ingestion of new matches.

However, the trained model can be applied to new matches as long as event data are available and processed through the same preprocessing pipeline.

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

## Applying PVNet to New Matches

PVNet can analyze new matches without retraining the model.

Required steps:

1. Obtain event data for the match.
2. Apply the same preprocessing and feature engineering steps.
3. Load trained model and scaler.
4. Compute probabilities and action values.
5. Export predictions in dashboard format.

In professional environments, these event datasets are typically provided by commercial providers such as Wyscout, Opta, or StatsBomb.

---

## License

MIT License.

---

## Author

Football analytics portfolio project combining machine learning and match analysis.
