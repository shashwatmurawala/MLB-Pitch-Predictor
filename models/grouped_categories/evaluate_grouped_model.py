import sys
import pandas as pd
import numpy as np
import joblib
import json
import re
from sklearn.metrics import accuracy_score

if len(sys.argv) < 2:
    print("Usage: python evaluate_grouped_model.py <path_to_test_csv>")
    sys.exit(1)

test_file = sys.argv[1]
print(f"Loading testing data from {test_file}...")

df = pd.read_csv(test_file)
df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
df = df.dropna(subset=['pitch_name'])

PITCH_GROUPS = {
    '4-Seam Fastball': 'Fastball',
    'Sinker': 'Fastball',
    'Cutter': 'Fastball',
    'Fastball': 'Fastball',
    'Slider': 'Breaking',
    'Curveball': 'Breaking',
    'Knuckle Curve': 'Breaking',
    'Slurve': 'Breaking',
    'Sweeper': 'Breaking',
    'Slow Curve': 'Breaking',
    'Changeup': 'Offspeed',
    'Split-Finger': 'Offspeed',
    'Knuckleball': 'Offspeed',
    'Forkball': 'Offspeed',
    'Screwball': 'Offspeed'
}
df['pitch_group'] = df['pitch_name'].map(PITCH_GROUPS)
df = df.dropna(subset=['pitch_group'])

batter_names = {}
event_rows = df.dropna(subset=['events'])
for _, row in event_rows.iterrows():
    desc = str(row['des'])
    match = re.match(r'^([A-Z][a-z]+(?:\s(?:de\s|De\s|La\s|Jr\.|Sr\.)?[A-Z][a-z]+)+)', desc)
    if match and row['batter'] not in batter_names:
        batter_names[int(row['batter'])] = match.group(1)

df = df.copy()
df['batter_name'] = df['batter'].map(lambda x: batter_names.get(int(x), f"Batter #{x}"))

df['on_1b'] = df['on_1b'].notna().astype(int)
df['on_2b'] = df['on_2b'].notna().astype(int)
df['on_3b'] = df['on_3b'].notna().astype(int)
df['home_score'] = df['home_score'].fillna(0).astype(int)
df['away_score'] = df['away_score'].fillna(0).astype(int)
df['prev_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(1).fillna('None')
df['stand'] = df['stand'].fillna('Unknown')
df['p_throws'] = df['p_throws'].fillna('Unknown')
df['inning_topbot'] = df['inning_topbot'].fillna('Unknown')
df['home_team'] = df['home_team'].fillna('Unknown')
df['away_team'] = df['away_team'].fillna('Unknown')

features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'home_score', 'away_score',
    'stand', 'p_throws', 'inning_topbot', 'prev_pitch_name',
    'player_name', 'home_team', 'away_team', 'batter_name'
]

X = df[features].copy()
y = df['pitch_group'].copy()
X['player_name'] = X['player_name'].astype(str)
X['batter_name'] = X['batter_name'].astype(str)

categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'prev_pitch_name',
                    'player_name', 'home_team', 'away_team', 'batter_name']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("Loading model artifacts...")
clf = joblib.load('rf_grouped_predictor.pkl')
target_encoder = joblib.load('target_encoder_grouped.pkl')
model_features = joblib.load('model_features_grouped.pkl')

print("Aligning features...")
X_aligned = pd.DataFrame(0, index=X.index, columns=model_features)
common_cols = list(set(X_encoded.columns).intersection(set(model_features)))
X_aligned[common_cols] = X_encoded[common_cols]

valid_idx = y.isin(target_encoder.classes_)
X_eval = X_aligned[valid_idx]
y_eval = y[valid_idx]
y_true = target_encoder.transform(y_eval)

print(f"Evaluating {len(y_eval)} pitches...")

# Because it's only predicting 3 classes, masking by pitcher isn't necessary for this particular evaluation, 
# although we could mask "Offspeed" if a pitcher only throws fastballs/breaking balls. 
# For true testing accuracy on groups, we just take the raw probabilities.
y_pred = clf.predict(X_eval)
acc = accuracy_score(y_true, y_pred)

print("\n--- Grouped Model Evaluation Results ---")
print(f"Accuracy (Exact Match out of 3): {acc * 100:.2f}%")

# Let's see category breakdowns
for category in target_encoder.classes_:
    cat_idx = (y_eval == category)
    if cat_idx.sum() > 0:
        cat_acc = accuracy_score(y_true[cat_idx], y_pred[cat_idx])
        print(f"Accuracy for {category}: {cat_acc * 100:.2f}% (Count: {cat_idx.sum()})")
