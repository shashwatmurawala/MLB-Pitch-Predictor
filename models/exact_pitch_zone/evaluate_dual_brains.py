import sys
import pandas as pd
import numpy as np
import joblib
import json
import re

if len(sys.argv) < 2:
    print("Usage: python evaluate_dual_brains.py <path_to_test_csv>")
    sys.exit(1)

test_file = sys.argv[1]
print(f"Loading testing data from {test_file}...")

df = pd.read_csv(test_file)
df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
df = df.dropna(subset=['pitch_name', 'zone'])

# Extract batter names
batter_names = {}
event_rows = df.dropna(subset=['events'])
for _, row in event_rows.iterrows():
    desc = str(row['des'])
    match = re.match(r'^([A-Z][a-z]+(?:\s(?:de\s|De\s|La\s|Jr\.|Sr\.)?[A-Z][a-z]+)+)', desc)
    if match and row['batter'] not in batter_names:
        batter_names[int(row['batter'])] = match.group(1)

df = df.copy()
df['batter_name'] = df['batter'].map(lambda x: batter_names.get(int(x), f"Batter #{x}"))

# Engineering
df['on_1b'] = df['on_1b'].notna().astype(int)
df['on_2b'] = df['on_2b'].notna().astype(int)
df['on_3b'] = df['on_3b'].notna().astype(int)

df['home_score'] = df['home_score'].fillna(0).astype(int)
df['away_score'] = df['away_score'].fillna(0).astype(int)

df['prev_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(1).fillna('None')
df['prev_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(1).fillna(0).astype(int)
df['prev2_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(2).fillna('None')
df['prev2_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(2).fillna(0).astype(int)

df['stand'] = df['stand'].fillna('Unknown')
df['p_throws'] = df['p_throws'].fillna('Unknown')
df['inning_topbot'] = df['inning_topbot'].fillna('Unknown')
df['home_team'] = df['home_team'].fillna('Unknown')
df['away_team'] = df['away_team'].fillna('Unknown')

df['zone_int'] = df['zone'].astype(int)
df['pitch_zone'] = df['pitch_name'] + ' | Zone ' + df['zone_int'].astype(str)

features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'home_score', 'away_score',
    'stand', 'p_throws', 'inning_topbot', 'prev_pitch_name',
    'prev_zone', 'prev2_pitch_name', 'prev2_zone',
    'player_name', 'home_team', 'away_team', 'batter_name'
]

X = df[features].copy()
y = df['pitch_zone'].copy()

X['player_name'] = X['player_name'].astype(str)
X['batter_name'] = X['batter_name'].astype(str)

categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'prev_pitch_name', 'prev2_pitch_name',
                    'player_name', 'home_team', 'away_team', 'batter_name']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Load artifacts
print("Loading Dual Brain artifacts...")
global_clf = joblib.load('rf_combined_predictor.pkl')
pitcher_brains = joblib.load('pitcher_brains.pkl')
batter_brains = joblib.load('batter_brains.pkl')
target_encoder = joblib.load('target_encoder_combined.pkl')
model_features = joblib.load('model_features.pkl')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

with open('pitcher_repertoires.json') as f:
    pitcher_repertoires = json.load(f)

# Align columns
X_aligned = pd.DataFrame(0, index=X.index, columns=model_features)
common_cols = list(set(X_encoded.columns).intersection(set(model_features)))
X_aligned[common_cols] = X_encoded[common_cols]

valid_idx = y.isin(target_encoder.classes_)
X_eval = X_aligned[valid_idx]
y_eval = y[valid_idx]
y_true = target_encoder.transform(y_eval)

pitcher_names = X['player_name'][valid_idx].values
batter_names = X['batter_name'][valid_idx].values

total = len(y_eval)
print(f"Evaluating Dual Brain Ensembling on {total} valid pitches...")

all_pitches = set()
for pitches in pitcher_repertoires.values():
    all_pitches.update(pitches)

class_to_pitch = {}
num_classes = len(target_encoder.classes_)
for j in range(num_classes):
    class_to_pitch[j] = target_encoder.inverse_transform([j])[0].split(' | Zone ')[0]

top1_correct = 0
top3_correct = 0
top5_correct = 0

# Helper to align local probs to global arr
def get_aligned_probs(local_model, X_slice):
    aligned = np.zeros((X_slice.shape[0], num_classes))
    local_probs = local_model.predict_proba(X_slice)
    if not isinstance(local_probs, np.ndarray):
        local_probs = np.array(local_probs)
    for i, c in enumerate(local_model.classes_):
        aligned[:, c] = local_probs[:, i]
    return aligned

X_eval_vals = X_eval.values

# Initialize with global fallback probabilities
print("Calculating Global Fallback Probabilities...")
p_probs = get_aligned_probs(global_clf, X_eval_vals)
b_probs = p_probs.copy()

print("Calculating Pitcher Brain Probabilities...")
unique_pitchers = np.unique(pitcher_names)
for pname in unique_pitchers:
    if pname in pitcher_brains:
        idx = (pitcher_names == pname)
        p_probs[idx, :] = get_aligned_probs(pitcher_brains[pname], X_eval_vals[idx])

print("Calculating Batter Brain Probabilities...")
unique_batters = np.unique(batter_names)
for bname in unique_batters:
    if bname in batter_brains:
        idx = (batter_names == bname)
        b_probs[idx, :] = get_aligned_probs(batter_brains[bname], X_eval_vals[idx])

print("Blending Dual Brain Probabilities...")
from copy import deepcopy
row_probs = (0.6 * p_probs) + (0.4 * b_probs)

print("Applying Repertoire Masking and Scoring...")
top1_correct = 0
top3_correct = 0
top5_correct = 0

for i in range(total):
    pname = pitcher_names[i]
    allowed_pitches = set(pitcher_repertoires.get(pname, all_pitches))
    
    probs = row_probs[i].copy()
    for j in range(num_classes):
        if class_to_pitch[j] not in allowed_pitches:
            probs[j] = 0.0
            
    ss = np.sum(probs)
    if ss > 0:
        probs = probs / ss
        
    top_n = probs.argsort()[::-1]
    
    true_class = y_true[i]
    if true_class == top_n[0]: top1_correct += 1
    if true_class in top_n[:3]: top3_correct += 1
    if true_class in top_n[:5]: top5_correct += 1

print("\n--- Dual Brain Evaluation Results ---")
print(f"Total Test Pitches:   {total:,}")
print(f"Exact Match (Top 1):  {top1_correct / total * 100:.2f}%")
print(f"In Top 3 Guesses:     {top3_correct / total * 100:.2f}%")
print(f"In Top 5 Guesses:     {top5_correct / total * 100:.2f}%")
