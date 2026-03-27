import sys
import pandas as pd
import numpy as np
import joblib
import json
import re

if len(sys.argv) < 2:
    print("Usage: python evaluate_model.py <path_to_test_csv>")
    sys.exit(1)

test_file = sys.argv[1]
print(f"Loading testing data from {test_file}...")

df = pd.read_csv(test_file)
df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
df = df.dropna(subset=['pitch_name', 'zone'])

# ── Extract batter names ──
batter_names = {}
event_rows = df.dropna(subset=['events'])
for _, row in event_rows.iterrows():
    desc = str(row['des'])
    match = re.match(r'^([A-Z][a-z]+(?:\s(?:de\s|De\s|La\s|Jr\.|Sr\.)?[A-Z][a-z]+)+)', desc)
    if match and row['batter'] not in batter_names:
        batter_names[int(row['batter'])] = match.group(1)

# To avoid setting values on slice warning, copy df
df = df.copy()
df['batter_name'] = df['batter'].map(lambda x: batter_names.get(int(x), f"Batter #{x}"))

# ── Feature Engineering ──
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
print("Running get_dummies...")
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Load artifacts
print("Loading model artifacts...")
clf = joblib.load('rf_combined_predictor.pkl')
target_encoder = joblib.load('target_encoder_combined.pkl')
model_features = joblib.load('model_features.pkl')

with open('pitcher_repertoires.json') as f:
    pitcher_repertoires = json.load(f)

# Align columns
print("Aligning features...")
X_aligned = pd.DataFrame(0, index=X.index, columns=model_features)
common_cols = list(set(X_encoded.columns).intersection(set(model_features)))
X_aligned[common_cols] = X_encoded[common_cols]

# Filter rows that have targets existing in the encoder
known_targets = set(target_encoder.classes_)
valid_idx = y.isin(known_targets)

if valid_idx.sum() == 0:
    print("No valid targets in test data to evaluate.")
    sys.exit(0)

print(f"Evaluating on {valid_idx.sum()} valid pitches...")
X_eval = X_aligned[valid_idx]
y_eval = y[valid_idx]
y_true = target_encoder.transform(y_eval)
pitcher_names = X['player_name'][valid_idx].values

print("Predicting probabilities... (This may take a moment)")
probs = clf.predict_proba(X_eval)

print("Applying repertoire masking and calculating metrics...")
all_pitches = set()
for pitches in pitcher_repertoires.values():
    all_pitches.update(pitches)

top1_correct = 0
top3_correct = 0
top5_correct = 0
total = len(y_eval)

# Pre-cache class labels for lightning fast lookup
class_to_pitch = {}
for j in range(len(target_encoder.classes_)):
    class_to_pitch[j] = target_encoder.inverse_transform([j])[0].split(' | Zone ')[0]

for i in range(total):
    pname = pitcher_names[i]
    allowed_pitches = set(pitcher_repertoires.get(pname, all_pitches))
    
    row_probs = probs[i].copy()
    
    for j in range(len(row_probs)):
        if class_to_pitch[j] not in allowed_pitches:
            row_probs[j] = 0.0
            
    if np.sum(row_probs) > 0:
        row_probs = row_probs / np.sum(row_probs)
    
    top_n = row_probs.argsort()[::-1]
    
    true_class = y_true[i]
    if true_class == top_n[0]: top1_correct += 1
    if true_class in top_n[:3]: top3_correct += 1
    if true_class in top_n[:5]: top5_correct += 1

print("\n--- Evaluation Results ---")
print(f"Total Test Pitches:   {total:,}")
print(f"Exact Match (Top 1):  {top1_correct / total * 100:.2f}%")
print(f"In Top 3 Guesses:     {top3_correct / total * 100:.2f}%")
print(f"In Top 5 Guesses:     {top5_correct / total * 100:.2f}%")
