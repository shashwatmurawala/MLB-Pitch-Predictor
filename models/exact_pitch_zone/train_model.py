import pandas as pd
import numpy as np
import json
import re
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")
files = glob.glob('../../data/Teams/*.csv')
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
print(f"Initial data shape: {df.shape}")

# Drop rows missing targets
df = df.dropna(subset=['pitch_name', 'zone'])

# ── Extract batter names from 'des' column ──────────────────────────
batter_names = {}
event_rows = df.dropna(subset=['events'])
for _, row in event_rows.iterrows():
    desc = str(row['des'])
    match = re.match(r'^([A-Z][a-z]+(?:\s(?:de\s|De\s|La\s|Jr\.|Sr\.)?[A-Z][a-z]+)+)', desc)
    if match and row['batter'] not in batter_names:
        batter_names[int(row['batter'])] = match.group(1)

# Fill missing batter names with ID
for bid in df['batter'].unique():
    if int(bid) not in batter_names:
        batter_names[int(bid)] = f"Batter #{bid}"

print(f"Extracted {len(batter_names)} batter names")

# Create batter_name column
df['batter_name'] = df['batter'].map(lambda x: batter_names.get(int(x), f"Batter #{x}"))

# ── Feature Engineering ──────────────────────────────────────────────
df['on_1b'] = df['on_1b'].notna().astype(int)
df['on_2b'] = df['on_2b'].notna().astype(int)
df['on_3b'] = df['on_3b'].notna().astype(int)

df['home_score'] = df['home_score'].fillna(0).astype(int)
df['away_score'] = df['away_score'].fillna(0).astype(int)

df['prev_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(1)
df['prev_pitch_name'] = df['prev_pitch_name'].fillna('None')
df['prev_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(1).fillna(0).astype(int)
df['prev2_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(2).fillna('None')
df['prev2_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(2).fillna(0).astype(int)

df['stand'] = df['stand'].fillna('Unknown')
df['p_throws'] = df['p_throws'].fillna('Unknown')
df['inning_topbot'] = df['inning_topbot'].fillna('Unknown')
df['home_team'] = df['home_team'].fillna('Unknown')
df['away_team'] = df['away_team'].fillna('Unknown')

# Combined target
df['zone_int'] = df['zone'].astype(int)
df['pitch_zone'] = df['pitch_name'] + ' | Zone ' + df['zone_int'].astype(str)

# ── Features ─────────────────────────────────────────────────────────
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

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"Features shape after encoding: {X_encoded.shape}")
print(f"Number of combined classes: {len(target_encoder.classes_)}")

# ── Train / Test ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

print("Training RandomForest Classifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Combined Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ── Save artifacts ───────────────────────────────────────────────────
print("Saving model artifacts...")
joblib.dump(clf, 'rf_combined_predictor.pkl')
joblib.dump(target_encoder, 'target_encoder_combined.pkl')
joblib.dump(list(X_encoded.columns), 'model_features.pkl')

# Export lists for frontend
pitcher_list = sorted(df['player_name'].dropna().unique().tolist())
pitch_names = sorted(df['pitch_name'].dropna().unique().tolist())
teams = sorted(set(df['home_team'].dropna().unique().tolist() + df['away_team'].dropna().unique().tolist()))

# Export pitcher repertoires (which pitches each pitcher actually throws)
repertoires_series = df.groupby('player_name')['pitch_name'].unique()
pitcher_repertoires = {p: list(pitches) for p, pitches in repertoires_series.items()}

# Batter list: {id: name} for the frontend
batter_list = [{"id": int(bid), "name": bname} for bid, bname in sorted(batter_names.items(), key=lambda x: x[1])]

with open('pitcher_list.json', 'w') as f:
    json.dump(pitcher_list, f)
with open('pitch_names.json', 'w') as f:
    json.dump(pitch_names, f)
with open('team_list.json', 'w') as f:
    json.dump(teams, f)
with open('batter_list.json', 'w') as f:
    json.dump(batter_list, f)
with open('pitcher_repertoires.json', 'w') as f:
    json.dump(pitcher_repertoires, f)

print(f"Saved {len(pitcher_list)} pitchers, {len(batter_list)} batters, {len(teams)} teams.")
print("Done!")
