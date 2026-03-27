import pandas as pd
import numpy as np
import joblib
import json
import glob
import re
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
files = glob.glob('../../data/Teams/*.csv')
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
df = df.dropna(subset=['pitch_name', 'zone'])

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

# ── Feature Engineering ──
batter_names = {}
event_rows = df.dropna(subset=['events'])
for _, row in event_rows.iterrows():
    desc = str(row['des'])
    match = re.match(r'^([A-Z][a-z]+(?:\s(?:de\s|De\s|La\s|Jr\.|Sr\.)?[A-Z][a-z]+)+)', desc)
    if match and row['batter'] not in batter_names:
        batter_names[int(row['batter'])] = match.group(1)

df['batter_name'] = df['batter'].map(lambda x: batter_names.get(int(x), f"Batter #{x}"))

df['on_1b'] = df['on_1b'].notna().astype(int)
df['on_2b'] = df['on_2b'].notna().astype(int)
df['on_3b'] = df['on_3b'].notna().astype(int)
df['home_score'] = df['home_score'].fillna(0).astype(int)
df['away_score'] = df['away_score'].fillna(0).astype(int)
df['stand'] = df['stand'].fillna('Unknown')
df['p_throws'] = df['p_throws'].fillna('Unknown')
df['inning_topbot'] = df['inning_topbot'].fillna('Unknown')
df['home_team'] = df['home_team'].fillna('Unknown')
df['away_team'] = df['away_team'].fillna('Unknown')

df['prev_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(1).fillna('None')
df['prev_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(1).fillna(0).astype(int)
df['prev2_pitch_name'] = df.groupby(['game_pk', 'at_bat_number'])['pitch_name'].shift(2).fillna('None')
df['prev2_zone'] = df.groupby(['game_pk', 'at_bat_number'])['zone'].shift(2).fillna(0).astype(int)

features = [
    'balls', 'strikes', 'outs_when_up', 'inning',
    'on_1b', 'on_2b', 'on_3b',
    'home_score', 'away_score',
    'stand', 'p_throws', 'inning_topbot', 'prev_pitch_name',
    'prev_zone', 'prev2_pitch_name', 'prev2_zone',
    'player_name', 'home_team', 'away_team', 'batter_name'
]

X = df[features].copy()
y = df['pitch_group'].copy()
X['player_name'] = X['player_name'].astype(str)
X['batter_name'] = X['batter_name'].astype(str)

categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'prev_pitch_name', 'prev2_pitch_name',
                    'player_name', 'home_team', 'away_team', 'batter_name']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Align columns to global grouped model
model_features = joblib.load('model_features_grouped.pkl')
target_encoder = joblib.load('target_encoder_grouped.pkl')

print("Aligning features...")
X_aligned = pd.DataFrame(0, index=X_encoded.index, columns=model_features)
common_cols = list(set(X_encoded.columns).intersection(set(model_features)))
X_aligned[common_cols] = X_encoded[common_cols]

valid_idx = y.isin(target_encoder.classes_)
X_final = X_aligned[valid_idx]
y_final = target_encoder.transform(y[valid_idx])

# Add original player names back as a series for grouping
pitcher_series = X['player_name'][valid_idx]
batter_series = X['batter_name'][valid_idx]

# ── Train Pitcher Grouped Brains ──
print("Counting pitches per pitcher...")
pitcher_counts = pitcher_series.value_counts()
valid_pitchers = pitcher_counts[pitcher_counts >= 200].index.tolist()

pitcher_brains = {}
print(f"Training {len(valid_pitchers)} Grouped Pitcher Brains...")
for i, pname in enumerate(valid_pitchers):
    if i > 0 and i % 50 == 0: print(f"  Trained {i}/{len(valid_pitchers)} pitchers...")
    idx = (pitcher_series == pname)
    
    clf = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    clf.fit(X_final[idx], y_final[idx])
    pitcher_brains[pname] = clf

print("Saving pitcher_brains_grouped.pkl...")
joblib.dump(pitcher_brains, 'pitcher_brains_grouped.pkl', compress=3)

# ── Train Batter Grouped Brains ──
print("Counting pitches per batter...")
batter_counts = batter_series.value_counts()
valid_batters = batter_counts[batter_counts >= 200].index.tolist()

batter_brains = {}
print(f"Training {len(valid_batters)} Grouped Batter Brains...")
for i, bname in enumerate(valid_batters):
    if i > 0 and i % 50 == 0: print(f"  Trained {i}/{len(valid_batters)} batters...")
    idx = (batter_series == bname)
    
    clf = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    clf.fit(X_final[idx], y_final[idx])
    batter_brains[bname] = clf

print("Saving batter_brains_grouped.pkl...")
joblib.dump(batter_brains, 'batter_brains_grouped.pkl', compress=3)

print("Dual Grouped Brains training complete!")
