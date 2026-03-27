import pandas as pd
import numpy as np
import joblib
import json
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")
files = glob.glob('../../data/Teams/*.csv')
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
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

# Extract batter names
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

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"Features shape after encoding: {X_encoded.shape}")
print(f"Number of combined classes: {len(target_encoder.classes_)} {target_encoder.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

print("Training Grouped Model...")
clf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20, 
    class_weight='balanced',
    min_samples_leaf=5,
    random_state=42, 
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Internal Validation Accuracy: {accuracy_score(y_test, y_pred):.4f}")

joblib.dump(clf, 'rf_grouped_predictor.pkl')
joblib.dump(target_encoder, 'target_encoder_grouped.pkl')
joblib.dump(list(X_encoded.columns), 'model_features_grouped.pkl')
print("Model saved as rf_grouped_predictor.pkl")
