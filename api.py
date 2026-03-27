import json
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model artifacts ─────────────────────────────────────────────
clf = joblib.load('models/exact_pitch_zone/rf_combined_predictor.pkl')
target_encoder = joblib.load('models/exact_pitch_zone/target_encoder_combined.pkl')
model_features = joblib.load('models/exact_pitch_zone/model_features.pkl')

try:
    pitcher_brains = joblib.load('models/exact_pitch_zone/pitcher_brains.pkl')
    batter_brains = joblib.load('models/exact_pitch_zone/batter_brains.pkl')
except Exception as e:
    pitcher_brains = {}
    batter_brains = {}

with open('data/meta/pitcher_list.json') as f:
    pitcher_list = json.load(f)
with open('data/meta/pitch_names.json') as f:
    pitch_names = json.load(f)
with open('data/meta/team_list.json') as f:
    team_list = json.load(f)
with open('data/meta/batter_list.json') as f:
    batter_list = json.load(f)

try:
    with open('data/meta/pitcher_repertoires.json') as f:
        pitcher_repertoires = json.load(f)
except FileNotFoundError:
    pitcher_repertoires = {}

ZONE_LABELS = {
    1: "High-Inside",   2: "High-Middle",    3: "High-Outside",
    4: "Mid-Inside",    5: "Mid-Middle",     6: "Mid-Outside",
    7: "Low-Inside",    8: "Low-Middle",     9: "Low-Outside",
    11: "Inside (Ball)", 12: "Above (Ball)",
    13: "Outside (Ball)", 14: "Below (Ball)"
}


@app.route('/api/pitchers', methods=['GET'])
def get_pitchers():
    return jsonify(pitcher_list)


@app.route('/api/pitch_names', methods=['GET'])
def get_pitch_names():
    return jsonify(pitch_names)


@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify(team_list)


@app.route('/api/batters', methods=['GET'])
def get_batters():
    return jsonify(batter_list)


@app.route('/api/zone_labels', methods=['GET'])
def get_zone_labels():
    return jsonify(ZONE_LABELS)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = {
        'balls': [int(data.get('balls', 0))],
        'strikes': [int(data.get('strikes', 0))],
        'outs_when_up': [int(data.get('outs', 0))],
        'inning': [int(data.get('inning', 1))],
        'on_1b': [1 if data.get('on_1b') else 0],
        'on_2b': [1 if data.get('on_2b') else 0],
        'on_3b': [1 if data.get('on_3b') else 0],
        'home_score': [int(data.get('home_score', 0))],
        'away_score': [int(data.get('away_score', 0))],
        'stand': [data.get('stand', 'R')],
        'p_throws': [data.get('p_throws', 'R')],
        'inning_topbot': [data.get('inning_topbot', 'Top')],
        'prev_pitch_name': [data.get('prev_pitch', 'None')],
        'prev_zone': [int(data.get('prev_zone') or 0)],
        'prev2_pitch_name': [data.get('prev2_pitch', 'None')],
        'prev2_zone': [int(data.get('prev2_zone') or 0)],
        'player_name': [data.get('pitcher_name', 'Unknown')],
        'home_team': [data.get('home_team', 'Unknown')],
        'away_team': [data.get('away_team', 'Unknown')],
        'batter_name': [data.get('batter_name', 'Unknown')]
    }

    df = pd.DataFrame(input_data)

    categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'prev_pitch_name', 'prev2_pitch_name',
                        'player_name', 'home_team', 'away_team', 'batter_name']
    X_encoded = pd.get_dummies(df, columns=categorical_cols)

    X_aligned = pd.DataFrame(0, index=[0], columns=model_features)
    for col in X_encoded.columns:
        if col in X_aligned.columns:
            X_aligned[col] = X_encoded[col].values

    import numpy as np
    
    num_classes = len(target_encoder.classes_)
    def get_aligned_probs(local_model, X_df):
        aligned = np.zeros(num_classes)
        local_probs = local_model.predict_proba(X_df)[0]
        for i, c in enumerate(local_model.classes_):
            aligned[c] = local_probs[i]
        return aligned
        
    pitcher_name = input_data['player_name'][0]
    batter_name = input_data['batter_name'][0]
    
    if pitcher_name in pitcher_brains:
        p_prob = get_aligned_probs(pitcher_brains[pitcher_name], X_aligned)
    else:
        p_prob = get_aligned_probs(clf, X_aligned)
        
    if batter_name in batter_brains:
        b_prob = get_aligned_probs(batter_brains[batter_name], X_aligned)
    else:
        b_prob = get_aligned_probs(clf, X_aligned)
        
    probabilities = (0.6 * p_prob) + (0.4 * b_prob)
    
    # ── Pitch Repertoire Masking ──
    pitcher_name = input_data['player_name'][0]
    allowed_pitches = set(pitcher_repertoires.get(pitcher_name, pitch_names))
    
    masked_probs = np.zeros_like(probabilities)
    for idx in range(len(probabilities)):
        label = target_encoder.inverse_transform([idx])[0]
        pitch_type = label.split(' | Zone ')[0]
        if pitch_type in allowed_pitches:
            masked_probs[idx] = probabilities[idx]
            
    total_prob = np.sum(masked_probs)
    if total_prob > 0:
        masked_probs = masked_probs / total_prob
    else:
        masked_probs = probabilities
        
    top_indices = masked_probs.argsort()[-5:][::-1]

    results = []
    for idx in top_indices:
        if masked_probs[idx] == 0:
            continue
            
        label = target_encoder.inverse_transform([idx])[0]
        parts = label.split(' | Zone ')
        pitch_type = parts[0]
        zone = int(parts[1])
        results.append({
            'pitch_type': pitch_type,
            'zone': zone,
            'zone_label': ZONE_LABELS.get(zone, 'Unknown'),
            'probability': round(float(masked_probs[idx]) * 100, 2),
            'label': label
        })

    return jsonify({'predictions': results})


if __name__ == '__main__':
    print("Starting MLB Pitch Predictor API on http://localhost:5001")
    app.run(debug=True, port=5001)
