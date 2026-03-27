import pandas as pd
import joblib
import argparse
import sys

def predict_pitch(args):
    try:
        clf_pitch = joblib.load('rf_pitch_predictor.pkl')
        target_encoder_pitch = joblib.load('target_encoder_pitch.pkl')
        
        clf_zone = joblib.load('rf_zone_predictor.pkl')
        target_encoder_zone = joblib.load('target_encoder_zone.pkl')
        
        model_features = joblib.load('model_features.pkl')
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("Please ensure you have run train_model.py first.")
        sys.exit(1)

    # Create a DataFrame from the input arguments for a single observation
    input_data = {
        'balls': [args.balls],
        'strikes': [args.strikes],
        'outs_when_up': [args.outs],
        'inning': [args.inning],
        'on_1b': [1 if args.on_1b else 0],
        'on_2b': [1 if args.on_2b else 0],
        'on_3b': [1 if args.on_3b else 0],
        'score_diff': [args.score_diff],
        'stand': [args.stand],
        'p_throws': [args.p_throws],
        'inning_topbot': [args.inning_topbot],
        'prev_pitch_name': [args.prev_pitch],
        'player_name': [args.pitcher_name]
    }
    
    df = pd.DataFrame(input_data)
    
    # Apply One-Hot Encoding
    categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'prev_pitch_name', 'player_name']
    X_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Align features with the model training features, filling missing with 0
    # Any columns from training not present in the new df will be added with 0
    X_aligned = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in X_encoded.columns:
            X_aligned[col] = X_encoded[col]
        else:
            X_aligned[col] = 0
            
    # Predict Pitch Type
    pred_encoded_p = clf_pitch.predict(X_aligned)
    pred_probabilities_p = clf_pitch.predict_proba(X_aligned)
    predicted_pitch = target_encoder_pitch.inverse_transform(pred_encoded_p)[0]
    
    # Predict Zone
    pred_encoded_z = clf_zone.predict(X_aligned)
    pred_probabilities_z = clf_zone.predict_proba(X_aligned)
    # Zone may be numeric floats, converting to int if necessary
    predicted_zone = target_encoder_zone.inverse_transform(pred_encoded_z)[0]
    
    print("\n--- Pitch Prediction ---")
    print(f"Predicted Pitch Type: {predicted_pitch}")
    print(f"Predicted Location (Zone): {int(predicted_zone)}\n")
    
    # Show top 3 likely pitches
    probs_p = pred_probabilities_p[0]
    top_3_indices_p = probs_p.argsort()[-3:][::-1]
    
    print("Top 3 Pitch Type Probabilities:")
    for idx in top_3_indices_p:
        pitch_name = target_encoder_pitch.inverse_transform([idx])[0]
        print(f"  {pitch_name}: {probs_p[idx]:.2%}")
        
    # Show top 3 likely zones
    probs_z = pred_probabilities_z[0]
    top_3_indices_z = probs_z.argsort()[-3:][::-1]
    
    print("\nTop 3 Location (Zone) Probabilities:")
    for idx in top_3_indices_z:
        zone_val = target_encoder_zone.inverse_transform([idx])[0]
        print(f"  Zone {int(zone_val)}: {probs_z[idx]:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the next MLB pitch based on game state.')
    parser.add_argument('--balls', type=int, default=0, help='Current balls (0-4)')
    parser.add_argument('--strikes', type=int, default=0, help='Current strikes (0-3)')
    parser.add_argument('--outs', type=int, default=0, help='Current outs (0-3)')
    parser.add_argument('--inning', type=int, default=1, help='Inning number')
    parser.add_argument('--on_1b', action='store_true', help='Runner on first base')
    parser.add_argument('--on_2b', action='store_true', help='Runner on second base')
    parser.add_argument('--on_3b', action='store_true', help='Runner on third base')
    parser.add_argument('--score_diff', type=int, default=0, help='Fielding team score minus Batting team score')
    parser.add_argument('--stand', type=str, default='R', choices=['L', 'R', 'Unknown'], help='Batter stand (L/R)')
    parser.add_argument('--p_throws', type=str, default='R', choices=['L', 'R', 'Unknown'], help='Pitcher throws (L/R)')
    parser.add_argument('--inning_topbot', type=str, default='Top', choices=['Top', 'Bot', 'Unknown'], help='Top or Bottom of inning')
    parser.add_argument('--prev_pitch', type=str, default='None', help='Name of the previous pitch (e.g. 4-Seam Fastball)')
    parser.add_argument('--pitcher_name', type=str, default='Unknown', help='Player name (e.g., "Ashcraft, Graham")')
    
    args = parser.parse_args()
    predict_pitch(args)
