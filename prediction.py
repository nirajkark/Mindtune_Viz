import pandas as pd
import numpy as np
import joblib
import time
from collections import deque

class MindTunePredictor:
    def __init__(self, model_path, scaler_path, window_size=5):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # --- NEW: Track start time to calculate session_time_sec ---
        self.start_time = time.time()
        
        self.label_map = {0: 'calm', 1: 'neutral', 2: 'stressed'}
        
        # EXACT column order from your training dataframe
        # Ensure 'session_time_sec' is included here in the exact position it was in X_train
        self.expected_features = self.model.feature_names_in_

    def _preprocess(self, raw_data):
        # 1. Calculate Session Time (The missing feature)
        current_time = time.time()
        session_time_sec = current_time - self.start_time
        
        # 2. Calculate Percentages
        bands = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']
        total_power = sum([raw_data.get(b, 0) for b in bands]) + 1e-6
        feat = {f"{b}_pct": raw_data.get(b, 0) / total_power for b in bands}
        
        # 3. Add base metrics
        feat['session_time_sec'] = session_time_sec
        feat['attention'] = float(raw_data.get('attention', 0))
        feat['meditation'] = float(raw_data.get('meditation', 0))
        feat['signal_quality'] = float(raw_data.get('signal_quality', 0))
        
        # 4. Markers
        marker_cols = [c for c in self.expected_features if 'marker_ev' in c]
        for col in marker_cols:
            feat[col] = float(raw_data.get(col, 0))

        # 5. Ratios
        eps = 1e-6
        feat['theta_beta_ratio'] = feat['theta_pct'] / (feat['low_beta_pct'] + feat['high_beta_pct'] + eps)
        feat['alpha_beta_ratio'] = (feat['low_alpha_pct'] + feat['high_alpha_pct']) / (feat['low_beta_pct'] + feat['high_beta_pct'] + eps)
        feat['slow_fast_ratio'] = (feat['delta_pct'] + feat['theta_pct']) / (feat['low_beta_pct'] + feat['high_beta_pct'] + feat['low_gamma_pct'] + feat['mid_gamma_pct'] + eps)
        
        return feat

    def predict(self, raw_data_dict):
        current_feat = self._preprocess(raw_data_dict)
        self.history.append(current_feat)
        
        if len(self.history) < self.window_size:
            return "Initializing buffer..."

        df_history = pd.DataFrame(list(self.history))
        final_features = current_feat.copy()

        # Rolling stats
        rolling_base = ['delta_pct', 'theta_pct', 'low_alpha_pct', 'high_alpha_pct', 
                        'low_beta_pct', 'high_beta_pct', 'attention', 'meditation']
        
        for col in rolling_base:
            final_features[f"{col}_roll_mean_5"] = df_history[col].mean()
            final_features[f"{col}_roll_std_5"] = df_history[col].std()

        # Convert to DF and force columns to match training EXACTLY
        X_input = pd.DataFrame([final_features])
        
        # Reorder columns to match the model's training order
        X_input = X_input[self.expected_features]
        
        # --- FIX FOR WARNING ---
        # 1. Scale the data
        scaled_data = self.scaler.transform(X_input)
        
        # 2. Put the scaled data back into a DataFrame with column names
        # This tells the model: "Yes, these are the correct features"
        X_scaled_df = pd.DataFrame(scaled_data, columns=self.expected_features)
        
        # 3. Predict using the DataFrame instead of the raw array
        prediction_idx = self.model.predict(X_scaled_df)[0]
        
        return self.label_map.get(prediction_idx, "Unknown")
# --- TEST ---
if __name__ == "__main__":
    predictor = MindTunePredictor('extra_trees_model.pkl', 'scaler.pkl')
    raw_sample = {
        'delta': 615630, 'theta': 302031, 'low_alpha': 23945, 'high_alpha': 22671,
        'low_beta': 14790, 'high_beta': 54734, 'low_gamma': 56355, 'mid_gamma': 21958,
        'attention': 50, 'meditation': 60, 'signal_quality': 200
    }

    for i in range(1, 7):
        print(f"Second {i}: {predictor.predict(raw_sample)}")