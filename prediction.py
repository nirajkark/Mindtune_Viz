import pandas as pd
import numpy as np
import joblib
from collections import deque

class MindTunePredictor:
    def __init__(self, model_path, scaler_path, window_size=5):
        # 1. Load the saved brain (Model) and the map (Scaler)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # 2. Initialize a Buffer for Rolling Statistics
        # We use deque to automatically "pop" old data when new data arrives
        self.history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Mapping for output
        self.label_map = {0: 'calm', 1: 'neutral', 2: 'stressed'}
        
        # Define the exact feature order used during training
        self.feature_order = [
            'delta_pct', 'theta_pct', 'low_alpha_pct', 'high_alpha_pct', 
            'low_beta_pct', 'high_beta_pct', 'low_gamma_pct', 'mid_gamma_pct', 
            'attention', 'meditation', 'signal_quality', 
            'marker_ev_praised_active', 'marker_ev_qna_active', 'marker_ev_question_active', 
            'marker_ev_scolded_active', 'marker_ev_speaking_active', 'marker_ev_tech_issue_active',
            'theta_beta_ratio', 'alpha_beta_ratio', 'slow_fast_ratio'
        ]

    def _preprocess(self, raw_data):
        """
        Transforms raw integers from hardware into engineered percentages and ratios.
        """
        # A. Calculate Percentages (Normalization)
        bands = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']
        total_power = sum([raw_data.get(b, 0) for b in bands]) + 1e-6
        
        processed = {f"{b}_pct": raw_data.get(b, 0) / total_power for b in bands}
        
        # B. Passthrough values
        processed['attention'] = raw_data.get('attention', 0)
        processed['meditation'] = raw_data.get('meditation', 0)
        processed['signal_quality'] = raw_data.get('signal_quality', 0)
        
        # C. Event Markers (0 or 1)
        marker_cols = [c for c in self.feature_order if 'marker_ev' in c]
        for col in marker_cols:
            processed[col] = raw_data.get(col, 0)

        # D. Mathematical Ratios
        eps = 1e-6
        processed['theta_beta_ratio'] = processed['theta_pct'] / (processed['low_beta_pct'] + processed['high_beta_pct'] + eps)
        processed['alpha_beta_ratio'] = (processed['low_alpha_pct'] + processed['high_alpha_pct']) / (processed['low_beta_pct'] + processed['high_beta_pct'] + eps)
        processed['slow_fast_ratio'] = (processed['delta_pct'] + processed['theta_pct']) / (processed['low_beta_pct'] + processed['high_beta_pct'] + processed['low_gamma_pct'] + processed['mid_gamma_pct'] + eps)
        
        return processed

    def predict(self, raw_data_dict):
        """
        Main entry point for prediction.
        """
        # 1. Transform raw integers to features
        current_features = self._preprocess(raw_data_dict)
        
        # 2. Add to Rolling Buffer
        self.history.append(current_features)
        
        # 3. Wait until buffer is full for high-accuracy rolling stats
        if len(self.history) < self.window_size:
            return "Initializing buffer..."

        # 4. Calculate Rolling Mean (Replicating training 'rolling_mean')
        df_history = pd.DataFrame(list(self.history))
        rolling_features = df_history.mean().to_frame().T
        
        # Ensure column order matches training exactly
        rolling_features = rolling_features[self.feature_order]
        
        # 5. Scale the features
        scaled_features = self.scaler.transform(rolling_features)
        
        # 6. Run Inference
        prediction_idx = self.model.predict(scaled_features)[0]
        
        return self.label_map.get(prediction_idx, "Unknown")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Initialize Predictor
    predictor = MindTunePredictor('extra_trees_model.pkl', 'scaler.pkl')

    # Simulated raw row from hardware
    raw_sample = {
        'delta': 615630, 'theta': 302031, 'low_alpha': 23945, 'high_alpha': 22671,
        'low_beta': 14790, 'high_beta': 54734, 'low_gamma': 56355, 'mid_gamma': 21958,
        'attention': 50, 'meditation': 60, 'signal_quality': 200,
        'marker_ev_speaking_active': 0, 'marker_ev_tech_issue_active': 0 # and other markers
    }

    # In a real app, you would call this inside your data-receive loop
    result = predictor.predict(raw_sample)
    print(f"Predicted State: {result}")