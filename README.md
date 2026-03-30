# Mindtune Stress/Calm/Neutral Classifier

This repo contains a minimal pipeline to:

- Collect labeled windows from a live feature stream (currently via mock CSV playback).
- Train a classifier (`stressed`, `calm`, `neutral`) on band-power features (+ optional attention/meditation).
- Run real-time streaming inference with probability smoothing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train From Existing Session CSVs

Your workspace already contains session-style CSVs with `emo_*` columns:

- `Downloads/session_meta/eeg_rows.csv`
- `Downloads/session_meta/eeg_rows_001.csv`

Train a baseline model:

```bash
python scripts/train_model.py --input Downloads/session_meta/eeg_rows.csv --input-type session_meta --model-out model.joblib --metrics-out metrics.json
```

## Real-Time Prediction (Mock Playback)

Play back the Mindtune-style CSV in real time and print the predicted state:

```bash
python scripts/realtime_predict.py --model model.joblib --backend mock_csv --csv-path Downloads/mindtune_full_eeg_data.csv
```

## Collect A Labeled Dataset (Self-Report)

This uses keyboard commands while the stream is running:

- `s` + Enter = stressed
- `c` + Enter = calm
- `n` + Enter = neutral
- `q` + Enter = quit

```bash
python scripts/data_collection.py --backend mock_csv --csv-path Downloads/mindtune_full_eeg_data.csv --output collected_dataset.csv
```

Then train:

```bash
python scripts/train_model.py --input collected_dataset.csv --input-type dataset --model-out model.joblib --metrics-out metrics.json
```

## Wiring The Real SDK

The package includes a generic callback adapter (`CallbackSdkStreamAdapter`) in `mindtune_stress_model/stream_adapter.py`.
To connect Mindtune’s real stream, we’ll need your SDK callback entry point (function/class name) and payload shape.

