# Scripts layout

This folder now uses grouped entrypoints by function:

- `scripts/train/`: training entrypoints
- `scripts/tune/`: hyperparameter tuning entrypoints
- `scripts/predict/`: prediction/inference entrypoints
- `scripts/evaluate/`: offline evaluation entrypoints
- `scripts/submission/`: Kaggle submission entrypoints

## Primary commands (new layout)

- LGBM train: `python scripts/train/train_lgbm.py --config configs/default.yaml`
- LGBM predict: `python scripts/predict/predict_lgbm.py --config configs/default.yaml`
- LGBM submit: `python scripts/submission/submit_lgbm.py --config configs/default.yaml`

- TFT tune: `python scripts/tune/tune_tft.py --config configs/default.yaml`
- TFT predict: `python scripts/predict/predict_tft.py --config configs/default.yaml`
- TFT evaluate: `python scripts/evaluate/evaluate_tft_test.py`

- N-BEATS tune: `python scripts/tune/tune_nbeats.py --config configs/default.yaml`
- N-BEATS predict: `python scripts/predict/predict_nbeats.py --config configs/default.yaml`
- N-BEATS evaluate: `python scripts/evaluate/evaluate_nbeats_test.py`

- PatchTST tune: `python scripts/tune/tune_patchtst.py --config configs/default.yaml`
- PatchTST predict: `python scripts/predict/predict_patchtst.py --config configs/default.yaml`

## Backward compatibility

Legacy root scripts are kept as wrappers and still work:

- `scripts/train.py`
- `scripts/predict.py`
- `scripts/submit.py`
- `scripts/tune_tft.py`
- `scripts/tune_nbeats.py`
- `scripts/tune_patchtst.py`
- `scripts/predict_tft.py`
- `scripts/predict_nbeats.py`
- `scripts/predict_patchtst.py`
- `scripts/evaluate_tft_test.py`
- `scripts/evaluate_nbeats_test.py`

They delegate to the new grouped entrypoints.
