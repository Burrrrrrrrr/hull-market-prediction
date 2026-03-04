# Hull Market Prediction

## 运行命令速查表

### 1) 环境准备（Windows）

```powershell
# 进入项目根目录
cd C:\Users\AAA\Desktop\hull-market-prediction

# 创建并激活虚拟环境（如未创建）
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
python -m pip install -r requirements.txt
```

> 如果你不激活 venv，也可直接用 `\.venv\Scripts\python.exe` 执行下列命令。

### 2) 推荐命令（新结构）

```powershell
# LGBM
python scripts/train/train_lgbm.py --config configs/default.yaml
python scripts/predict/predict_lgbm.py --config configs/default.yaml
python scripts/submission/submit_lgbm.py --config configs/default.yaml

# TFT
python scripts/tune/tune_tft.py --config configs/default.yaml
python scripts/predict/predict_tft.py --config configs/default.yaml
python scripts/evaluate/evaluate_tft_test.py

# N-BEATS
python scripts/tune/tune_nbeats.py --config configs/default.yaml
python scripts/predict/predict_nbeats.py --config configs/default.yaml
python scripts/evaluate/evaluate_nbeats_test.py

# PatchTST
python scripts/tune/tune_patchtst.py --config configs/default.yaml
python scripts/predict/predict_patchtst.py --config configs/default.yaml
```

### 3) 兼容命令（旧入口，仍可用）

```powershell
# LGBM
python scripts/train.py --config configs/default.yaml
python scripts/predict.py --config configs/default.yaml
python scripts/submit.py --config configs/default.yaml

# TFT / N-BEATS / PatchTST
python scripts/tune_tft.py --config configs/default.yaml
python scripts/predict_tft.py --config configs/default.yaml
python scripts/evaluate_tft_test.py

python scripts/tune_nbeats.py --config configs/default.yaml
python scripts/predict_nbeats.py --config configs/default.yaml
python scripts/evaluate_nbeats_test.py

python scripts/tune_patchtst.py --config configs/default.yaml
python scripts/predict_patchtst.py --config configs/default.yaml
```

### 4) 常见产物位置

- 预测文件：`artifacts/preds/`
- 调参结果：`artifacts/reports/*_best_params.json`, `*_best_value.json`
- 评估结果：`artifacts/reports/*_metrics.json`, `*_curve.csv`, `*_curve.png`
- 提交文件：`artifacts/submission/`

### 5) 快速自检

```powershell
python scripts/train.py --help
python scripts/predict.py --help
python scripts/tune_tft.py --help
python scripts/predict_tft.py --help
```

---

更多脚本分组说明见 [scripts/README.md](scripts/README.md)。
