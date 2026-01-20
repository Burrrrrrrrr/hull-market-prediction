from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _to_1d(a):
    if isinstance(a, (pd.Series, pd.DataFrame)):
        return np.asarray(a).reshape(-1)
    return np.asarray(a).reshape(-1)


def spearman_ic(y_true, y_pred) -> float:
    y = _to_1d(y_true)
    p = _to_1d(y_pred)
    return float(spearmanr(p, y, nan_policy="omit").correlation)


def newey_west_var(x, lag=5) -> float:
    x = _to_1d(x)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    var = gamma0
    L = int(lag)
    if L <= 0:
        return var
    for k in range(1, min(L, n - 1) + 1):
        w = 1.0 - k / (L + 1.0)
        gamma_k = np.dot(x[k:], x[:-k]) / n
        var += 2.0 * w * gamma_k
    return var


def adjusted_sharpe(daily_rets, sf=252.0, nw_lag=5, eps=1e-12) -> float:
    r = _to_1d(daily_rets)
    r = r[~np.isnan(r)]
    if len(r) < 5:
        return np.nan
    mu = np.mean(r)
    var_hac = newey_west_var(r, lag=nw_lag)
    if var_hac is None or np.isnan(var_hac) or var_hac <= eps:
        return np.nan
    return (mu / np.sqrt(var_hac + eps)) * np.sqrt(sf)


def strategy_returns_from_preds(y_true, pred, date_id=None):
    y = _to_1d(y_true)
    p = _to_1d(pred)
    pos = np.sign(p)
    pnl = pos * y

    if date_id is None:
        return pnl

    tmp = pd.DataFrame({"date_id": date_id, "pnl": pnl})
    daily = tmp.groupby("date_id", sort=True)["pnl"].mean()
    return daily.values


def compute_fold_metrics(y_true, pred, date_id=None, sf=252.0, nw_lag=5):
    daily = strategy_returns_from_preds(y_true, pred, date_id=date_id)
    return {
        "adj_sharpe": adjusted_sharpe(daily, sf=sf, nw_lag=nw_lag),
        "ic": spearman_ic(y_true, pred),
        "mse": float(np.mean(((_to_1d(y_true)) - (_to_1d(pred))) ** 2)),
    }
