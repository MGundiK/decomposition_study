# analysis/backtest.py
import numpy as np
from .diagnostics import crps_normal

def rolling_backtest(x, decomp_fn, horizons=(1,6,24), train_frac=0.6,
                     resid_model="naive", resid_kwargs=None):
    from .resid_forecasters import resid_naive, resid_ar
    resid_models = {"naive": resid_naive, "ar": resid_ar}
    f_res = resid_models[resid_model]
    resid_kwargs = resid_kwargs or {}

    n = len(x); t0 = int(train_frac*n)
    out = {h: {"y":[], "yhat":[], "sigma":[]} for h in horizons}

    for t in range(t0, n - max(horizons)):
        tr, se, r = decomp_fn(x[:t+1])
        mu = tr[-1] + se[-1]
        # trailing residual std for predictive sigma
        win = r[max(0, len(r)-400):]
        sig = float(np.std(win) if len(win)>10 else np.std(r) + 1e-6)

        for h in horizons:
            rhat = f_res(r, h, **resid_kwargs)
            yhat = mu + rhat
            out[h]["yhat"].append(yhat)
            out[h]["y"].append(x[t+h])
            out[h]["sigma"].append(sig)

    metrics = {}
    for h, d in out.items():
        y = np.array(d["y"]); yhat = np.array(d["yhat"]); sig = np.array(d["sigma"])
        err = yhat - y
        rmse = float(np.sqrt(np.mean(err**2)))
        mae  = float(np.mean(np.abs(err)))
        smape= float(np.mean(2*np.abs(yhat-y) / (np.abs(yhat)+np.abs(y)+1e-12)))
        crps = float(np.mean(crps_normal(y, yhat, sig)))
        z = (y - yhat) / (sig + 1e-12)
        pit = 0.5*(1 + np.erf(z/np.sqrt(2)))  # OK: np.erf exists on older NumPy; if not, you can reuse the _erf_array
        metrics[h] = {"RMSE": rmse, "MAE": mae, "sMAPE": smape, "CRPS(N)": crps, "PIT": pit}
    return metrics
