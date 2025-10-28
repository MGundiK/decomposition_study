# analysis/run_study.py
import os, csv, numpy as np
from .periods import default_periods
from .decomp_wrappers import decomp_stl_causal, decomp_kalman_level
from .diagnostics import acf, ljung_box_from_acf, welch_psd, gph_d, surrogate_test_mi_abs_lag, leakage_future_shuffle
from .backtest import rolling_backtest

def get_univariate_series(data_set, col_idx=0):
    # Your Dataset_ classes return scaled data arrays; here we just flatten one channel.
    x = data_set.data_y[:, col_idx].astype(float)
    return x

def run_one_series(x, data_name, freq, horizons, out_csv, decomp_kind="stl", periods=None,
                   resid_model="ar"):
    periods = periods or default_periods(data_name, freq)
    if decomp_kind == "stl":
        def deco_fn(x_np): return decomp_stl_causal(x_np, periods=periods, ema_alpha=0.05, hist_cycles=5)
    elif decomp_kind == "kalman":
        def deco_fn(x_np): return decomp_kalman_level(x_np, level_var=0.05, obs_var=1.0)
    else:
        raise ValueError("decomp_kind must be 'stl' or 'kalman' (or plug your Torch adapter).")

    # components & residuals on full series (for diagnostics)
    tr, se, r = deco_fn(x)

    # residual diagnostics
    L = 40
    ac_r  = acf(r, L); Qr, df1  = ljung_box_from_acf(ac_r, len(r))
    ac_r2 = acf(r**2, L); Qr2, df2 = ljung_box_from_acf(ac_r2, len(r))
    dvol  = gph_d(r**2)
    s_obs, surr, p_surr = surrogate_test_mi_abs_lag(r, lag=10, bins=24, nsurr=9)

    # leakage (shuffle last 20% and compare pre-cut components)
    cut = int(0.8*len(x))
    leak_tr, leak_se = leakage_future_shuffle(x, deco_fn, cut)

    # rolling-origin backtests on original series combining decomp+residual model
    bt = rolling_backtest(x, deco_fn, horizons=horizons, train_frac=0.6,
                          resid_model=resid_model, resid_kwargs={"p":12, "W":600})

    # write to CSV
    header = [
        "dataset","decomp","resid_model","h",
        "RMSE","MAE","sMAPE","CRPS_N",
        "Q_resid","df_resid","Q_resid2","df_resid2",
        "dvol","surr_MI_abs_lag10","surr_p",
        "leak_tr","leak_se","periods"
    ]
    newfile = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if newfile: w.writerow(header)
        for h, m in bt.items():
            w.writerow([
                data_name, decomp_kind, resid_model, int(h),
                m["RMSE"], m["MAE"], m["sMAPE"], m["CRPS(N)"],
                Qr, df1, Qr2, df2,
                dvol, s_obs, p_surr,
                leak_tr, leak_se, "|".join(map(str, periods))
            ])
