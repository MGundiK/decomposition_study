# analysis/decomp_wrappers.py
import numpy as np

# -------- simple causal baselines (NumPy) --------
def _ema(x, alpha):
    y = np.zeros_like(x, float); y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    return y

def _seasonal_mean_one_sided(x, period, hist_cycles=5):
    n = len(x); s = np.zeros(n)
    for t in range(n):
        idx = list(range(t-period, -1, -period))[:hist_cycles]
        s[t] = np.mean(x[idx]) if idx else 0.0
    return s

def decomp_stl_causal(x, periods, ema_alpha=0.05, hist_cycles=5):
    x = np.asarray(x, float).ravel()
    trend = _ema(x, ema_alpha)
    seas_list = []
    for p in periods:
        seas_list.append(_seasonal_mean_one_sided(x - trend, p, hist_cycles))
    seasonal = np.sum(np.vstack(seas_list), axis=0) if seas_list else np.zeros_like(x)
    resid = x - trend - seasonal
    return trend, seasonal, resid

def decomp_kalman_level(x, level_var=0.05, obs_var=1.0):
    x = np.asarray(x, float).ravel()
    a, P = 0.0, 1e6
    trend = np.zeros_like(x)
    for t, xt in enumerate(x):
        # predict
        P = P + level_var
        # update
        y = xt - a
        S = P + obs_var
        K = P / S
        a = a + K * y
        P = (1 - K) * P
        trend[t] = a
    seasonal = np.zeros_like(x)
    resid = x - trend
    return trend, seasonal, resid

# -------- adapter for your PyTorch decomposition(s) --------
def make_torch_decomposer(torch_model_ctor, device="cuda"):
    """
    Wrap a user-supplied PyTorch decomposition model so it returns (trend, seasonal, resid)
    using past-only information. We assume the ctor returns a model with .forward(x_past)
    that returns components aligned to x_past.
    """
    import torch
    model = torch_model_ctor().to(device)
    model.eval()

    @torch.no_grad()
    def decompose(x_np: np.ndarray):
        x = torch.tensor(x_np, dtype=torch.float32, device=device).view(1, -1)  # (B=1, T)
        # Optional: your model may need extra marks; adapt here as needed.
        comps = model(x)  # e.g., returns (trend, seas) or dict
        if isinstance(comps, dict):
            tr = comps["trend"].squeeze(0).detach().cpu().numpy()
            se = comps["seasonal"].squeeze(0).detach().cpu().numpy()
        elif isinstance(comps, (tuple, list)) and len(comps) >= 2:
            tr = comps[0].squeeze(0).detach().cpu().numpy()
            se = comps[1].squeeze(0).detach().cpu().numpy()
        else:
            raise ValueError("Your decomposition model must return (trend, seasonal) or dict with those keys.")
        r = x_np.ravel() - tr - se
        return tr, se, r

    return decompose
