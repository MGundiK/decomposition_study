# analysis/diagnostics.py
import numpy as np

def acf(x, L):
    x = np.asarray(x, float); x -= x.mean()
    n = len(x); ac = np.correlate(x, x, mode="full")
    return ac[n-1:n+L] / (ac[n-1] + 1e-12)

def ljung_box_from_acf(ac, n):
    L = len(ac) - 1
    ks = np.arange(1, L+1)
    Q = n*(n+2)*np.sum((ac[1:]**2)/(n-ks))
    return float(Q), int(L)

def _erf_array(z):
    try:
        from scipy.special import erf as sp_erf
        return sp_erf(z)
    except Exception:
        from math import erf as m_erf
        return np.vectorize(m_erf)(z)

def crps_normal(y, mu, sigma):
    y = np.asarray(y, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu)/sigma
    phi = (1./np.sqrt(2*np.pi))*np.exp(-0.5*z**2)
    Phi = 0.5*(1 + _erf_array(z/np.sqrt(2)))
    return sigma*( z*(2*Phi-1) + 2*phi - 1/np.sqrt(np.pi) )

def _periodogram(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    Xf = np.fft.rfft(x); P = (np.abs(Xf)**2)/n
    f = np.fft.rfftfreq(n, 1.0)
    return f[1:], P[1:]

def welch_psd(x, block=256):
    x = np.asarray(x, float)
    n = len(x) - (len(x)%block)
    if n <= 0: return np.array([np.nan]), np.array([np.nan])
    x = x[:n]; K = n//block; acc = None
    for k in range(K):
        f, P = _periodogram(x[k*block:(k+1)*block])
        acc = P if acc is None else (acc + P)
    return f, acc/K

def gph_d(x):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    Xf = np.fft.fft(x); Iw = (np.abs(Xf[:n//2])**2)/(2*np.pi*n)
    w = 2*np.pi*np.arange(1, n//2)/n
    m = int(np.floor(n**0.5))
    if m < 10: return np.nan
    w = w[:m]; Iw = Iw[1:m+1]
    y = np.log(Iw + 1e-12); xr = np.log(2*np.sin(w/2) + 1e-12)
    X = np.vstack([np.ones_like(xr), -2*xr]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1])

def hist_mi(x, y, bins=24):
    x = np.asarray(x, float); y = np.asarray(y, float)
    H, _, _ = np.histogram2d(x, y, bins=bins)
    P = H/np.sum(H); Px = P.sum(1, keepdims=True); Py = P.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((P>0)&(Px>0)&(Py>0), P/(Px*Py), 1.0)
        MI = np.sum(np.where(P>0, P*np.log(ratio), 0.0))
    return float(MI)

def iaaft_surrogate(x, iters=150, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    x = np.asarray(x, float)
    X_sorted = np.sort(x)
    target_mag = np.abs(np.fft.fft(x))
    y = x.copy(); rng.shuffle(y)
    for _ in range(iters):
        Y = np.fft.fft(y); Y = target_mag * np.exp(1j*np.angle(Y))
        y = np.fft.ifft(Y).real
        ranks = np.argsort(y); y[ranks] = X_sorted
    return y

def surrogate_test_mi_abs_lag(x, lag=10, bins=24, nsurr=9, rng=None):
    def stat(z):
        if len(z) <= lag: return np.nan
        return hist_mi(np.abs(z[:-lag]), np.abs(z[lag:]), bins=bins)
    s_obs = stat(x)
    S = []
    for _ in range(nsurr):
        y = iaaft_surrogate(x, iters=150, rng=rng)
        S.append(stat(y))
    S = np.array(S, float)
    p = (np.sum(np.abs(S - S.mean()) >= np.abs(s_obs - S.mean())) + 1) / (nsurr + 1)
    return float(s_obs), S, float(p)

def leakage_future_shuffle(x, decomp_fn, last_train_idx, rng=None):
    if rng is None: rng = np.random.default_rng(7)
    x = np.asarray(x, float)
    x_shuf = x.copy()
    tail = x[last_train_idx+1:].copy()
    rng.shuffle(tail)
    x_shuf[last_train_idx+1:] = tail
    tr1, se1, _ = decomp_fn(x)
    tr2, se2, _ = decomp_fn(x_shuf)
    pre = slice(None, last_train_idx+1)
    return float(np.mean(np.abs(tr1[pre]-tr2[pre]))), float(np.mean(np.abs(se1[pre]-se2[pre])))
