# analysis/resid_forecasters.py
import numpy as np

def resid_naive(resid, h):
    return 0.0

def resid_ar(resid, h, p=12, W=600):
    r = np.asarray(resid, float).ravel()
    W = min(W, len(r))
    if W <= p + 1: return 0.0
    y = r[-W:]; Y = y[p:]
    X = np.column_stack([y[p-k-1:-k-1] for k in range(p)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    buf = y[-p:].tolist()
    for _ in range(h):
        yhat = float(np.dot(beta, buf[::-1][:p]))
        buf.append(yhat); buf = buf[-p:]
    return buf[-1]
