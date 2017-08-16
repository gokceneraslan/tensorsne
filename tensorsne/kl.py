import numpy as np

from  . import _cpp

def gradKL(P, Y):
    assert P.dtype == Y.dtype, "P and Y must be of same dtype"
    dtype = Y.dtype

    ts = _cpp.BH_SNE()
    return ts.computeGradientExact(P, Y).astype(dtype)


def KL(P, Y):
    assert P.dtype == Y.dtype, "P and Y must be of same dtype"
    dtype = Y.dtype

    ts = _cpp.BH_SNE()
    return np.array(ts.evaluateErrorExact(P, Y), dtype=dtype)


def gradKLsparse(rows, cols, data, Y, theta=0.5):
    assert data.dtype == Y.dtype, "data and Y must be of same dtype"
    dtype = Y.dtype

    ts = _cpp.BH_SNE()
    return ts.computeGradient(rows, cols, data, Y, theta).astype(dtype)


def KLsparse(rows, cols, data, Y, theta=0.5):
    assert data.dtype == Y.dtype, "data and Y must be of same dtype"
    dtype = Y.dtype

    ts = _cpp.BH_SNE()
    return np.array(ts.evaluateError(rows, cols, data, Y, theta), dtype=dtype)

