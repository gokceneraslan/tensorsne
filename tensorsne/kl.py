import numpy as np

from  . import _cpp

def gradKL(P, Y):
    ts = _cpp.BH_SNE()
    return ts.computeGradientExact(P, Y)


def KL(P, Y):
    ts = _cpp.BH_SNE()
    return ts.evaluateErrorExact(P, Y)


def gradKLsparse(rows, cols, data, Y, theta=0.5):
    ts = _cpp.BH_SNE()
    return ts.computeGradient(rows, cols, data, Y, theta)


def KLsparse(rows, cols, data, Y, theta=0.5):
    ts = _cpp.BH_SNE()
    return ts.evaluateError(rows, cols, data, Y, theta)

