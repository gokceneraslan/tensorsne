from .kl import KL, gradKL, KLsparse, gradKLsparse

import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.python.framework import ops


def __py_func(func, inp, Tout, stateful=False, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def __tsne_grad_op(op, grad):
    P, Y = op.inputs
    grdop = tf.py_func(gradKL, [P, Y], [tf.float32], stateful=False)[0]
    return None, grad*grdop


def __tsne_sparse_grad_op(op, grad):
    rows, cols, data, Y, theta = op.inputs
    grdop = tf.py_func(gradKLsparse, [rows, cols, data, Y, theta],
                       [tf.float32], stateful=False)[0]
    return None, None, None, grad*grdop, None


def tsne_op(P, Y, theta=0.5, name=None):
    if isinstance(P, sp.sparse.csr_matrix):
        op = __py_func(KLsparse, [P.indptr, P.indices, P.data, Y, theta],
                       [tf.float32], name=name, grad=__tsne_sparse_grad_op)[0]
    elif isinstance(P, tuple):
        op = __py_func(KLsparse, [P[0], P[1], P[2], Y, theta],
                       [tf.float32], name=name, grad=__tsne_sparse_grad_op)[0]
    else:
        op = __py_func(KL, [P, Y], [tf.float32], name=name, grad=__tsne_grad_op)[0]

    op.set_shape((1,))
    return op

