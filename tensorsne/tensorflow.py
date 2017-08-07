from .kl import KL, gradKL, KLsparse, gradKLsparse

import numpy as np
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


def tsne_op(P, Y, name=None):
    op = __py_func(KL, [P, Y], [tf.float32], name=name, grad=__tsne_grad_op)[0]
    op.set_shape((1,))
    return op


def __tsne_sparse_grad_op(op, grad):
    rows, cols, data, Y = op.inputs
    grdop = tf.py_func(gradKLsparse, [rows, cols, data, Y], [tf.float32], stateful=False)[0]
    return None, None, None, grad*grdop


def tsne_sparse_op(rows, cols, data, Y, name=None):
    op = __py_func(KLsparse, [rows, cols, data, Y], [tf.float32], name=name, grad=__tsne_sparse_grad_op)[0]
    op.set_shape((1,))
    return op

