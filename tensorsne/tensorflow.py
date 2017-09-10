from .kl import KL, gradKL, KLsparse, gradKLsparse
from .x2p import x2p, __find_betas

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
    grdop = tf.py_func(gradKL, [P, Y], [Y.dtype], stateful=False)[0]
    return None, grad*grdop


def __tsne_sparse_grad_op(op, grad):
    rows, cols, data, Y, theta = op.inputs
    grdop = tf.py_func(gradKLsparse, [rows, cols, data, Y, theta],
                       [Y.dtype], stateful=False)[0]
    return None, None, None, grad*grdop, None


def tsne_op(P, Y, theta=0.5, name=None):

    if isinstance(P, sp.sparse.csr_matrix):
        op = __py_func(KLsparse, [P.indptr, P.indices, P.data, Y, theta],
                       [Y.dtype], name=name, grad=__tsne_sparse_grad_op)[0]
    elif isinstance(P, tuple):
        op = __py_func(KLsparse, [P[0], P[1], P[2], Y, theta],
                       [Y.dtype], name=name, grad=__tsne_sparse_grad_op)[0]
    else:
        op = __py_func(KL, [P, Y], [Y.dtype],
                       name=name, grad=__tsne_grad_op)[0]

    op.set_shape((1,))
    return op


def __knn_bruteforce(X, k=50):
    # calculate euclidean distances
    r = tf.reduce_sum(X*X, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)
    D = tf.matrix_set_diag(D, tf.constant(1e32, dtype=X.dtype,
                           shape=(X.shape[0],)))
    D = tf.sqrt(D)

    #find kNNs
    distances, indices = tf.nn.top_k(-D, k)
    return -distances, indices


def __find_betas_op(D, perplexity=50, name=None):

    P, beta = tf.py_func(__find_betas, [D, perplexity],
                         [D.dtype, D.dtype], name=name)

    P.set_shape(D.shape)
    beta.set_shape([D.shape[0],])

    return P, beta


def x2p_op(X, perplexity=50):
    k = 3 * perplexity
    N = X.shape[0]

    distances, indices = __knn_bruteforce(X, k)
    P, beta = __find_betas_op(tf.square(distances), perplexity)

    return P
