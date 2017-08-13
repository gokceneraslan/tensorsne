# coding: utf-8

from tensorsne import x2p, KL, gradKL, KLsparse, gradKLsparse
from tensorsne.knn import knn
from tensorsne.tsne import tsne

import numpy as np
from pytest import approx
import scipy as sp
np.random.seed(71)

from keras.datasets import mnist
from sklearn.decomposition import PCA


def get_mnist(n_train=5000, n_test=500, pca=True, d=50):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n, row, col = X_train.shape
    channel = 1

    X_train = X_train.reshape(-1, channel * row * col)
    X_test = X_test.reshape(-1, channel * row * col)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train = X_train[:n_train] - X_train[:n_train].mean(axis=0)
    X_test = X_test[:n_test] - X_test[:n_test].mean(axis=0)

    if pca:
        pcfit = PCA(n_components=d)

        X_train = pcfit.fit_transform(X_train)
        X_test = pcfit.transform(X_test)

    y_train = y_train[:n_train]
    y_test = y_test[:n_test]

    return X_train, y_train, X_test, y_test


def test_x2p():
    N = 1000
    N_test = 500
    D = 50
    Dlow = 2
    perplexity = 50

    X, y, X_test, y_test = get_mnist(N, N_test, True, D)

    P1 = x2p(X, perplexity, method='exact')
    P2 = x2p(X, perplexity, method='knn')
    P3 = x2p(X, perplexity, method='approx')

    assert np.mean(np.argmax(P1, axis=0) == np.argmax(P2, axis=0)) > 0.95
    assert np.mean(np.argmax(P2, axis=0) == np.argmax(P3, axis=0)) > 0.99


def test_kl():
    N = 6000
    N_test = 500
    D = 50
    Dlow = 2
    perplexity = 50

    X, y, X_test, y_test = get_mnist(N, N_test, True, D)

    P = x2p(X, perplexity, method='knn')
    Y = np.random.normal(0, 1e-1, (N, Dlow))

    kl1 = KL(P.toarray(), Y)
    kl2 = KLsparse(P.indptr, P.indices, P.data, Y, 0.1)
    kl3 = KLsparse(P.indptr, P.indices, P.data, Y, 0.5)

    assert approx(kl1, kl2)
    assert approx(kl1, kl3)
    assert abs(kl1-kl2) < abs(kl1-kl3)

    dY1 = gradKL(P.toarray(), Y)
    dY2 = gradKLsparse(P.indptr, P.indices, P.data, Y, 0.1)
    dY3 = gradKLsparse(P.indptr, P.indices, P.data, Y, 0.5)

    assert np.all(np.corrcoef(dY1.flatten(), dY2.flatten()) > 0.99)
    assert np.all(np.corrcoef(dY1.flatten(), dY3.flatten()) > 0.99)

    # mini gd test
    for i in range(50):
        dY = gradKLsparse(P.indptr, P.indices, P.data, Y, theta=0.5)
        Y -= (200.*dY)

    kl4 = KLsparse(P.indptr, P.indices, P.data, Y, 0.5)
    assert kl4 < kl3


def test_tensorflow():
    from tensorsne.tensorflow import tsne_op
    import tensorflow as tf

    N = 1000
    N_test = 500
    D = 50
    Dlow = 2
    perplexity = 100

    X, y, X_test, y_test = get_mnist(N, N_test, True, D)

    # test tsne_op
    P = x2p(X, perplexity, method='exact')
    Y = np.random.normal(0, 1e-4, (N, Dlow)).astype(np.float32)
    kl1 = KL(P, Y)
    klgrad1 = gradKL(P, Y)

    with tf.Session() as sess:
        Yc = tf.constant(Y)
        tf.global_variables_initializer().run()

        yop = tsne_op(P, Yc)
        yop_gr = tf.gradients(yop, Yc)[0]

        kl2 = yop.eval()
        klgrad2 = yop_gr.eval()

    assert kl1 == kl2
    assert np.allclose(klgrad1, klgrad2)

    # test tsne_sparse_op
    P = x2p(X, perplexity, method='knn')
    Y = np.random.normal(0, 1e-4, (N, Dlow)).astype(np.float32)
    kl1 = KLsparse(P.indptr, P.indices, P.data, Y)
    klgrad1 = gradKLsparse(P.indptr, P.indices, P.data, Y)

    with tf.Session() as sess:
        Yc = tf.constant(Y)
        tf.global_variables_initializer().run()

        yop = tsne_op(P, Yc)
        yop_gr = tf.gradients(yop, Yc)[0]

        kl2 = yop.eval()
        klgrad2 = yop_gr.eval()

    assert approx(kl1, kl2)
    assert np.allclose(klgrad1, klgrad2)

    # test tf ops via GD
    with tf.Session() as sess:
        Y_var = tf.Variable(Y)

        opt = tf.train.MomentumOptimizer(learning_rate=200., momentum=0.8)
        loss = tsne_op(P, Y_var)
        update = opt.minimize(loss, var_list=[Y_var])

        tf.global_variables_initializer().run()
        for i in range(50):
            update.run()

        Y_gd = Y_var.eval()

    kl1 = KLsparse(P.indptr, P.indices, P.data, Y)
    kl2 = KLsparse(P.indptr, P.indices, P.data, Y_gd)

    assert kl2 < kl1


def test_tsne():
    N = 100
    N_test = 1
    perplexity = 30

    X, _, _, _ = get_mnist(N, N_test, False)
    res = tsne(X, dim=2, perplexity=perplexity, verbose=True)

    assert res['loss'][-1] < 0.5
    assert res['Y'].shape[0] == N
    assert res['Y'].shape[1] == 2

