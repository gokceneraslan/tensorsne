from .tensorflow import tsne_op
from .x2p import x2p

import time
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

import numpy as np
import scipy as sp

from sklearn.decomposition import PCA


def tsne(X,
         perplexity=50,
         dim=2,
         theta=0.5,
         knn_method='knnparallel',
         pca_dim=50,
         exag = 12.,
         exag_iter=250,
         max_iter=1000,
         verbose=False,
         print_iter=50,
         lr=200.,
         init_momentum=0.5,
         final_momentum=0.8,
         save_snapshots=False,
         optimizer='momentum',
         tf_optimizer='AdamOptimizer',
         seed=42):

    X -= X.mean(axis=0)
    N = X.shape[0]
    result = {}

    assert optimizer in ('momentum', 'tensorflow', 'bfgs'), 'Available options: momentum, tensorflow and bfgs'

    if pca_dim is not None:
        result['PCA'] = PCA(n_components=pca_dim)
        X = result['PCA'].fit_transform(X)

    P = x2p(X, perplexity=perplexity, method=knn_method, verbose=verbose)
    result['P'] = P
    result['exag_iter'] = exag_iter
    result['print_iter'] = print_iter
    result['loss'] = []
    if save_snapshots:
        result['snapshots'] = []

    tf.reset_default_graph()
    tf.set_random_seed(seed)

    with tf.Session() as sess:
        step = 1

        def step_callback(Y_var):
            nonlocal step
            if step % print_iter == 0:
                print('Step: %d, error: %.16f' %(step, result['loss'][-1]))
                if save_snapshots:
                    result['snapshots'].append(Y_var.reshape((N, dim)).copy())
            if step == exag_iter:
                sess.run(tf.assign(exag_var, 1.))

            #zero mean
            sess.run(tf.assign(Y, Y-tf.reduce_mean(Y, axis=0)))
            step += 1

        def loss_callback(err):
            result['loss'].append(err)

        stddev = 1. if optimizer == 'bfgs' else 0.01
        Y = tf.Variable(tf.random_normal((N, dim),
                                         stddev=stddev, dtype=X.dtype))
        exag_var = tf.Variable(exag, dtype=P.dtype)

        if isinstance(P, sp.sparse.csr_matrix):
            loss = tsne_op((P.indptr, P.indices, P.data*exag_var), Y)
        else:
            loss = tsne_op(P*exag_var, Y)

        if optimizer == 'bfgs':
            opt = ScipyOptimizerInterface(loss, var_list=[Y], method='L-BFGS-B',
                                          options={'eps': 1., 'gtol': 0.,
                                                   'ftol': 0., 'disp': False,
                                                   'maxiter': max_iter,
                                                   'maxls': 100})
            tf.global_variables_initializer().run()
            opt.minimize(sess, fetches=[loss],
                         loss_callback=loss_callback, step_callback=step_callback)
            Y_final = Y.eval()

        else:
            zero_mean = tf.assign(Y, Y-tf.reduce_mean(Y, axis=0))

            if optimizer == 'tensorflow':
                opt = getattr(tf.train, tf_optimizer)(learning_rate=lr)
                update = opt.minimize(loss, var_list=[Y])
            else:
                mom_var = tf.Variable(init_momentum, dtype=X.dtype)
                uY = tf.Variable(tf.zeros((N, dim), dtype=X.dtype))
                gains = tf.Variable(tf.ones((N, dim), dtype=X.dtype))
                dY = tf.gradients(loss, [Y])[0]

                gains = tf.assign(gains,
                                  tf.where(tf.equal(tf.sign(dY), tf.sign(uY)),
                                                    gains * .8, gains + .2))

                gains = tf.assign(gains, tf.maximum(gains, 0.01))
                uY = tf.assign(uY, mom_var*uY - lr*gains*dY)

                update = tf.assign_add(Y, uY)

            tf.global_variables_initializer().run()

            t = time.time()
            for i in range(1, max_iter+1):
                if i == exag_iter:
                    if optimizer == 'momentum': sess.run(tf.assign(mom_var, final_momentum))
                    sess.run(tf.assign(exag_var, 1.))

                sess.run(update)
                sess.run(zero_mean)

                if i % print_iter == 0:
                    kl = loss.eval()
                    result['loss'].append(kl)
                    if verbose:
                        print('Step: %d, error: %f (in %f sec.)' % (i, kl, (time.time()-t)))
                        t = time.time()
                    if save_snapshots:
                        result['snapshots'].append(Y.eval())
            Y_final = Y.eval()

    result['Y'] = Y_final
    return result

