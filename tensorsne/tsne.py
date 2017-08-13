from .tensorflow import tsne_op
from .x2p import x2p

import time
import tensorflow as tf
import numpy as np
import scipy as sp

from sklearn.decomposition import PCA


def tsne(X,
         perplexity=50,
         dim=2,
         theta=0.5,
         knn_method='approx',
         pca_dim=50,
         exag = 12.,
         exag_iter=300,
         max_iter=2000,
         verbose=False,
         print_iter=100,
         lr=200.,
         init_momentum=0.5,
         final_momentum=0.8,
         seed=42):

    X -= X.mean(axis=0)
    N = X.shape[0]
    result = {}

    if pca_dim is not None:
        result['PCA'] = PCA(n_components=pca_dim)
        X = result['PCA'].fit_transform(X)

    P = x2p(X, perplexity=perplexity, method=knn_method, verbose=verbose)
    result['P'] = P
    result['exag_iter'] = exag_iter
    result['print_iter'] = print_iter
    result['loss'] = []

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    with tf.Session() as sess:
        mom_var, exag_var = tf.Variable(init_momentum), tf.Variable(exag)
        Y = tf.Variable(tf.random_normal((N, dim), stddev=1e-4))
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mom_var)

        if isinstance(P, sp.sparse.csr_matrix):
            loss = tsne_op((P.indptr, P.indices, P.data*exag_var), Y)
        else:
            loss = tsne_op(P*exag_var, Y)

        update = opt.minimize(loss, var_list=[Y])
        tf.global_variables_initializer().run()

        t = time.time()
        for i in range(max_iter):
            if i == exag_iter:
                sess.run(tf.assign(mom_var, final_momentum))
                sess.run(tf.assign(exag_var, 1.))

            update.run()

            if i % print_iter == 0:
                kl = loss.eval()
                result['loss'].append(kl)
                if verbose:
                    print('Error: %f (%d iter. in %f seconds)' % (kl, print_iter, (time.time()-t)))
                    t = time.time()
        Y_final = Y.eval()

    result['Y'] = Y_final
    return result

