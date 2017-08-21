import nmslib
import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors


def knn(data, k=50, method='sklearn', verbose=False, **kwargs):
    assert method in ('sklearn', 'nmslib'), 'No such knn method'

    if method == 'sklearn':
        return __knn_sklearn(data, k, verbose=verbose)
    else:
        return __knn_nmslib(data, k, verbose=verbose)


def __knn_nmslib(data, k=50, method='vptree', verbose=False, **kwargs):

    # see https://github.com/searchivarius/nmslib/issues/230
    data = np.ascontiguousarray(data)

    if isinstance(data, sp.sparse.csr_matrix):
        space = 'l2_sparse'
        data_type=nmslib.DataType.SPARSE_VECTOR
    else:
        space = 'l2'
        data_type=nmslib.DataType.DENSE_VECTOR

    index = nmslib.init(method=method, space=space, data_type=data_type)
    index.addDataPointBatch(data)
    index.createIndex(print_progress=verbose, **kwargs)

    if verbose:
        print('knn: Indexing done.')

    neig = index.knnQueryBatch(data, k=k+1)
    if verbose:
        print('knn: Query done.')

    indices = np.vstack(x[0][1:] for x in neig)  # exclude self
    distances = np.vstack(x[1][1:] for x in neig)

    # two N x k matrices
    return distances.astype(data.dtype), indices


def __knn_sklearn(X, k, n_jobs=-1, verbose=False, **kwargs):

    nn = NearestNeighbors(n_neighbors=k+1, n_jobs=n_jobs,
                          algorithm='ball_tree', **kwargs)
    nn.fit(X)

    if verbose:
        print('Indexing done.')
    dist, ind = nn.kneighbors(X, k+1, return_distance=True)

    if verbose:
        print('Query done.')

    return dist[:,1:].astype(X.dtype), ind[:,1:]

