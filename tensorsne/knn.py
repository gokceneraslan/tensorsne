import nmslib
import numpy as np
import scipy as sp

def knn(data, k=50, method='vptree', verbose=False):

    if isinstance(data, sp.sparse.csr_matrix):
        space = 'l2_sparse'
        data_type=nmslib.DataType.SPARSE_VECTOR
    else:
        space = 'l2'
        data_type=nmslib.DataType.DENSE_VECTOR

    index = nmslib.init(method=method, space=space, data_type=data_type)
    index.addDataPointBatch(data)
    index.createIndex(print_progress=verbose)

    if verbose:
        print('knn: Indexing done.')

    neig = index.knnQueryBatch(data, k=k+1)
    if verbose:
        print('knn: Query done.')

    indices = np.vstack(x[0][1:] for x in neig)  # exclude self
    distances = np.vstack(x[1][1:] for x in neig)

    # two N x k matrices
    return distances.astype(data.dtype), indices

