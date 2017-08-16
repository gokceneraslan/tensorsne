# distutils: language = c++
import numpy as np
import scipy as sp
cimport numpy as np
cimport cython
from libcpp cimport bool

cdef extern from "tsne.h":
    cdef cppclass TSNE:
        TSNE()
        void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity)
        void computeGaussianPerplexity(double* X, int N, int D, unsigned int*row_P, unsigned int* col_P, double* val_P, double perplexity, int K)

        void computeExactGradient(double* P, double* Y, int N, int D, double* dC)
        void computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)

        double evaluateError(double* P, double* Y, int N, int D)
        double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)


cdef class BH_SNE:
    cdef TSNE* thisptr # hold a C++ instance

    def __cinit__(self):
        self.thisptr = new TSNE()

    def __dealloc__(self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def computeGaussianPerplexityExact(self, X, perplexity):
        N = X.shape[0]
        D = X.shape[1]

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] P = np.zeros((N, N), dtype=np.float64)

        self.thisptr.computeGaussianPerplexity(&_X[0,0], N, D, &P[0,0],perplexity)

        # symmetrize P
        P += P.T
        P /= P.sum()

        return P

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def computeGaussianPerplexity(self, X, perplexity):
        K = int(3 * perplexity)
        N = X.shape[0]
        D = X.shape[1]

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X, dtype=np.float64)
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] rows = np.zeros(N+1, dtype=np.uint32)
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] cols = np.zeros(N*K, dtype=np.uint32)
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] data = np.zeros(N*K, dtype=np.float64)

        self.thisptr.computeGaussianPerplexity(&_X[0,0], N, D,
                                               &rows[0], &cols[0],
                                               &data[0], perplexity, K)
        P = self.tuple2sparse(rows, cols, data)

        # symmetrize P
        P += P.T
        P /= P.sum()

        return P


    def tuple2sparse(self, rows, cols, data):

        rows = rows.flatten()
        cols = cols.flatten()
        data = data.flatten()

        N = rows.shape[0] - 1
        P = sp.sparse.csr_matrix((data, cols, rows), shape=(N, N))

        return P


    def sparse2tuple(self, P):
        rows, cols, data = P.indptr, P.indices, P.data

        return rows, cols, data


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def computeGradientExact(self, P, Y):
        N = Y.shape[0]
        D = Y.shape[1]

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _P = np.ascontiguousarray(P, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] dY = np.zeros((N, D), dtype=np.float64)

        self.thisptr.computeExactGradient(&_P[0,0], &_Y[0,0], N, D, &dY[0,0])

        return dY


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def computeGradient(self, rows, cols, data, Y, theta):
        N = Y.shape[0]
        D = Y.shape[1]

        cdef double* _P = NULL
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _rows = np.ascontiguousarray(rows, dtype=np.uint32)
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _cols = np.ascontiguousarray(cols, dtype=np.uint32)
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] _data = np.ascontiguousarray(data, dtype=np.float64)

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] dY = np.zeros((N, D), dtype=np.float64)

        self.thisptr.computeGradient(_P,
                                     &_rows[0],
                                     &_cols[0],
                                     &_data[0],
                                     &_Y[0,0], N, D, &dY[0,0], theta)

        return dY


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluateErrorExact(self, P, Y):
        N = Y.shape[0]
        D = Y.shape[1]

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _P = np.ascontiguousarray(P, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y, dtype=np.float64)

        err = self.thisptr.evaluateError(&_P[0,0], &_Y[0,0], N, D)

        return err


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluateError(self, rows, cols, data, Y, theta):
        N = Y.shape[0]
        D = Y.shape[1]

        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _rows = np.ascontiguousarray(rows, dtype=np.uint32)
        cdef np.ndarray[np.uint32_t, ndim=1, mode='c'] _cols = np.ascontiguousarray(cols, dtype=np.uint32)
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] _data = np.ascontiguousarray(data, dtype=np.float64)

        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y, dtype=np.float64)

        err =  self.thisptr.evaluateError(&_rows[0],
                                          &_cols[0],
                                          &_data[0],
                                          &_Y[0,0], N, D, theta)

        return err

