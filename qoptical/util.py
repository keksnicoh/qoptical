# -*- coding: utf-8 -*-
""" some utilities
    :author: keksnicoh
"""
from . import settings
import numpy as np
from collections import namedtuple


def eigh(h0):
    ev, _ = np.linalg.eigh(h0)
    return ev, _.T


def is_square(a):
    """ test if some ``a`` is a square number """
    return a ** 0.5 == int(a ** 0.5)


def is_sqmat(a):
    """ test if some ``a`` is a square matrix
        meaning that it must be a numpy ndarray
        with a two component shape where both
        components are the same. """
    return isinstance(a, np.ndarray) \
           and len(a.shape) == 2 \
           and a.shape[0] == a.shape[1]


def sqmat(mat):
    """ normalizes and validates an input ``mat``
        and returns an instance of `np.matrix` such that
        `shape[0] == shape[1]`.

        Example:
        --------

        sqmat([1,2,3,4])

        >> np.matrix([[1,2],[3,4]])

    """

    if isinstance(mat, np.ndarray) or isinstance(mat, list):
        try:
            mat = np.matrix(mat)
        except ValueError as e:
            raise ValueError((
                'cannot be interpreted as a numpy matrix: {}'
            ).format(e))

    assert isinstance(mat, np.matrix)

    if mat.shape[0] == 1:
        assert is_square(mat.shape[1]), "not square"
        sqint = int(np.sqrt(mat.shape[1]))
        return mat.reshape((sqint, sqint))
    elif len(mat.shape) == 2:
        assert mat.shape[0] == mat.shape[1], "not square"
        return mat
    else:
        raise ValueError((
            'could not understand object of '
            'shape {} as square matrix'
        ).format(mat.shape))


def ketbra(s, i, j = None):
    """ the (``i``, ``j``) ketbra for a given
        array of states ``s``

        for states |s_1>, |s_2>, ..., |s_n> in ``s``
        the ketbra |s_i><s_j| is given by this function.

        if ``j`` is None then ``j`` <- ``i``
    """
    return np.outer(s[i].T.conj(), s[j or i])


def boson_stat(T):
    """ mean bosonic occupation number distribution

        N = 1/(exp(hw)-1)

        higher order function creates a distribution
        at termperature ``T``.

        Example:
        --------

        n = boson_stat(T=10)
        n(E=4)
        >> float XXX
    """
    if T == 0:
        return lambda E: 0.0
    try:
        return lambda E: 1.0 / (np.exp(E / T) - 1 + 0.0000000001)
    except RuntimeWarning:
        raise RuntimeWarning("E={}, T={}".format(E, T))

def vectorize(x, dtype=None):
    if isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        x = np.array([x])

    assert len(x.shape) == 1, 'input must have trivial shape.'
    try:
        return x if dtype is None else x.astype(dtype)
    except TypeError:
        raise RuntimeError('could not vectorize {} as dtype {}'.format(x, dtype))

def unvectorize(x):

    assert isinstance(x, list) \
        or isinstance(x, np.ndarray)
    return x[0] if len(x) == 1 else np.array(x)

def thermal_dist(E, T):
    if T == 0.0:
        d = np.zeros(len(E), dtype=settings.DTYPE_FLOAT)
        d[0] = 1.0
        return d
    Z = sum(np.exp(-1.0/T * e) for e in E)
    return np.array([np.exp(-1.0/T * e) / Z for e in E], settings.DTYPE_FLOAT)


def list_fillup_like(a, b):
    assert isinstance(a, list)
    assert isinstance(b, list)
    if len(b) == 1:
        return [b[0]] * len(a)
    elif len(b) == len(a):
        return b
    raise ValueError()



def npmat_manylike(a, b):
    """ normalizes a mixed input `b` in context of a given square matrix (`(M x M)`) `a`
        such that a `np.ndarray` of `n` matricies (shape=`(n, M, M)`).

        valid input:

        - `b` is a `list`: `[np.ndarray(...), np.ndarray(...), ...]`
        - `b` is a np.ndarray of shape=`(M*M, )`
        - `b` is a np.ndarray of shape=`(M, M)`
        - `b` is a np.ndarray of shape=`(N, M, M)`
        - `b` is a np.ndarray of shape=`(N, M * M)`
        """
    assert is_sqmat(a)

    if isinstance(b, list):
        b = np.array(b, dtype=settings.DTYPE_COMPLEX)

    assert isinstance(b, np.ndarray)

    if len(b.shape) == 1:
        b = np.array([sqmat(b), ], dtype=settings.DTYPE_COMPLEX)
    elif len(b.shape) == 2 and b.shape[0] != b.shape[1]:
        b = np.array([sqmat(c) for c in b], dtype=settings.DTYPE_COMPLEX)
    elif len(b.shape) == 2 and b.shape[0] == b.shape[1]:
        b = np.array([b, ], dtype=settings.DTYPE_COMPLEX)
    elif len(b.shape) == 3:
        b = np.array([sqmat(c) for c in b], dtype=settings.DTYPE_COMPLEX)
    else:
        raise NotImplementedError('cant understand this')

    if not b.shape[1:3] == a.shape:
        raise ValueError((
            'shape missmatch: all given matricies should have shape {}'
        ).format(a.shape))
    return b

def nparr_manylike(a, b, dtype):

    assert isinstance(a, np.ndarray)
    assert len(a.shape) == 3
    assert a.shape[1] == a.shape[2]

    # normalize type
    if isinstance(b, list):
        b = np.array(b, dtype=dtype)
    if not isinstance(b, np.ndarray):
        b = np.array([b], dtype=dtype)

    # expand single element so that it fits
    # the length of a
    if b.shape[0] == 1 and a.shape[0] != 1:
        b = np.array([b[0]] * a.shape[0], dtype=dtype)


    if b.shape[0] != a.shape[0]:
        raise ValueError(
            'shape missmatch: expected {} items, got {} items.'.format(
                a.shape[0], b.shape[0]))

    return b

class InconsistentVectorSizeError(Exception):
    def __init__(self, msg, vectors):
        self.vectors = vectors
        description = "\n".join("    {:10}: {}".format(v[0], v[1].shape) for v in vectors)
        super().__init__("{}:\n{}".format(msg, description))
