# -*- coding: utf-8 -*-
""" some utilities
    :author: keksnicoh
"""
import numpy as np
from . import settings


def fmap_nested_list(f, lst):
    # XXXXXXX TEST & FINALIZE
    return [
        f(a) if not isinstance(a, list) else fmap_nested_list(f, a)
        for a in lst
    ]

def reshape_list(lst, shape):
    """ reshapes a python list into a nested lists
        of lists of a given shape

        Arguments:
        ----------

        :lst: list to be reshaped

        :shape: tuple shape

        Returns:
        --------

        reshaped list

        """

    if len(shape) == 1:
        return lst

    cs = shape[-1]

    n = len(lst) / cs
    if not int(n) == n:
        err = 'list of length {} cannot be reshaped into {}'
        raise ValueError(err.format(len(lst), shape))

    return reshape_list(
        [lst[(i * cs):((i + 1) * cs)] for i in range(0, int(n))],
        shape[0:-1]
    )


def H(a):
    return a.conj().T

def is_H(a):
    return np.allclose(H(a), a, **settings.QOP.CLOSE_TOL)


def eigh(h0):
    """ returns eigensystem such that the states
        can be used to construct density operators
        from it.
        """
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


def sqmat(a, dtype=None):
    """ normalizes a mixed input to a single
        numpy array of shape (d, d). If not possible
        a `ValueError` is raised.

        Arguments:
        ----------

        :a: square-matrix-like input, e.g.

            [1, 2, 3, 4]
            (1, 2, 3, 4)
            [(1, 3), (4, 2)],
            np.array([1,2,3,4])
            ...

        Returns:
        --------

        np.ndarray with shape (d, d)

        """

    if isinstance(a, list):
        try:
            a = np.array(a, dtype=dtype or settings.QOP.T_COMPLEX)
        except ValueError as e:
            err = 'cannot be interpreted as a numpy matrix: {}'
            raise ValueError(err.format(e))

    if not isinstance(a, (np.ndarray, tuple)):
        err = "value of type {} not accepted."
        raise ValueError(err.format(type(err)))

    b = np.array(a)

    if len(b.shape) == 1:
        if not is_square(b.shape[0]):
            err = '1-dim shape (d, ): d must be square number, {} given.'
            raise ValueError(err.format(b.shape))

        sqint = int(np.sqrt(b.shape[0]))
        return b.reshape((sqint, sqint))

    if len(b.shape) == 2 and b.shape[0] == 1:
        if not is_square(b.shape[1]):
            err = '2-dim shape (1, d): d must be square number, {} given.'
            raise ValueError(err.format(b.shape))

        sqint = int(np.sqrt(b.shape[1]))
        return b.reshape((sqint, sqint))

    elif len(b.shape) == 2:
        if b.shape[0] != b.shape[1]:
            err = '2-dim shape (d, e): d must equal e, {} given.'
            raise ValueError(err.format(b.shape))

        return b

    elif len(b.shape) == 3 and b.shape[0] == 1:
        if b.shape[1] != b.shape[2]:
            err = '2-dim shape (1, d, e): d must equal e, {} given.'
            raise ValueError(err.format(b.shape))

        return b.reshape(b.shape[1:3])

    err = 'could not understand object of shape {} as square matrix'
    raise ValueError(err.format(b.shape))


def ketbra(s, i, j=None):
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
    elif np.isinf(T):
        return np.ones(len(E), dtype=settings.DTYPE_FLOAT)
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



def time_gatter(t0, tf, dt):
    # XXX Test me & doc me
    n = int(np.floor((tf - t0) / dt))
    if np.isclose(n * dt, tf - t0):
        return t0 + np.arange(n + 1) * dt
    return t0 + np.arange(n + 2) * dt



class InconsistentVectorSizeError(Exception):
    def __init__(self, msg, vectors):
        self.vectors = vectors
        description = "\n".join("    {:10}: {}".format(v[0], v[1].shape) for v in vectors)
        super().__init__("{}:\n{}".format(msg, description))
