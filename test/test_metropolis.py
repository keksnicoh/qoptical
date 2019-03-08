# -*- coding: utf-8 -*-
""" test some of the components of opme.
    :author: keksnicoh
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from qoptical import metropolis
from qoptical import settings
from qoptical.settings import QOP

class RandMock():
    def __init__(self, dataset):
        if len(dataset) > 0:
            assert np.min(dataset) >= 0
            assert np.max(dataset) <= 1
        self.dataset = dataset
        self.idx = 0

    def __call__(self, l):
        if isinstance(l, int):
            l = (l, )

        if len(self.dataset) <= self.idx:
            err = 'RandMock was not expected to be invoked more than {} times.'
            raise Exception(err.format(len(self.dataset)))

        current = np.array(self.dataset[self.idx])
        if current.shape != l:
            err = 'RandMock expected first argument to be {} on invokation #{}, {} given'
            raise Exception(err.format(current.shape, self.idx + 1, l))

        self.idx += 1
        return current


def test_mc_param_no_boundaries():
    """
    this test mocks the random generator such that we
    can control the paramer field and the sign of change.
    """

    # prepare
    memc = metropolis.MetropolisMC(fields=['a', 'b', 'c'], beta=1)
    KM, p = memc.keymap, \
            memc.zero_param(2)

    # initialize
    p[0, KM['a']] = (1.0, 0.1, -np.inf, np.inf, 0.3)
    p[1, KM['a']] = (2.0, 0.5, -np.inf, np.inf, 0.1)
    p[0, KM['b']] = (0.0, 0.3, -np.inf, np.inf, 2.0)
    p[1, KM['b']] = (5.0, 0.1, -np.inf, np.inf, 1.0)
    p[0, KM['c']] = (3.0, 0.3, -np.inf, np.inf, 1.0)
    p[1, KM['c']] = (3.0, 0.1, -np.inf, np.inf, 2.0)

    ## first ------------------------------------------------

    memc.rand = RandMock([
        [0, 0], # first parameters
        [0, 0], # decrease
    ])
    p1 = memc.mc_param(p)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([
        [0, 0], # first parameters
        [.49, .49], # decrease
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.8, 1.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([
        [0.32, .032], # first parameters
        [.501, .501], # increase
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    ## second ------------------------------------------------

    memc.rand = RandMock([
        [0.34, 0.34], # second parameters
        [.501, .501], # increase
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.3, 5.1], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([
        [0.49, 0.66], # second parameters
        [.499, .499], # decrease
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    ## third ------------------------------------------------

    memc.rand = RandMock([
        [0.67, 0.77], # third parameters
        [.501, .501], # increase
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.3, 3.1], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([
        [0.67, 0.9999], # third parameters
        [.499, .499], # decrease
    ])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([0.9, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 5.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    ## mixed ------------------------------------------------

    memc.rand = RandMock([[0.0, 0.3411], [.501, .499]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.0, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 4.9], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[0.0, 0.3411], [.501, .499]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.1, 1.5], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 4.8], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([3.0, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[0.8, 0.1], [.1, .89]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.1, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 4.8], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)
    assert_allclose([2.7, 3.0], p1[:, KM['c']]['p'], **QOP.TEST_TOLS)


def test_mc_param_boundaries():
    """
    test whether the it is not possible to overshoot
    the parameter boundaries
    """

    # prepare
    memc = metropolis.MetropolisMC(fields=['a', 'b'], beta=1)
    KM, p = memc.keymap, \
            memc.zero_param(2)

    # initialize
    p[0, KM['a']] = (1.05, 0.1, 0.5, 1.2, 0.3)
    p[1, KM['a']] = (2.0, 0.2, 1.9, 2.4, 0.2)
    p[0, KM['b']] = (0.0, 0.3, 0, .5, 2.0)
    p[1, KM['b']] = (1.0, 1.0, -1, 1, 1.0)

    ## first ------------------------------------------------

    PA, PB   = 0.49, 0.51
    DEC, INC = 0.40, 0.60

    memc.rand = RandMock([[PA, PB], [INC, INC]])
    p1 = memc.mc_param(p)
    assert_allclose([1.15, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 0.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[PA, PB], [INC, INC]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.2, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 1.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[PA, PB], [INC, INC]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.1, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 0.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[PB, PA], [DEC, DEC]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.1, 1.9], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.3, 0.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[PB, PB], [DEC, DEC]])
    p, p1 = p1, memc.mc_param(p1)
    assert_allclose([1.1, 1.9], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, -1.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)


def test_mc_param_forbidden():
    """
    test behavior if forbidden values are defined.

    basically forbidden values avoid going back
    in the parameter evolution.
    """

    # prepare
    memc = metropolis.MetropolisMC(fields=['a', 'b'], beta=1)
    KM, p = memc.keymap, \
            memc.zero_param(2)

    # initialize
    p[0, KM['a']] = (1.05, 0.1, 0.5, 1.2, 0.3)
    p[1, KM['a']] = (2.0, 0.2, 1.9, 2.4, 0.2)
    p[0, KM['b']] = (0.0, 0.3, 0, .5, 2.0)
    p[1, KM['b']] = (1.0, 1.0, -1, 1, 1.0)

    ## first ------------------------------------------------

    PA, PB   = 0.49, 0.51
    DEC, INC = 0.40, 0.60

    memc.rand = RandMock([[PA, PB], [INC, INC]])
    p1 = memc.mc_param(p)
    assert_allclose([1.15, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, 0.0], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)

    memc.rand = RandMock([[PA, PB], [DEC, DEC]])
    p, p1 = p1, memc.mc_param(p1, p['p'])
    assert_allclose([1.2, 2.0], p1[:, KM['a']]['p'], **QOP.TEST_TOLS)
    assert_allclose([0.0, -1], p1[:, KM['b']]['p'], **QOP.TEST_TOLS)


@pytest.mark.parametrize("beta, random_numbers, optimize, x0, x1, xe", [
    [np.inf, None, metropolis.Optimize.MAXIMIZE, [0, 0, 0], [0, -0.001, 0.001], [0, 0, 0.001]],
    [np.inf, None, metropolis.Optimize.MINIMIZE, [0, 0, 0], [0, -0.001, 0.001], [0, -0.001, 0]],
    [
        1,
        np.array([0, 0.4999999, 0.5, 0.5000001]).reshape((4,)),
        metropolis.Optimize.MAXIMIZE,
        [0, 0,          0,          0],
        [0, -np.log(2), -np.log(2), -np.log(2)],
        [0, -np.log(2), 0, 0]
    ],
    [
        3.0,
        np.array([0, 0.5**3 - 0.001, 0.5**3, 0.5**3 + 0.001]).reshape((4,)),
        metropolis.Optimize.MAXIMIZE,
        [0, 0,          0,          0],
        [0, -np.log(2), -np.log(2), -np.log(2)],
        [0, -np.log(2), -np.log(2), 0]
    ],
])
def test_arg_discard(beta, random_numbers, optimize, x0, x1, xe):
    """
    test whether the correct indices are discared for different
    configurations.
    """
    x0, x1 = np.array(x0), np.array(x1)

    # test plain functional api --------------------
    rand     = RandMock([random_numbers]) if random_numbers is not None else None
    didx     = metropolis.arg_discard(optimize.value * (x0 - x1), beta, rand)
    xs       = np.array(x1, copy=True)
    xs[didx] = x0[didx]
    assert_array_equal(xe, xs)

    # test object api ----------------------------
    rand      = RandMock([random_numbers]) if random_numbers is not None else None
    memc      = metropolis.MetropolisMC(fields=['a', 'b'], beta=beta, optimize=optimize)
    memc.rand = rand

    didx     = memc.arg_discard(x0, x1)
    xs       = np.array(x1, copy=True)
    xs[didx] = x0[didx]
    assert_array_equal(xe, xs)


def test_arg_eps_bad_shape():
    """
    test some argument validation
    """
    try:
        metropolis.arg_eps(np.zeros((2,3)), np.zeros((2,4)))
    except ValueError as e:
        assert str(e) == "p.shape (2, 3) does not equals eps.shape (2, 4)"
        return
    assert False


def test_arg_eps_zero():
    """
    test minimal not working example
    """
    P = np.array([[[1]], [[2]]])
    EPS = np.array([[[0.4]], [[0.4]]])
    idx = metropolis.arg_eps(P, EPS)
    assert_array_equal([[0],], idx)


def test_arg_eps():
    """
    test multiple complex trajectories
    """
    KM, p = metropolis.zero_param(4, ['a', 'b'])

    _ = (.0, .0, .0)
    p[:, KM['a']] = [(0.1, *_, 0.1),   # will just increase by constant 0.1
                     (0.1, *_, 0.15),  # hopping around
                     (0.0, *_, 0.0),   # must be equal
                     (0.0, *_, 0.1)]   # good, but the last will be bad
    p[:, KM['b']] = [(1.0, *_, 0.1),   # will converge
                     (2.0, *_, 0.3),   # + 0.1, mean = 2.3
                     (3.0, *_, 0.3),   # + 0.1, mean = 3.3, eps adapative (1)
                     (4.0, *_, 0.3)]   # + 0.1, mean = 3.3, eps adaptive (2)
    P = np.array(p.reshape(1, *p.shape), copy=True)
    p[:, KM['a']] = [(0.2, *_, 0.1),
                     (-0.1, *_, 0.15),
                     (0.001, *_, 0.0),
                     (0.05, *_, 0.1)]
    p[:, KM['b']] = [(1.2, *_, 0.1),
                     (2.1, *_, 0.3),
                     (3.1, *_, 0.2),
                     (4.0, *_, 0.1)]
    P = np.concatenate((P, p.reshape(1, *p.shape)))
    p[:, KM['a']] = [(0.3, *_, 0.1),
                     (0.1, *_, 0.15),
                     (0.1, *_, 0.0),
                     (0.0, *_, 0.1)]
    p[:, KM['b']] = [(1.3, *_, 0.1),
                     (2.2, *_, 0.3),
                     (3.2, *_, 0.1),
                     (4.0, *_, 0.1)]
    P = np.concatenate((P, p.reshape(1, *p.shape)))
    p[:, KM['a']] = [(0.4, *_, 0.1),
                     (-0.1, *_, 0.15),
                     (0.11, *_, 0.0),
                     (0.05, *_, 0.1)]
    p[:, KM['b']] = [(1.35, *_, 0.1),
                     (2.3, *_, 0.3),
                     (3.3, *_, 0.0),
                     (4.0, *_, 0.0)]
    P = np.concatenate((P, p.reshape(1, *p.shape)))
    p[:, KM['a']] = [(0.5, *_, 0.1),
                     (0.1, *_, 0.15),
                     (0.111, *_, 0.0),
                     (0.0, *_, 0.1)]
    p[:, KM['b']] = [(1.40, *_, 0.1),
                     (2.4, *_, 0.3),
                     (3.4, *_, 0.1),
                     (4.0, *_, np.inf)]
    P = np.concatenate((P, p.reshape(1, *p.shape)))
    p[:, KM['a']] = [(0.6, *_, 0.1),
                     (-0.1, *_, 0.15),
                     (0.1111, *_, 0.0),
                     (0.05, *_, 0.1)]
    p[:, KM['b']] = [(1.42, *_, 0.1),
                     (2.5, *_, 0.3),
                     (3.5, *_, 0.2),
                     (4.0, *_, 0.2)]
    P = np.concatenate((P, p.reshape(1, *p.shape)))
    p[:, KM['a']] = [(0.7, *_, 0.1),
                     (0.1, *_, 0.15),
                     (0.1111, *_, 0.0),
                     (0.151, *_, 0.1)]
    p[:, KM['b']] = [(1.41, *_, 0.1),
                     (2.6, *_, 0.3),
                     (3.6, *_, 0.3),
                     (4.0, *_, np.inf)] # infinite should be allowed too
    P = np.concatenate((P, p.reshape(1, *p.shape)))

    # usually, the P records are ordered in time, so we must
    # invert them. Note that we build this test near the real
    # usage which is why we build a full parameter dtype
    # instead of only the 2 required fields.
    idx = metropolis.arg_eps(P['p'][::-1], P['eps'][::-1])

    assert_array_equal([
        [2, 5],
        [6, 6],
        [1, 3],
        [3, 6],
    ], idx)

