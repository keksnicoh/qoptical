# -*- coding: utf-8 -*-
""" test some of the components of opme.
    :author: keksnicoh
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from qoptical import hamilton
from qoptical import settings

@pytest.mark.parametrize("args, kwargs, valid", [
    # valid
    [[[1, 1, 1, 1]], {}, True],
    [[np.array([1, 1, 1, 1])], {}, True],
    [[np.array([1, 1, 1, 1]).reshape((2, 2)),], {}, True],
    [[np.mat([1, 1, 1, 1])], {}, True],
    [[np.mat([1, 1, 1, 1]).reshape((2, 2))], {}, True],
    [[np.mat([1, 1, 1, 1]).reshape((2, 2)), [1, 1, 1, 1]], {}, True],
    [[np.mat([1, 1, 1, 1]).reshape((2, 2)), [1, 1, 1, 1]], {}, True],

    # invalid
    [[np.mat([1, 1, 1, 1]).reshape((2, 2))], {'dipole': [1, 2, 3]}, False],
    [[np.mat([1, 1, -1, 1]).reshape((2, 2))], {}, False],
    [[np.mat([1, 1, 1, 1]).reshape((2, 2)), [1, -1, 1, 1]], {}, False],
  ])
def test_invalid_constructor_args(args, kwargs, valid):
    """ test instantiation of ``ReducedSystem`` """
    if valid:
        hamilton.ReducedSystem(*args, **kwargs)
    else:
        try:
            hamilton.ReducedSystem(*args, **kwargs)
        except ValueError:
            return
        assert False

def test_trivial():
    """ test what happens if no dipole operator
        is given. The coupling should be 1 for all transitions.
        """
    h0 = np.diag([1, 2, 3])
    h0c = hamilton.ReducedSystem(h0)
    assert h0c.dipole is None
    assert h0c.dimH == 3

    # the most trivial dipole in eigenbase is full of 1 except
    # diagonal elements
    assert_allclose(h0c.dipole_eb(), np.array([
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    ]).reshape((3, 3)))

    # from 1-2,2-3 is w=1,
    # from 1-3     is w=2
    jumps = h0c.get_jumps()

    assert len(jumps) == 3

    assert jumps[0].shape == (1,)
    assert jumps[0]['w'][0] == 2
    assert jumps[0]['d'][0] == 1 # if no dipole given, than 1.0 should be default coupling
    assert jumps[0]['I'][0][0] == 0
    assert jumps[0]['I'][0][1] == 2

    assert jumps[1].shape == (2,)
    assert jumps[1]['w'][0] == 1
    assert jumps[1]['w'][1] == 1
    assert jumps[1]['d'][0] == 1
    assert jumps[1]['d'][1] == 1
    assert jumps[1]['I'][0][0] == 0
    assert jumps[1]['I'][0][1] == 1
    assert jumps[1]['I'][1][0] == 1
    assert jumps[1]['I'][1][1] == 2

def test_non_eb():
    """ test non eigenbase h0
        """
    h0 = np.array([
        1, 2j, 3j,
        -2j, 4, 4j,
        -3j, -4j, 9
    ])
    h0c = hamilton.ReducedSystem(h0)
    assert h0c.dipole is None
    assert h0c.dimH == 3

    # test eigensystem
    assert_allclose(h0c.h0 @ h0c.s[0].T, h0c.ev[0] * h0c.s[0].T, atol=1e-5, rtol=1e-7)
    assert_allclose(h0c.h0 @ h0c.s[1].T, h0c.ev[1] * h0c.s[1].T, atol=1e-5, rtol=1e-7)
    assert_allclose(h0c.h0 @ h0c.s[2].T, h0c.ev[2] * h0c.s[2].T, atol=1e-5, rtol=1e-7)


    # -- thermal_state() method tests
    rho_tinf = h0c.s[0:1].conj().T @ h0c.s[0:1]\
             + h0c.s[1:2].conj().T @ h0c.s[1:2]\
             + h0c.s[2:3].conj().T @ h0c.s[2:3]
    rho_tzero = h0c.s[0:1].conj().T @ h0c.s[0:1]

    # test thermal state at t=0
    assert_allclose(h0c.thermal_state(T=0), rho_tzero, atol=1e-5, rtol=1e-7)

    # test thermal state at t=inf (equally distributed)
    assert_allclose(h0c.thermal_state(T=np.inf), rho_tinf, atol=1e-5, rtol=1e-7)

    # does it work for multiple temperatures as well?
    rho = h0c.thermal_state(T=[0, np.inf])
    assert len(rho) == 2
    assert_allclose(rho[0], rho_tzero, atol=1e-5, rtol=1e-7)
    assert_allclose(rho[1], rho_tinf, atol=1e-5, rtol=1e-7)

    # -- rho2eb() method tests

    # test if op2eb works properly
    rho_eb_tinf = np.array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).reshape((3, 3))

    rho_eb_tzero = np.array([
        1, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ]).reshape((3, 3))

    assert_allclose(h0c.op2eb(h0c.thermal_state(T=0)), rho_eb_tzero, atol=1e-5, rtol=1e-7)
    assert_allclose(h0c.op2eb(h0c.thermal_state(T=np.inf)), rho_eb_tinf, atol=1e-5, rtol=1e-7)

    # does it work with multiple operators?
    rho_eb = h0c.op2eb(h0c.thermal_state(T=[0, np.inf]))
    assert_allclose(rho_eb[0], rho_eb_tzero, atol=1e-5, rtol=1e-7)
    assert_allclose(rho_eb[1], rho_eb_tinf, atol=1e-5, rtol=1e-7)

    # -- eb2op()
    # eb2op . op2eb = id
    op = np.array([[1,2,3,4,5,6,7,8,9], [1,3,2,5,2,7,2,2,5]]).reshape((2, 3, 3))
    assert_allclose(h0c.eb2op(h0c.op2eb(op)), op, atol=1e-5, rtol=1e-7)

    #print(h0c.ev[0] * h0c.s[0].reshape((3,)))
    jumps = h0c.get_jumps()

    assert len(jumps) == 3
    # XXX test jump setup, add dipole

def test_jumps_3gen():
    """ we let the system create the most general jumping
        pattern in this test. """
    # we expect the following jump pattern
    #
    # unique jumps
    # w = 6:
    #       4->0
    # w = 7:
    #       5->2
    # w = 8:
    #       5->1
    # w = 9:
    #       5->0
    # degenerated jumps (two jumps with same transition freq)
    # w = 1:
    #       1->0, 2->1
    # w = 2:
    #       3->1, 5->4
    # w = 4:
    #       3->0, 4->2
    # w = 5:
    #       4->1, 5->3
    # degenerates jumps (three jumps with same transition freq)
    # w = 2:
    #       2->0, 3->2, 4->3
    h0 = np.diag([
        -1,    #0
        0,     #1
        1,     #2
        3,     #3
        5,     #4
        8,     #5
    ])
    system = hamilton.ReducedSystem(h0)

    # the calculated spectrum should be equal to diagonal
    # elements of the given hamiltonian h0
    ev = np.diagonal(h0).astype(system.ev.dtype)
    assert np.all(system.ev == ev)

    freqs = system.get_possible_tw()
    jumps = system.get_jumps()

    # test unique jumps
    assert len(jumps[0]) == 4
    assert len(set(jumps[0][:]['w'])) == len(jumps[0])

    assert np.all(jumps[0][0]['I'] == (0, 4))
    assert ev[4] - ev[0] == jumps[0][0]['w']
    assert np.all(jumps[0][1]['I'] == (2, 5))
    assert ev[5] - ev[2] == jumps[0][1]['w']
    assert np.all(jumps[0][2]['I'] == (1, 5))
    assert ev[5] - ev[1] == jumps[0][2]['w']
    assert np.all(jumps[0][3]['I'] == (0, 5))
    assert ev[5] - ev[0] == jumps[0][3]['w']

    # test 2-degen jumps
    assert len(jumps[1]) == 4 * 2
    assert 2 * len(set(jumps[1][:]['w'])) == len(jumps[1])

    assert np.all(jumps[1][0]['I'] == (0, 1))
    assert np.all(jumps[1][1]['I'] == (1, 2))
    assert jumps[1][0]['w'] == jumps[1][1]['w']
    assert jumps[1][0]['w'] == ev[1] - ev[0]

    assert np.all(jumps[1][2]['I'] == (1, 3))
    assert np.all(jumps[1][3]['I'] == (4, 5))
    assert jumps[1][2]['w'] == jumps[1][3]['w']
    assert jumps[1][2]['w'] == ev[3] - ev[1]

    assert np.all(jumps[1][4]['I'] == (0, 3))
    assert np.all(jumps[1][5]['I'] == (2, 4))
    assert jumps[1][4]['w'] == jumps[1][5]['w']
    assert jumps[1][4]['w'] == ev[3] - ev[0]

    assert np.all(jumps[1][6]['I'] == (1, 4))
    assert np.all(jumps[1][7]['I'] == (3, 5))
    assert jumps[1][6]['w'] == jumps[1][7]['w']
    assert jumps[1][6]['w'] == ev[4] - ev[1]

    # test 3-degen jumps
    assert len(jumps[2]) == 3
    assert 3 * len(set(jumps[2][:]['w'])) == len(jumps[2])

    assert np.all(jumps[2][0]['I'] == (0, 2))
    assert np.all(jumps[2][1]['I'] == (2, 3))
    assert np.all(jumps[2][2]['I'] == (3, 4))
    assert jumps[2][0]['w'] == jumps[2][1]['w']
    assert jumps[2][0]['w'] == jumps[2][2]['w']
    assert jumps[2][0]['w'] == ev[2] - ev[0]


