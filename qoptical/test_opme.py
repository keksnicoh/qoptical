# -*- coding: utf-8 -*-
""" test some of the components of opme.
    :author: keksnicoh
"""

from . import opme
from . import settings
import numpy as np
import pytest

@pytest.mark.parametrize("args, kwargs, valid", [
    # valid
    [[[1,1,1,1]], {}, True],
    [[np.array([1,1,1,1])], {}, True],
    [[np.array([1,1,1,1]).reshape((2,2)),], {}, True],
    [[np.mat([1,1,1,1])], {}, True],
    [[np.mat([1,1,1,1]).reshape((2,2))], {}, True],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1]], {}, True],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1]], {}, True],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], 1, 0, []], {}, True],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], 0, 1, [1,2,3]], {}, True],

    # invalid
    [[np.mat([1,1,1,1]).reshape((2,2))], {'dipole': [1,2,3]}, False],
    [[np.mat([1,1,-1,1]).reshape((2,2))], {}, False],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,-1,1,1]], {}, False],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], -2, 5, []], {}, False],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], 2, -1, []], {}, False],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], 0, 0, [1,0]], {}, False],
    [[np.mat([1,1,1,1]).reshape((2,2)), [1,1,1,1], 0, 0, [-1]], {}, False],
  ] )
def test_reduced_system(args, kwargs, valid):
    """ test instantiation of ``ReducedSystem`` """
    if valid:
        opme.ReducedSystem(*args, **kwargs)
    else:
        try:
            opme.ReducedSystem(*args, **kwargs)
        except AssertionError:
            return
        assert False

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
    system = opme.ReducedSystem(h0)

    # the calculated spectrum should be equal to diagonal
    # elements of the given hamiltonian h0
    ev = np.diagonal(h0).astype(system.ev.dtype)
    assert np.all(system.ev == ev)

    freqs = system.get_possible_tw()
    jumps = system.get_jumps()

    # test unique jumps
    assert len(jumps[0]) == 4
    assert len(set(jumps[0][:]['w'])) == len(jumps[0])

    assert np.all(jumps[0][0]['I'] == (4, 0))
    assert ev[4] - ev[0] == jumps[0][0]['w']
    assert np.all(jumps[0][1]['I'] == (5, 2))
    assert ev[5] - ev[2] == jumps[0][1]['w']
    assert np.all(jumps[0][2]['I'] == (5, 1))
    assert ev[5] - ev[1] == jumps[0][2]['w']
    assert np.all(jumps[0][3]['I'] == (5, 0))
    assert ev[5] - ev[0] == jumps[0][3]['w']

    # test 2-degen jumps
    assert len(jumps[1]) == 4 * 2
    assert 2 * len(set(jumps[1][:]['w'])) == len(jumps[1])

    assert np.all(jumps[1][0]['I'] == (1, 0))
    assert np.all(jumps[1][1]['I'] == (2, 1))
    assert jumps[1][0]['w'] == jumps[1][1]['w']
    assert jumps[1][0]['w'] == ev[1] - ev[0]

    assert np.all(jumps[1][2]['I'] == (3, 1))
    assert np.all(jumps[1][3]['I'] == (5, 4))
    assert jumps[1][2]['w'] == jumps[1][3]['w']
    assert jumps[1][2]['w'] == ev[3] - ev[1]

    assert np.all(jumps[1][4]['I'] == (3, 0))
    assert np.all(jumps[1][5]['I'] == (4, 2))
    assert jumps[1][4]['w'] == jumps[1][5]['w']
    assert jumps[1][4]['w'] == ev[3] - ev[0]

    assert np.all(jumps[1][6]['I'] == (4, 1))
    assert np.all(jumps[1][7]['I'] == (5, 3))
    assert jumps[1][6]['w'] == jumps[1][7]['w']
    assert jumps[1][6]['w'] == ev[4] - ev[1]

    # test 3-degen jumps
    assert len(jumps[2]) == 3
    assert 3 * len(set(jumps[2][:]['w'])) == len(jumps[2])

    assert np.all(jumps[2][0]['I'] == (2, 0))
    assert np.all(jumps[2][1]['I'] == (3, 2))
    assert np.all(jumps[2][2]['I'] == (4, 3))
    assert jumps[2][0]['w'] == jumps[2][1]['w']
    assert jumps[2][0]['w'] == jumps[2][2]['w']
    assert jumps[2][0]['w'] == ev[2] - ev[0]


def test_opmesolve_thermal_state():
    result = opme.opmesolve(
        H=[0, 0, 0,
           0, 1, 0,
           0, 0, 2],
        rho0=[
            [0.5, 0, 0, 0, 0, 0, 0, 0, 0.5],
            [1, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        t_bath=[0.0, 1.0],
        y_0=2.0,
        tlist=np.arange(0, 3, 0.0025))

    # test if state converged to thermal state within 0.001 tol
    rho_e1 = np.diag([1, 0, 0])
    assert np.all(np.abs(rho_e1 - result.state[0]) < 0.001)

    Z = 1 + np.exp(-1.0) + np.exp(-2.0)
    rho_e2 = np.diag([1.0 / Z, np.exp(-1.0) / Z, np.exp(-2.0) / Z])
    assert np.all(np.abs(rho_e2 - result.state[1]) < 0.001)


def test_qutip_kernel_case_1():
    j0f = [[0.-0.j, 0.-2.j, 0.-0.j, 0.-0.j, 0.-0.j],
           [0.+2.j, 0.-0.j, 0.-2.44948974j, 0.-0.j, 0.-0.j,],
           [0.-0.j, 0.+2.44948974j, 0.-0.j, 0.-2.44948974j, 0.-0.j,],
           [0.-0.j, 0.-0.j, 0.+2.44948974j, 0.-0.j, 0.-2.j,],
           [0.-0.j, 0.-0.j, 0.-0.j, 0.+2.j, 0.-0.j,]]
    h0f = [[ 6.        +0.j,  -2.        +0.j,  -0.        +0.j,  -0.        +0.j, -0.        +0.j],
           [-2.        +0.j,   3.        +0.j,  -2.44948974+0.j,  -0.        +0.j, -0.        +0.j,],
           [-0.        +0.j,  -2.44948974+0.j,   2.        +0.j,  -2.44948974+0.j, -0.        +0.j,],
           [-0.        +0.j,  -0.        +0.j,  -2.44948974+0.j,   3.        +0.j, -2.        +0.j,],
           [-0.        +0.j,  -0.        +0.j,  -0.        +0.j,  -2.        +0.j, 6.         +0.j,],]
    rho0 = [[0.5, 0, 0, 0, -.5j,],
            [0,   0, 0, 0, 0,],
            [0,   0, 0, 0, 0,],
            [0,   0, 0, 0, 0,],
            [.5j,   0, 0, 0, 0.5,]]
    tlist = np.arange(0, 0.2, 0.00005)
    opme.opmesolve(H=h0f,
                        rho0=rho0,
                        dipole=j0f,
                        tlist=tlist,
                        t_bath=10,
                        y_0=0.04,
                        kernel="QuTip")