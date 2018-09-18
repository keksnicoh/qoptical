# -*- coding: utf-8 -*-
""" test some of the components of opme.
    :author: keksnicoh
"""

from qoptical import opme
from qoptical import settings
import numpy as np
import pytest

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
        tr=(0, 3, 0.0025))

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

    opme.opmesolve(H=h0f,
                        rho0=rho0,
                        dipole=j0f,
                        tr=(0, 0.2, 0.00005),
                        t_bath=10,
                        y_0=0.04,
                        kernel="QuTip")