# -*- coding: utf-8 -*-
""" contains tests with the harmonic oscillator.
    :author: keksnicoh
"""

import numpy as np
from qoptical.hamilton import ReducedSystem
from qoptical.kernel_qutip import QutipKernel
from qoptical.kernel_opencl import OpenCLKernel
from qoptical.util import thermal_dist, time_gatter
from numpy.testing import assert_allclose
RTOL = 0.00001

def op_a(n):
    """ 0          sqrt(1)     0           0           0
        0          0           sqrt(2)     0           0
        0          0           0           sqrt(3)     0
        0          0           0           0           sqrt(4)
        0          0           0           0           0
        """
    Oa = np.zeros(n ** 2, dtype=np.complex64).reshape((n, n))
    ar = np.arange(n-1)
    Oa[ar, ar+1] = np.sqrt(ar + 1)
    return Oa

def test_complex_dipole_complex():
    """ test whether complex components in dipole
        operator are interpreted correctly.
        """

    # system parameters
    dimH        = 5
    t_bath      = [0.0, 0.2, 0.5, 0.6]
    y_0         = 0.15
    Omega       = 1.4
    tr          = (0, 16.52, 0.005)
    state0      = np.zeros(dimH**2).reshape((dimH, dimH))
    state0[0,0] = 1

    # operators
    Oa  = op_a(dimH)
    Oad = Oa.conj().T
    On  = Oad @ Oa
    Ox  = Oa + Oad

    # non-complex dipole
    dipole = [
        0,              2+1j,       0,          3-np.pi*0.1j,   0,
        2-1j,           0,          -1j,        0,              4,
        0,              1j,         0,          5,              0,
        3+np.pi*0.1j,   0,          5,          0,              6j,
        0,              4,          0,          -6j,             0,
    ];
    # reduced system
    rs = ReducedSystem(Omega * On, dipole=dipole)

    # -- Run with QuTip
    kernel = QutipKernel(rs)
    kernel.compile()
    kernel.sync(t_bath=t_bath, y_0=y_0, state=[state0]*4)
    tlist = time_gatter(*tr)
    result = kernel.run(tlist).tstate

    # -- Run with OpenCL kernel
    kernelCL = OpenCLKernel(rs)
    kernelCL.compile()
    kernelCL.sync(t_bath=t_bath, y_0=y_0, state=state0.flatten())
    tlist_cl, resultCL = kernelCL.reader_rho_t(kernelCL.run(tr, steps_chunk_size=132))


    # -- compare all states at all times
    assert_allclose(tlist, tlist_cl)
    assert_allclose(resultCL, result, atol=1e-5, rtol=1e-7)
