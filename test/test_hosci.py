# -*- coding: utf-8 -*-
""" contains tests with the harmonic oscillator.
    :author: keksnicoh
"""

import numpy as np
from qoptical.hamilton import ReducedSystem
from qoptical.kernel_qutip import QutipKernel
from qoptical.kernel_opencl import OpenCLKernel
from numpy.testing import assert_allclose
from qoptical.util import thermal_dist, time_gatter
from qoptical.settings import QOP

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

def test_thermalization():
    """ in this test we setup a harmonic oscillator

            H = \Omega n

        such that it couples thru x operator

            D = x = a^\dagger + a

        and check whether a list of ground states
        thermalize correctly.

        Testes Kernels:
        - QuTip
        - OpenCL
        """

    # system parameters
    dimH        = 5
    t_bath      = [0.0, 0.2, 1.0, 1.5]
    y_0         = 1.25
    Omega       = 1.0
    tr          = (0, 4.00, 0.005)
    state0      = np.zeros(dimH**2, dtype=np.complex64).reshape((dimH, dimH))
    state0[0,0] = 1
    Ee          = [Omega * k for k in range(dimH)]
    tlist       = time_gatter(tr[0], tr[1], tr[2])

    # operators
    Oa  = op_a(dimH)
    Oad = Oa.conj().T
    On  = Oad @ Oa
    Ox  = Oa + Oad

    # reduced system
    rs = ReducedSystem(Omega * On)

    # the expected mean occupation numbers
    #
    #     <E> = Tr(H rho_th) = Tr(H sum_i p_i e^(beta*Omega*i))
    #
    expected_En = [
        sum(e_i * p_i for (e_i, p_i) in zip(Ee, thermal_dist(Ee, t)))
        for t in t_bath
    ]

    # -- Run with QuTip
    kernel = QutipKernel(rs)
    kernel.compile()
    kernel.sync(t_bath=t_bath, y_0=y_0, state=[state0]*4)
    tlist = time_gatter(tr[0], tr[1], tr[2])
    (_, _, tstate, _) = kernel.run(tlist)

    # test whether states have thermalized
    En = np.trace(tstate[-1,:]@On, axis1=1, axis2=2).real
    assert np.allclose(expected_En, En, **QOP.TEST_TOLS)

    # -- Run with OpenCL kernel
    kernelCL = OpenCLKernel(rs)
    kernelCL.compile()
    kernelCL.sync(t_bath=t_bath, y_0=y_0, state=state0.flatten())
    tlist_cl, resultCL = kernelCL.reader_rho_t(kernelCL.run(tr))

    # test whether states have thermalized
    assert_allclose(tlist, tlist_cl)
    EnCL = np.trace(resultCL[-1,:]@On, axis1=1, axis2=2).real
    assert np.allclose(expected_En, EnCL, **QOP.TEST_TOLS), "{}".format(expected_En - EnCL)

