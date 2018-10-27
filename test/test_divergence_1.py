# -*- coding: utf-8 -*-
""" for a system with dimH=13 we observed divergence of
    the state within OpenCL integrator when y>0. The QuTip
    reference implementation does not show such a behavior.
    This test reproduced the scanrio.

    Solution:
        - OpenCL Integrator uses hermitian property of density operator.
            rho[item_idx] = result
            rho[item_idxT] = conj(result)

        - It seems that we get some stable race condition because for
          diagonal elements the result was assigned twice.

        - Fix:
            rho[item_idx] = result
            if (not_diag_idx) rho[item_idxT] = conj(result)

        XXX

        - Why did the result not fluctuate in any sense but was
          completely stable?!

    :author: keksnicoh
"""

import numpy as np
from numpy.testing import assert_allclose
import qoptical as qo
from qoptical.kernel_qutip import QutipKernel
from qoptical.kernel_opencl import OpenCLKernel

def Ovoid(n, dtype):
    return np.zeros(n ** 2, dtype=dtype).reshape((n, n))

def Odn2(n, dtype):
    o, odd, m2 = Ovoid(n, dtype), n%2 != 0, int(n / 2)
    kr = np.array([i for i in range(-m2, m2 + 1) if odd or i != 0])
    o[np.arange(n), np.arange(n)] = kr**2
    return o

def Ocos(n, dtype):
    o, r = Ovoid(n, dtype), np.arange(n-1)
    o[r, r + 1] = o[r + 1, r] = 0.5
    return o

def Osin(n, dtype):
    o, r = Ovoid(n, dtype), np.arange(n-1)
    o[r, r + 1] = -0.5j
    o[r + 1, r] = 0.5j
    return o

def test_divergence_1():
    config = {
        'dimH':    13,
        'EC':      1,
        'Omega':   3,
        'y_0':     0.02,
        't_bath':  0.0,
        'tr':      (0, 20, 0.001),
        't_rho0':  [0],
    }

    pf  = 1.0 / config['EC'] * config['Omega'] ** 2
    Oh0 = 0.5 * config['EC'] * Odn2(config['dimH'], qo.QO.T_COMPLEX) \
        - pf * Ocos(config['dimH'], qo.QO.T_COMPLEX)
    rs  = qo.ReducedSystem(Oh0, dipole=Osin(config['dimH'], qo.QO.T_COMPLEX))

    # -- Run with QuTip
  #  kernel = QutipKernel(rs)
  #  kernel.compile()
  #  kernel.sync(t_bath=config['t_bath'], y_0=config['y_0'], state=rs.thermal_state(T=config['t_rho0']))
  #  tlist = qo.time_gatter(*config['tr'])
  #  result = kernel.run(tlist).tstate

    # -- Run with OpenCL kernel
    kernelCL = OpenCLKernel(rs)
    kernelCL.optimize_jumps = True
    kernelCL.compile()
    print(kernelCL.c_kernel)
    kernelCL.sync(t_bath=config['t_bath'], y_0=config['y_0'], state=rs.thermal_state(T=config['t_rho0']))
    tlist_cl, resultCL = kernelCL.reader_rho_t(kernelCL.run(config['tr'], steps_chunk_size=1e4))

    # -- compare all states at all times
    assert_allclose(tlist, tlist_cl)
    assert_allclose(resultCL, result, **qo.QO.TEST_TOLS)

