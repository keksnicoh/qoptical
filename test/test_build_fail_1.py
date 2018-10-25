# -*- coding: utf-8 -*-
"""
    :author: keksnicoh
"""

import numpy as np
from numpy.testing import assert_allclose
import qoptical as qo
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

def test_build_fail_1():
    dimH = 11

    Oh0 = Odn2(dimH, qo.QO.T_COMPLEX) - Ocos(dimH, qo.QO.T_COMPLEX)
    rs  = qo.ReducedSystem(Oh0, dipole=Osin(dimH, qo.QO.T_COMPLEX))

    # -- Run with OpenCL kernel
    kernelCL = OpenCLKernel(rs)
    kernelCL.compile()
