# -*- coding: utf-8 -*-
""" global settings
   :author: keksnicoh
"""
import numpy as np
import os

class QOP():
    DEBUG = os.environ.get('QOP_DEBUG', '0') == '1'

    # precision section
    DOUBLE_PRECISION = os.environ.get('QOP_DOUBLE_PRECISION', '0') == '1'
    T_FLOAT = np.float32 if not DOUBLE_PRECISION else np.float64
    T_INT = np.int32 if not DOUBLE_PRECISION else np.int32
    T_COMPLEX  = np.complex64 if not DOUBLE_PRECISION else np.complex128
    CLOSE_TOL = {'atol': 1e-5, 'rtol': 1e-7}
    COMPLEX_ZERO_TOL = float(os.environ.get('QOP_TOL_COMPLEX', 1e-7))
    # the absolute and relative tolerance for two numbers to be equal.
    TEST_TOLS = {'atol': 1e-5, 'rtol': 1e-7}

def print_debug(msg, *args, **kwargs):
    print(('[\033[95m...\033[0m] ' + msg).format(*args, **kwargs))

# XXX old, remove me when nobody is accessing this anymore.
EQ_COMPARE_TOL = 0.0000001
DTYPE_FLOAT    = np.float32
DTYPE_INT      = np.int32
DTYPE_COMPLEX  = np.complex64
DEBUG = True
