# -*- coding: utf-8 -*-
""" global settings
   :author: keksnicoh
"""
import numpy as np

EQ_COMPARE_TOL = 0.0000001
DTYPE_FLOAT    = np.float32
DTYPE_INT      = np.int32
DTYPE_COMPLEX  = np.complex64
DEBUG = True

def print_debug(msg, *args, **kwargs):
    print(('[\033[95m...\033[0m] ' + msg).format(*args, **kwargs))