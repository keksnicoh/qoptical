# -*- coding: utf-8 -*-
""" test the f2cl module
    :author: keksnicoh
"""
import pytest
from . import f2cl
import numpy as np
import math
import pyopencl as cl
import pyopencl.tools
from numpy import cos

devices     = cl.get_platforms()[0].get_devices()
gpu_devices = list(d for d in devices if d.type == cl.device_type.GPU)
ctx         = cl.Context(devices=gpu_devices)
queue       = cl.CommandQueue(ctx)

@pytest.mark.parametrize("f", [
    lambda t: 1.0,
    lambda t: 5*t,
    lambda t: np.sin(t),
    lambda t: np.cos(t),
    lambda t: np.tan(t),
    lambda t: np.sinh(t),
    lambda t: np.cosh(t),
    lambda t: np.tanh(t),
    lambda t: np.arcsin(t),
    lambda t: np.arccos(t),
    lambda t: np.arctan(t),
    lambda t: math.sin(t),
    lambda t: math.cos(t),
    lambda t: math.tan(t),
    lambda t: math.sinh(t),
    lambda t: math.cosh(t),
    lambda t: math.tanh(t),
    lambda t: math.asin(t),
    lambda t: math.acos(t),
    lambda t: math.atan(t),
    lambda t: np.sin(t)*2-1+3*np.cos(np.cos(t*t)),
])
def test_f2cl(f):
    r_clf = f2cl.f2cl(f, "bork_from_ork")
    # compile
    try:
        prg = cl.Program(ctx, r_clf + """
        __kernel void testf (
            __global const float *x,
            __global float *res
        ) {
          int gid = get_global_id(0);
          res[gid] = bork_from_ork(x[gid]);
        }
        """).build()
    except Exception as e:
        msg = "Could not compile. C-function:\n{}".format(r_clf)
        assert False, msg

    # run
    mf       = cl.mem_flags
    h_x      = np.arange(-1, 1, 0.1).astype(np.float32)
    h_return = np.empty_like(h_x)
    b_x      = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_x)
    b_return = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_return)
    prg.testf(queue, h_x.shape, None, b_x, b_return)
    cl.enqueue_copy(queue, h_return, b_return)

    # test
    for i, x in enumerate(h_x):
        a = h_return[i]
        b = f(x)
        if np.isnan(a) and np.isnan(b):
            continue
        msg = "{}-th item differs ({}, {}). C-function:\n{}".format(i, a, b, r_clf)
        assert np.isclose(a, b), msg


@pytest.mark.parametrize("f, param", [
    [lambda t: 1.0, None],
    [lambda t: 5*t, None],
    [lambda t: t+1-np.sin(math.cos(t))*45/np.tanh(0.5), None],
    # test differen names for the first arg
    [lambda x: 2.0 * x, None],
    [lambda x, p: np.sin(p['a'] * np.cos(x*6))-p['f'], (2, 0.4)],
    [lambda x, p: p['a'], (2, 0.4)],
])
def test_f2cl_param(f, param):
    dtype_param = np.dtype([('a', np.int32), ('f', np.float32) ]);
    r_clf = f2cl.f2cl(f, "bork_from_ork", "t_parameters")
    _, r_strct = cl.tools.match_dtype_to_c_struct(ctx.devices[0], "t_parameters", dtype_param)

    # compile
    try:
        prg = cl.Program(ctx, r_strct + "\n" + r_clf + """
        __kernel void testf (
            __global const float *x,
            __global const t_parameters *param,
            __global float *res
        ) {
          int gid = get_global_id(0);
          res[gid] = bork_from_ork(x[gid], param[0]);
        }
        """).build()
    except Exception as e:
        msg = "Could not compile. C-function:\n{}".format(r_clf)
        assert False, msg

    # run
    mf       = cl.mem_flags
    h_x      = np.arange(-1, 1, 0.1).astype(np.float32)
    h_return = np.empty_like(h_x)
    h_param  = np.array([param or (0, 0.0)], dtype=dtype_param)
    b_x      = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_x)
    b_return = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=h_return)
    b_param  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_param)
    prg.testf(queue, h_x.shape, None, b_x, b_param, b_return)
    cl.enqueue_copy(queue, h_return, b_return)

    # test
    pf = lambda x: f(x, h_param[0]) if param is not None else f(x)
    for i, x in enumerate(h_x):
        a = h_return[i]
        b = pf(x)
        if np.isnan(a) and np.isnan(b):
            continue
        msg = "{}-th item differs ({}, {}). C-function:\n{}".format(i, a, b, r_clf)
        assert np.isclose(a, b), msg