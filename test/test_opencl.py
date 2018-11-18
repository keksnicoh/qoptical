# -*- coding: utf-8 -*-
""" this test tries to find all available gpu devices
    and performs a simple kernel. After calculation
    the result is compared to numpy result.
    :author: keksnicoh
"""

import pyopencl as cl
import numpy as np
import pytest

def test_gpu():
    # test platforms
    platforms = cl.get_platforms()
    assert len(platforms) > 0
    platform = platforms[0]

    # test gpu devices
    devices = platform.get_devices()
    assert len(devices) > 0
    gpu_devices = list(d for d in devices if d.type == cl.device_type.GPU)
    assert len(gpu_devices) > 0

    # test context & queue creation
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx)

    # create two buffers with 50k random numbers
    # (from pyopencl getting started tutorial)
    # src: https://documen.tician.de/pyopencl/
    a_np = np.random.rand(5000).astype(np.float32)
    b_np = np.random.rand(5000).astype(np.float32)
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # compare numpy CPU vs. GPU result
    assert np.all(res_np - (a_np + b_np) == np.zeros_like(a_np))
