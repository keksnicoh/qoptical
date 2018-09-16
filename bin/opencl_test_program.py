#!/bin/python3
# -*- coding: utf-8 -*-
""" runs a parametrized opencl kernel for testing purpose.
    The program will calculate the integratl

    $$
        (2\\pi)^{-1} \\int_0^{2\\pi} dx f_{\\phi}(x)
    $$

    where $\\phi$ is a the value of the current matrix element
    (represented as a work-item).

    Example:
    --------

        calculate - in parrallel - a list of N MxM matricies
        which contains the parametrization

        M_1 = Matrix(phi^1_11, phi^1_12,
                     phi^1_21, phi^1_22)
        M_2 = Matrix(phi^2_11, phi^2_12,
                     phi^2_21, phi^2_22)
        ... = ...
        M_N = Matrix(...)

        The function to be calculated is

        f_a(x) = sin(x)*sin(x + a)

    The integration is performed numercally in the
    most simple (an not most accurate fashion)

    $$
        \\int_0^{2\\pi}dx -> \\Delta x \\sum_{k=0}^{(2\\pi)/(\\Delta x)}
    $$

    The integral has the following solution

    $$
        0.5 * \\cos(\\phi)
    $$

    :author: keksnicoh
    """

from time import time
import getopt
import sys
import pyopencl as cl
import numpy as np


def get_platform(platforms):
    """ select the platform from a list of **platforms**
        by a magical and overcomplicated rule.
        """
    return next(iter(platforms))


def print_debug_platforms(platforms):
    """ print platforms + devices
        """
    for p in platforms:
        print('PLATFORM {}'.format(p))
        for d in p.get_devices():
            print('- DEVICE {}'.format(d))


def get_gpu_devices(platform):
    """ get gpu devices for given **platform**
        or fails if not gpu devies were found.
        """
    gpu_devices = list(d for d in platform.get_devices() if d.type == cl.device_type.GPU)
    assert len(gpu_devices) > 0
    return gpu_devices


def run(config, double=False):
    """ runs the program by given config...
        if **double** then the program executed with
        double precision.
        """
    if double:
        fdtype = np.float64
        fcltype = 'double'
    else:
        fdtype = np.float32
        fcltype = 'float'

    # create ctx+queue
    platforms = cl.get_platforms()
    print_debug_platforms(platforms)

    ctx = cl.Context(devices=get_gpu_devices(get_platform(platforms)))
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, """
    #define M_PI 3.14159265358979323846
    __kernel void integrate(
        __global const $(float) *phi,
                 const $(float) dx,
        __global       $(float) *result
    ) {
        int gid = get_group_id(0) * get_local_size(1) * get_local_size(2);
        int tx  = get_local_id(1);
        int ty  = get_local_id(2);
        int idx = gid + get_local_size(1) * tx + ty;

        $(float) lphi = phi[idx];
        $(float) integratedf = 0.0;
        for ($(float) x = 0; x < 2 * M_PI; x += dx) {
            integratedf += sin(x) * sin(x + lphi);
        }

        result[idx] = dx / (2 * M_PI) * integratedf;
    }
    """.replace('$(float)', fcltype)).build()

    # RAM
    shape = (config['N'], config['M'], config['M'])
    dphi = 2 * np.pi / (config['N'] * config['M'] ** 2)
    host_phi = np.arange(0, 2 * np.pi, dphi, dtype=fdtype).reshape(shape)
    host_result = np.empty_like(host_phi)

    # GPU
    mf = cl.mem_flags
    b_phi = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_phi)
    b_result = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=host_result)
    dx = fdtype(2 * np.pi / config['n_int'])

    # -- kernel
    for i in range(config['n_kernel']):
        t0 = time()
        prg.integrate(queue, shape, (1, *shape[1:]), b_phi, dx, b_result)
        queue.finish()
        print('...invoke #{}: took {:.2f}s'.format(i, time() - t0))

    cl.enqueue_copy(queue, host_result, b_result)
    queue.finish()
    result = host_result.flatten()
    return result


def main():
    """ main main main!
        """
    thelp = 'wad? try: python3 opencl_test_program.py --n_int=234 --n_kernel=2 -M 2 -N 5 --double'
    double = False
    try:
        opts = getopt.getopt(sys.argv[1:], "M:N:n:k:d", ['n_int=', 'n_kernel=', 'double'])[0]
    except getopt.GetoptError:
        print(thelp)
        sys.exit(2)

    config = {
        'M': 2,
        'N': 4,
        'n_int': 1000,
        'n_kernel': 10,
    }

    for opt, arg in opts:
        if opt == '-h':
            print(thelp)
            sys.exit(0)
        elif opt in ("-M", ):
            config['M'] = int(arg)
        elif opt in ("-N", ):
            config['N'] = int(arg)
        elif opt in ("-n", "--n_int"):
            config['n_int'] = int(arg)
        elif opt in ("-k", "--n_kernel"):
            config['n_kernel'] = int(arg)
        elif opt in ("-d", "--double"):
            double = True
        else:
            raise NotImplementedError('sorryyyyyyyy!!1')

    result = run(config, double)
    dphi = 2 * np.pi / (config['N'] * config['M'] ** 2)
    dtype = np.float64 if double else np.float32
    expected = 0.5 * np.cos(np.arange(0, 2 * np.pi, dphi), dtype=dtype)

    print(result)
    print(expected)
    print(expected - result)

    sys.exit(0)


if __name__ == '__main__':
    main()
