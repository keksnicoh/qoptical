#!/bin/python3
# -*- coding: utf-8 -*-
"""
    :author: keksnicoh
    """

from time import time
import getopt
import sys
import os
import numpy as np
import qoptical as qo
from qoptical.kernel_opencl import OpenCLKernel

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


def run(config):

    tr = (0, 2, 0.00005)
    n_steps = len(qo.time_gatter(*tr))
    kernel = create_kernel(config['dimH'], tr)
    trhol = [list(np.arange(1, 1.01 + i * 0.01, 0.01)) for i in range(30)]

    report = {
        'dimH':    config['dimH'],
        'n_steps': n_steps,
        'trhol':   trhol,
        'benched': [],
        'mean':    [],
    }

    # wrap the inner OpenCL kernel invoke function to measure the time
    kernel_prg_invoke = kernel.prg.opmesolve_rk4_eb
    state = []
    def wrapped_invoke(*args, **kwargs):
        t0 = time()
        kernel_prg_invoke(*args, **kwargs)
        kernel.queue.finish()
        state.append(time() - t0)

    # go
    for trho in trhol:
        kernel.sync(state=kernel.system.thermal_state(trho))
        kernel.prg.opmesolve_rk4_eb = kernel_prg_invoke

        # warmup
        print('[...] warmup')
        kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=n_steps))
        kernel.queue.finish()
        kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=n_steps))
        kernel.queue.finish()

        kernel.prg.opmesolve_rk4_eb = wrapped_invoke

        benched = []
        for i in range(config['n_measure']):
            tlist_cl, resultCL = kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=n_steps))
            benched.append(state[-1])
       #     print('{} {} {:.4f}s'.format(len(trho), i, state[-1]))

        print('---------------------')
        print('{:.6f}s'.format(np.mean(benched)))
        report['benched'].append(benched)
        report['mean'].append(np.mean(benched))

    qo.persist_fs(config['fname'], report=report)


def create_kernel(dimH, tr):
    dtype_param = np.dtype([('a', np.float32)]);
    coeff = lambda t, p: p['a'] * np.cos(p['a'] * t)

    Oa  = op_a(dimH)
    Oad = Oa.conj().T
    On  = Oad @ Oa
    Ox  = Oa + Oad
    rs = qo.ReducedSystem(On, dipole=Ox)

    kernel = OpenCLKernel(system=rs, t_sysparam=dtype_param, ht_coeff=[coeff])
    kernel.compile()
    kernel.sync(
        t_bath=1.0,
        y_0=0.1,
        sysparam=np.array([(1, ), ], dtype=dtype_param),
        state=rs.thermal_state(0),
        htl=[Oa + Oad]
    )
    return kernel


def main():
    """ main main main!
        """
    thelp = 'wad? try: python3 opeDORK'
    double = False
    try:
        opts = getopt.getopt(sys.argv[1:], "h", ['dimH=', 'fname=', 'n_measure='])[0]
    except getopt.GetoptError as e:
        sys.exit(2)

    config = {
        'dimH': 5,
        'fname': None,
    }

    for opt, arg in opts:
        if opt == '-h':
            print(thelp)
            sys.exit(0)
        elif opt in ("--dimH"):
            config['dimH'] = int(arg)
        elif opt in ("--fname"):
            config['fname'] = arg
        elif opt in ("--n_measure"):
            config['n_measure'] = int(arg)
        else:
            print('HM?!')
            exit(1)

    if os.path.exists(config['fname']):
        print('{} exists.'.format(config['fname']))
        exit(42)

    run(config)
    sys.exit(0)


if __name__ == '__main__':
    main()
