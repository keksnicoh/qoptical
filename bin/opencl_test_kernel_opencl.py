#!/bin/python3
# -*- coding: utf-8 -*-
"""
    :author: keksnicoh
    """

from time import time
import getopt
import sys
import pyopencl as cl
import numpy as np
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('..', '..', 'lib'))
if module_path not in sys.path:
    sys.path.append(module_path)
from qoptical.math import dft_single_freq_window
from lib.phasespace import *
from lib.probing import opmesolve_probe_voltage
import qoptical as qo
DFLOAT   = np.float32
DCOMPLEX = np.complex64


EC = 1
tp0 = 0
wp = [0.025 / 2, 0.05 / 2]#, 0.05, 0.025
Omega = 3

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


def run(cconfig):
    """ runs the program by given config...
        if **double** then the program executed with
        double precision.
        """

    # create ctx+queue
    platform = cl.get_platforms()[cconfig['platform']]
    device = get_gpu_devices(platform)[cconfig['device']]
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    result = []
    fnamepre = "jeff/EC{:.4f}OM{:.4f}".format(cconfig['EC'], cconfig['Omega'])
    fname = cconfig['fname']
    if not os.path.exists(fname):
        config = {
            'dimH':   cconfig['dimH'],
            'EC':     cconfig['EC'],
            'Omega':  cconfig['Omega'],
            'y_0':    0.02,
            't_bath': cconfig['TBath'] * cconfig['Omega'],
            'wp':     cconfig['wp'],
            'drv':    [(float(wd1), 0.3) for wd1 in np.arange(
                cconfig['wd0'],
                cconfig['wd1'],
                cconfig['dwd'])
            ],
            'tp0':    0,
            'tr':     (0, int(np.ceil(min_time_wp(cconfig['wp'], cconfig['Omega'], cconfig['dp']))) + tp0, 0.0005),
        }
        qo.persist_fs(fname, **opmesolve_jeff_cl_driving_phasespace(config, ctx, queue))

    result.append(qo.load_fs(fname, ['sys_ids', 'jeff', 'config', 'systems', 'texpect_v100']))

def min_time_wp(wp, Omega, Np):
    return Np * 2.0 * np.pi / (Omega * np.min(wp))



def opmesolve_jeff_cl_driving_phasespace(config, ctx, queue):
    pf       = 1.0 / config['EC'] * config['Omega'] ** 2
    Oh0      = 0.5 * config['EC'] * Odn2(config['dimH'], DCOMPLEX) \
             - pf * Ocos(config['dimH'], DCOMPLEX)
    Ohp      = Ophi(config['dimH'], DCOMPLEX)
    Ov       = config['EC'] * Odn(config['dimH'], DCOMPLEX)
    hu_drive = [Ocos(config['dimH'], DCOMPLEX), lambda t, p: p['Ad1'] * np.sin(p['wd1'] * t)]
    hu_probe = [Ohp, lambda t, p: p['Ap'] * np.sin(p['wp'] * t)]
    tr       = config['tr']

    systems = np.array([
        (wp * config['Omega'], 0.01, w * config['Omega'], pf * a)
        for (w, a) in config['drv']
        for wp in config['wp']
    ], dtype=np.dtype([
        ('wp', DFLOAT),
        ('Ap', DFLOAT),
        ('wd1', DFLOAT),
        ('Ad1', DFLOAT),
    ]))

    reduced_system = qo.ReducedSystem(Oh0, dipole=Osin(config['dimH'], DCOMPLEX))

    result = opmesolve_jeff(
        tr=(tr[0], tr[1] + tr[2], tr[2]),
        reduced_system=reduced_system,
        hu_probe=hu_probe,
        hu_drive=hu_drive,
        Ov=1.0j * Ov,
        systems=systems,
        t_bath=config['t_bath'],
        y_0=config['y_0'],
        tp0=config['tp0'],
        rec_skip=16,
        ctx=ctx,
        queue=queue
    )

    result.update({'config': config})
    return result

def opmesolve_jeff(tr, reduced_system, hu_probe, hu_drive, Ov, systems, t_bath, y_0, tp0, rec_skip, ctx=None, queue=None):
    texpect_v = opmesolve_probe_voltage(
        (tr[0], tr[1] + tr[2], tr[2]),
        reduced_system,
        hu_probe,
        hu_drive,
        Ov,
        systems,
        t_bath,
        y_0,
        rec_skip=rec_skip,
        ctx=ctx,
        queue=queue
    )
    Nwp = len(set(systems['wp']))
    Ndrv = int(systems.shape[0] / Nwp)
    vwp = dft_single_freq_window(texpect_v, systems['wp'], dt=rec_skip * tr[2], tperiod=(tp0, tr[1]))
    sigma = 1.0 / vwp
    return {
        'jeff':         np.sum((sigma.imag * systems['wp']).reshape((Ndrv, Nwp)), axis=1),
        'sys_ids':      np.arange(len(systems))[::Nwp],
        'texpect_v100': texpect_v,
        'systems':      systems,
    }


def main():
    """ main main main!
        """
    thelp = 'wad? try: python3 opencl_test_program.py --n_int=234 --n_kernel=2 -M 2 -N 5 --double'
    double = False
    try:
        opts = getopt.getopt(sys.argv[1:], "hd:p:", ['fname=', 'device=', 'platform=', 'dp=', 'TBath=', 'wd0=', 'wd1=', 'dwd=', 'dimH=', 'EC=', 'Omega=', 'wp='])[0]
    except getopt.GetoptError as e:
        print(e)
        print(thelp)
        sys.exit(2)

    config = {
        'device': None,
        'platform': None,
    }

    for opt, arg in opts:
        if opt == '-h':
            print(thelp)
            sys.exit(0)
        elif opt in ("-d", "--device"):
            config['device'] = int(arg)
        elif opt in ("-p", "--platform"):
            config['platform'] = int(arg)
        elif opt in ("--wd0"):
            config['wd0'] = float(arg)
        elif opt in ("--wd1"):
            config['wd1'] = float(arg)
        elif opt in ("--dwd"):
            config['dwd'] = float(arg)
        elif opt in ("--dimH"):
            config['dimH'] = int(arg)
        elif opt in ("--dp"):
            config['dp'] = int(arg)
        elif opt in ("--EC"):
            config['EC'] = float(arg)
        elif opt in ("--Omega"):
            config['Omega'] = float(arg)
        elif opt in ("--TBath"):
            config['TBath'] = float(arg)
        elif opt in ("--fname"):
            config['fname'] = arg
        elif opt in ("--wp"):
            config['wp'] = [float(a.strip()) for a in arg.split(',')]
        else:
            print(opt)
            raise NotImplementedError('sorryyyyyyyy!!1')

    run(config)
    sys.exit(0)


if __name__ == '__main__':
    main()
