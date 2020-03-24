# -*- coding: utf-8 -*-
""" test the QuTip reference kernel implementation.
    :author: keksnicoh
"""
import numpy as np
import pytest
from qoptical.hamilton import ReducedSystem
from qoptical.kernel_qutip import QutipKernel
from qoptical.util import ketbra, eigh

EQ_COMPARE_TOL = 0.000001

def test_jump():
    """ test whether the qutip kernel is able to read
        jumps from system and creates the expected list of
        lindblad operators """
    # 1->0, 2->1    @ w=1
    #
    # 0   1   0   0
    # 0   0   1   0
    # 0   0   0   0
    # 0   0   0   0
    #
    # 2->0, 3->2    @ w=2
    #
    # 0   0   1   0
    # 0   0   0   0
    # 0   0   0   1
    # 0   0   0   0
    #
    # 4->1          @ w=3
    # 0   0   0   0
    # 0   0   0   1
    # ...
    #
    # 4->0          @ w=4
    # 0   0   0   1
    # 0   0   0   0
    # ...
    h0 = np.diag([0, 1, 2, 4])
    kernel = QutipKernel(ReducedSystem(h0))
    kernel.compile()
    lindblads = kernel.q_L

    assert 3.0 == lindblads[0][0]
    assert np.all(
        np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]) == lindblads[0][1].full())

    assert 4.0 == lindblads[1][0]
    assert np.all(
        np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]) == lindblads[1][1].full())

    assert 1.0 == lindblads[2][0]
    assert np.all(
        np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]) == lindblads[2][1].full())

    assert 2.0 == lindblads[3][0]
    assert np.all(
        np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]) == lindblads[3][1].full())

def test_qutip_kernel_single_frequency_transtions():
    """ in this test we setup a three state
        system where the energy levels are equidistant
        with deltaE = w0

        e3  ------------ 2 * w0

        e1  ------------ 1 * w0

        e0  ------------ 0

        Only jumps with w0 are allowed:
            e3 -> e2
            e2 -> e1
    """
    # like hosci
    T2, w0 = 1, 1.4
    h0 = [
        (0 * w0), 0, 0,
        0, (1 * w0), 0,
        0, 0, (2 * w0),
    ]

    # create system, only w0 transitions are allowed
    system = ReducedSystem(h0, [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0
    ])

    # create & compile qutip kernel
    kernel = QutipKernel(system)
    kernel.compile()

    # set two different initial states and two differen
    # bath temperatures. the global damping y0 is the
    # same for both systems.
    kernel.sync(
        state  = [
            [0, 0, 0,
             0, 1, 0,
             0, 0, 0],
            [1, 0, 0,
             0, 0, 0,
             0, 0, 0]],
        t_bath = [0, T2],
        y_0    = 2.5,
    )

    # the first state should go into groundstate
    expected_state_1 = np.array([
        1, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ]).reshape((3, 3))

    # the second state should converge to a thermal state at T2
    Z = np.exp(-1.0 / T2 * 0 * w0) \
      + np.exp(-1.0 / T2 * 1 * w0) \
      + np.exp(-1.0 / T2 * 2 * w0)
    expected_state_2 = np.diag([
        1.0 / Z * np.exp(-1.0 / T2 * 0 * w0),
        1.0 / Z * np.exp(-1.0 / T2 * 1 * w0),
        1.0 / Z * np.exp(-1.0 / T2 * 2 * w0),
    ]).reshape((3, 3))

    # we test if the kernel keeps track of the state
    # properly (sync_state=True). Executing a time interval
    # multiple times should be the same as perform
    # the integration once: V(t)V(t)...V(t) = V(t+t+...+t)
    required_steps = None
    times = np.arange(0.0, 0.1, 0.0025)
    for i in range(25):
        kernel.run(times, sync_state=True)
        close_1 = np.all(np.abs(kernel.state[0] - expected_state_1) < EQ_COMPARE_TOL)
        close_2 = np.all(np.abs(kernel.state[1] - expected_state_2) < EQ_COMPARE_TOL)
        if close_1 and close_2:
            required_steps = i
            break

    assert required_steps is not None, 'did not converge...'
    assert required_steps > 1, 'we want more than one step, please decrease y_0...'

def test_qutip_kernel_simple_htl():
    def coeff(t, args):
        return 1.0

    kernel = QutipKernel(ReducedSystem([
        0, 0, 0,
        0, 2, 0,
        0, 0, 4,
    ], [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
    ]), n_htl=1, n_e_ops=2)
    kernel.compile()
    kernel.sync(
        state = [
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
        ],
        t_bath=2, y_0=1,
        htl=[
            [[0, 1, 0,
              1, 0, 1,
              0, 1, 0], coeff]
        ],

        # expectation values Tr(0.5*rho) = 0.5, Tr(0.8*rho) = 0.8
        e_ops=[
            [.5, 0, 0, 0, .5, 0, 0, 0, .5],
            [.8, 0, 0, 0, .8, 0, 0, 0, .8]
        ]
    )

    tlist = np.arange(0, 2, 0.005)
    (_, _, _, texpect) = kernel.run(tlist)
    assert np.all(texpect[0, 0] == 0.5)
    assert np.all(texpect[0, 1] == 0.8)

def test_qutip_run_stateless():
    kernel = QutipKernel(ReducedSystem([
        0, 0, 0,
        0, 2, 0,
        0, 0, 4,
    ], [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
    ]))
    kernel.compile()
    kernel.sync(
        state  = [0, 0, 0, 0, 1, 0, 0, 0, 0],
        t_bath = 2,
        y_0    = 1)
    # test if both are the same (no sync)
    (_, fstate_a, _, _) = kernel.run(np.arange(0, .1, 0.005))
    (_, fstate_b, _, _) = kernel.run(np.arange(0, .1, 0.005))
    assert np.all(fstate_a[0][-1] == fstate_b[0][-1])

def test_qutip_fail_due_sync_with_e_ops():
    """ test is kernel.run fails whenever e_ops
        are given and sync_state is True """
    kernel = QutipKernel(ReducedSystem([
        0, 0, 0,
        0, 2, 0,
        0, 0, 4,
    ], [
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
    ]), n_e_ops=1)
    kernel.compile()
    kernel.sync(
        state  = [0, 0, 0, 0, 1, 0, 0, 0, 0],
        t_bath = 2,
        y_0    = 1,
        e_ops  = [[1, 0, 0, 0, 0, 0, 0, 0, 0]])
    try:
        kernel.run(np.arange(0, .1, 0.005), sync_state=True)
    except ValueError:
        pass


def test_qutip_kernel_nontrivial_basis():
    """ this test aims to test a system
        where h0 is non-diagonal.
    """
    h0 = np.array([-1, 0, 1, 0,
                   0, 1, 0, 0,
                   1, 0, 4, 1,
                   0, 0, 1, 12]).reshape((4, 4))
    ev, st = eigh(h0)
    T      = 0.5
    # partition sum
    Z = np.exp(-1.0 / T * ev[0]) \
      + np.exp(-1.0 / T * ev[1]) \
      + np.exp(-1.0 / T * ev[2]) \
      + np.exp(-1.0 / T * ev[3])
    # expected thermal state
    thermal_state = np.exp(-1.0 / T * ev[0]) / Z * ketbra(st, 0) \
                  + np.exp(-1.0 / T * ev[1]) / Z * ketbra(st, 1) \
                  + np.exp(-1.0 / T * ev[2]) / Z * ketbra(st, 2) \
                  + np.exp(-1.0 / T * ev[3]) / Z * ketbra(st, 3)
    # initial state
    rho0 = [1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,]

    kernel = QutipKernel(ReducedSystem(h0))
    kernel.compile()
    kernel.sync(state=rho0, t_bath=T, y_0=3)
    (_, fstate, _, _) = kernel.run(np.arange(0, 0.5, 0.001))
    assert np.all(np.abs(fstate.real - thermal_state) < EQ_COMPARE_TOL)

