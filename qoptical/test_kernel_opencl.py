# -*- coding: utf-8 -*-
""" OpenCL Kernel implementation tests.
"""
from .opme import ReducedSystem, opmesolve
from .kernel_qutip import QutipKernel
from .kernel_opencl import OpenCLKernel
from .util import ketbra, eigh
import pytest
import numpy as np

def test_von_neumann():
    """ we integrate von Neumann equation to test the
        following content:

        - reduced system with no transitions
          => von Neumann
        - evolve multiple states
        - all states at all times t should be recorded
          and be available in `result.tstate`
        - we test some physical properties of the results
          i)  desity operator properties at all t
          ii) behavior of coherent elements (rotate at certain w_ij)

        """

    PRECISION_DT_ANGLE = 6
    tr = (0.0, 13.37, 0.01)

    h0 = [0, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 3, 0,
          0, 0, 0, 5.5,]
    system = ReducedSystem(h0, tw=[])
    kernel = OpenCLKernel(system)
    kernel.compile()

    # we confige a state whith 3 coherent elements.
    # we expect that the diagonal elements are constant
    # in time while the coherent elements rotate at
    # the transition frrquency, meaning
    #
    #     d arg(<0|rho(t)|1>) * dt d = (w_1 - w_0) * 0.1 = 0.1
    #     d arg(<0|rho(t)|2>) * dt d = (w_2 - w_0) * 0.1 = 0.3
    #     d arg(<2|rho(t)|3>) * dt d = (w_3 - w_2) * 0.1 = 0.25
    #
    expect_w10, expect_w20, expect_w32 = 1.0, 3.0, 2.5
    ground_state = [
        0.7,  0.25, 0.5, 0.0,
        0.25, 0.2,  0.0, 0.0,
        0.5,  0.0,  0.0, 0.3,
        0.0,  0.0,  0.3, 0.1
    ]

    # this groundstate should be stationary.
    gs2 = [1, 0, 0, 0,
           0, 0, 0, 0,
           0, 0, 0, 0,
           0, 0, 0, 0,]
    kernel.sync(state=[ground_state, gs2], t_bath=0, y_0=0)
    result = kernel.run(tr)

    # Debug:
    #qkernel = QutipKernel(system)
    #qkernel.compile()
    #qkernel.sync(state=[ground_state, gs2], t_bath=0, y_0=1)
    #result = qkernel.run(np.arange(*tr))
    #print(np.round(result.state, 2))

    ts = result.tstate
    assert tstate_rho_hermitian(ts)
    assert tstate_rho_trace(1.0, ts)

    # test diagonal elements, r_00(t+dt) - r_00(t) = 0 for all t
    assert np.allclose(ts[:,0,0,0][:-1] - ts[:,0,0,0][1:], 0)
    assert np.allclose(ts[:,0,1,1][:-1] - ts[:,0,1,1][1:], 0)
    assert np.allclose(ts[:,0,2,2][:-1] - ts[:,0,2,2][1:], 0)
    assert np.allclose(ts[:,0,3,3][:-1] - ts[:,0,3,3][1:], 0)

    # test rotation of coherent elements by
    # calulating(r_01(t+dt) - r_01(t))/dt
    r10 = np.round(
        (np.angle(ts[:,0,1,0][:-1]) - np.angle(ts[:,0,1,0][1:])) % np.pi,
        PRECISION_DT_ANGLE
    )
    assert np.all(r10 == expect_w10 * tr[2])

    r20 = np.round(
        (np.angle(ts[:,0,2,0][:-1]) - np.angle(ts[:,0,2,0][1:])) % np.pi,
        PRECISION_DT_ANGLE
    )
    assert np.all(r20 == expect_w20 * tr[2])

    r32 = np.round(
        (np.angle(ts[:,0,3,2][:-1]) - np.angle(ts[:,0,3,2][1:])) % np.pi,
        PRECISION_DT_ANGLE
    )
    assert np.all(r32 == expect_w32 * tr[2])

    # test result data
    assert np.allclose(result.state, result.tstate[-1])


def tstate_rho_hermitian(ts):
    return np.all(np.abs(np.transpose(ts, (0, 1, 3, 2)).conj() - ts) < 0.0000001)


def tstate_rho_trace(expected, ts):
    trace = np.trace(ts, axis1=2, axis2=3).reshape((ts.shape[0] * 2))
    return np.allclose(trace, expected)


def test_von_neumann_basis():
    """ we integrate a system which is not provided in eigenbase.
        two states are tests:

            1. stationary (and pure) state |i><i|
            2. some non stationary state

        test checks basic integrator and density operator
        properties and compares the result against QuTip
        reference solver.
        """
    REF_TOL = 0.0001
    tr = (0, 1, 0.001)
    h0 = [
        1,   1.5,  0,
        1.5, 1.42, 3,
        0,   3,    2.11,
    ];

    ev, s = np.linalg.eigh(np.array(h0).reshape((3, 3)))
    s = s.T
    rho1 = np.outer(s[0].conj().T, s[0])
    rho2 = np.array([
        0.5, 0,   0,
        0,   0.5, 0,
        0,   0,   0
    ], dtype=np.complex64).reshape((3, 3))
    states = [rho1, rho2]

    system = ReducedSystem(h0, tw=[])
    kernel = OpenCLKernel(system)
    kernel.compile()
    kernel.sync(state=states, y_0=0, t_bath=0)
    result = kernel.run(tr)

    # test density operator
    ts = result.tstate
    assert tstate_rho_hermitian(ts[1:2])
    assert tstate_rho_trace(1.0, ts)

    # test result data
    assert np.allclose(result.state, result.tstate[-1])

    print(np.round(result.state[0], 4))
    print(np.round(rho1, 4))
    # test stationary state
   # assert np.allclose(result.state[0], rho1)

    # test against reference
    resultr = opmesolve(h0, states, 0, 0, tw=[], tlist=np.arange(*tr), kernel="QuTip")
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)

def test_two_level_TZero():
    """ most simple dissipative case.
        two level system with at T=0:

          d rho / dt = -i[H,rho] + y_0 \Omega^3 D[A(\Omega)]

        """
    REF_TOL = 0.0001
    OMEGA = 2.0
    tr = (0, 1.0, 0.001)
    y_0 = [0.5, 0.5, 0.25]
    h0 = [
        0, 0,
        0, OMEGA
    ]
    states = [
        # T=inf
        [0.5, 0.0, 0.0, 0.5],
        # T=0
        [1.0, 0.0, 0.0, 0.0],
        # T=t + coherence
        [0.75, 0.5, 0.5, 0.25],
    ]
    sys = ReducedSystem(h0, tw=[OMEGA])
    kernel = OpenCLKernel(ReducedSystem(h0, tw=[OMEGA]))
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=0)
    result = kernel.run(tr)

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=y_0, tw=[OMEGA], tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result.state[2] - resultr.state[2]) < REF_TOL)

def test_three_level_TZero():
    """ two different annihilation processes A(Omega), A(2*Omega) at T=0:

        - two possible jumps
        - no dipole
        - eigenbase
        - compared optimized vs. reference
        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 0.1, 0.001)
    tw      = [OMEGA, 2*OMEGA]

    h0 = [
        0.0, 0, 0,
        0, OMEGA, 0,
        0, 0, 2 * OMEGA,
    ]

    states = [[
        # T=inf
        1.0/3.0, 0.0, 0.0,
        0.0, 1.0/3.0, 0.0,
        0.0, 0.0, 1.0/3.0
    ], [
        # T=0
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], [
        # T=t + coherence
        0.4, 0.4, 0.6,
        0.4, 0.2, 0.2,
        0.6, 0.2, 0.4
    ]]
    sys = ReducedSystem(h0, tw=tw)
    kernel = OpenCLKernel(sys)
    kernel.compile()
    kernel.sync(state=states, y_0=1.0, t_bath=0)
    result = kernel.run(tr)

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=1.0, tw=tw, tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result.state[2] - resultr.state[2]) < REF_TOL)


def test_four_level_TZero():
    """ four level system at T=0.

        - all possible jumps
        - no dipole
        - eigenbase
        - compared optimized + non-optimized vs. reference
        """

    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 0.1, 0.001)
    h0 = [
        0.0, 0, 0, 0,
        0, 1.0, 0, 0,
        0, 0, 2.0, 0,
        0, 0, 0, 3.0,
    ]
    states = [[
        # T=0
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ], [
        # some weird state
        0.4, 0.4, 0.6, 0.3,
        0.4, 0.3, 0.2, 0.2,
        0.6, 0.2, 0.1, 0.6,
        0.3, 0.2, 0.6, 0.2,
    ]]

    sys = ReducedSystem(h0)

    kernel = OpenCLKernel(sys)
    kernel.compile()
    kernel.sync(state=states, y_0=0.15, t_bath=0)
    result = kernel.run(tr)

    kernel2 = OpenCLKernel(sys)
    kernel2.optimize_jumps = False
    kernel2.compile()
    kernel2.sync(state=states, y_0=0.15, t_bath=0)
    result2 = kernel2.run(tr)

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=0.15, tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result2.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result2.state[1] - resultr.state[1]) < REF_TOL)


def test_two_level_T():
    """ most simple dissipative case at finite temperature:
        two level system at T > 0:

          d rho / dt = -i[H,rho] + y_0 * \Omega^3 * (1 + N(\Omega)) * D[A(\Omega)]
                                 + y_0 * \Omega^3 * N(\Omega) * D[A^\dagger(\Omega)]

        - single jump
        - no dipole
        - eigenbase
        - compared optimized vs. reference
        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 1.0, 0.001)
    y_0     = 0.5
    t_bath  = 1.0

    h0 = [
        0, 0,
        0, OMEGA
    ]
    states = [[
        # T=inf
        0.5, 0.2-0.4j,
        0.2+0.4j, 0.5
    ], [
        # T=0
        1.0, 0.0,
        0.0, 0.0
    ]]

    sys = ReducedSystem(h0, tw=[OMEGA])

    kernel = OpenCLKernel(ReducedSystem(h0, tw=[OMEGA]))
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=t_bath)
    result = kernel.run(tr)


    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=y_0, tw=[OMEGA], tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)

def test_three_level_T():
    """ three level system at finite temperature.

        - two jumps (1*Omega, 2*Omega)
        - no dipole
        - eigenbase
        - compare optimized vs. reference

        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 0.5, 0.001)
    tw      = [OMEGA, 2*OMEGA]
    t_bath  = 1.0
    h0 = [
        0.0, 0, 0,
        0, OMEGA, 0,
        0, 0, 2 * OMEGA,
    ]
    states = [[
        # T=inf
        1.0/3.0, 0.0, 0.0,
        0.0, 1.0/3.0, 0.0,
        0.0, 0.0, 1.0/3.0
    ], [
        # T=0
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], [
        # T=t + coherence
        0.4, 0.4+0.25j, 0.6-0.5j,
        0.4-0.25j, 0.2, -0.2j,
        0.6+0.5j, 0.2j, 0.4
    ]]

    sys = ReducedSystem(h0, tw=tw)

    kernel = OpenCLKernel(sys)
    assert kernel.optimize_jumps
    kernel.compile()
    kernel.sync(state=states, y_0=1.0, t_bath=t_bath)
    result = kernel.run(tr)

    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=1.0, tw=tw, tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result.state[2] - resultr.state[2]) < REF_TOL)

def test_four_level_T():
    """ four level system at finite temperature T

        - all possible jumps
        - no dipole
        - eigenbase
        - compare optimized and non-optimized vs. reference
        """
    REF_TOL = 0.0001
    tr      = (0, 1.0, 0.001)
    t_bath  = [1.0, 0.5]
    y_0     = [1.3, 2.4]

    h0 = [
        0.0, 0, 0, 0,
        0, 1.0, 0, 0,
        0, 0, 2.0, 0,
        0, 0, 0, 6.0,
    ]

    states = [[
        # T=0
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ], [
        # some weird state
        0.4, 0.4, 0.6, 0.3,
        0.4, 0.3, 0.2, 0.1,
        0.6, 0.2, 0.1, 0.6,
        0.3, 0.2, 0.6, 0.2,
    ]]

    sys = ReducedSystem(h0)
    kernel = OpenCLKernel(sys)
    kernel.optimize_jumps = True
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=t_bath)
    result = kernel.run(tr)

    kernel2 = OpenCLKernel(sys)
    kernel2.optimize_jumps = False
    kernel2.compile()
    kernel2.sync(state=states, y_0=y_0, t_bath=t_bath)
    result2 = kernel2.run(tr)

    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=y_0, tlist=np.arange(*tr), kernel="QuTip")

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result2.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result2.state[1] - resultr.state[1]) < REF_TOL)


def test_two_level_T_driving():
    """ two level system at finite temperature with
        time dependent hamiltonian compared to reference
        implementation.
        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 1.0, 0.001)
    y_0     = 0.5
    t_bath  = 1.0
    h0      = [0, 0, 0, OMEGA]
    states  = [[1.0, 0.0, 0.0, 0.0]] * 3
    param   = np.array([
        (0.0, 0.0),
        (1.0, 2.0),
        (1.0, 2.5)
    ], dtype=np.dtype([
        ('A', np.float32, ),
        ('b', np.float32, ),
    ]))

    sys = ReducedSystem(h0, tw=[OMEGA])

    kernel = OpenCLKernel(ReducedSystem(h0, tw=[OMEGA]))
    kernel.t_sysparam = param.dtype
    kernel.ht_coeff = [lambda t, p: p['A'] * np.sin(p['b'] * t / np.pi)]
    kernel.compile()

    kernel.sync(state=states, y_0=y_0, t_bath=t_bath, sysparam=param, htl=[1, 1, 1, 1])
    result = kernel.run(tr)

    # reference result
    resultr = opmesolve(
        [h0, [[1, 1, 1, 1], kernel.ht_coeff[0]]],
        states,
        t_bath=t_bath,
        y_0=y_0,
        tw=[OMEGA],
        tlist=np.arange(*tr),
        kernel="QuTip",
        args=param)

    # test against reference
    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result.state[2] - resultr.state[2]) < REF_TOL)


def test_three_level_T_driving():
    """ three level system at finite temperature with
        time dependent hamiltonian compared to reference
        implementation.
        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 0.1, 0.0001)
    y_0     = 0.5
    t_bath  = 1.0
    h0      = [
        0, 0, 0,
        0, OMEGA, 0,
        0, 0, 4 * OMEGA
    ]
    states  = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
    param   = np.array([
        (0.0, 0.0),
        (1.0, 2.0),
        (1.0, 2.5),
    ], dtype=np.dtype([
        ('A', np.float32, ),
        ('b', np.float32, ),
    ]))
    htl = [
        0, 1+0.5j, -0.33j,
        1-0.5j, 0, 1,
        0.33j, 1, 0,
    ]
    sys = ReducedSystem(h0, tw=[OMEGA])

    kernel = OpenCLKernel(ReducedSystem(h0, tw=[OMEGA]))
    kernel.t_sysparam = param.dtype
    kernel.ht_coeff = [lambda t, p: p['A'] * np.sin(p['b'] * t * np.pi)]
    kernel.compile()

    kernel.sync(state=states, y_0=y_0, t_bath=t_bath, sysparam=param, htl=htl)
    result = kernel.run(tr)

    # reference result
    resultr = opmesolve(
        [h0, [htl, kernel.ht_coeff[0]]],
        states,
        t_bath=t_bath,
        y_0=y_0,
        tw=[OMEGA],
        tlist=np.arange(*tr),
        kernel="QuTip",
        args=param)

    assert np.all(np.abs(result.state[0] - resultr.state[0]) < REF_TOL)
    assert np.all(np.abs(result.state[1] - resultr.state[1]) < REF_TOL)
    assert np.all(np.abs(result.state[2] - resultr.state[2]) < REF_TOL)
