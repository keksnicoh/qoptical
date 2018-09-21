# -*- coding: utf-8 -*-
""" OpenCL Kernel implementation tests.
"""
from qoptical.opme import opmesolve
from qoptical.hamilton import ReducedSystem
from qoptical.kernel_qutip import QutipKernel
from numpy.testing import assert_allclose
from qoptical.kernel_opencl import OpenCLKernel
from qoptical.util import ketbra, eigh
import pyopencl as cl
import pytest
import numpy as np

TOLS = {'atol': 1e-5, 'rtol': 1e-7}

def test_von_neumann():
    """ integrate von Neumann equation to test the following:

        - reduced system with no transitions => von Neumann
        - evolve multiple states
        - all states at all times t should be recorded
          and be available in `result.tstate`
        - we test some physical properties of the results
          i)  desity operator properties at all t
          ii) behavior of coherent elements (rotatation at certain w_ij)

        """

    PRECISION_DT_ANGLE = 6
    tr = (0.0, 13.37, 0.01)

    h0 = [0, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 3, 0,
          0, 0, 0, 5.5,]

    # dipole coupling = 0 => no dissipative dynamics
    system = ReducedSystem(h0, np.zeros_like(h0))
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

    # note that for y_0=0.0 the dissipator would vanish as well.
    kernel.sync(state=[ground_state, gs2], t_bath=0, y_0=1.0)
    tlist, ts = kernel.reader_rho_t(kernel.run(tr))

    # test times
    assert_allclose(np.arange(tr[0], tr[1] + tr[2], tr[2]), tlist)

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

    system = ReducedSystem(h0)
    kernel = OpenCLKernel(system)
    kernel.compile()

    # we archive von Neumann by setting global damping to y_0=
    # which leads to supression of dissipative terms
    kernel.sync(state=states, y_0=0, t_bath=0)
    tlist, ts = kernel.reader_rho_t(kernel.run(tr))

    # test times
    assert_allclose(np.arange(tr[0], tr[1] + tr[2], tr[2]), tlist)

    # test density operator
    assert tstate_rho_hermitian(ts[1:2])
    assert tstate_rho_trace(1.0, ts)

    # test stationary state
    assert_allclose(ts[-1][0], rho1, **TOLS)

    # test against reference
    resultr = opmesolve(h0, states, 0, 0, tw=[], tr=tr, kernel="QuTip")
    assert_allclose(ts[-1][0], resultr.state[0], **TOLS)
    assert_allclose(ts[-1][1], resultr.state[1], **TOLS)

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
    kernel = OpenCLKernel(ReducedSystem(h0, [
        0, 1,
        1, 0,
    ]))
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=0)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=y_0, tw=[OMEGA], tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof[2], resultr.state[2], **TOLS)

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
    sys = ReducedSystem(h0, [
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    ])
    kernel = OpenCLKernel(sys)
    kernel.compile()
    kernel.sync(state=states, y_0=1.0, t_bath=0)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=1.0, tw=tw, tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof[2], resultr.state[2], **TOLS)

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
    print(kernel.c_kernel)
    kernel.sync(state=states, y_0=0.15, t_bath=0)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    kernel2 = OpenCLKernel(sys)
    kernel2.optimize_jumps = False
    kernel2.compile()
    kernel2.sync(state=states, y_0=0.15, t_bath=0)
    tf, rhof2 = kernel.reader_tfinal_rho(kernel2.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=0, y_0=0.15, tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof2[0], resultr.state[0], **TOLS)
    assert_allclose(rhof2[1], resultr.state[1], **TOLS)

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

    kernel = OpenCLKernel(ReducedSystem(h0, [
        0, 1,
        1, 0,
    ]))
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=t_bath)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=y_0, tw=[OMEGA], tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)


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

    sys = ReducedSystem(h0, [
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    ])

    kernel = OpenCLKernel(sys)
    assert kernel.optimize_jumps
    kernel.compile()
    kernel.sync(state=states, y_0=1.0, t_bath=t_bath)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=1.0, tw=tw, tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof[2], resultr.state[2], **TOLS)

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
        0.4, 0.3, 0.2, 0.2,
        0.6, 0.2, 0.1, 0.6,
        0.3, 0.2, 0.6, 0.2,
    ]]

    sys = ReducedSystem(h0)
    kernel = OpenCLKernel(sys)
    kernel.optimize_jumps = True
    kernel.compile()
    kernel.sync(state=states, y_0=y_0, t_bath=t_bath)
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=1111))

    # test final time
    assert np.isclose(tf, tr[1])

    kernel2 = OpenCLKernel(sys)
    kernel2.optimize_jumps = False
    kernel2.compile()
    kernel2.sync(state=states, y_0=y_0, t_bath=t_bath)
    tf, rhof2 = kernel.reader_tfinal_rho(kernel2.run(tr))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(h0, states, t_bath=t_bath, y_0=y_0, tr=tr, kernel="QuTip")

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof2[0], resultr.state[0], **TOLS)
    assert_allclose(rhof2[1], resultr.state[1], **TOLS)


def test_time_gatter():
    """ in this test two debug buffers are injected into the
        OpenCL kernel:
        1. Time Buffer   -  to read out internal time at each time step
        2. Index Buffer  -  to read out internal output index at each time step.
        we compare the values, chunkwise, against expected times and indices.
        Also, the run() generator yields a triplet which we also test
        in this test.
        """
    rs = ReducedSystem([0, 0, 0, 1], [0, 1, 1, 0])
    kernel = OpenCLKernel(rs)
    kernel.c_debug_hook_1 = "time_gatter[2*n+get_global_id(0)] = t + 1337 * get_global_id(0);\n" \
        + "index_gatter[2*n+get_global_id(0)] = __out_len * n + __in_offset;\n"

    # we want 5 steps & two systems, we use a buffer of shape (7, 2) to check
    # whether the kernel overflow the 5 expected items.
    h_time = np.zeros((7, 2), dtype=np.float32)
    b_time = kernel.arr_to_buf(h_time)
    h_gatter = np.zeros((7, 2), dtype=np.int32)
    b_gatter = kernel.arr_to_buf(h_gatter)

    # hook & compile
    kernel.cl_debug_buffers = [
        ('__global float *time_gatter', b_time),
        ('__global int *index_gatter', b_gatter),
    ]
    kernel.compile()

    # syn
    kernel.sync(state=[1,0,0,0], y_0=1, t_bath=[1,1])

    expected_cl_tlists = [
        np.array([.000, .001, .002, .003, 0.004]),
        np.array([.005, .006, .007, .008, 0.009]),
        np.array([.010, .011, .012]),
    ]

    expected_tlists = [
        np.array([.001, .002, .003, .004, 0.005]),
        np.array([.006, .007, .008, .009, 0.010]),
        np.array([.011, .012, .013]),
    ]

    dt = 0.001
    # the first index (0) is allready occupied by the initial state
    # given at kernel.sync we therefore expect the idx to start from 1.
    current_index = 1;
    for j, (idx, tlist, rho_eb) in enumerate(kernel.run((0, 0.013, dt), steps_chunk_size=5)):
        # -- test index
        assert idx[0] == current_index
        i1 = idx[1] - idx[0]
        current_index += i1 + 1

        # -- test time used inside the kernel
        # note: we measure the time before increasing it, thus
        #  we expect a lattice like 0, 1, 2, 3 while the yielde
        #  tlist should be 1, 2, 3, 4 as tlist corresponds to the
        #  time at which the state rho_eb is.
        expected_tlist_cl = expected_cl_tlists[j]
        l = len(expected_tlist_cl)
        cl.enqueue_copy(kernel.queue, h_time, b_time)
        assert_allclose(expected_tlist_cl, h_time[:l, 0])
        assert_allclose(expected_tlist_cl + 1337, h_time[:l, 1])
        # test that buffer did not overflow
        assert_allclose(np.zeros(7 - l), h_time[l:, 0])
        assert_allclose(np.zeros(7 - l), h_time[l:, 1])
        # reset time buffer
        h_time = np.zeros_like(h_time)
        b_time = kernel.arr_to_buf(h_time)
        kernel.cl_debug_buffers[0] = (kernel.cl_debug_buffers[0], b_time)

        # -- test time yielded from python
        expected_tlist = expected_tlists[j]
        l = len(expected_tlist)
        assert_allclose(expected_tlist, tlist)

        # -- test index gatter
        cl.enqueue_copy(kernel.queue, h_gatter, b_gatter)
        expected_gatter = np.arange(l * 2).reshape((l, 2)) * 4
        assert_allclose(expected_gatter, h_gatter[0:len(expected_gatter)])
        expected_empty_gatter = np.zeros((7 - l) * 2).reshape((7 - l, 2))
        assert_allclose(expected_empty_gatter, h_gatter[len(expected_gatter):])

        # reset gatter buffer
        h_gatter = np.zeros_like(h_gatter)
        b_gatter = kernel.arr_to_buf(h_gatter)
        kernel.cl_debug_buffers[1] = (kernel.cl_debug_buffers[1], b_gatter)

        # test rho_eb
        assert rho_eb.shape[0] == l


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

    kernel = OpenCLKernel(ReducedSystem(h0, [
        0, 1,
        1, 0,
    ]))
    kernel.t_sysparam = param.dtype
    kernel.ht_coeff = [lambda t, p: p['A'] * np.sin(p['b'] * t / np.pi)]
    kernel.compile()

    kernel.sync(state=states, y_0=y_0, t_bath=t_bath, sysparam=param, htl=[[1, 1, 1, 1]])
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=1234))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(
        [h0, [[1, 1, 1, 1], kernel.ht_coeff[0]]],
        states,
        t_bath=t_bath,
        y_0=y_0,
        tw=[OMEGA],
        tr=tr,
        kernel="QuTip",
        args=param)

    # test against reference
    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof[2], resultr.state[2], **TOLS)

def test_three_level_T_driving():
    """ three level system at finite temperature with
        time dependent hamiltonian compared to reference
        implementation.
        """
    REF_TOL = 0.0001
    OMEGA   = 2.0
    tr      = (0, 1.0, 0.01)
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

    kernel = OpenCLKernel(ReducedSystem(h0, [
        0, 1, 0,
        1, 0, 0,
        0, 0, 0,
    ]))
    kernel.t_sysparam = param.dtype
    kernel.ht_coeff = [lambda t, p: p['A'] * np.sin(p['b'] * t * np.pi)]
    kernel.compile()

    kernel.sync(state=states, y_0=y_0, t_bath=t_bath, sysparam=param, htl=[htl])
    tf, rhof = kernel.reader_tfinal_rho(kernel.run(tr, steps_chunk_size=431))

    # test final time
    assert np.isclose(tf, tr[1])

    # reference result
    resultr = opmesolve(
        [h0, [htl, kernel.ht_coeff[0]]],
        states,
        t_bath=t_bath,
        y_0=y_0,
        tw=[OMEGA],
        tr=tr,
        kernel="QuTip",
        args=param)


    assert_allclose(rhof[0], resultr.state[0], **TOLS)
    assert_allclose(rhof[1], resultr.state[1], **TOLS)
    assert_allclose(rhof[2], resultr.state[2], **TOLS)

