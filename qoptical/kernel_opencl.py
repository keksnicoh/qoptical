# -*- coding: utf-8 -*-
""" module containing OpenCL kernel.

    The OpenCL kernel computes a state (M x M) component
    per work-item. Many states can be computed in parallel.

    The precompiler analyzed the structure of the dynamics
    to generate the lines of code which interchange the
    state components. For the unitary operators one line
    of code per operator is generated:

       CELL = f(CELL, IDX_0, H_0)
       CELL = f(CELL, IDX_0, H_1, f_1(t, p))
       CELL = f(CELL, IDX_0, H_1, f_1(t, p))

       CELL = f(CELL, IDX_1, H_0)
       CELL = f(CELL, IDX_1, H_1, f_1(t, p))
       CELL = f(CELL, IDX_1, H_1, f_1(t, p))

       ...

       CELL = f(CELL, IDX_M, H_0)
       CELL = f(CELL, IDX_M, H_1, f_1(t, p))
       CELL = f(CELL, IDX_M, H_1, f_1(t, p))

    The dissipators are evaluated such that all resulting
    contributions into a CELL are known. On runtime,
    the temperature dependent prefactors are calculated and the
    values are accumulated such. For each jump one line of code
    is generated

       CELL += JUMP[0].PF * RHO[JUMP[0].IDX]
       CELL += JUMP[1].PF * RHO[JUMP[1].IDX]

       ...

       CELL += JUMP[N_J].PF * RHO[JUMP[N_J].IDX]

    with N_J max number of jumps to be performed (after optimization).

    Using this concept the kernel internally avoids many unwanted
    branchings.

    :author: keksnicoh
"""
import os
import sys
from time import time
import itertools
import numpy as np
import pyopencl as cl
import pyopencl.tools
from .f2cl import f2cl
from .settings import QOP, print_debug
from .util import (
    vectorize, npmat_manylike, InconsistentVectorSizeError,
    boson_stat, time_gatter)

mf = cl.mem_flags

def opmesolve_cl_expect(tg,
                        reduced_system,
                        t_bath,
                        y_0,
                        rho0,
                        Oexpect,
                        yt_coeff=None,
                        OHul=None,
                        params=None,
                        rec_skip=1,
                        kappa=None,
                        ctx=None,
                        queue=None,
                        steps_chunk_size=None):
    """ evolves expectation value on time gatte `tr`.

        Parameters:
        -----------

        :tg:             time gatter `(t0, tf, dt)`

        :reduced_system: Instance of the reduced system which configures the
                         jumping and coupling to the bath

        :t_bath:         float or vector of temperatures

        :y_0:            float or vector of global damping

        :rho0:           matrix or vector of matricies representing state at `tr[0]`

        :Oexpect:        the expectation value of this operator is the result
                         of this function.

        :OHul:           list of unitary hamiltonians

        :params:         vector or single system parameter tuple.
                         Must have a specific numpy dtype, e.g.:
                            ```python
                            p = np.array(
                                [(1,2)],
                                dtype=np.dtype([
                                    ('a', QOP.T_FLOAT),
                                    ('b', QOP.T_FLOAT)
                                ])
                            )
                            ```

        Result:
        -------

        an np.array of shape (len(time_gatter), N) where N is the number of systems.

        XXX: unit test this function

        """

    t_param = None
    if params is not None:
        t_param = params.dtype

    OHul = OHul or []
    ht_coeff = [ht[1] for ht in OHul]
    ht_op = [ht[0] for ht in OHul]
    kernel = OpenCLKernel(reduced_system,
                          t_sysparam=t_param,
                          ht_coeff=ht_coeff,
                          ctx=ctx,
                          queue=queue)

    if kappa is not None:
        kernel.kappa = kappa

    if yt_coeff is not None:
        kernel.yt_coeff = yt_coeff

    kernel.compile()

    kernel.sync(state=rho0, t_bath=t_bath, y_0=y_0, sysparam=params, htl=ht_op)

    # result reader / expectation value
    tlist = np.arange(*tg)[::rec_skip]
    result = np.zeros((len(tlist), kernel.N), dtype=np.complex64)
    Oeb = kernel.system.op2eb(Oexpect)

    # run
    for idx, tlist, rho_eb in kernel.run(tg, steps_chunk_size=steps_chunk_size or 1e4):
        sidx = (int(np.ceil(idx[0] / rec_skip)), int(np.ceil(idx[1] / rec_skip)))
        nrho = result[sidx[0]:sidx[1]+1].shape[0]
        result[sidx[0]:sidx[1]+1] = np.trace(rho_eb[::rec_skip][:nrho] @ Oeb, axis1=2, axis2=3)

    return result


def kappa(T, p, w):
    if T == 0:
        return w ** 3 * np.heaviside(w, 0)
    return w**3 * (1.0 + 1.0 / (np.exp(1.0 / T * w) - 1 + 0.0000000001))


class OpenCLKernel():
    """ Renders & compiles an OpenCL GPU kernel to
        integrate Quantum Optical Master Eqaution.
        """

    # kernel code file name
    TEMPLATE_NAME = 'kernel_opencl.tmpl.c'

    # before creating the final jump instructions (`DTYPE_T_JMP`),
    # this dtype is used to create a non-optimized buffer.
    # when all jumps are analyzed the resulting buffer is
    # accumulated (grouped by IDX column) to get the final
    # contribution of a matrix element into the work-item.
    # (See `cl_jmp_acc_pf`)
    DTYPE_JUMP_RAW = np.dtype([
        ('IDX', QOP.T_INT),    # source idx
        ('PF', QOP.T_COMPLEX), # prefactor (dipole)
        ('W', QOP.T_FLOAT),    # transition frequency
    ])

    # instruction to add a matrix element rho[IDX]
    # by a given, complex, prefactor. This is the final
    # (and optimized) representation of the dissipator
    # contribution from a matrix element into the
    # work-item's matrx element.
    DTYPE_T_JMP = np.dtype([
        ('IDX', QOP.T_INT),
        ('PF', QOP.T_COMPLEX),
    ])

    def __init__(self,
                 system,
                 ctx=None,
                 queue=None,
                 t_sysparam=None,
                 ht_coeff=None,
                 optimize_jumps=True,
                 debug=None):
        """ create QutipKernel for given ``system`` """

        self.system = system
        self.optimize_jumps = optimize_jumps
        self.t_sysparam = t_sysparam
        self.ctx = ctx or _ctx()
        self.queue = queue or cl.CommandQueue(self.ctx)
        self.debug = debug or QOP.DEBUG

        self._mb = None # base transformation matrix

        self.prg = None
        self.c_kernel = None # generated kernel c code
        self.cl_local_size = None # due to hermicity we only need M*(M+1)/2 work-items
                                  # where M is dimH

        # -- debug members
        # debug_hook_1 is rendered at the end of the loop just
        # before time inc.
        self.c_debug_hook_1 = None
        # extra buffer configuration,
        #
        # e.g.
        #
        #   [
        #       ('__global const float *dork', some_cl_buffer),
        #       ...
        #   ]
        self.cl_debug_buffers = []

        self.dimH = None
        self.N = None

        # time dependent hamiltonian
        self.t_sysparam_name = None
        self.ht_coeff = ht_coeff
        self.yt_coeff = None # time dependent global damping

        self.jmp_n = None # the number of jumping instructions due to dissipators
        self.jmp_instr = None # jumping instructions for all cells

        # synced, vectorized state
        self.hu = None
        self.state = None
        self.t_bath = None
        self.y_0 = None
        self.htl = None
        self.sysparam = None
        self.kappa = kappa

        # host buffers and gpu buffers
        self.h_sysparam = None
        self.h_htl = None
        self.h_hu = None
        self.h_state = None
        self.h_cl_jmp = None
        self.h_y_0 = None
        self.h_t_bath = None
        self.h_cl_jmp = None
        self.b_htl = None
        self.b_sysparam = None
        self.b_hu = None
        self.b_cl_jmp = None

        self.result = None

        self.init()


    def init(self):
        """ implicit initialization """
        self.hu = npmat_manylike(self.system.h0, [self.system.h0])
        self._mb = np.array(self.system.s, dtype=self.system.s.dtype)
        if self.debug:
            # because everyone likes ascii cats
            print_debug("")
            print_debug('rawrr!                              ,-""""""-.    OpenCL kernel v0.0')
            print_debug("                                 /\\j__/\\  (  \\`--.    RK4 integrator")
            print_debug("Compile me,                      \\`\033[36m@\033[0m_\033[31m@\033[0m'/  _)  >--.`.")
            print_debug("give me data                    _{{.:Y:_}}_{{{{_,'    ) )")
            print_debug("and I'll work that out 4u!     {{_}}`-^{{_}} ```     (_/")
            print_debug("")


    def __del__(self):
        if self.debug:
            print_debug('release buffers')
        if self.b_hu is not None:
            self.b_hu.release()
        if self.b_htl is not None:
            for b_ht in self.b_htl:
                b_ht.release()
        if self.b_cl_jmp is not None:
            self.b_cl_jmp.release()
        if self.b_sysparam is not None:
            self.b_sysparam.release()


    def compile(self):
        """ renders OpenCL kernel from given state and reduced
            system. Since the kernel renders code depending on
            the jump-layout it must be known at compile-time and
            cannot be changed without recompile. Currently, only
            one jump layout and only one set of time dependent
            Hamiltons are supported.

            In this approach, the kernel is designed to be "branchless".
            This may lead to sume unnecessary assignments.

            After compilation the rendered kernel code is accessible
            via the `OpenCLKernel.c_kernel`.

            """
        M = self.system.h0.shape[0]
        self.cl_local_size = int(M * (M + 1) / 2)

        # create jump instructions. Since the number of instructions
        # depends on the jumping instructions we need to calculate
        # them at this early stage.
        self.create_jmp_instr()

        # read template
        tpl_path = os.path.join(
            os.path.dirname(__file__),
            self.__class__.TEMPLATE_NAME
        )
        with open(tpl_path) as f:
            src = f.read()

        # ------ Initialize Rendering Parts

        # coefficient functions
        r_cl_coeff = ""
        # private buffer time dependent hamilton
        # matrix element allocations
        r_htl_priv = ""
        # kernel arguments for time dependent
        # hamilton buffer
        r_arg_htl = ""
        # coeff_tpl
        tpl_coeff = ""
        r_local_coeff = ""
        r_sysp = ""
        # coeff y(t)
        r_yt_coeff = ""
        # macro to refresh the value of time dependent
        # hamilton at a specific time.
        r_htl = ""
        # system parameters kernel argument
        r_arg_sysparam = ""
        # macro containg full dynamics at each timestep.
        r_define_rk = ""
        # rendered constants
        r_define = ""
        # thread layout
        r_tl = ""
        # debug code
        r_arg_debug = ""
        r_debug_hook_1 = self.c_debug_hook_1 or ''

        # ---- self.debug ARGS

        if len(self.cl_debug_buffers):
            r_arg_debug = "\n    " + ",\n    ".join(x[0] for x in self.cl_debug_buffers) + ','

        # ---- STRUCTS

        structs = [
            self._compile_struct('t_jump', self.__class__.DTYPE_T_JMP),
        ]

        # ---- SYSTEM PARAMETERS DTYPE

        if self.t_sysparam is not None:
            if not isinstance(self.t_sysparam, np.dtype):
                msg = 't_sysparam must be numpy.dtype: {} given'
                raise ValueError(msg.format(type(self.t_sysparam)))

            if self.debug:
                msg = "gonna compile system parameters struct as {}."
                print_debug(msg.format("t_sysparam"))

            structs += (self._compile_struct('t_sysparam', self.t_sysparam), )
            self.t_sysparam_name = "t_sysparam"
            r_arg_sysparam = "\n    __global const t_sysparam *sysparam,"


        # ---- DYNAMIC DAMPING

        if self.yt_coeff is not None:
            rtype = "t_sysparam" if self.t_sysparam is not None else None
            r_yt_coeff = f2cl(self.yt_coeff, "ytcoeff", rtype)
            if self.debug:
                msg = "gonna compile y(t) coefficient function."
                print_debug(msg)

        # ---- DYNAMIC HAMILTON
        n_htl = 0
        r_htl = '#define HTL()\\\n   _hu[__item] = _h0[__item];'
        if self.ht_coeff is not None and len(self.ht_coeff) > 0:
            n_htl = len(self.ht_coeff)
            # compile coefficient function to OpenCL
            r_cl_coeff = "\n".join(
                f2cl(f, "htcoeff{}".format(i), "t_sysparam")
                for (i, f) in enumerate(self.ht_coeff))

            # render HTL makro contributions due to H(T)
            cx = ("_hu[__item] = $(cfloat_add)(_hu[__item], "
                  "$(cfloat_rmul)(coeff[{i}], "
                  "htl[{i}]));")
            r_htl += '\\\n   ' \
                   + '\\\n   '.join([cx.format(i=i) for i in range(n_htl)])

            # private buffer for matrix elements of H(T)
            cx = "htl[{i}] = ht{i}[__item];"
            r_htl_priv = "$(cfloat_t) htl[{}];".format(len(self.ht_coeff)) \
                       + '\n    ' \
                       + '\n    '.join(cx.format(i=i) for i in range(n_htl))

            # kernel arguments
            cx = "__global const $(cfloat_t) *ht{i},"
            r_arg_htl = '\n    ' \
                      + '\n    '.join(cx.format(i=i) for i in range(n_htl))

        tpl_coeff = ""
        if self.yt_coeff or self.ht_coeff is not None:
            r_coeff = []
            if self.yt_coeff is not None:
                if self.t_sysparam is not None:
                    r_coeff += ["ytc = pow(ytcoeff(/*{t}*/, sysp), 2);"]
                else:
                    r_coeff += ["ytc = pow(ytcoeff(/*{t}*/), 2);"]

            if self.ht_coeff is not None and len(self.ht_coeff) > 0:
                coeffx = "coeff[{i}] = htcoeff{i}(/*{{t}}*/, sysp);"
                r_coeff += [coeffx.format(i=i) for i in range(n_htl)]

            tpl_coeff = '\n/*{s}*/if (isfirst) {'\
                      + '\n/*{s}*/    '\
                      + '\n/*{s}*/    '.join(r_coeff)\
                      + '\n/*{s}*/}'

            r_local_coeff = "\n"
            if self.t_sysparam is not None:
                r_sysp = "\n        sysp = sysparam[GID];"
                r_local_coeff += "    __local t_sysparam sysp;\n"
            if self.ht_coeff is not None and len(self.ht_coeff) > 0:
                r_local_coeff += "    __local $(float) coeff[{}];\n".format(n_htl)
            if self.yt_coeff is not None:
                r_local_coeff += "    __local $(float) ytc;\n"

        # -- MAIN MAKRO

        # render R(K) macro content.

        if self.debug:
            print_debug(
                "precomiler: generating 1 x H_un, {} x H_t, {} x jump operations.",
                n_htl,
                self.jmp_n
            )

        cx_unitary = ("K = $(cfloat_add)(K, $(cfloat_mul)(ihbar, "
                      "$(cfloat_sub)($(cfloat_mul)(HU(__idx, {i}), R({i}, __idy)), "
                      "$(cfloat_mul)(HU({i}, __idy), R(__idx, {i})))));")

        if self.yt_coeff is not None:
            cx_jump = ("K = $(cfloat_add)(K, $(cfloat_mul)"
                       "($(cfloat_rmul)(ytc, jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].PF), "
                       "_rky[jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].IDX]));")
        else:
            cx_jump = ("K = $(cfloat_add)(K, $(cfloat_mul)"
                       "(jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].PF, "
                       "_rky[jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].IDX]));")

        r_define_rk = '\\\n    '.join(
            [cx_unitary.format(i=r_clint(i)) for i in range(M)]
            + [cx_jump.format(i=r_clint(i)) for i in range(self.jmp_n)])

        # -- Contants

        r_define = '\n'.join('#define {} {}'.format(*df) for df in [
            ('NATURE_HBAR', r_clfloat(1)),
            ('IN_BLOCK_SIZE', r_clint(M**2)),
            ('THREAD_X', r_clint(M)),
            ('N_JUMP', r_clint(self.jmp_n)),
            ('N_JUMP2', r_clint(self.jmp_n**2)),
            ('HDIM', r_clint(M)),
            ('LOCAL_SIZE', r_clint(self.cl_local_size)),
            # butcher
            ('b1', r_clfrac(1.0, 6.0)),
            ('b2', r_clfrac(4.0, 6.0)),
            ('b3', r_clfrac(1.0, 6.0)),
            ('a21', r_clfloat(0.5)),
            ('a31', r_clfloat(-1.0)),
            ('a32', r_clfloat(2.0)),
        ])

        # -- Thread Layout
        kij = enumerate((i, j) for i in range(M) for j in range(M) if i < j + 1)
        r_tl = '\n'.join(
            ['__local int2 _idx[LOCAL_SIZE];']
            + ['    _idx[{}] = (int2)({}, {});'.format(k, i, j) for (k, (i, j)) in kij]
        )

        # -- compile full kernel

        self.c_kernel = r_tmpl(src,
                               define=r_define,
                               structs="\n".join(structs),
                               rk_macro=r_define_rk,
                               arg_sysparam=r_arg_sysparam,
                               ht_coeff=r_cl_coeff,
                               htl_priv=r_htl_priv,
                               htl_macro=r_htl,
                               htl_coefft0=r_tmpl(tpl_coeff, t='t0', s='    '),
                               htl_coefftdt2=r_tmpl(tpl_coeff, t='t + dt / 2.0f', s='    ' * 2),
                               htl_coefftdt=r_tmpl(tpl_coeff, t='t + dt', s='    ' * 2),
                               local_coeff=r_local_coeff,
                               sysp=r_sysp,
                               arg_htl=r_arg_htl,
                               arg_debug=r_arg_debug,
                               debug_hook_1=r_debug_hook_1,
                               yt_coeff=r_yt_coeff,
                               tl=r_tl)

        if self.debug:
            print_debug(
                "generated kernel: {} lines. Compiling...",
                self.c_kernel.count("\n") + 1
            )

        if QOP.ECHO_COMPILED_KERNEL:
            print(self.c_kernel)

        self.prg = cl.Program(self.ctx, self.c_kernel).build()


    def sync(self, state=None, t_bath=None, y_0=None,
             hu=None, htl=None, sysparam=None):
        """ sync data into gpu buffers. All host buffers are either
            one-component data (single valued) or some kind of vector.
            If vectors are given, they must have the same length or a
            length of 1. The single component buffers are then normalized
            to the length of the non-single component buffers.

            :arg state:    initial state (rho_0)
            :arg t_bath:   bath temperatures
            :arg y_0:      damping coefficients
            :arg hu:       unitary stationary hams
            :arg htl:      unitary non-stationary hams
            :arg sysparam: system parameters (used by htl coefficient functions)

            """
        if state is not None:
            self.state = npmat_manylike(self.system.h0, state)

        if y_0 is not None:
            self.y_0 = vectorize(y_0, dtype=QOP.T_FLOAT)
            assert np.all(self.y_0 >= 0)

        if t_bath is not None:
            self.t_bath = vectorize(t_bath, dtype=QOP.T_FLOAT)
            assert np.all(self.t_bath >= 0)

        if hu is not None:
            self.hu = npmat_manylike(self.system.h0, hu)

        if sysparam is not None:
            self.sysparam = vectorize(sysparam, dtype=self.t_sysparam)

        self.normalize_vbuffers()

        # if y_0 and t_bath is normalized we can create the jumping
        # instructions buffer.
        if self.h_y_0 is not None and self.h_t_bath is not None:
            if self.debug:
                print_debug('compute jumping structure...')
            self.h_cl_jmp = self.create_h_cl_jmp()
            self.b_cl_jmp = None
            if self.h_cl_jmp.shape[-1] > 0:
                self.b_cl_jmp = self.arr_to_buf(self.h_cl_jmp, readonly=True)

        # XXX
        if htl is not None:
            self.h_htl = [
                self.system.op2eb(
                    np.array(ht, dtype=np.complex64).reshape(1, *self.system.h0.shape)
                ) for ht in htl
            ]
            self.b_htl = [
                self.arr_to_buf(h_ht, readonly=True)
                for h_ht in self.h_htl
            ]

    def run(self, tg, steps_chunk_size=1e4, parallel=True):
        """ runs the evolutions within given time gatter **tg**

            Arguments:
            ----------

            :tg: the time gatter (t0, tf, dt) where the final
                state of the calculation will be rho(tf) and the given
                initial state is rho(t0). (see util.time_gatter())

            :steps_chunk_size: how many RK4 loops are executed
                within one OpenCL kernel invokation.

            This function returns a generator so the actual progress
            can be controlled from outside.

            Example:

            ```python
                kernel.sync(state=rho, ...)
                for idx, tlist, rho_eb in kernel.run((0, 0.1, 0.01), steps_chunk_size=4):
                    print(idx)
                    print(tlist)
                    # do something with rho_eb...

                # result:
                # (1, 4)
                # [0.001 0.002 0.003 0.004]
                # (5, 8)
                # [0.005 0.006 0.007 0.008]
                # (9, 10)
                # [0.009 0.01 ]

            ```
            """
        assert self.h_t_bath is not None, "t_bath was not synced."
        assert self.h_y_0 is not None, "y_0 was not synced."
        assert self.h_state is not None, "state was not synced."
        assert self.h_hu is not None, "hu was not synced."
        assert self.h_cl_jmp is not None, "h_cl_jmp was not synced."
        assert_rho_hermitian(self.h_state)

        steps_chunk_size = int(steps_chunk_size)
        assert steps_chunk_size > 0

        # integration configuration,
        #             t0     dt     n_int : n_int * dt = tf
        rk4_config = (tg[0], tg[2], len(time_gatter(*tg)) - 1)

        # result buffer (rho(t)) for one chunk
        h_rhot = np.zeros((steps_chunk_size, *self.h_state.shape), dtype=self.h_state.dtype)
        b_rhot = self.arr_to_buf(h_rhot, writeonly=True)

        # rho0 initial state
        h_rhot[-1] = self.h_state

        # prepare kernel args
        # each of the N states is evolved inside a work-group.
        work_layout = (self.N, self.cl_local_size), (1, self.cl_local_size)

        bufs = (self.b_hu, )
        if self.b_htl is not None:
            bufs += (*self.b_htl, )
        bufs += (self.b_cl_jmp, )
        if self.b_sysparam is not None:
            bufs += (self.b_sysparam, )
        bufs += (b_rhot, )

        arg_dt = QOP.T_FLOAT(rk4_config[1])
        last_arg_n_int = None
        for i in np.arange(0, rk4_config[2] - 1, steps_chunk_size):
            t0 = time()

            # upload new t0 state and create buffer tuple
            b_rho0 = self.arr_to_buf(h_rhot[-1], readonly=True)

            # (for custom injected code via debug hooks for example).
            b_external = [x[1] for x in self.cl_debug_buffers]

            # run kernel
            arg_t0 = QOP.T_FLOAT(rk4_config[0] + i * rk4_config[1])
            arg_n_int = QOP.T_INT(np.minimum(steps_chunk_size, rk4_config[2] - i))
            vargs = (*bufs, *b_external, b_rho0, arg_t0, arg_dt, arg_n_int)
            self.prg.opmesolve_rk4_eb(self.queue, *work_layout, *vargs)

            # GPU + CPU parallel readout
            if parallel:
                t1 = time()
                if last_arg_n_int is not None:
                    yield (idx[0], idx[1]), tlist, h_rhot[0:last_arg_n_int]
                last_arg_n_int = arg_n_int
                dt1 = time() - t0
                self.queue.finish()

            cl.enqueue_copy(self.queue, h_rhot, b_rhot)
            self.queue.finish()
            b_rho0.release()

            # the idx represents the index of the full non-chunkwise evolution
            idx = i + 1, i + arg_n_int
            tlist = rk4_config[0] + np.arange(idx[0], idx[1] + 1) * arg_dt

            # -- GPU+CPU sequential readout
            if not parallel:
                t1 = time()
                yield idx, tlist, h_rhot[0:arg_n_int]
                dt1 = time() - t0

            dt0 = time() - t0

            # print something interesting
            if self.debug:
                progress = str(int(np.round(100 * (i + arg_n_int) / rk4_config[2])))
                dbg_args = (progress, *idx, *work_layout, dt0, dt1, )
                dmsg = "\033[s\033[36m{:>3s}\033[0m % [{}-{}] global={}, local={} "\
                     + "GPU+CPU \033[34m{:.4f}s\033[0m "\
                     + "CPU \033[34m{:.4f}s\033[0m\033[1A\033[u"
                print_debug(dmsg.format(*dbg_args))

        # if GPU+CPU parallel readout the GPU kernel is invoked twice
        # before CPU code is executed. Thus, the missing yield must
        # be performed outside the loop
        if parallel:
            yield idx, tlist, h_rhot[0:arg_n_int]


    def arr_to_buf(self, arr, readonly=False, writeonly=False):
        """ helper to create a new buffer from np array.
            """
        try:
            flags = mf.COPY_HOST_PTR
            if readonly:
                flags |= mf.READ_ONLY
            if writeonly:
                flags |= mf.WRITE_ONLY
            return cl.Buffer(self.ctx, flags, hostbuf=arr)
        except:
            errmsg = 'could not allocate buffer of size {:.2f}mb'
            raise RuntimeError(errmsg.format(arr.size / 1024**2))


    def normalize_vectors(self, vector_names):
        vectors = [(name, getattr(self, name))
                   for name in vector_names
                   if getattr(self, name) is not None]
        vlens = set(v[1].shape[0] for v in vectors)
        if len(vlens) > 2:
            raise InconsistentVectorSizeError("too many different vector dimensions", vectors)

        if len(vlens) == 2:
            vl_min, vl_max = min(vlens), max(vlens)
            if vl_min != 1 and vl_max != 1:
                msg = ("sryy cannot normalize {} dimensions vs. {}"
                       " dimensions. One dimension must be 1 to be normalizable.")
                raise InconsistentVectorSizeError(msg.format(vl_min, vl_max), vectors)

            return {n: v if v.shape[0] == vl_max else \
                    np.array([v] * vl_max, dtype=v.dtype).reshape((vl_max, *v.shape[1:]))
                    for (n, v) in vectors}

        return dict(vectors)


    def reader_tfinal_rho(self, g):
        """ returns the final state in basis
            of the reduced system.
            """

        r = None
        for r in g:
            pass

        if r is None:
            raise RuntimeError('empty generator given.')

        t, rho_eb = r[1:3]
        return t[-1], self._mb.T @ rho_eb[-1] @ self._mb.conj()


    def reader_rho_t(self, g):
        """ reads out rho at all times in basis of the
            reduced system.
            """
        rho0 = self._mb.T @ self.h_state @ self._mb.conj()
        rhot = rho0.reshape((1, *rho0.shape))
        tg = np.zeros(1)
        for tlist, rho_eb in (x[1:3] for x in g):
            rhot = np.concatenate((rhot, self._mb.T @ rho_eb @ self._mb.conj()))
            tg = np.concatenate((tg, tlist))

        # this is because we need 2 time steps in order to restore t0.
        # XXX Think about apply time gatter via. sync() so we can access
        #     the time gatter here at this point.
        if len(tg) < 3:
            raise RuntimeError('must proceed at least 2 steps.')

        tg[0] = 2 * tg[1] - tg[2]

        return tg, rhot


    def normalize_vbuffers(self):
        """ normalized vectorized buffers such that they have
            all the same number of components. This method will
            only normalize the allready synced buffers. """
        vnorm = self.normalize_vectors(['state', 'y_0', 't_bath', 'hu', 'sysparam'])
        if 'state' in vnorm:
            self.h_state = self.system.op2eb(vnorm['state'])

        if 'hu' in vnorm:
            self.h_hu = self.system.op2eb(vnorm['hu'])
            self.b_hu = self.arr_to_buf(self.h_hu, readonly=True)

        if 'sysparam' in vnorm:
            self.h_sysparam = vnorm['sysparam']
            self.b_sysparam = self.arr_to_buf(self.h_sysparam, readonly=True)

        if 'y_0' in vnorm:
            self.h_y_0 = vnorm['y_0']

        if 't_bath' in vnorm:
            self.h_t_bath = vnorm['t_bath']

        self.N, self.dimH = self.h_state.shape[0:2]


    def get_flat_jumps(self):
        fj = []
        for nw, jumps in enumerate(self.system.get_jumps()):
            if jumps is None:
                continue
            jumps = jumps.reshape((int(len(jumps) / (nw + 1)), nw + 1))
            for jump in jumps:
                fj.append(jump)
        return fj


    def create_h_cl_jmp(self):
        """ create cl_jmp host buffer """
        N = self.h_state.shape[0]
        cl_jmp = np.zeros((N, *self.jmp_instr.shape), dtype=self.__class__.DTYPE_T_JMP)
        for i in range(len(self.h_state)):
            p = self.h_sysparam[i] if self.h_sysparam is not None else {}
            cl_jmp[i]['IDX'] = self.jmp_instr['IDX']
            cl_jmp[i]['PF']  = self.h_y_0[i] \
                             * self.jmp_instr['PF'] \
                             * self.kappa(self.h_t_bath[i], p, self.jmp_instr['W'])

        if self.jmp_n == 0 or not self.optimize_jumps:
            return cl_jmp

        optimized = self.cl_jmp_acc_pf(cl_jmp)
        if self.debug:
            dbg = 'optimized h_cl_jmp: old shape {} to new shape {}.'
            print_debug(dbg.format(cl_jmp.shape, optimized.shape))

        return optimized


    def create_jmp_instr(self):
        """ create """
        M = self.system.h0.shape[0]
        idx = lambda i, j: M * i + j
        jelem = [[] for _ in range(M ** 2)]
        flat_jumps = self.get_flat_jumps()
        all_idx = [(i, j) for i in range(M) for j in range(M)]

        for ((i, j), jump) in itertools.product(all_idx, flat_jumps):
            tidx = idx(i, j)
            jx, jy = jump[np.where(jump['I'][:, 0] == i)[0]], \
                     jump[np.where(jump['I'][:, 0] == j)[0]]
            for j1, j2 in itertools.product(jx, jy):
                fidx = idx(j1['I'][1], j2['I'][1])
                jelem[tidx].append((fidx, j1['d'] * j2['d'].conj(), j1['w'])) # A rho Ad
                jelem[fidx].append((tidx, j1['d'].conj() * j2['d'], -j1['w'])) # Ad rho A
            # -1/2 {rho, Ad A}
            jy = jump[np.where(jump['I'][:, 1] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:, 0] == j2['I'][0])[0]]:
                    fidx = idx(i, j1['I'][1])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], j1['w']))
            jx = jump[np.where(jump['I'][:, 1] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:, 0] == j1['I'][0])[0]]:
                    fidx = idx(j2['I'][1], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'] * j2['d'].conj(), j1['w']))
            # -1/2 {rho, A Ad}
            jx = jump[np.where(jump['I'][:, 0] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:, 1] == j1['I'][1])[0]]:
                    fidx = idx(j2['I'][0], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], -j1['w']))
            jy = jump[np.where(jump['I'][:, 0] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:, 1] == j2['I'][1])[0]]:
                    fidx = idx(i, j1['I'][0])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], -j1['w']))

        # structurize the data as (M, M, n_max_jump) numpy array
        jmp_n = len(flat_jumps)
        jmp_n_max = max(len(_) for _ in jelem)
        jmp_instr = np.zeros((M, M, max(1, jmp_n_max)), dtype=self.__class__.DTYPE_JUMP_RAW)
        for (i, j) in [(i, j) for j in range(M) for i in range(M)]:
            l = len(jelem[idx(i, j)])
            jmp_instr[i, j, :l] = jelem[idx(i, j)]

        # filter out non-contributing jumps
        sidx = (-np.abs(jmp_instr['PF'])).argsort(axis=2)
        s_jmp_instr = np.take_along_axis(jmp_instr, sidx, axis=2)
        s, n_max_instr = s_jmp_instr.shape, 0
        for a in s_jmp_instr.reshape((s[0] * s[1], s[2])):
            n_instr = len(np.argwhere(np.abs(a['PF']) > QOP.COMPLEX_ZERO_TOL))
            n_max_instr = max(n_max_instr, n_instr)
        c_jmp_instr = s_jmp_instr[:,:,:n_max_instr]

        # count contributions
        jmp_n_opt, c_jmp_n_opt = 0, 0
        for (i, j) in [(i, j) for j in range(M) for i in range(M)]:
            l = len(jelem[idx(i, j)])
            jmp_n_opt = max(jmp_n_opt, len(set(jmp_instr[i, j]['IDX'])))
            c_jmp_n_opt = max(c_jmp_n_opt, len(set(c_jmp_instr[i, j]['IDX'])))

        if self.debug:
            msg = "prepared {} jumps. Require {} operations per work-item."
            print_debug(msg, jmp_n, jmp_n_max)
            msg = "the jumps can be optimized such that at most {} operations are required"
            print_debug(msg, jmp_n_opt)
            msg = "... {} transitions have non-zero (>{}) contribution."
            print_debug(msg, c_jmp_n_opt, QOP.COMPLEX_ZERO_TOL)

        self.jmp_n = c_jmp_n_opt if self.optimize_jumps else n_max_instr
        self.jmp_instr = c_jmp_instr


    def cl_jmp_acc_pf(self, cl_jmp):
        """ accumulates prefactors PF for cells with same IDX.

            Example:
            --------

                [[[[(8,  65.19407   +0.j) (0,  -0.5970355 +0.j) (0,  -0.5970355 +0.j)
                    (4,   9.252141  +0.j) (0,  -0.62607056+0.j) (0,  -0.62607056+0.j)]
                   [(1,  -0.5970355 +0.j) (5,   9.252141  +0.j) (1,  -4.6260705 +0.j)
                    (1,  -0.62607056+0.j) (1,  -0.62607056+0.j) (0,   0.        +0.j)]
                   [(2, -32.597034  +0.j) (2,  -0.5970355 +0.j) (2,  -4.6260705 +0.j)
                    (2,  -0.62607056+0.j) (0,   0.        +0.j) (0,   0.        +0.j)]],
                 ...
                ]

                is accumulated to to

                [[[[(8,  65.19407  +0.j) (0,  -2.4462123+0.j) (4,   9.252141 +0.j)]
                   [(1,  -6.475247 +0.j) (5,   9.252141 +0.j) (0,   0.       +0.j)]
                   [(2, -38.44621  +0.j) (0,   0.       +0.j) (0,   0.       +0.j)]],
                 ...
                ]
        """
        N, M = self.h_state.shape[0:2]
        cl_jmp_opt = np.zeros((*cl_jmp.shape[0:3], self.jmp_n), dtype=cl_jmp.dtype)
        allidx = [(i, j) for j in range(M) for i in range(M)]

        for (k, (i, j)) in itertools.product(range(N), allidx):
            # contributing cell indices
            jcell = cl_jmp[k, i, j]
            jcell0 = jcell[np.where(np.abs(jcell['PF']) > 0)[0]]
            cidx = list(
                np.where(jcell0['IDX'] == idx)[0]
                for idx in set(jcell0['IDX']))

            # accumulate prefactors, assign
            jcell_acc = list(
                (jcell0[indx[0]]['IDX'], np.sum(jcell0[indx]['PF']))
                for indx in cidx)
            cl_jmp_opt[k, i, j][:len(jcell_acc)] = jcell_acc

        return cl_jmp_opt


    def _compile_struct(self, name, dtype):
        """ compiles a `dtype` to c-struct with `name`.
            """
        _, c_decl = cl.tools.match_dtype_to_c_struct(
            self.ctx.devices[0],
            name,
            dtype
        )
        return c_decl


def r_clfloat(f, prec=None):
    """ renders an OpenCL float representation """
    if prec is not None:
        raise NotImplementedError()
    sf = str(f)
    return ''.join([sf, '' if '.' in sf else '.', 'f'])


def r_clfrac(p, q, prec=None):
    """ renders an OpenCL fractional representation """
    return "{}/{}".format(r_clfloat(p, prec), r_clfloat(q, prec))


def r_clint(f):
    """ renders an OpenCL integer representation """
    assert isinstance(f, int)
    return str(f)


R_CLTYPES_POSTFIX = ['t', 'mul', 'rmul', 'sub', 'new', 'add', 'neg', 'conj', 'fromreal']
def r_cltypes(src, double_precision=False):
    """ renders cltype placeholders like $(cfloat_mul), $(float), ...
        into correpsonding cl identifier depending on whether double
        precision is on or not.

        Example:

          $(float)      ->       double         Double Precision ON
          $(float)      ->       float          Double Precision OFF

        """
    ctype = 'cdouble' if double_precision else 'cfloat'
    subst = [
        ('$(cfloat_{})'.format(pf), '{}_{}'.format(ctype, pf))
        for pf in R_CLTYPES_POSTFIX]
    subst += [
        ('$(float)', 'double' if double_precision else 'float'),
        ('$(cf_)', 'cfloat_'),
    ]

    for k, v in subst:
        src = src.replace(k, v)

    return src


def _ctx():
    """ returs the first available gpu context
        """
    platforms = cl.get_platforms()
    assert len(platforms) > 0
    gpu_devices = list(d for d in platforms[0].get_devices() if d.type == cl.device_type.GPU)
    assert len(gpu_devices) > 0
    return cl.Context(devices=gpu_devices)


def r_tmpl(src, **kwargs):
    """ renders a string containing placeholders like

        /*{NAME}*/

        such that it subsitutes `kwargs['NAME']` into it.
        """
    for k, v in kwargs.items():
        src = src.replace('/*{'+k+'}*/', v)

    return r_cltypes(src)

def assert_rho_hermitian(state):
    """ asserts that **state** is hermitian.
        """
    idx = np.where(np.abs(np.transpose(state, (0, 2, 1)).conj() - state) > 0.000001)[0]
    assert len(idx) == 0, "safety abort - state[{}] not hermitian".format(idx[0])
