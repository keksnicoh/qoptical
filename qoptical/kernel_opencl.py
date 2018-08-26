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
import pyopencl as cl
import pyopencl.tools
import numpy as np
import itertools, os, dis
from .f2cl import f2cl
from .settings import *
from .util import *
from .result import OpMeResult


mf = cl.mem_flags

DTYPE_TIME_RANGE = np.dtype([
    ('INT_T0',  np.float32),
    ('INT_T',  np.float32),
    ('INT_DT', np.float32),
])

DTYPE_INTEGRATOR_PARAM = np.dtype([
    ('INT_T0', np.float32),
    ('INT_DT', np.float32),
    ('INT_N',  np.int32),
])


def opmesolve_opencl():
    pass


class OpenCLKernel():
    """ Renders & compiles an OpenCL GPU kernel to
        integrate Quantum Optical Master Eqaution.
        """

    """ kernel code file name """
    TEMPLATE_NAME = 'kernel_opencl.tmpl.c'

    """ before creating the final jump instructions (`DTYPE_T_JMP`),
        this dtype is used to create a non-optimized buffer.
        when all jumps are analyzed the resulting buffer is
        accumulated (grouped by IDX column) to get the final
        contribution of a matrix element into the work-item.
        (See `cl_jmp_acc_pf`)
        """
    DTYPE_JUMP_RAW = np.dtype([
        ('IDX', np.int32),     # source idx
        ('PF', DTYPE_COMPLEX), # prefactor (dipole)
        ('SE', np.int32),      # sponanious emission (1 or 0)
        ('W', DTYPE_FLOAT),    # transition frequency
    ])

    """ instruction to add a matrix element rho[IDX]
        by a given, complex, prefactor. This is the final
        (and optimized) representation of the dissipator
        contribution from a matrix element into the
        work-item's matrx element.
        """
    DTYPE_T_JMP = np.dtype([
        ('IDX', np.int32),
        ('PF', DTYPE_COMPLEX),
    ])

    def __init__(self, system, ctx=None, queue=None, t_sysparam=None, optimize_jumps=True):
        """ create QutipKernel for given ``system`` """

        self.system         = system
        self.optimize_jumps = optimize_jumps
        self.t_sysparam     = t_sysparam
        self.ctx            = ctx or self._ctx()
        self.queue          = queue or cl.CommandQueue(self.ctx)

        self.ev  = None # eigh
        self._mb = None # base transformation matrix

        self.c_kernel      = None # generated kernel c code
        self.cl_local_size = None # due to hermicity we only need M*(M+1)/2 work-items
                                  # where M is dimH

        # time dependent hamiltonian
        self.t_sysparam_name = None
        self.ht_coeff        = None

        self.jmp_n     = None # the number of jumping instructions due to dissipators
        self.jmp_instr = None # jumping instructions for all cells

        # synced, vectorized state
        self.hu       = None
        self.state    = None
        self.t_bath   = None
        self.y_0      = None
        self.htl      = None
        self.sysparam = None

        # host buffers and gpu buffers
        self.h_sysparam = None
        self.h_htl      = None
        self.h_hu       = None
        self.h_state    = None
        self.h_cl_jmp   = None
        self.h_y_0      = None
        self.h_t_bath   = None
        self.h_cl_jmp   = None
        self.b_htl      = None
        self.b_sysparam = None
        self.b_hu       = None
        self.b_state    = None
        self.b_cl_jmp   = None

        self.init()


    def init(self):
        """ implicit initialization """
        self.hu  = npmat_manylike(self.system.h0, [self.system.h0])
        self.ev  = self.system.ev
        self._mb = np.array(self.system.s, dtype=self.system.s.dtype)


    def compile(self):
        """ renders OpenCL kernel from given state and reduced
            system. Since the kernel renders code depending on
            the jump-layout it must be known at compile-time and
            cannot be changed without recompile. Currently, only
            one jump layout and only one set of time dependent
            Hamiltons are supported.
            XXX: support set of timedependent hamilton matrices
                 per system.

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
            self.__class__.TEMPLATE_NAME)
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

        # ---- STRUCTS

        structs = [
            self._compile_struct('t_int_parameters', DTYPE_INTEGRATOR_PARAM),
            self._compile_struct('t_jump',           self.__class__.DTYPE_T_JMP),
        ]

        # ---- SYSTEM PARAMETERS DTYPE

        if self.t_sysparam is not None:
            if not isinstance(self.t_sysparam, np.dtype):
                msg = 't_sysparam must be numpy.dtype: {} given'
                raise ValueError(msg.format(type(self.t_sysparam)))

            if DEBUG:
                msg = "gonna compile system parameters struct as {}."
                print_debug(msg.format("t_sysparam"))

            structs += self._compile_struct('t_sysparam', self.t_sysparam),
            self.t_sysparam_name = "t_sysparam"
            r_arg_sysparam = "\n    __global const t_sysparam *sysparam,"

        # ---- DYNAMIC HAMILTON

        r_htl = '#define HTL(T)\\\n   _hu[__item] = _h0[__item];'
        if self.ht_coeff is not None:
            n_htl = len(self.ht_coeff)
            # compile coefficient function to OpenCL
            r_cl_coeff = "\n".join(
                f2cl(f, "htcoeff{}".format(i), "t_sysparam")
                for (i, f) in enumerate(self.ht_coeff))

            # render HTL makro contributions due to H(T)
            cx = ("_hu[__item] = $(cfloat_add)(_hu[__item], "
                  "$(cfloat_rmul)(htcoeff{i}(T, sysparam[GID]), "
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

        # -- MAIN MAKRO

        # render R(K) macro content.
        DEBUG and print_debug("precomiler: generating 1 x H_un, 0 x H_t, {} x jump operations.", self.jmp_n)
        cx_unitary = ("K = $(cfloat_add)(K, $(cfloat_mul)(ihbar, "
                      "$(cfloat_sub)($(cfloat_mul)(HU(__idx, {i}), R({i}, __idy)), "
                      "$(cfloat_mul)(HU({i}, __idy), R(__idx, {i})))));");
        cx_jump = ("K = $(cfloat_add)(K, $(cfloat_mul)"
                   "(jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].PF, "
                   "_rky[jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].IDX]));")
        r_define_rk = '\\\n    '.join(
            [cx_unitary.format(i=r_clint(i)) for i in range(M)]
            + [cx_jump.format(i=r_clint(i)) for i in range(self.jmp_n)])

        # -- Contants

        r_define = '\n'.join('#define {} {}'.format(*df) for df in [
            ('NATURE_HBAR',   r_clfloat(1)),
            ('IN_BLOCK_SIZE', r_clint(M**2)),
            ('THREAD_X',      r_clint(M)),
            ('N_JUMP',        r_clint(self.jmp_n)),
            ('N_JUMP2',       r_clint(self.jmp_n**2)),
            ('HDIM',          r_clint(M)),
            ('LOCAL_SIZE',    r_clint(self.cl_local_size)),
            # butcher
            ('b1',            r_clfrac(1.0, 6.0)),
            ('b2',            r_clfrac(4.0, 6.0)),
            ('b3',            r_clfrac(1.0, 6.0)),
            ('a21',           r_clfloat(0.5)),
            ('a31',           r_clfloat(-1.0)),
            ('a32',           r_clfloat(2.0)),
        ])

        # -- Thread Layout

        r_tl = '\n'.join(['int2 _idx[LOCAL_SIZE];'] + [
           '    _idx[{}] = (int2)({}, {});'.format(k, i, j)
           for (k, (i, j)) in enumerate((i, j) for i in range(M)
                                               for j in range(M) if i < j+1)])

        # -- compile full kernel

        self.c_kernel = r_tmpl(src, define       = r_define,
                                    structs      = "\n".join(structs),
                                    rk_macro     = r_define_rk,
                                    arg_sysparam = r_arg_sysparam,
                                    ht_coeff     = r_cl_coeff,
                                    htl_priv     = r_htl_priv,
                                    htl_macro    = r_htl,
                                    arg_htl      = r_arg_htl,
                                    tl           = r_tl)

        DEBUG and print_debug(
            "generated kernel: {} lines. Compiling...",
            self.c_kernel.count("\n") + 1)

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
            self.y_0 = vectorize(y_0, dtype=DTYPE_FLOAT)
            assert np.all(self.y_0 >= 0)

        if t_bath is not None:
            self.t_bath = vectorize(t_bath, dtype=DTYPE_FLOAT)
            assert np.all(self.t_bath >= 0)

        if hu is not None:
            self.hu = npmat_manylike(self.system.h0, hu)

        if sysparam is not None:
            self.sysparam = vectorize(sysparam, dtype=self.t_sysparam)

        self.normalize_vbuffers()

        # if y_0 and t_bath is normalized we can create the jumping
        # instructions buffer.
        if self.h_y_0 is not None and self.h_t_bath is not None:
            DEBUG and print_debug('compute jumping structure...')
            self.h_cl_jmp = self.create_h_cl_jmp()
            self.b_cl_jmp = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_cl_jmp)

        #XXX
        if htl is not None:
            self.h_htl = np.array(htl, dtype=np.complex64)
            self.b_htl = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_htl)

    def run(self, trange, sync_state=False):

        assert self.h_t_bath is not None, "t_bath was not synced."
        assert self.h_y_0 is not None,    "y_0 was not synced."
        assert self.h_state is not None,  "state was not synced."
        assert self.h_hu is not None,     "hu was not synced."
        assert self.h_cl_jmp is not None, "h_cl_jmp was not synced."

        trange = self.normalize_trange(trange)
        h_int_param = self._create_int_param(trange)
        res_len = max(r['INT_N'] for r in h_int_param)

        h_tstate = np.zeros((res_len, *self.state.shape), dtype=self.state.dtype)
        h_debug  = -np.ones(res_len + 2, dtype=np.float32)

        b_int_param = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=h_int_param)
        b_tstate    = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_tstate)
        b_debug     = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_debug)

        # create argument buffer tuple
        bufs = (self.b_hu, )
        if self.b_htl is not None:
            bufs += (self.b_htl, )
        bufs += (self.b_cl_jmp, b_int_param, )
        if self.b_sysparam is not None:
            bufs += (self.b_sysparam, )
        bufs += (self.b_state, b_tstate, b_debug)

        # run
        work_layout = (self.state.shape[0], self.cl_local_size), \
                      (1, self.cl_local_size)
        self.prg.opmesolve_rk4_eb(self.queue, *work_layout, *bufs)

        res_state = np.empty_like(self.state)
        cl.enqueue_copy(self.queue, res_state, self.b_state)
        cl.enqueue_copy(self.queue, h_tstate, b_tstate)
        cl.enqueue_copy(self.queue, h_debug, b_debug)

        h_tstate = self._be(h_tstate.reshape(((res_len, *self.state.shape))))
        h_state = self._be(res_state)

        texpect = None
        # copy date to avoid references
        return OpMeResult(tlist=[np.arange(*r) for r in h_int_param],
                          state=h_state[:] if res_state  is not None else None,
                          tstate=h_tstate[:] if h_tstate  is not None else None,
                          texpect=texpect[:] if texpect is not None else None)


    def normalize_vectors(self, vector_names):
        vectors = [(name, getattr(self, name))
            for name in vector_names
            if getattr(self, name) is not None]
        vlens = set([v[1].shape[0] for v in vectors])
        if len(vlens) > 2:
            raise InconsistentVectorSizeError("too many different vector dimensions", vectors)

        if len(vlens) == 2:
            vl_min, vl_max = min(vlens), max(vlens)
            if vl_min != 1 and vl_max != 1:
                msg = ("sryy cannot normalize {} dimensions vs. {}"
                       " dimensions. One dimension must be 1 to be normalizable.")
                raise InconsistentVectorSizeError(msg.format(vl_min, vl_max), vectors)

            return {n: v if v.shape[0] == vl_max else np.array([v] * vl_max, dtype=v.dtype)
                for (n, v) in vectors}
        else:
            return dict(vectors)


    def get_flat_jumps(self):
        fj = []
        for nw, jumps in enumerate(self.system.get_jumps()):
            if jumps is None:
                continue
            jumps = jumps.reshape((int(len(jumps) / (nw+1)), nw+1))
            for jump in jumps:
                fj.append(jump)
        return fj


    def normalize_vbuffers(self):
        """ normalized vectorized buffers such that they have
            all the same number of components. This method will
            only normalize the allready synced buffers. """
        vnorm = self.normalize_vectors(['state', 'y_0', 't_bath', 'hu', 'sysparam'])
        if 'state' in vnorm:
            self.h_state = self._eb(vnorm['state'])
            self.b_state = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.h_state)

        if 'hu' in vnorm:
            self.h_hu = self._eb(vnorm['hu'])
            self.b_hu = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_hu)

        if 'sysparam' in vnorm:
            self.h_sysparam = vnorm['sysparam']
            self.b_sysparam = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_sysparam)

        if 'y_0' in vnorm:
            self.h_y_0 = vnorm['y_0']

        if 't_bath' in vnorm:
            self.h_t_bath = vnorm['t_bath']


    def create_h_cl_jmp(self):
        """ create cl_jmp host buffer """
        N, M = self.state.shape[0:2]
        dst = [boson_stat(t) for t in self.h_t_bath]
        cl_jmp = np.zeros((N, *self.jmp_instr.shape), dtype=self.__class__.DTYPE_T_JMP)
        for i in range(0, len(self.state)):
            cl_jmp[i]['IDX'] = self.jmp_instr['IDX']
            cl_jmp[i]['PF'] = self.h_y_0[i] \
                            * self.jmp_instr['PF'] \
                            * self.jmp_instr['W']**3 \
                            * (self.jmp_instr['SE'] + dst[i](self.jmp_instr['W']))

        if self.jmp_n == 0 or not self.optimize_jumps:
            return cl_jmp
        else:
            optimized = self.cl_jmp_acc_pf(cl_jmp)
            if DEBUG:
                dbg = 'optimized h_cl_jmp: old shape {} to new shape {}.'
                print_debug(dbg.format(cl_jmp.shape, cl_jmp.shape))

            return optimized


    def create_jmp_instr(self):
        """ create """
        M     = self.system.h0.shape[0]
        idx   = lambda i, j: M * i + j
        jelem = [[] for _ in range(M ** 2)]

        flat_jumps = self.get_flat_jumps()
        all_idx    = [(i, j) for i in range(M) for j in range(M)]
        for ((i, j), jump) in itertools.product(all_idx, flat_jumps):
            tidx = idx(i, j)
            jx, jy = jump[np.where(jump['I'][:,0] == i)[0]], \
                     jump[np.where(jump['I'][:,0] == j)[0]]
            for j1, j2 in itertools.product(jx, jy):
                fidx = idx(j1['I'][1], j2['I'][1])
                jelem[tidx].append((fidx, j1['d'] * j2['d'].conj(), 1, j1['w'])) # A rho Ad
                jelem[fidx].append((tidx, j1['d'].conj() * j2['d'], 0, j1['w'])) # Ad rho A

            # -1/2 {rho, Ad A}
            jy = jump[np.where(jump['I'][:,1] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:,0] == j2['I'][0])[0]]:
                    fidx = idx(i, j1['I'][1])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 1, j1['w']))
            jx = jump[np.where(jump['I'][:,1] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:,0] == j1['I'][0])[0]]:
                    fidx = idx(j2['I'][1], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'] * j2['d'].conj(), 1, j1['w']))
            # -1/2 {rho, A Ad}
            # XXX can be merged with the block above?!
            jx = jump[np.where(jump['I'][:,0] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:,1] == j1['I'][1])[0]]:
                    fidx = idx(j2['I'][0], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 0, j1['w']))
            jy = jump[np.where(jump['I'][:,0] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:,1] == j2['I'][1])[0]]:
                    fidx = idx(i, j1['I'][0])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 0, j1['w']))

        # structurize the data as (M, M, n_max_jump) numpy array
        jmp_n_max = max(len(_) for _ in jelem)
        jmp_instr = np.zeros((M, M, max(1, jmp_n_max)), dtype=self.__class__.DTYPE_JUMP_RAW)
        jmp_n_opt = 0
        for (i, j) in [(i, j) for j in range(M) for i in range(M)]:
            l = len(jelem[idx(i, j)])
            jmp_instr[i,j,:l] = jelem[idx(i, j)]
            jmp_n_opt = max(jmp_n_opt, len(set(jmp_instr[i,j,:l]['IDX'])))

        if DEBUG:
            print_debug("prepared jumps for each work-item. Requires {} operations per work-item.", jmp_n_max)
            print_debug("the jumps can be optimized such that at most {} operations are required", jmp_n_opt)

        self.jmp_n     = jmp_n_opt if self.optimize_jumps else jmp_n_max
        self.jmp_instr = jmp_instr


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
        N, M       = self.state.shape[0:2]
        cl_jmp_opt = np.zeros((*cl_jmp.shape[0:3], self.jmp_n), dtype=cl_jmp.dtype)
        allidx     = [(i, j) for j in range(M) for i in range(M)]

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


    def _eb(self, op):
        """ transforms `op` into eigenbase.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self._mb.conj() @ op @ self._mb.T


    def _be(self, op):
        """ transforms `op` back into original basis.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self._mb.T @ op @ self._mb.conj()


    def _ctx(self):
        """ returs the first available gpu content
            """
        platforms = cl.get_platforms()
        assert len(platforms) > 0
        gpu_devices = list(d for d in platforms[0].get_devices() if d.type == cl.device_type.GPU)
        assert len(gpu_devices) > 0
        return cl.Context(devices=gpu_devices)


    def _compile_struct(self, name, dtype):
        """ compiles a `dtype` to c-struct with `name`.
            """
        _, c_decl = cl.tools.match_dtype_to_c_struct(
            self.ctx.devices[0],
            name,
            dtype
        )
        return c_decl


    def normalize_trange(self, trange):
        if isinstance(trange, tuple):
            return self.normalize_trange([trange])
        if not isinstance(trange, np.ndarray):
            if not isinstance(trange, list):
                raise ValueError()
            return self.normalize_trange(
                np.array(trange, dtype=DTYPE_TIME_RANGE)
            )
        if not trange.dtype == DTYPE_TIME_RANGE:
            raise ValueError()
        if len(trange) == 1 and len(self.state) != 1:
            return np.array([trange[0]] * len(self.state), dtype=DTYPE_TIME_RANGE)
        if len(trange) != len(self.state):
            raise ValueError()
        return trange


    def _create_int_param(self, trange):
        """ create integrator parameters from normalized `trange` """
        return np.array([
            (tr['INT_T0'], tr['INT_DT'], len(np.arange(*tr)))
            for tr in trange
        ], dtype=DTYPE_INTEGRATOR_PARAM)


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


def r_tmpl(src, **kwargs):
    """ renders a string containing placeholders like

        /*{NAME}*/

        such that it subsitutes `kwargs['NAME']` into it.
        """
    for k, v in kwargs.items():
        src = src.replace('/*{'+k+'}*/', v)

    return r_cltypes(src)

