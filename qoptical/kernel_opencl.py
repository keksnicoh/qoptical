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

# a jump within the OpenCL kernel:
# it contains the matrix-element id IDX which should be
# added into the cell by a prefactor PF
DTYPE_T_JMP = np.dtype([
    ('IDX', np.int32),
    ('PF', DTYPE_COMPLEX),
])

# a single jump instruction
DTYPE_JMP_INSTR = np.dtype([
    # source idx
    ('IDX', np.int32),
    # prefactor (dipole)
    ('PF', DTYPE_COMPLEX),
    # sponanious emission
    ('SE', np.int32),
    # transition frequency
    ('W', DTYPE_FLOAT),
])

class OpenCLKernel():

    def __init__(self, system, ctx=None):
        """ create QutipKernel for given ``system`` """
        # api state
        self.system = system
        self.state  = None
        self.t_bath = None
        self.y_0    = None
        self.htl    = None

        self.optimize_jumps = True

        # generated kernel c code
        self.c_kernel = None
        self.hu       = npmat_manylike(self.system.h0, [self.system.h0])
        self.ctx      = ctx or self._ctx()
        self.ev       = self.system.ev
        self._mb      = np.array(self.system.s, dtype=self.system.s.dtype)

        self.cl_local_size = None

        # time dependent hamiltonian
        self.t_sysparam = None
        self.t_sysparam_name = None
        self.ht_coeff = None

        # the number of jumping instructions due to dissipators
        self.jmp_n = None

        # jumping instructions for all cells
        self.jmp_instr = None


    def compile(self):
        M = self.system.h0.shape[0]
        self.cl_local_size = int(M * (M + 1) / 2)

        # create jump instructions
        self.create_jmp_instr()

        # read template
        with open(os.path.join(os.path.dirname(__file__), 'kernel_opencl.tmpl.c')) as f:
            src = f.read()

        # ---- STRUCTS

        # render structures
        structs = [
            self._compile_struct('t_int_parameters', DTYPE_INTEGRATOR_PARAM),
            self._compile_struct('t_jump', DTYPE_T_JMP),
        ]

        # ---- SYSTEM PARAMETERS DTYPE

        # system parameters
        r_arg_sysparam = ""
        if self.t_sysparam is not None:
            if not isinstance(self.t_sysparam, np.ndarray):
                msg = 't_sysparam must be numpy ndarray: {} given'
                raise ValueError(msg.format(gettype(self.t_sysparam)))

            if DEBUG:
                msg = "gonna compile system parameters struct as {}."
                print_debug(msg.format("t_sysparam"))

            structs += self._compile_struct('t_sysparam', self.t_sysparam.dtype),
            self.t_sysparam_name = "t_sysparam"
            r_arg_sysparam = "\n    __global const t_sysparam *sysparam,"

        # ---- DYNAMIC HAMILTON

        r_cl_coeff = ""
        r_htl_priv = ""
        r_arg_htl = ""
        r_htl = '#define HTL(T)\\\n   _hu[__item] = _h0[__item];'
        CX_HTLI = ("_hu[__item] = $(cfloat_add)(_hu[__item], "
                   "$(cfloat_rmul)(htcoeff{i}(T, sysparam[GID]), "
                   "htl[{i}]));")
        if self.ht_coeff is not None:
            # compile coefficient function to OpenCL
            r_cl_coeff = "\n".join(
                f2cl(f, "htcoeff{}".format(i), "t_sysparam")
                for (i, f) in enumerate(self.ht_coeff))

            # render HTL makro contributions due to H(T)
            r_htl += '\\\n   ' \
                   + '\\\n   '.join([CX_HTLI.format(i=i) for i in range(len(self.ht_coeff))])

            # private buffer for matrix elements of H(T)
            r_htl_priv = "$(cfloat_t) htl[{}];".format(len(self.ht_coeff))
            for i in range(len(self.ht_coeff)):
                r_htl_priv += "\n    htl[{i}] = ht{i}[__item];".format(i=i)

                # kernel arguments
                r_arg_htl  += "\n    __global const $(cfloat_t) *ht{i},".format(i=i)

        # -- MAIN MAKRO

        # render R(K) macro content.
        DEBUG and print_debug("precomiler: generating 1 x H_un, 0 x H_t, {} x jump operations.", self.jmp_n)
        CX_UNITARY = ("K = $(cfloat_add)(K, $(cfloat_mul)(ihbar, "
                      "$(cfloat_sub)($(cfloat_mul)(HU(__idx, {i}), R({i}, __idy)), "
                      "$(cfloat_mul)(HU({i}, __idy), R(__idx, {i})))));");
        CX_JUMP = ("K = $(cfloat_add)(K, $(cfloat_mul)"
                   "(jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].PF, "
                   "_rky[jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item+{i}].IDX]));")
        r_define_rk = '\\\n    '.join(
            [CX_UNITARY.format(i=r_clint(i)) for i in range(M)]
            + [CX_JUMP.format(i=r_clint(i)) for i in range(self.jmp_n)])

        # render constants
        r_define = '\n'.join('#define {} {}'.format(*df) for df in [
            ('NATURE_HBAR',   r_clfloat(1)),
            ('IN_BLOCK_SIZE', r_clint(M**2)),
            ('THREAD_X',      r_clint(M)),
            ('N_JUMP',        r_clint(self.jmp_n)),
            ('N_JUMP2',       r_clint(self.jmp_n**2)),
            ('HDIM',          r_clint(M)),
            ('LOCAL_SIZE',    r_clint(self.cl_local_size)),
            # butcher
            ('b1', r_clfrac(1.0, 6.0)),
            ('b2', r_clfrac(4.0, 6.0)),
            ('b3', r_clfrac(1.0, 6.0)),
            ('a21', r_clfloat(0.5)),
            ('a31', r_clfloat(-1.0)),
            ('a32', r_clfloat(2.0)),
        ])

        # render thread layout
        r_tl = '\n'.join(['int2 _idx[LOCAL_SIZE];'] + [
           '    _idx[{}] = (int2)({}, {});'.format(k, i, j)
           for (k, (i, j)) in enumerate((i, j) for i in range(M)
                                               for j in range(M) if i < j+1)
        ])

        # render kernel
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
        print(self.c_kernel)
        self.prg = cl.Program(self.ctx, self.c_kernel).build()


    def get_flat_jumps(self):
        fj = []
        for nw, jumps in enumerate(self.system.get_jumps()):
            if jumps is None:
                continue
            jumps = jumps.reshape((int(len(jumps) / (nw+1)), nw+1))
            for jump in jumps:
                fj.append(jump)
        return fj


    def sync(self,
             state=None,
             t_bath=None,
             y_0=None,
             hu=None,
             htl=None,
             e_ops=None,
             sysparam=None):
        self.state = npmat_manylike(self.system.h0, state)
        # XXX normalize:
        # self.t_bath
        # self.y_0
        # self.hu
        # self.htl
        # self.e_ops
        # self.sysparam
        #
        #
        #
        # XXX final buffer
        # self.h_hu
        # self.h_htl
        # self.h_state
        # self.h_sysparam
        # self.h_cl_jump
        #

        if y_0 is not None:
            self.y_0 = vectorize(y_0, dtype=DTYPE_FLOAT)
            assert np.all(self.y_0 >= 0)
        N, M = self.state.shape[0:2]
        self.buf_state = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self._eb(self.state))
        self.buf_hu = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self._eb(self.state.shape[0]*[self.hu]))

        self.h_sysparam = None
        self.b_sysparam = None
        if self.t_sysparam is not None and sysparam is not None:
            self.h_sysparam = sysparam
            self.b_sysparam = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=sysparam)

        self.h_htl = None
        self.b_htl = None
        if htl is not None:
            self.h_htl = np.array(htl, dtype=np.complex64)
            self.b_htl = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_htl)

        y_0 = self.y_0
        if y_0 is None:
            y_0 = [0.0]
        if len(y_0) == 1:
            y_0 = np.array([y_0] * N, dtype=DTYPE_FLOAT)

        if t_bath is None:
            t_bath = [0.0]
        if not isinstance(t_bath, list):
            t_bath = [t_bath]
        if len(t_bath) == 1:
            t_bath = np.array([t_bath] * N, dtype=DTYPE_FLOAT)

        dst = [boson_stat(t) for t in t_bath]
        cl_jmp = np.zeros((N, *self.jmp_instr.shape), dtype=DTYPE_T_JMP)
        for i in range(0, len(self.state)):
            cl_jmp[i]['IDX'] = self.jmp_instr['IDX']
            cl_jmp[i]['PF'] = y_0[i] \
                            * self.jmp_instr['PF'] \
                            * self.jmp_instr['W']**3 \
                            * (self.jmp_instr['SE'] + dst[i](self.jmp_instr['W']))

        if self.jmp_n == 0 or not self.optimize_jumps:
            self.h_cl_jmp = cl_jmp
        else:
            self.h_cl_jmp = self.cl_jmp_acc_pf(cl_jmp)
            if DEBUG:
                dbg = 'optimized h_cl_jmp: old shape {} to new shape {}.'
                print_debug(dbg.format(cl_jmp.shape, self.h_cl_jmp.shape))


    def run(self, trange, sync_state=False):
        trange = self.normalize_trange(trange)
        h_int_param = self._create_int_param(trange)
        res_len = max(r['INT_N'] for r in h_int_param)

        h_tstate = np.zeros((res_len, *self.state.shape), dtype=self.state.dtype)
        h_debug = -np.ones(res_len + 2, dtype=np.float32)

        self.b_int_param = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=h_int_param)
        b_tstate = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_tstate)
        b_debug = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_debug)
        b_jump = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_cl_jmp)

        # run
        queue = cl.CommandQueue(self.ctx)
        bufs = (self.buf_hu, )
        if self.b_htl is not None:
            bufs += (self.b_htl, )
        bufs += (b_jump, self.b_int_param, )
        if self.b_sysparam is not None:
            bufs += (self.b_sysparam, )
        bufs += (self.buf_state, b_tstate, b_debug)
        self.prg.opmesolve_rk4_eb(queue, *self._work_layout(), *bufs)

        res_state = np.empty_like(self.state)
        cl.enqueue_copy(queue, res_state, self.buf_state)
        cl.enqueue_copy(queue, h_tstate, b_tstate)
        cl.enqueue_copy(queue, h_debug, b_debug)

        h_tstate = self._be(h_tstate.reshape(((res_len, *self.state.shape))))
        h_state = self._be(res_state)

        texpect = None
        # copy date to avoid references
        return OpMeResult(tlist=[np.arange(*r) for r in h_int_param],
                          state=h_state[:] if res_state  is not None else None,
                          tstate=h_tstate[:] if h_tstate  is not None else None,
                          texpect=texpect[:] if texpect is not None else None)

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
        jmp_instr = np.zeros((M, M, max(1, jmp_n_max)), dtype=DTYPE_JMP_INSTR)
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
            jcell = cl_jmp[k, i, j]

            # cells which contribute must have a prefactor
            jcell0 = jcell[np.where(np.abs(jcell['PF']) > 0)[0]]

            # all indices of cell items with the same source idx
            cidx_pf = list(
                np.where(jcell0['IDX'] == idx)[0]
                for idx in set(jcell0['IDX']))

            # accumulate prefactors, assign
            jcell_acc = list(
                (jcell0[indx[0]]['IDX'], np.sum(jcell0[indx]['PF']))
                for indx in cidx_pf)
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


    def _work_layout(self):
        """ returns a global layout of `(N, (d, d))` and a local
            layout of `(1, (d, d))` where `N` is the number of systems
            and d the dimension of the Hilber space.
            """
        local_size = (1, *self.hu.shape[1:])
        local_size = (1, self.cl_local_size)
        global_size = (self.state.shape[0], local_size[1])
        return global_size, local_size


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
    src = r_cltypes(src)
    return src

