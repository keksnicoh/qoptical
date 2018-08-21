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
        self.system    = system
        self.state     = None
        self.t_bath    = None
        self.y_0       = None
        self.htl       = None

        self.optimize_jumps = True

        # generated kernel c code
        self.c_kernel = None
        self.hu        = npmat_manylike(self.system.h0, [self.system.h0])
        self.ctx       = ctx or self._ctx()
        self.ev        = self.system.ev
        self._mb       = np.array(self.system.s, dtype=self.system.s.dtype)

        self.cl_local_size = None

        # the number of jumping instructions due to dissipators
        self.jmp_n = None

        # jumping instructions for all cells
        self.jmp_instr = None


    def compile(self):
        M = self.system.h0.shape[0]
        self.cl_local_size = int(M * (M + 1) / 2)

        # read template
        with open(os.path.join(os.path.dirname(__file__), 'kernel_opencl.tmpl.c')) as f:
            src = f.read()

        # render structures
        r_structs = "\n".join([
            self._compile_struct('t_int_parameters', DTYPE_INTEGRATOR_PARAM),
            self._compile_struct('t_jump', DTYPE_T_JMP),
        ])

        # create jump instructions
        self.create_jmp_instr()

        # render R(K) macro content.
        DEBUG and print_debug("precomiler: generating 1 x H_un, 0 x H_t, {} x jump operations.", self.jmp_n)
        CX_UNITARY = ("KI(K) = $(cfloat_add)(KI(K), $(cfloat_mul)(ihbar, "
                      "$(cfloat_sub)($(cfloat_mul)(HU(__idx, {i}), R({i}, __idy)), "
                      "$(cfloat_mul)(HU({i}, __idy), R(__idx, {i})))));");
        CX_JUMP = ("KI(K) = $(cfloat_add)(KI(K), $(cfloat_mul)"
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
        self.c_kernel = r_tmpl(src, define = r_define,
                                    structs = r_structs,
                                    rk_macro = r_define_rk,
                                    tl=r_tl)

        DEBUG and print_debug(
            "generated kernel: {} lines. Compiling...",
            self.c_kernel.count("\n") + 1)
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


    def sync(self, state=None, t_bath=None, y_0=None, hu=None, htl=None, e_ops=None):
        self.state = npmat_manylike(self.system.h0, state)
        if y_0 is not None:
            self.y_0 = vectorize(y_0, dtype=DTYPE_FLOAT)
            assert np.all(self.y_0 >= 0)
        N, M = self.state.shape[0:2]
        self.buf_state = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self._eb(self.state))
        self.buf_hu = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self._eb(self.state.shape[0]*[self.hu]))

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
        bufs = (self.buf_hu, b_jump, self.b_int_param, self.buf_state, b_tstate, b_debug)
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
            w = jump[0]['w']
            jx, jy = jump[np.where(jump['I'][:,0] == i)[0]], \
                     jump[np.where(jump['I'][:,0] == j)[0]]
            for j1, j2 in itertools.product(jx, jy):
                fidx = idx(j1['I'][1], j2['I'][1])
                jelem[tidx].append((fidx, j1['d'] * j2['d'].conj(), 1, w)) # A rho Ad
                jelem[fidx].append((tidx, j1['d'].conj() * j2['d'], 0, w)) # Ad rho A

            # -1/2 {rho, Ad A}
            jy = jump[np.where(jump['I'][:,1] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:,0] == j2['I'][0])[0]]:
                    fidx = idx(i, j1['I'][1])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 1, w))
            jx = jump[np.where(jump['I'][:,1] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:,0] == j1['I'][0])[0]]:
                    fidx = idx(j2['I'][1], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'] * j2['d'].conj(), 1, w))
            # XXX -1/2 {rho, A Ad} can be merged with the block above?!
            jx = jump[np.where(jump['I'][:,0] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:,1] == j1['I'][1])[0]]:
                    fidx = idx(j2['I'][0], j)
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 0, w))
            jy = jump[np.where(jump['I'][:,0] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:,1] == j2['I'][1])[0]]:
                    fidx = idx(i, j1['I'][0])
                    jelem[tidx].append((fidx, -0.5 * j1['d'].conj() * j2['d'], 0, w))

        # structurize the data as (M, M, n_max_jump) numpy array
        jmp_n_max = max(len(_) for _ in jelem)
        jmp_instr = np.zeros((M, M, max(1, jmp_n_max)), dtype=DTYPE_JMP_INSTR)
        jmp_n_opt = 0
        for (i, j) in [(i, j) for j in range(M) for i in range(M)]:
            jmp_instr[i, j, 0:len(jelem[idx(i, j)])] = jelem[idx(i, j)]
            jmp_n_opt = max(jmp_n_opt, len(set(jmp_instr[i, j, 0:len(jelem[idx(i, j)])]['IDX'])))

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
        N, M = self.state.shape[0:2]

        cl_jmp_opt = np.zeros((*cl_jmp.shape[0:3], self.jmp_n), dtype=cl_jmp.dtype)
        for (k, (i, j)) in itertools.product(range(N), [(i, j) for j in range(M) for i in range(M)]):
            jcell = cl_jmp[k, i, j]
            # cells which contribute must have a prefactor
            jcell0 = jcell[np.where(np.abs(jcell['PF']) > 0)[0]]
            # all indices of cell items with the same source idx
            cidx_pf = list(np.where(jcell0['IDX'] == idx)[0] for idx in set(jcell0['IDX']))
            # accumulate prefactors, assign
            jcell_acc = list((jcell0[indx[0]]['IDX'], np.sum(jcell0[indx]['PF'])) for indx in cidx_pf)
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
    subst = [('$(cfloat_{})'.format(pf), '{}_{}'.format(ctype, pf)) for pf in R_CLTYPES_POSTFIX]
    subst += [('$(float)', 'double' if double_precision else 'float'), ('$(cf_)', 'cfloat_')]
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

T_VAL = 'VAL'
T_VAR = 'VAR'
T_BINARY_ADD = 'ADD'
T_BINARY_ADD = 'ADD'
MODE_BIN = 1

def f2cl_instruction_scalar(instruction):
    if not isinstance(instruction, dis.Instruction):
        return instruction
    elif instruction.opname == "LOAD_CONST":
        return ('T_VAL', instruction.argval)
    elif instruction.opname == "LOAD_GLOBAL":
        return ('T_SYMBOLE', instruction.argval)
    elif instruction.opname == "LOAD_FAST":
        return ('T_SYMBOLE', instruction.argval)
    else:
        raise ValueError(instruction)
BIN_MAP = {
    'BINARY_ADD': '+',
    'BINARY_MULTIPLY': '*',
    'BINARY_POWER': '**',
    'BINARY_SUBTRACT': '-',
    'BINARY_TRUE_DIVIDE': '/',
    'BINARY_MODULO': '%',
}

def f2cl(f):
    f2cl_ctree_print(f2cl_ctree(f))



def f2cl_ctree_print(a, l=0):
    if isinstance(a, dis.Instruction):
        print(l * ' ' + str(a))
    elif not isinstance(a, tuple):
        return '!?'
    elif a[0] == 'T_FUNC':
        print(l * ' ' + a[0] + "(" + str(a[1]) + ")")
        for b in a[2]:
            f2cl_ctree_print(b, l+1)
    elif a[0] == 'T_BIN':
        print(l * ' ' + a[0] + '(' + a[1] + ')')
        for b in a[2]:
            f2cl_ctree_print(b, l+1)
    elif a[0] == 'T_SYMBOLE':
        print(l * ' ' + a[0] + '(' + str(a[1]) + ')')
    elif a[0] == 'T_VAL':
        print(l * ' ' + a[0] + '(<' + str(a[1].__class__.__name__) + '>' + str(a[1]) + ')')
    elif a[0] == 'T_RETURN':
        print(l * ' ' + 'T_RETURN')
        for b in a[1]:
            f2cl_ctree_print(b, l+1)

def f2cl_ctree(f):
    mode = MODE_BIN
    args = []
    read = []
    current = None
    log = []

    for a in dis.get_instructions(f):
        log.append(a)
        read.append(a)
        if a.opname in ['LOAD_CONST']:
            read[-1] = f2cl_instruction_scalar(a)
        elif a.opname in ['LOAD_FAST']:
            read[-1] = f2cl_instruction_scalar(a)
        elif a.opname in ['LOAD_GLOBAL']:
            read[-1] = f2cl_instruction_scalar(a)
        elif a.opname == 'LOAD_ATTR':
            read[-2] = (read[-2][0], read[-2][1] + '.' + a.argval)
            read = read[:-1]
        elif a.opname in BIN_MAP:
            current = ('T_BIN', BIN_MAP[a.opname], [
                f2cl_instruction_scalar(read[-3]),
                f2cl_instruction_scalar(read[-2])])
            read = read[:-3]
            read.append(current)
        elif a.opname == 'CALL_FUNCTION':
            fargs = read[-a.arg-1:-1]
            fname = read[-a.arg-2]
            current = ('T_FUNC', fname, [f2cl_instruction_scalar(fa) for fa in fargs])
            read = read[:-a.arg-2]
            read.append(current)
        elif a.opname == 'RETURN_VALUE':
            current = ('T_RETURN', [f2cl_instruction_scalar(fa) for fa in read[:-1]])
        else:
            raise RuntimeError('did not understand?!\n\n'+'\n'.join('>>> ' + str(l) for l in log))

    return current
