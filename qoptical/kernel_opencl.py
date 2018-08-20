# -*- coding: utf-8 -*-
""" contains qutip kernels for solving optical
    master equation.
    :author: keksnicoh
"""
import pyopencl as cl
import pyopencl.tools
import numpy as np
import itertools
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


def r_clfloat(f, prec=None):
    """ renders an OpenCL float representation """
    if prec is not None:
        raise NotImplementedError()
    sf = str(f)
    return ''.join([sf, '' if '.' in sf else '.', 'f'])


def r_clfrac(p, q, prec):
    """ renders an OpenCL fractional representation """
    return "{}/{}".format(r_clfloat(p, prec), r_clfloat(q, prec))


def r_clint(f):
    """ renders an OpenCL integer representation """
    assert isinstance(f, int)
    return str(f)


def r_cltypes(src, double_precision=False):

    if double_precision:
        src = src.replace('$(cfloat_t)', 'cdouble_t') \
                 .replace('$(cfloat_mul)', 'cdouble_mul') \
                 .replace('$(cfloat_rmul)', 'cdouble_rmul') \
                 .replace('$(cfloat_sub)', 'cdouble_sub') \
                 .replace('$(cfloat_new)', 'cdouble_new') \
                 .replace('$(cfloat_add)', 'cdouble_add') \
                 .replace('$(cfloat_neg)', 'cdouble_neg') \
                 .replace('$(cfloat_conj)', 'cdouble_conj') \
                 .replace('$(cfloat_fromreal)', 'cdouble_fromreal') \
                 .replace('$(float)', 'double') \
                 .replace('$(cf_)', 'cdouble_')
    else:
        src = src.replace('$(cfloat_t)', 'cfloat_t') \
                 .replace('$(cfloat_mul)', 'cfloat_mul') \
                 .replace('$(cfloat_sub)', 'cfloat_sub') \
                 .replace('$(cfloat_rmul)', 'cfloat_rmul') \
                 .replace('$(cfloat_new)', 'cfloat_new') \
                 .replace('$(cfloat_add)', 'cfloat_add') \
                 .replace('$(cfloat_neg)', 'cfloat_neg') \
                 .replace('$(cfloat_conj)', 'cfloat_conj') \
                 .replace('$(cfloat_fromreal)', 'cfloat_fromreal') \
                 .replace('$(float)', 'float') \
                 .replace('$(cf_)', 'cfloat_')

    return src

class KernelEnvironment():


    def __init__(self,
                 kernel_args = None,
                 structs     = None,
                 includes    = None,
                 defines     = None,
    ):
        self.kernel_args = kernel_args or []
        self.structs     = structs     or []
        self.includes    = includes    or []
        self.defines     = defines     or []


    def __radd__(self, a):
        kenv  = KernelEnvironment()

        kenv.kernel_args = self.kernel_args + a.kernel_args
        kenv.structs     = self.structs     + a.structs
        kenv.includes    = self.includes    + a.includes
        kenv.defines     = self.defines     + a.defines

        return kenv

    def __iadd__(self, a): return self.__radd__(a)

    def add_struct(self, name, struct):
        keys = [k for k, v in self.structs]
        if name in keys:
            raise ValueError('struct name "{}" allready in use.'.format(name))
        self.structs.append((name, struct))
        return self

    def render_structs(self):
        return "\n".join(v for k, v in self.structs)


class OpenCLKernel():

    def __init__(self, system, ctx=None):
        """ create QutipKernel for given ``system`` """
        # api state
        self.system    = system
        self.state     = None
        self.t_bath    = None
        self.y_0       = None
        self.htl       = None
        self.hu        = npmat_manylike(self.system.h0, [self.system.h0])
        self.ctx       = ctx or self._ctx()
        self.ev        = self.system.ev
        self._mb       = np.array(self.system.s, dtype=self.system.s.dtype)

    def _ctx(self):
        platforms = cl.get_platforms()
        assert len(platforms) > 0
        gpu_devices = list(d for d in platforms[0].get_devices() if d.type == cl.device_type.GPU)
        assert len(gpu_devices) > 0
        return cl.Context(devices=gpu_devices)

    def _compile_int_parameters(self):
        strct_name   = 't_int_parameters'
        _, c_decl    = cl.tools.match_dtype_to_c_struct(
            self.ctx.devices[0],
            strct_name,
            DTYPE_INTEGRATOR_PARAM
        )
        return KernelEnvironment().add_struct(strct_name, c_decl)

    def _compile_struct(self, name, dtype):
        _, c_decl    = cl.tools.match_dtype_to_c_struct(
            self.ctx.devices[0],
            name,
            dtype
        )
        return KernelEnvironment().add_struct(name, c_decl)

    def compile(self):
        kenv = KernelEnvironment()
        DTYPE_T_JUMP = np.dtype([
            ('IDX', np.int32),
            ('PF', DTYPE_COMPLEX),
        ]);
        kenv += self._compile_int_parameters()
        kenv += self._compile_struct('t_jump', DTYPE_T_JUMP)
        src = """
#include <pyopencl-complex.h>
// butcher schema
#define b1 1.0f/6.0f
#define b2 4.0f/6.0f
#define b3 1.0f/6.0f
#define a21 0.5f
#define a31 -1
#define a32 2
/*{define}*/

#define LX get_local_id(1)
#define LY get_local_id(2)
#define R(X,Y) _rky[THREAD_X*(X)+(Y)]
#define HU(X,Y) _hu[THREAD_X*(X)+(Y)]
#define GID get_group_id(0)
#define KI(K) K[__item]

#define RK(K) \\
    /*{R(K)}*/
/*{structs}*/

__kernel void opmesolve_rk4_eb(
    __global const $(cfloat_t) *hu,
    __global const t_jump *jb,
    __global const t_int_parameters *int_parameters,
    __global const $(float) *y_0,
    __global $(cfloat_t) *rho,
    __global $(cfloat_t) *result,
    __global $(float) *test_buffer
) {
    $(float) t;
    int n;

    int __in_offset, __item, __out_len;

    $(cfloat_t) ihbar = $(cfloat_new)(0.0f, -1.0f / NATURE_HBAR);

    __local $(cfloat_t) k1[IN_BLOCK_SIZE];
    __local $(cfloat_t) k2[IN_BLOCK_SIZE];
    __local $(cfloat_t) k3[IN_BLOCK_SIZE];
    __local $(cfloat_t) _hu[IN_BLOCK_SIZE];
    __local $(cfloat_t) _rho[IN_BLOCK_SIZE];
    __local $(cfloat_t) _rky[IN_BLOCK_SIZE];
    __local t_int_parameters int_prm;

    __in_offset = IN_BLOCK_SIZE * get_global_id(0);
    __item      = THREAD_X * LX + LY;
    __out_len   = get_num_groups(0) * IN_BLOCK_SIZE;
    // init local memory
    int_prm      = int_parameters[GID];
    _rho[__item] = rho[__in_offset + __item];
    _hu[__item]  = hu[__in_offset + __item];

    // t=0
    result[__in_offset + __item] = _rho[__item];

    // loop init
    t  = int_prm.INT_T0;
    n  = 1;
    while (n < int_prm.INT_N) {
        // k1
        k1[__item] = $(cfloat_fromreal)(0.0f);
        _rky[__item] = _rho[__item];

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k1)

        // k2
        barrier(CLK_LOCAL_MEM_FENCE);
        k2[__item] = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho[__item].real
                          + a21 * int_prm.INT_DT * k1[__item].real;
        _rky[__item].imag = _rho[__item].imag
                          + a21 * int_prm.INT_DT * k1[__item].imag;

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k2)

        // k3
        barrier(CLK_LOCAL_MEM_FENCE);
        k3[__item] = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho[__item].real
                          + a31 * int_prm.INT_DT * k1[__item].real
                          + a32 * int_prm.INT_DT * k2[__item].real;
        _rky[__item].imag = _rho[__item].imag
                          + a31 * int_prm.INT_DT * k1[__item].imag
                          + a32 * int_prm.INT_DT * k2[__item].imag;

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k3)
        barrier(CLK_LOCAL_MEM_FENCE);

        _rho[__item].real += int_prm.INT_DT * (
            b1 * k1[__item].real
            + b2 * k2[__item].real
            + b3 * k3[__item].real
        );
        _rho[__item].imag += int_prm.INT_DT * (
            b1 * k1[__item].imag
            + b2 * k2[__item].imag
            + b3 * k3[__item].imag
        );

        result[__out_len * n + __in_offset + __item] = _rho[__item];

        t = int_prm.INT_T0 + (++n) * int_prm.INT_DT;

        // debug buffer
        if (GID == 0) {
            test_buffer[n] = n-1;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rho[__in_offset + __item] = _rho[__item];
}
        """

        M = self.system.h0.shape[0]


        self.h_jump = self.create_h_jump(DTYPE_T_JUMP)
        n_jump = self.h_jump.shape[2]
        # render R(K) macro content.
        CX_UNITARY = ("KI(K) = $(cfloat_add)(KI(K), $(cfloat_mul)(ihbar, "
                      "$(cfloat_sub)($(cfloat_mul)(HU(LX, {i}), R({i}, LY)), "
                      "$(cfloat_mul)(HU({i}, LY), R(LX, {i})))));");
        CX_JUMP = ("KI(K) = $(cfloat_add)(KI(K), $(cfloat_rmul)(y_0[GID] * (1.0+0.0), "
                   "$(cfloat_mul)(jb[N_JUMP * __item+{i}].PF, "
                   "_rky[jb[N_JUMP * __item+{i}].IDX])));")
        r_define_rk = '\\\n    '.join(
            [CX_UNITARY.format(i=r_clint(i)) for i in range(M)]
            + [CX_JUMP.format(i=r_clint(i)) for i in range(n_jump)])

        # render constants
        r_define = '\n'.join('#define {} {}'.format(*df) for df in [
            ('NATURE_HBAR',   r_clfloat(1)),
            ('IN_BLOCK_SIZE', r_clint(M**2)),
            ('THREAD_X',      r_clint(M)),
            ('N_JUMP',        r_clint(n_jump)),
        ])

        src = src.replace('/*{define}*/',  r_define) \
                 .replace('/*{structs}*/', kenv.render_structs()) \
                 .replace('/*{R(K)}*/',    r_define_rk)
        src = r_cltypes(src)
        print(src)
       # assert False
     #   print(src)
        self.prg = cl.Program(self.ctx, src).build()


    def create_h_jump(self, dtype):

        # go thru all jumps to create matrix element
        # interchange stacks for all (i, j) elements of the state.
        M          = self.system.h0.shape[0]
        idx        = lambda i, j: M * i + j
        jelem      = [[] for _ in range(M ** 2)]
        all_idx    = [(i, j) for i in range(M) for j in range(M)]
        flat_jumps = self.get_flat_jumps()

        for ((i, j), jump) in itertools.product(all_idx, flat_jumps):
            w3 = jump['w'][0] ** 3
            tidx = idx(i, j)

            # A rho Ad
            jx, jy = jump[np.where(jump['I'][:,1] == i)[0]], \
                     jump[np.where(jump['I'][:,1] == j)[0]]
            for j1, j2 in itertools.product(jx, jy):
                fidx = idx(j1['I'][0], j2['I'][0])
                jelem[tidx].append((fidx, w3 * j1['d'] * j2['d'].conj()))

            # -1/2 {rho, Ad A}
            jy = jump[np.where(jump['I'][:,0] == j)[0]]
            for j2 in jy:
                for j1 in jump[np.where(jump['I'][:,1] == j2['I'][1])[0]]:
                    fidx = idx(i, j1['I'][0])
                    jelem[tidx].append((fidx, -0.5 * w3 * j1['d'].conj() * j2['d']))
            jx = jump[np.where(jump['I'][:,0] == i)[0]]
            for j1 in jx:
                for j2 in jump[np.where(jump['I'][:,1] == j1['I'][1])[0]]:
                    fidx = idx(j2['I'][0], j)
                    jelem[tidx].append((fidx, -0.5 * w3 * j1['d'].conj() * j2['d']))

        n_jump = max(len(_) for _ in jelem)
        h_jump = np.zeros((M, M, max(1, n_jump)), dtype=dtype)
        for (i, j) in [(i, j) for j in range(M) for i in range(M)]:
            h_jump[i, j, 0:len(jelem[idx(i, j)])] = jelem[idx(i, j)]

        return h_jump

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

        self.buf_state = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self._eb(self.state))
        self.buf_hu = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self._eb(self.state.shape[0]*[self.hu]))

        y_0 = self.y_0
        if y_0 is None:
            y_0 = [0.0]
        if len(y_0) == 1:
            y_0 = np.array([y_0] * self.state.shape[0], dtype=DTYPE_FLOAT)
        self.buf_y0 = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=y_0)

    def run(self, trange, sync_state=False):
        trange = self.normalize_trange(trange)
        h_int_param = self._create_int_param(trange)
        res_len = max(r['INT_N'] for r in h_int_param)

        h_tstate = np.zeros((res_len, *self.state.shape), dtype=self.state.dtype)
        h_debug = -np.ones(res_len + 2, dtype=np.float32)

        self.b_int_param = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=h_int_param)
        b_tstate = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_tstate)
        b_debug = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=h_debug)
        b_jump = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=self.h_jump)

        # run
        queue = cl.CommandQueue(self.ctx)
        bufs = (self.buf_hu, b_jump, self.b_int_param, self.buf_y0, self.buf_state, b_tstate, b_debug)
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

        pass


    def _eb(self, op):
        """ transforms `op` into eigenbase.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self._mb.conj().T @ op @ self._mb


    def _be(self, op):
        """ transforms `op` back into original basis.
            `op` must be ndarray of shape `(M,M)` or `(N,M,M)`
            """
        return self._mb @ op @ self._mb.conj().T


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
        global_size = (self.state.shape[0], *self.hu.shape[1:])
        return global_size, local_size


    def _create_int_param(self, trange):
        """ create integrator parameters from normalized `trange` """
        return np.array([
            (tr['INT_T0'], tr['INT_DT'], len(np.arange(*tr)))
            for tr in trange
        ], dtype=DTYPE_INTEGRATOR_PARAM)
