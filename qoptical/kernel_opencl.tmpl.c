#include <pyopencl-complex.h>

/*{define}*/

#define LX      get_local_id(1)
#define R(X,Y)  _rky[THREAD_X*(X)+(Y)]
#define HU(X,Y) _hu[THREAD_X*(X)+(Y)]
#define GID     get_group_id(0)
#define KI(K)   K[__item]
#define RK(K) \
    /*{rk_macro}*/
/*{htl_macro}*/

/*{structs}*/

/*{ht_coeff}*/

__kernel void opmesolve_rk4_eb(
    __global const $(cfloat_t) *hu,/*{arg_htl}*/
    __global const t_jump *jb,/*{arg_sysparam}*/
    __global __write_only $(cfloat_t) *result,/*{arg_debug}*/
    __global const $(cfloat_t) *rho0,
    const $(float) t0,
    const $(float) dt,
    const int n_int
) {
    $(float) t;
    int n;
    /*{tl}*/
    $(cfloat_t) ihbar = $(cfloat_new)(0.0f, -1.0f / NATURE_HBAR);
    $(cfloat_t) k1, k2, k3, _rho;
    __local $(cfloat_t) _hu[IN_BLOCK_SIZE];
    __local $(cfloat_t) _h0[IN_BLOCK_SIZE];
    __local $(cfloat_t) _rky[IN_BLOCK_SIZE];
    __local t_jump _jb[IN_BLOCK_SIZE * N_JUMP];

    // thread layout related private scope
    int __in_offset, __item, __itemT, __out_len, __idx, __idy;
    __in_offset = IN_BLOCK_SIZE * get_global_id(0);
    __idx       = _idx[LX].x;
    __idy       = _idx[LX].y;
    __item      = HDIM * __idx + __idy;
    __itemT     = HDIM * __idy + __idx;
    __out_len   = get_num_groups(0) * IN_BLOCK_SIZE;

    // upload the jumping structure into local buffer.
    // note: this enhanced the performance around 20-50%
    //       (feature/clkernel-performance-1)
    for (int k = 0; k < N_JUMP; ++k) {
        _jb[N_JUMP * __item + k] = jb[GID * N_JUMP * IN_BLOCK_SIZE + N_JUMP * __item + k];
    }

    // init local memory
    _rho         = rho0[__in_offset + __item]; // t=0
    _h0[__item]  = hu[__in_offset + __item];
    _h0[__itemT] = hu[__in_offset + __itemT];
    /*{htl_priv}*/

    HTL(t0)
    barrier(CLK_LOCAL_MEM_FENCE);

    bool non_diag = true;
    if (__idx == __idy) {
        non_diag = false;
    }

    // loop init
    //t  = t0 + n0 * dt;
    t = t0;
    n = 0;
    while (n < n_int) {
        /*{debug_hook_0}*/
        // k1
        k1 = $(cfloat_fromreal)(0.0f);
        _rky[__item] = _rho;
        if (non_diag) _rky[__itemT] = $(cfloat_conj)(_rho);
        if (non_diag) _hu[__itemT] = $(cfloat_conj)(_hu[__item]);

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k1)
        barrier(CLK_LOCAL_MEM_FENCE);

        // k2
        k2 = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho.real + a21 * dt * k1.real;
        _rky[__item].imag = _rho.imag + a21 * dt * k1.imag;
        if (non_diag) _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        HTL(t + dt / 2.0f)
        if (non_diag) _hu[__itemT] = $(cfloat_conj)(_hu[__item]);

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k2)
        barrier(CLK_LOCAL_MEM_FENCE);

        // k3
        k3 = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho.real + a31 * dt * k1.real + a32 * dt * k2.real;
        _rky[__item].imag = _rho.imag + a31 * dt * k1.imag + a32 * dt * k2.imag;
        if (non_diag) _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        HTL(t + dt)
        if (non_diag) _hu[__itemT] = $(cfloat_conj)(_hu[__item]);

        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k3)
        barrier(CLK_LOCAL_MEM_FENCE);

        _rho.real += dt * (b1 * k1.real + b2 * k2.real + b3 * k3.real);
        _rho.imag += dt * (b1 * k1.imag + b2 * k2.imag + b3 * k3.imag);
        result[__out_len * n + __in_offset + __item] = _rho;
        if (non_diag) result[__out_len * n + __in_offset + __itemT] = $(cfloat_conj)(_rho);
        /*{debug_hook_1}*/
        t = t0 + (++n) * dt;
    }
}