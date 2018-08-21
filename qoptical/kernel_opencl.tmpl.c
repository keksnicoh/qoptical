#include <pyopencl-complex.h>

/*{define}*/

#define LX      get_local_id(1)
#define R(X,Y)  _rky[THREAD_X*(X)+(Y)]
#define HU(X,Y) _hu[THREAD_X*(X)+(Y)]
#define GID     get_group_id(0)
#define KI(K)   K[__item]
#define RK(K) \
    /*{rk_macro}*/

/*{structs}*/

__kernel void opmesolve_rk4_eb(
    __global const $(cfloat_t) *hu,
    __global const t_jump *jb,
    __global const t_int_parameters *int_parameters,
    __global $(cfloat_t) *rho,
    __global $(cfloat_t) *result,
    __global $(float) *test_buffer
) {
    $(float) t;
    int n;
    /*{tl}*/
    $(cfloat_t) ihbar = $(cfloat_new)(0.0f, -1.0f / NATURE_HBAR);

    __local $(cfloat_t) k1[IN_BLOCK_SIZE];
    __local $(cfloat_t) k2[IN_BLOCK_SIZE];
    __local $(cfloat_t) k3[IN_BLOCK_SIZE];
    __local $(cfloat_t) _hu[IN_BLOCK_SIZE];
    __local $(cfloat_t) _rho[IN_BLOCK_SIZE];
    __local $(cfloat_t) _rky[IN_BLOCK_SIZE];
    __local t_int_parameters prm;

    // thread layout related private scope
    int __in_offset, __item, __itemT, __out_len, __idx, __idy;
    __in_offset = IN_BLOCK_SIZE * get_global_id(0);
    __idx       = _idx[LX].x;
    __idy       = _idx[LX].y;
    __item      = HDIM * __idx + __idy;
    __itemT     = HDIM * __idy + __idx;
    __out_len   = get_num_groups(0) * IN_BLOCK_SIZE;

    // init local memory
    prm           = int_parameters[GID];
    _rho[__item]  = rho[__in_offset + __item];
    _rho[__itemT] = rho[__in_offset + __itemT];
    _hu[__item]   = hu[__in_offset + __item];
    _hu[__itemT]  = hu[__in_offset + __itemT];

    // t=0
    result[__in_offset + __item] = _rho[__item];
    result[__in_offset + __itemT] = _rho[__itemT];

    // loop init
    t  = prm.INT_T0;
    n  = 1;
    while (n < prm.INT_N) {
        // k1
        k1[__item] = k1[__itemT] = $(cfloat_fromreal)(0.0f);
        _rky[__itemT] = _rky[__item] = _rho[__item];
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k1)

        // k2
        barrier(CLK_LOCAL_MEM_FENCE);
        k2[__item] = k2[__itemT] = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho[__item].real
                          + a21 * prm.INT_DT * k1[__item].real;
        _rky[__item].imag = _rho[__item].imag
                          + a21 * prm.INT_DT * k1[__item].imag;
        _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k2)

        // k3
        barrier(CLK_LOCAL_MEM_FENCE);
        k3[__item] = k3[__itemT] = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho[__item].real
                          + a31 * prm.INT_DT * k1[__item].real
                          + a32 * prm.INT_DT * k2[__item].real;
        _rky[__item].imag = _rho[__item].imag
                          + a31 * prm.INT_DT * k1[__item].imag
                          + a32 * prm.INT_DT * k2[__item].imag;
        _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k3)

        _rho[__item].real += prm.INT_DT * (b1 * k1[__item].real + b2 * k2[__item].real + b3 * k3[__item].real);
        _rho[__item].imag += prm.INT_DT * (b1 * k1[__item].imag + b2 * k2[__item].imag + b3 * k3[__item].imag);
        _rho[__itemT] = $(cfloat_conj)(_rho[__item]);
        result[__out_len * n + __in_offset + __item] = _rho[__item];
        result[__out_len * n + __in_offset + __itemT] = _rho[__itemT];

        t = prm.INT_T0 + (++n) * prm.INT_DT;
    }
    rho[__in_offset + __item] = _rho[__item];
    rho[__in_offset + __itemT] = _rho[__itemT];
}