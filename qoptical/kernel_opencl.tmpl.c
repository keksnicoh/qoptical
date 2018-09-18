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
    __global const t_jump *jb,
    __global const t_int_parameters *int_parameters,/*{arg_sysparam}*/
    __global $(cfloat_t) *result,/*{arg_debug}*/
    const int n0
) {
    $(float) t;
    int n;
    /*{tl}*/
    $(cfloat_t) ihbar = $(cfloat_new)(0.0f, -1.0f / NATURE_HBAR);
    $(cfloat_t) k1, k2, k3, _rho;
    __local $(cfloat_t) _hu[IN_BLOCK_SIZE];
    __local $(cfloat_t) _h0[IN_BLOCK_SIZE];
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
    prm          = int_parameters[0];
    //_rho         = result[__out_len * (n0 - 1) + __in_offset + __item]; // t=0
    _rho         = result[ __in_offset + __item]; // t=0
    _h0[__item]  = hu[__in_offset + __item];
    _h0[__itemT] = hu[__in_offset + __itemT];
    /*{htl_priv}*/

    // loop init
    //t  = prm.INT_T0 + n0 * prm.INT_DT;
    t  = prm.INT_T0;
    n  = 0;
    //n  = n0;
    while (n < prm.INT_N) {
        /*{debug_hook_0}*/
        // k1
        k1 = $(cfloat_fromreal)(0.0f);
        _rky[__item] = _rho;
        _rky[__itemT] = $(cfloat_conj)(_rho);
        HTL(t)
        _hu[__itemT] = $(cfloat_conj)(_hu[__item]);
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k1)

        // k2
        barrier(CLK_LOCAL_MEM_FENCE);
        k2 = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho.real + a21 * prm.INT_DT * k1.real;
        _rky[__item].imag = _rho.imag + a21 * prm.INT_DT * k1.imag;
        _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        HTL(t)
        _hu[__itemT] = $(cfloat_conj)(_hu[__item]);
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k2)

        // k3
        barrier(CLK_LOCAL_MEM_FENCE);
        k3 = $(cfloat_fromreal)(0.0f);
        _rky[__item].real = _rho.real + a31 * prm.INT_DT * k1.real + a32 * prm.INT_DT * k2.real;
        _rky[__item].imag = _rho.imag + a31 * prm.INT_DT * k1.imag + a32 * prm.INT_DT * k2.imag;
        _rky[__itemT] = $(cfloat_conj)(_rky[__item]);
        HTL(t)
        _hu[__itemT] = $(cfloat_conj)(_hu[__item]);
        barrier(CLK_LOCAL_MEM_FENCE);
        RK(k3)

        _rho.real += prm.INT_DT * (b1 * k1.real + b2 * k2.real + b3 * k3.real);
        _rho.imag += prm.INT_DT * (b1 * k1.imag + b2 * k2.imag + b3 * k3.imag);
        result[__out_len * (n + 1) + __in_offset + __item] = _rho;
        result[__out_len * (n + 1) + __in_offset + __itemT] = $(cfloat_conj)(_rho);
        /*{debug_hook_1}*/
        barrier(CLK_LOCAL_MEM_FENCE);
        t = prm.INT_T0 + (++n) * prm.INT_DT;
    }
}