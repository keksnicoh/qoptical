# CHANGELOG

## 0.0
- 2018-09-12:
    - improved OpenCL integrator such that the calculation is performed chunk-wise
      to allow integrations where the buffers are bigger than the physical memory
      of the device.
    - Added `kernel_opencl.opmesolve_cl_expect()`.
    - added `math.dft_single_freq_window()`.
    - added `fstools.persist_fs()`, `fstools.load_fs()`.
- 2018-08-06: initial version 0.0. Prototype QuTip and OpenCL kernel implementation.