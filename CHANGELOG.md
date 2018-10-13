# CHANGELOG

## 0.0
- **2018-10.12**:
    - Fixed bug which lead to uncontrolable divergence in the OpenCL integrator.
- **2018-09-21**:
    - fixed bug in rk4 butcher schema of OpenCLKernel.
    - optimized OpenCLKernel by using local memory for jumping buffers.
    - Added `settings.QOP`.
- **2018-09-18**: (Refactoring & BugFixes)
    - Fixed time-gatter bugs in OpenCL kernel, added/improved test
    - moved test_*.py to test/*.py
    - replaced `qoptical.opme.ReducedSyste` by `qoptical.hamilton.ReducedSystem`.
    - added `setup.py`.
- **2018-09-12**: (New Features)
    - improved OpenCL integrator such that the calculation is performed chunk-wise
      to allow integrations where the buffers are bigger than the physical memory
      of the device.
    - added `kernel_opencl.opmesolve_cl_expect()`.
    - added `math.dft_single_freq_window()`.
    - added `fstools.persist_fs()`, `fstools.load_fs()`.
- **2018-08-06**: initial version 0.0. Prototype QuTip and OpenCL kernel implementation.