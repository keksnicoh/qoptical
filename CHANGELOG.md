# CHANGELOG

## 0.01

- **2020-03-24**:
    - cleanup and port to python 3.7.
    - Removed metropolis stuff
    - Removed result.py

## 0.0
- **2019-05-24**:
    - fixed usage of `cl.tools.match_dtype_to_c_struct` such that the current context is passed.
- **2019-05-01**:
    - added DISSIPATOR_DISABLE_JUMP_TERM, DISSIPATOR_DISABLE_ACOMM_TERM flags
- **2019-04-16**:
    - fixed bug for dynamic coefficient function when no sysparams where defined.
- **2019-01-20**:
    - added custom kappa function support in OpenCL integrator
- **2018-11-17**:
    - improved f2cl so that closures and +/- unary instructions work
    - added time dependend damping y(t)
- **2018-11-13**:
    - optimized jumping buffer such that non-contributing jumps are removed.
- **2018-10-30**:
    - GPU/CPU parallel execution in OpenCL Kernel
- **2018-10-26**:
    - removed performance feature since the allocated shared memory exceeded.
- **2018-10-25**:
    - removed np.matrix()
- **2018-10-12**:
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
