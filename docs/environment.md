# Environment
The following environment variables can be exported to change the behavior of the lib

## QOP_DEBUG

Enables debug output to stdout, e.g.:

```bash
export QOP_DEBUG=1
```

## QOP_TOL_COMPLEX

a complex number is interpreted as zero if its absolute value is lower than this tolerance

```bash
export QOP_TOL_COMPLEX=0.0001
```

the OpenCL kernel uses this value to identify non-contributing operations due to the lindblad dissipators.
If `QOP_DEBUG` is enabled then the integrator provides some information about the inner jumping structure:

```bash
[...] prepared 90 jumps. Require 90 operations per work-item.
[...] the jumps can be optimized such that at most 19 operations are required
[...] ... 9 transitions have non-zero (>0.0001) contribution.
```

## QOP_ECHO_COMPILED_KERNEL

echo compiled OpenCL kernel after compilation

```bash
export QOP_ECHO_COMPILED_KERNEL=1
```
