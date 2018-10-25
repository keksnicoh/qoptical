from setuptools import setup, find_packages

setup(
    name="qoptical",
    packages=find_packages(),
    requires=["numpy", "scipy", "mako", "pybind11", "pyopencl", "pytest", "cython", "qutip"],
])