from setuptools import setup, find_packages

setup(
    name="qoptical",
    packages=find_packages(),
    setup_requires=[
        'setuptools>=18.0',
        'cython',
        "pybind11",
        "mako"
    ],
    install_requires=["numpy", "scipy", "pytest", "qutip"],
    #install_requires=["numpy", "scipy", "mako", "pybind11", "pyopencl", "pytest", "cython", "qutip"],
)
