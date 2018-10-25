from setuptools import setup, find_packages

setup(
    name="qoptical",
    packages=find_packages(),
    install_requires=[
        "cython",
        "mako",
        "pybind11",
        "numpy",
        "scipy",
        "pytest",
        "pytools",
        "pyopencl",
        "qutip",
    ],
    package_data={
        "": [
            "qoptical/kernel_opencl.tmpl.c",
        ]
    },
    include_package_data=True,
)
