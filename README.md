# kgeo

[![License: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/achael/kgeo/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8092968-blue.svg)](https://doi.org/10.5281/zenodo.8092968)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/achael/kgeo/actions/workflows/ci.yml)

[![CI](https://github.com/achael/kgeo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/achael/kgeo/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/achael/kgeo/main)](https://codecov.io/gh/achael/kgeo/branch/main)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This is a relatively simple implementation of raytracing null geodesics in the Kerr metric using the formalism of [Gralla and Lupsasca 2019](https://arxiv.org/abs/1910.12881).

In addition to some standard python libraries, this code requires the latest version of [scipy](https://github.com/scipy/scipy) to perform elliptic integrals of the third kind.

Alternatively, elliptic integrals may be performed with bindings to GSL (which are much slower). To use GSL, you need to have the python `ctypes` module installed. Then verify that the following works:

```bash
ctypes.CDLL('libgsl.so')
```

Then modify the following global variables at the top of `kgeo/kerr_raytracing_ana.py`:

```python
SCIPY = False
GSL = True
```

To run a simple example, try:

```python
from kgeo.kerr_raytracing_ana import *
out = raytrace_ana(plotdata=True)
```
