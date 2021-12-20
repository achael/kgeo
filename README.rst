This is a relatively simple implementation of raytracing null geodesics in the Kerr metric using the formalism of `Gralla and Lupsasca 2019 <https://arxiv.org/abs/1910.12881>`_

In addition to some standard python libraries, this code requires the latest version of `scipy<https://github.com/scipy/scipy>`_ to perform elliptic integrals of the third kind. 

Alternatively, elliptic integrals may be perofmed with bindings to GSL (which are much slower). To use GSL, you need to have the python ``ctypes`` module installed. Then verify that the following works:

.. code-block:: bash

  ctypes.CDLL('libgsl.so') 
  
Then modify the following global variables at the top of kerr_raytracing_ana.py

.. code-block:: bash

  SCIPY = False
  GSL = True

To run a simple example, try

.. code-block:: bash

  from kerr_raytracing_ana import *
  out = raytrace_ana(plotdata=True)
