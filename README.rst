This is a relatively simple implementation of raytracing null geodesics in the Kerr metric using the formalism of `Gralla and Lupsasca 2019 <https://arxiv.org/abs/1910.12881>`_

In addition to some standard python libraries, this code uses GSL bindings for evaluating elliptic integrals of the third kind. You need to have GSL and the python ``ctypes`` module installed installed. Then verify that the following works:

.. code-block:: bash

  ctypes.CDLL('libgsl.so') 
  
To try a simple example, try

.. code-block:: bash

  from kerr_raytracing_ana import *
  out = raytrace_ana(plotdata=True)
