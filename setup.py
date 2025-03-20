import os
from setuptools import setup
#read
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="kgeo",

          version = "0.0.1",

          author = "Andrew Chael",
          author_email = "achael@princeton.edu",
          description = "Analytic Kerr Raytracing",
          long_description=read('README.rst'),
          license = "GPLv3",
          keywords = "astronomy EHT black holes",
          url = "https://github.com/achael/kgeo",
          download_url = "https://github.com/achael/kgeo/archive/v0.0.1.tar.gz",
          packages = ["kgeo"],
          install_requires=["numpy",
                            "scipy",
                            "mpmath",
                            "tqdm",
                            "matplotlib",
                            "ehtim",
                            "h5py"
                          ],
          classifiers=[
            'Development Status :: 3 - Alpha',     
            'Intended Audience :: Developers',    
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3.8',
          ],
          include_package_data=True,
          package_data={'': ['*.dat','*.csv']},

         )

