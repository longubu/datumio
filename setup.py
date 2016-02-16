import os
import re
from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str
    with open(os.path.join(here, 'datumio', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search("__version__ = '(.*)'", init_py).groups()[0]
except Exception:
    version = ''

setup(name='datumio',
      version=version,
      description='Real-time augmentation of data for inputs into deep learning models.',
      author='Long Van Ho',
      author_email='longvho916@gmail.com',
      url='https://github.com/longubu/datumio',
      download_url='---',
      license='MIT',
      extras_require={
          '': [''],
      },
      packages=find_packages())
