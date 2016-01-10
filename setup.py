from setuptools import setup
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext


setup(name='datumio',
      version='0.1.0',
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
