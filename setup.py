# Adapted from https://github.com/10XDev/tsne

import sys
import platform

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from Cython.Build import build_ext

if sys.platform == 'darwin':
    # OS X
    version, _, _ = platform.mac_ver()
    parts = version.split('.')
    v1 = int(parts[0])
    v2 = int(parts[1])
    v3 = int(parts[2]) if len(parts) == 3 else None

    if v2 >= 10:
        # More than 10.10
        extra_compile_args=['-I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers']
    else:
        extra_compile_args=['-I/System/Library/Frameworks/vecLib.framework/Headers']

    ext_modules = [Extension(name='tensorsne._cpp',
                   sources=['bhtsnecpp/sptree.cpp', 'bhtsnecpp/tsne.cpp', 'tensorsne/_cpp.pyx'],
                   include_dirs=[numpy.get_include(), 'bhtsnecpp/'],
                   extra_compile_args=extra_compile_args + ['-ffast-math', '-O3'],
                   extra_link_args=['-Wl,-framework', '-Wl,Accelerate', '-lcblas'],
                   language='c++')]

else:
    # LINUX
    ext_modules = [Extension(name='tensorsne._cpp',
                   sources=['bhtsnecpp/sptree.cpp', 'bhtsnecpp/tsne.cpp', 'tensorsne/_cpp.pyx'],
                   include_dirs=[numpy.get_include(), '/usr/include', '/usr/local/include', 'bhtsnecpp/'],
                   library_dirs=['/usr/lib', '/usr/local/lib', '/usr/lib64/atlas'],
                   extra_compile_args=['-msse2', '-O3', '-fPIC', '-w', '-ffast-math'],
                   extra_link_args=['-lblas'],
                   language='c++')]

ext_modules = cythonize(ext_modules)

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='tensorsne',
      version='0.1',
      cmdclass={'build_ext': build_ext},
      author='Gokcen Eraslan',
      author_email='gokcen.eraslan@gmail.com',
      url='https://github.com/gokceneraslan/tensorsne',
      description='Barnes-Hut tSNE in Python',
      license='MIT',
      packages=find_packages(),
      ext_modules=ext_modules,
      install_requires=required
)
