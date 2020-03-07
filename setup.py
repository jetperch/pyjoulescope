# Copyright 2018 Jetperch LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Joulescope python setuptools module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
import setuptools
from distutils.command.build import build as build_orig
import platform
import os
import struct
import sys


MYPATH = os.path.abspath(os.path.dirname(__file__))
VERSION_PATH = os.path.join(MYPATH, 'joulescope', 'version.py')


try:
    from Cython.Build import cythonize
    USE_CYTHON = os.path.isfile(os.path.join(MYPATH, 'joulescope', 'stream_buffer.pyx'))
except ImportError:
    USE_CYTHON = False


def _to_int_safe(s):
    try:
        return int(s)
    except:
        return s


def _version_parse(s):
    return [_to_int_safe(k) for k in s.split('.')]


def _platform_check():
	if struct.calcsize("P") != 8:
	    raise RuntimeError('pyjoulescope only supports 64-bit Python')
	if _version_parse(platform.python_version()) < _version_parse('3.6.0'):
	    raise RuntimeError('pyjoulescope only supports Python 3.6+')


_platform_check()


def _version_get():
    with open(VERSION_PATH, 'rt') as fv:
        for line in fv:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip()[1:-1]
    raise RuntimeError('VERSION not found!')


ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    setuptools.Extension('joulescope.stream_buffer',
        sources=[
            'joulescope/stream_buffer' + ext,
            'joulescope/native/running_statistics.c',
        ],
        include_dirs=[],
    ),
    setuptools.Extension('joulescope.filter_fir',
        sources=[
            'joulescope/filter_fir' + ext,
            'joulescope/native/filter_fir.c',
        ],
        include_dirs=[],
    ),
    setuptools.Extension('joulescope.pattern_buffer',
        sources=['joulescope/pattern_buffer' + ext],
        include_dirs=[],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, compiler_directives={'language_level': '3'})  # , annotate=True)


# Get the long description from the README file
with open(os.path.join(MYPATH, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


if sys.platform.startswith('win'):
    PLATFORM_INSTALL_REQUIRES = ['pypiwin32>=223']
else:
    PLATFORM_INSTALL_REQUIRES = []


# Hack to install numpy before numpy.get_include()
# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class Build(build_orig):

    def finalize_options(self):
        super().finalize_options()
        # I stole this line from ead's answer:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        self.distribution.include_dirs.append(numpy.get_include())


setuptools.setup(
    name='joulescope',
    version=_version_get(),
    description='Joulescope™ host driver and utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://www.joulescope.com',
    author='Jetperch LLC',
    author_email='joulescope-dev@jetperch.com',
    license='Apache',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Operating systems
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: Microsoft :: Windows :: Windows 8.1',
        'Operating System :: Microsoft :: Windows :: Windows 8',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',

        # Supported Python versions
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        
        # Topics
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Embedded Systems',
        'Topic :: Software Development :: Testing',
        'Topic :: System :: Hardware :: Hardware Drivers',
        'Topic :: Utilities',
    ],

    keywords='joulescope driver',

    packages=setuptools.find_packages(exclude=['native', 'docs', 'test', 'dist', 'build']),
    ext_modules=extensions,
    cmdclass={
        'build': Build,
    },
    include_dirs=[],
    
    # See https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='~=3.6',

    setup_requires=[
        'numpy>=1.15.2',
        'Cython>=0.29.3',
    ],

    # See https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy>=1.15.2',
        'psutil',
        'python-dateutil>=2.7.3',
        'pymonocypher>=0.1.3',
    ] + PLATFORM_INSTALL_REQUIRES,

    extras_require={
        'dev': ['check-manifest', 'coverage', 'Cython', 'wheel', 'sphinx', 'm2r'],
    },   

    entry_points={
        'console_scripts': [
            'joulescope=joulescope.entry_points.runner:run',
        ],
    },
    
    project_urls={
        'Bug Reports': 'https://github.com/jetperch/pyjoulescope/issues',
        'Funding': 'https://www.joulescope.com',
        'Twitter': 'https://twitter.com/joulescope',
        'Source': 'https://github.com/jetperch/pyjoulescope/',
    },
)
