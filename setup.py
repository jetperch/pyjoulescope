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
import os
import numpy as np

MYPATH = os.path.abspath(os.path.dirname(__file__))

try:
    from Cython.Build import cythonize
    USE_CYTHON = os.path.isfile(os.path.join(MYPATH, 'joulescope', 'stream_buffer.pyx'))
except ImportError:
    USE_CYTHON = False


ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    setuptools.Extension('joulescope.stream_buffer',
        sources=['joulescope/stream_buffer' + ext],
        include_dirs=[np.get_include()],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)  # , annotate=True)


# Get the long description from the README file
with open(os.path.join(MYPATH, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='joulescope',
    version='0.1.5',
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
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Supported Python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='joulescope driver',

    packages=setuptools.find_packages(exclude=['native', 'docs', 'test', 'dist', 'build']),
    ext_modules=extensions,
    include_dirs=[np.get_include()],

    # See https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy>=1.15.2',
        'pypiwin32>=223',
        'python-dateutil>=2.7.3',
        'pymonocypher>=0.1.0',
    ],

    extras_require={
        'dev': ['check-manifest', 'Cython', 'coverage'],
    },   

    entry_points={
        'console_scripts': [
            'joulescope_cmd=joulescope.command.runner:run',
        ],
    },
    
    project_urls={
        'Bug Reports': 'https://github.com/jetperch/pyjoulescope/issues',
        'Funding': 'https://www.joulescope.com',
        'Twitter': 'https://twitter.com/joulescope',
        'Source': 'https://github.com/jetperch/pyjoulescope/',
    },
)
