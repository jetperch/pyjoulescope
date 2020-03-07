.. _install:

*******
Install
*******

If you want to use released versions of pyjoulescope, **Install from PyPI**.
It's quick and easy!  If you want unrelease pyjoulescope software or you
want to develop pyjoulescope, see **Install from Source**.


Install Python
==============

You will need to install Python 3.6+ x64 for your platform, if it is not already
on your system.


Windows
-------

Download the latest 
`Python release for Windows <https://www.python.org/downloads/windows/>`_.
If in doubt, select the "Windows x86-64 executable installer".  Install Python
and allow it to update your path.  Open a Command Prompt and type::

    > python -VV
    Python 3.7.6 (tags/v3.7.6:43364a7ae0, Dec 19 2019, 00:42:30) [MSC v.1916 64 bit (AMD64)]

Note that Windows usually installs python 3 as python.exe.  This documentation
often uses the executable "python3".


macOS
-----

The easiest way to install Python on macOS is through 
`Homebrew <https://brew.sh/>`_.  Follow the instructions to install brew,
then type::

    $ brew install python3
    $ python3 -VV


Linux
-----

Use your package manager to install python.  On Debian-based systems, including
Ubuntu, use apt:

    $ sudo apt install python3-dev python3-pip
    $ python3 -VV


Install from PyPI
=================

PyPI is the "Python Package Index" which distributes most python packages.
To install this python package from PyPI using `pip` (or `pipenv`)::

    pip3 install -U joulescope

That's it!  The extra "-U" will upgrade to the latest version, just in case.

The joulescope package requires Python 3.6+ and 64-bit Python.
If your system does not meet these requirements, then the installation will
fail with an error message.


Install from Source
===================

The pyjoulescope package repository is open-source on GitHub.  You can clone 
the repository::

    git clone https://github.com/jetperch/pyjoulescope.git
    
You can also download the tarbal (Linux, macOS)::

    curl -Ol https://github.com/jetperch/pyjoulescope/tarball/master
    
or the ZIP file (Windows)::

    https://github.com/jetperch/pyjoulescope/archive/master.zip
    
Change directory into the extracted file.  Install the developer dependencies::

    pip install -U -e .[dev]

If you plan on using pyjoulescope, you should install it::

    pip install -U .
    
If you plan on developing pyjoulescope, you can adjust your PYTHONPATH to
use the source directly.  For Linux and macOS::

    python setup.py build_ext --inplace
    export PYTHONPATH=`cwd`

For Windows::

    python setup.py build_ext --inplace
    set PYTHONPATH=%cd%
