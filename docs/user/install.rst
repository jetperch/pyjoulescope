.. _install:

************
Installation
************

If you want to use released versions of pyjoulescope, **Install from PyPI**.
It's quick and easy!  If you want unrelease pyjoulescope software or you
want to develop pyjoulescope, see **Install from Source**.

.. contents::  :local:


Install Python
==============

You will need to install Python 3.9+ x64 for your platform, if it is not already
on your system.


Windows
-------

Download the latest 
`Python release for Windows <https://www.python.org/downloads/windows/>`_.
If in doubt, select the "Windows x86-64 executable installer".  Install Python
and allow it to update your path.  Open a Command Prompt and type::

    > python -VV
    Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]

Note that Windows usually installs python 3 as python.exe.  This documentation
often uses the executable "python3".


macOS
-----

The easiest way to install Python on macOS is through 
`Homebrew <https://brew.sh/>`_.  Follow the instructions to install brew,
then type::

    $ brew install python3 libusb
    $ python3 -VV


Linux
-----

Use your package manager to install python.  On Debian-based systems, including
Ubuntu, use apt::

    $ sudo apt install python3-dev python3-pip libusb-1.0
    $ python3 -VV
    
The user must have sufficient permissions to access Joulescopes.
Many linux systems, including Debian, Ubuntu, and Raspberry Pi OS, use udev.
We include a 
`udev script <https://github.com/jetperch/pyjoulescope/blob/master/99-joulescope.rules>`_ 
in the repository to give permissions to all users.  To install it::

    $ wget https://raw.githubusercontent.com/jetperch/pyjoulescope/master/99-joulescope.rules
    $ sudo cp 99-joulescope.rules /etc/udev/rules.d/
    $ sudo udevadm control --reload-rules


Install from PyPI
=================

PyPI is the "Python Package Index" which distributes most python packages.
To install this python package from PyPI using `pip` (or `pipenv`)::

    pip3 install -U joulescope

That's it!  The extra "-U" will upgrade to the latest version, just in case.

The joulescope package requires Python 3.6+ and 64-bit Python.
If your system does not meet these requirements, then the installation will
fail with an error message.


Troubleshooting
---------------

If you get an error showing "CERTIFICATE_VERIFY_FAILED", your corporate 
IT policy is likely blocking the pypi SSL certificate.  You have several
options:

A. Contact your IT department to correct the problem.

B. Ignore the SSL validation (slight security risk)::

    pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -U joulescope

For more discussion, see `StackOverflow <https://stackoverflow.com/questions/25981703/pip-install-fails-with-connection-error-ssl-certificate-verify-failed-certi>`.



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

    pip3 install -U -e .[dev]

If you plan on using pyjoulescope, you should install it::

    pip3 install -U .


Using virtualenv
================

Although not required, the developers recommend using 
[virtualenv](https://virtualenv.pypa.io/en/latest/) to minimize any
dependency conflicts with different python applications.

First install virtualenv::

    pip3 install -U virtualenv
    
And then create a new virtual environment::

    virtualenv ~/venv/joulescope

You need to activate the virtual environment whenever you start
a new terminal.
    
On POSIX (Linux, Mac OS X with homebrew)::

    source ~/venv/joulescope/bin/activate
    
On Windows::

    virtualenv c:\venv\joulescope
    source c:\venv\joulescope\Scripts\activate


Developing Joulescope
=====================

If you plan on developing pyjoulescope, you probably want to run directly
from the source code directory.  First, install the dependencies::

    pip3 install -U -r requirements.txt
    
You can then build the native modules and adjust your PYTHONPATH so that
python finds the joulescope package.


For Linux and macOS::

    python setup.py build_ext --inplace
    export PYTHONPATH=`cwd`

For Windows::

    python setup.py build_ext --inplace
    set PYTHONPATH=%cd%
