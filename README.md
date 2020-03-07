
# ![Joulescope](https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png "Joulescope Logo")

[![Build Status](https://travis-ci.org/jetperch/pyjoulescope.svg?branch=master)](https://travis-ci.org/jetperch/pyjoulescope)
[![Docs Status](https://readthedocs.org/projects/joulescope/badge/?version=latest)](https://joulescope.readthedocs.io/)

Welcome to Joulescope™!  Joulescope is an affordable, precision DC energy 
analyzer that enables you to build better products. 
Joulescope accurately and simultaneously measures the voltage and current 
supplied to your target device, and it then computes power and energy. 
For more information on Joulescope, see 
[www.joulescope.com](https://www.joulescope.com).

This pyjoulescope python package contains the driver and command-line 
utilities that  run on a host computer and communicates with a Joulescope 
device over USB. You can use this package to automate and script Joulescope 
operation. You can also incorporate Joulescope into a custom application.
Most users will run the graphical user interface which is in the 
[pyjoulescope_ui](https://github.com/jetperch/pyjoulescope_ui) package. 
The majority of code is written in Python 3.6+, but a small amount is in 
Cython for better performance. 

This package runs under Windows 10, Linux (Ubuntu is tested) and Mac OS X.
On Windows, the USB communication is performed using 
[WinUSB](https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/winusb),
which is included with Windows 10.
On Linux and Mac OS X, the USB communication uses 
[libusb-1.0](https://libusb.info/).

For the list of changes by release, see the [Changelog](CHANGELOG.md).

If you just want to use Joulescope, you can 
[download](https://www.joulescope.com/download) the application.


## Quick start

Install [Python](https://www.python.org/) 3.6+ 64-bit.  If you already have
Python installed, verify 3.6+ and 64-bit:

    python3 -VV

Install this python package from pypi:

    pip3 install joulescope

The Joulescope python package includes command line tools:

    python3 -m joulescope --help
    
For example, to capture 1 second of contiguous data:

    python3 -m joulescope capture --contiguous 1.0 mycapture.jls
    
You can also import the Joulescope python package in your own programs.
For example, this script opens the joulescope instrument, reads 1/4 second 
of data, and then display the averaged values:

    import joulescope
    import numpy as np

    js = joulescope.scan_require_one()
    js.open()
    try:
        js.parameter_set('source', 'on')
        js.parameter_set('i_range', 'auto')
        data = js.read(contiguous_duration=0.25)
    finally:
        js.close()

    current, voltage = np.mean(data, axis=0)
    print(f'{current} A, {voltage} V')

For more examples, see 
[pyjoulescope_examples](https://github.com/jetperch/pyjoulescope_examples)

## Developer

Install [Python](https://www.python.org/) 3.6+ 64-bit. 


### Configure virtualenv

Although not required, the developers recommend using 
[virtualenv](https://virtualenv.pypa.io/en/latest/).

First install virtualenv:

    pip3 install virtualenv
    
And then create a new virtual environment.

    virtualenv ~/venv/joulescope

You need to activate the virtual environment whenever you start
a new terminal.
    
On POSIX (Linux, Mac OS X with homebrew):

    source ~/venv/joulescope/bin/activate
    
On Windows:

    virtualenv c:\venv\joulescope
    source c:\venv\joulescope\Scripts\activate

### Configure packages
    
Install development dependencies:

    pip3 install -r requirements.txt


### Use Joulescope
    
Joulescope includes PYX files that must be compiled to native libraries using
Cython. You can use the setup script to allow development in place:

    python3 setup.py build_ext --inplace
    
You should then be able to execute joulescope:

    python3 -m joulescope --help

If you want to switch directories, you may need to set your 
[PYTHONPATH](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)
environment variable.

If you would rather build and install Joulescope:

    python setup.py sdist
    pip3 install dist/joulescope_[version].tar.gz


## License

All pyjoulescope code is released under the permissive Apache 2.0 license.
See the [License File](LICENSE.txt) for details.
