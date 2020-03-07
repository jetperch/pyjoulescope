
# ![Joulescope](https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png "Joulescope Logo")

[![Build Status](https://travis-ci.org/jetperch/pyjoulescope.svg?branch=master)](https://travis-ci.org/jetperch/pyjoulescope)
[![Docs Status](https://readthedocs.org/projects/joulescope/badge/?version=latest)](https://joulescope.readthedocs.io/)

Welcome to the Joulescope™!  Joulescope is an affordable, precision DC energy 
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

If you just want to use your Joulescope, you can 
[download](https://www.joulescope.com/download) the application.


## Documentation

For more information, see the 
`documentation <https://joulescope.readthedocs.io/en/latest/>`_.


## License

All pyjoulescope code is released under the permissive Apache 2.0 license.
See the [License File](LICENSE.txt) for details.
