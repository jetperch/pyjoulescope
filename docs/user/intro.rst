.. _intro:

Introduction
============

Welcome to the Joulescope™ driver! 
Joulescope is an affordable, precision DC energy 
analyzer that enables you to build better products. 
Joulescope accurately and simultaneously measures the voltage and current 
supplied to your target device, and it then computes power and energy. 
For more information on Joulescope, see 
`joulescope.com <https://www.joulescope.com>`_.

This `joulescope <https://github.com/jetperch/pyjoulescope>`_
python package contains the driver and command-line 
utilities that  run on a host computer and communicates with a Joulescope 
device over USB. You can use this package to automate and script Joulescope 
operation. You can also incorporate Joulescope into a custom application.
Most users will run the graphical user interface which is in the 
`pyjoulescope_ui <https://github.com/jetperch/pyjoulescope_ui>`_ package. 
The majority of code is written in Python 3.6+, but it does contain some C and
Cython for better performance. 

This package runs under Windows 10, Linux (Ubuntu is tested) and Mac OS X.
On Windows, the USB communication is performed using 
`WinUSB <https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/winusb>`_,
which is included with Windows 10.
On Linux and Mac OS X, the USB communication uses 
`libusb-1.0 <https://libusb.info/>`_.

For the list of changes by release, see the :ref:`Changelog <changelog>`.

If you just want to use Joulescope, you can 
`download <https://www.joulescope.com/download>`_ the application.


Naming
------

The GitHub repository is "pyjoulescope" to differentiate it as a Python
package.  The actual Python package name is simply "joulescope".
