
# ![Joulescope](https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png "Joulescope Logo")

[![Build Status](https://travis-ci.org/jetperch/pyjoulescope.svg?branch=master)](https://travis-ci.org/jetperch/pyjoulescope)
[![Docs Status](https://readthedocs.org/projects/joulescope/badge/?version=latest)](https://joulescope.readthedocs.io/)

Welcome to the Joulescope™!  Joulescope is an affordable, precision DC energy 
analyzer that enables you to build better products. 
Joulescope accurately and simultaneously measures the voltage and current 
supplied to your target device, and it then computes power and energy. 

This pyjoulescope python package enables you to
automate Joulescope operation and easily measure current, voltage, power and
energy within your own Python programs on Windows,
Linux, and macOS.
With the Joulescope driver, controlling your Joulescope is easy.  This 
example captures 0.1 seconds of data and then prints the average current
and voltage:

    import joulescope
    import numpy as np
    with joulescope.scan_require_one(config='auto') as js:
        data = js.read(contiguous_duration=0.1)
    current, voltage = np.mean(data, axis=0)
    print(f'{current} A, {voltage} V')

This package also installs the "joulescope" command line tool:

    joulescope --help

Most users will run the graphical user interface which is in the 
[pyjoulescope_ui](https://github.com/jetperch/pyjoulescope_ui) package. 
The majority of code is written in Python 3.6+, but a small amount is in 
Cython for better performance. 

For the list of changes by release, see the [Changelog](CHANGELOG.md).

If you just want to use your Joulescope, you can 
[download](https://www.joulescope.com/download) the application.


## Documentation

For more information, see the 
[documentation](https://joulescope.readthedocs.io/en/latest/).


## License

All pyjoulescope code is released under the permissive Apache 2.0 license.
See the [License File](LICENSE.txt) for details.
