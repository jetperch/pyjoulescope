.. _quickstart:

Quick Start Guide
=================


The joulescope package also installs a command-line tool, "joulescope".  You
can use the coammand line tool to ensure that your connected Joulescope is
working properly with the package::

    joulescope capture --contiguous 1.0 mycapture.jls
    
To get help for all available commands::
    
    joulescope --help
    
You can also import the Joulescope python package in your own programs.
For example, this script opens the joulescope instrument, reads 1/4 second 
of data, and then display the averaged values::

    import joulescope
    import numpy as np
    with joulescope.scan_require_one(config='auto') as js:
        data = js.read(contiguous_duration=0.25)
    current, voltage = np.mean(data, axis=0)
    print(f'{current} A, {voltage} V')

For more examples, see 
[pyjoulescope_examples](https://github.com/jetperch/pyjoulescope_examples)