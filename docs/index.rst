.. Joulescope Driver documentation master file, created by
   sphinx-quickstart on Fri Mar  6 16:24:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png
    :target: https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png

The most affordable and easy-to-use precision DC energy analyzer.

Version \ |version| (:ref:`Installation <install>`)

.. image:: https://pepy.tech/badge/joulescope
    :target: https://pypi.org/project/joulescope/
    
.. image:: https://img.shields.io/pypi/l/joulescope.svg
    :target: https://pypi.org/project/joulescope/

.. image:: https://img.shields.io/pypi/wheel/joulescope.svg
    :target: https://pypi.org/project/joulescope/

.. image:: https://img.shields.io/pypi/pyversions/joulescope.svg
    :target: https://pypi.org/project/joulescope/


-------------------

Welcome to the Joulescope Driver's documentation.  This driver allows you to
include Joulescope functionality into your own Python programs.  
With the Joulescope driver, it is easy to automate your energy measurements::

    import joulescope
    import numpy as np
    with joulescope.scan_require_one(config='auto') as js:
        data = js.read(contiguous_duration=0.1)
    current, voltage = np.mean(data, axis=0)
    print(f'{current} A, {voltage} V')

Visit `joulescope.com <https://joulescope.com>`_ to purchase a Joulescope.
You can also `download <https://joulescope.com/download>`_ the Joulescope UI,
which is a Windows, macOS and Linux application.


.. toctree::
    :maxdepth: 2

    user/index
    api/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
