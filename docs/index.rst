.. Joulescope Driver documentation master file, created by
   sphinx-quickstart on Fri Mar  6 16:24:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png
    :target: https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png

The most affordable and easy-to-use precision DC energy analyzer.

Version \ |version| (:ref:`install`)

.. image:: https://pepy.tech/badge/joulescope
    :target: https://pypi.org/project/joulescope/
    
.. image:: https://img.shields.io/pypi/l/joulescope.svg
    :target: https://pypi.org/project/joulescope/

.. image:: https://img.shields.io/pypi/wheel/joulescope.svg
    :target: https://pypi.org/project/joulescope/

.. image:: https://img.shields.io/pypi/pyversions/joulescope.svg
    :target: https://pypi.org/project/joulescope/


-------------------

Welcome to the Joulescope™ driver's documentation.  This driver enables you to
automate Joulescope operation and easily measure current, voltage, power and
energy within your own Python programs on Windows,
Linux, and macOS.
With the Joulescope driver, controlling your Joulescope is easy.  This 
example captures 0.1 seconds of data and then prints the average current
and voltage::

    import joulescope
    import numpy as np
    with joulescope.scan_require_one(config='auto') as js:
        data = js.read(contiguous_duration=0.1)
    current, voltage = np.mean(data, axis=0, dtype=np.float64)
    print(f'{current} A, {voltage} V')

Visit `joulescope.com <https://joulescope.com>`_ to purchase a Joulescope.
You can also `download <https://joulescope.com/download>`_ the Joulescope UI.

In addition to this documentation, you can:

* Visit the `Joulescope support page <https://www.joulescope.com/pages/support>`_
* Read the `Joulescope User's Guide <http://download.joulescope.com/docs/JoulescopeUsersGuide/index.html>`_
* Visit the `Joulescope forum <https://forum.joulescope.com/>`_
* Submit issues on `GitHub <https://github.com/jetperch/pyjoulescope/issues>`_


Table of Contents
=================

.. toctree::
    :maxdepth: 2

    user/index
    api/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
