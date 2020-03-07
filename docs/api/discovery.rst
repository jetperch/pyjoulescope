.. _api_discovery:


Device Discovery
================

The device discovery functions allow the application to find Joulescopes
connected to the host.  Most scripts that only support a single Joulescopes
will use::

    joulescope.scan_require_one(config='auto')

To support multiple Joulescopes, use::

    joulescope.scan(config='auto')
    
See :class:`joulescope.driver.Device`.


.. autofunction:: joulescope.driver.scan
.. autofunction:: joulescope.driver.scan_require_one
.. autofunction:: joulescope.driver.scan_for_changes
