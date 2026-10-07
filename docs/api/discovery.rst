.. _api_discovery:


Device Discovery
================

The device discovery functions allow the application to find Joulescopes
connected to the host.  Most scripts that only support a single Joulescopes
will use::

    joulescope.scan_require_one(config='auto')

To support multiple Joulescopes, use::

    joulescope.scan(config='auto')

To select devices, provide device specifications, the same as
pyjoulescope_driver ``Driver.device_paths()``::

    joulescope.scan_require_one('js320', config='auto')  # the only JS320
    joulescope.scan_require_one('8W2A')                  # by serial number
    joulescope.scan('u/js220/000415, js320')             # comma-separated
    joulescope.scan(['js220', 'js320'])                  # list

A specification is a full device path, such as "u/js320/8W2A", the
backend and model "u/js320", the model and serial number "js320/8W2A" or
"js320-8W2A", the model "js320", or the serial number "8W2A".  Matching
is case-insensitive.  The joulescope package only returns Joulescope
instruments, so other devices that the driver reports, such as
MiniBitty devices, never match.

Scans exclude devices in bootloader mode, unless you request them with
"bootloader" or a "&" model specification, such as "&js220".  For
backwards compatibility, "Joulescope" selects all devices.

:func:`scan_require_one` raises :class:`joulescope.v1.driver.ScanError`
when zero or multiple devices match.  Its message lists the matching and
available devices.  ScanError is both a ValueError and, for backwards
compatibility, a RuntimeError.

See :class:`joulescope.v1.device.Device`.


scan
----

.. autofunction:: joulescope.scan


scan_require_one
----------------

.. autofunction:: joulescope.scan_require_one


scan_for_changes
----------------

.. autofunction:: joulescope.scan_for_changes


ScanError
---------

.. autoclass:: joulescope.v1.driver.ScanError
