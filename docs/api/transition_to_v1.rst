.. _api_transition_v1:


Transition to v1
================

The launch of the JS220 in 2022 introduced a new API based upon
`publish-subscribe <https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern>`_.
You can find the source code at 
`joulescope_driver <https://github.com/jetperch/joulescope_driver>`_.  We
encourage all new code to use this API directly.

However, this project provides a backwards-compatible v0 convenience wrapper.
Access the wrapper through any of the scan functions:

* **scan**
* **scan_require_one**
* **scan_for_changes**

These scan functions return **Driver** wrapper instances compatible with the
v0 API.  The **Driver** wrapper removes the following methods found in the 
v0 API:

* **Driver.usb_device**
* **Driver.bootloader**
* **Driver.run_from_bootloader**
* **Driver.calibration_program**
* **Driver.enter_test_mode**

The **Driver.serial_number** now returns **Driver.device_serial_number**.

Parameter support has also changed.
The following parameters have modified behavior:

.. csv-table:: 
    :file: transition_to_v1.csv
    :widths: 50 25 25
    :header-rows: 1

New JS220 topics are available as parameters.  However, the parameter metadata
is retrieved dynamically and is not available until the device is opened.
