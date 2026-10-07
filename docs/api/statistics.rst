.. _api_statistics:


Statistics API
==============

The statistics API consists of the statistics data structure,
which is provided periodically to registered callbacks.
The data structure contains the following top-level keys:

-   **time**: The time information which includes:

    -   **range**: The (start, stop) time range for this data structure.
    -   **delta**: The total duration which is equal to (stop - start).
    -   **samples**: The total number of samples combined into this data.
    
-   **signals**: The signal values over the previous statistics time window.
    The keys include **current**, **voltage**, and **power**.  Each key 
    contains a map with keys:
    
    - **µ**: The mean (average) value.
    - **σ2**: The variance value.
    - **min**: The minimum value.
    - **max**: The maximum value.
    - **p2p**: The peak-to-peak value = (**max** - **min**)
    - **∫**: The integrated value, only for **current** and **power**.
    
-   **accumulators**: The integrated charge and energy values.
-   **source**: Where the statistics were computed.  The v1 backend
    provides **sensor** (on the instrument) or **host** (by the host
    driver from the full-rate sample stream).  The legacy v0 backend
    provides **sensor** or **stream_buffer**.

Statistics source
-----------------

The model and the scan config select where the statistics are computed.
:attr:`Device.statistics_source <joulescope.v1.device.Device.statistics_source>`
reports the source for a device.

=========  ====================  ==========================================
Model      scan config           Source
=========  ====================  ==========================================
JS110      'auto', 'ignore',     host: computed by the host driver from the
           None                  2 Msps sample stream.
JS110      'off'                 sensor: computed on the instrument.
JS220      any                   sensor: computed on the instrument.
JS320      any                   sensor: computed on the instrument.
=========  ====================  ==========================================

The ``source`` argument of ``statistics_callback_register`` and
``statistics_callback_unregister`` is deprecated and ignored.  It issues
a DeprecationWarning on first use.


Statistics on your thread
-------------------------

Statistics callbacks run on the USB thread, so they must return quickly
and must not call back into the device.  Most scripts instead want the
statistics on their own thread.  Use
:meth:`Device.statistics_get <joulescope.v1.device.Device.statistics_get>`
or :meth:`Device.statistics_iter <joulescope.v1.device.Device.statistics_iter>`,
which block until the next value arrives::

    import joulescope

    with joulescope.scan_require_one(config='auto') as device:
        device.parameter_set('reduction_frequency', '2 Hz')
        for stats in device.statistics_iter(count=20):  # 10 seconds
            i = stats['signals']['current']['µ']['value']
            print(f'{i:.9f} A')

The first call starts buffering, so consecutive calls return consecutive
values without gaps.  Both wait 2 seconds for each value by default and
raise TimeoutError if none arrives.  ``statistics_iter`` without a count
continues until the device closes.


Example
-------

Here is an example statistics data structure::

    {
      "time": {
        "range": {"value": [29.975386, 29.999424], "units": "s"},
        "delta": {"value": 0.024038, "units": "s"},
        "samples": {"value": 48076, "units": "samples"}
      },
      "signals": {
        "current": {
          "µ": {"value": 0.000299379503657111, "units": "A"},
          "σ2": {"value": 2.2021878912979553e-12, "units": "A"},
          "min": {"value": 0.00029360855114646256, "units": "A"},
          "max": {"value": 0.0003051375679206103, "units": "A"},
          "p2p": {"value": 1.1529016774147749e-05, "units": "A"},
          "∫": {"value": 0.008981212667119223, "units": "C"}
        },
        "voltage": {
          "µ": {"value": 2.99890387873055,"units": "V"},
          "σ2": {"value": 1.0830626821348923e-06, "units": "V"},
          "min": {"value": 2.993824005126953, "units": "V"},
          "max": {"value": 3.002903699874878, "units": "V"},
          "p2p": {"value": 0.009079694747924805, "units": "V"}
        },
        "power": {
          "µ": {"value": 0.000897810357252683, "units": "W"},
          "σ2": {"value": 1.9910494110256852e-11, "units": "W"},
          "min": {"value": 0.0008803452947176993, "units": "W"},
          "max": {"value": 0.0009152597631327808, "units": "W"},
          "p2p": {"value": 3.49144684150815e-05, "units": "W"},
          "∫": {"value": 0.026933793578814716, "units": "J"}
        },
        "current_range": {
          "µ": {"value": 4.0, "units": ""},
          "σ2": {"value": 0.0, "units": ""},
          "min": {"value": 4.0, "units": ""},
          "max": {"value": 4.0, "units": ""},
          "p2p": {"value": 0.0, "units": ""}
        },
        "current_lsb": {
          "µ": {"value": 0.5333222397870035, "units": ""},
          "σ2": {"value": 0.24889270730539995, "units": ""},
          "min": {"value": 0.0, "units": ""},
          "max": {"value": 1.0, "units": ""},
          "p2p": {"value": 1.0, "units": ""}
        },
        "voltage_lsb": {
          "µ": {"value": 0.5333430401863711, "units": ""},
          "σ2": {"value": 0.24889309698100895, "units": ""},
          "min": {"value": 0.0, "units": ""},
          "max": {"value": 1.0, "units": ""},
          "p2p": {"value": 1.0, "units": ""}
        }
      },
      "accumulators": {
        "charge": {"value": 0.0, "units": "C"},
        "energy": {"value": 0.0, "units": "J"}
      },
      "source": "sensor"
    }      

