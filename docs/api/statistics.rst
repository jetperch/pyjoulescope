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
-   **source**: Either **sensor** (on instrument) or **stream_buffer** (on host).

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

