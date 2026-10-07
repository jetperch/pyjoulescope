.. _api_streaming:


Stream Process API
==================

.. autoclass:: joulescope.v0.driver.StreamProcessApi
    :members:


Threads
-------

The stream process methods, such as ``stream_notify``, the ``stop_fn``
given to :meth:`Device.start <joulescope.v1.device.Device.start>`, and
statistics callbacks all run on the USB thread.  Keep them fast, since
slow callbacks drop samples, and do not call back into the device from
them.  Pass results to your thread with a ``queue.Queue`` or a
``threading.Event``.  For statistics, use
:meth:`Device.statistics_iter <joulescope.v1.device.Device.statistics_iter>`
instead of a callback.  See :ref:`api_statistics`.


Run length
----------

There are three ways to decide how long a capture runs:

1.  **Sample count** (recommended): ``device.start(stop_fn, duration)``
    stops after ``duration`` seconds of samples, at the sample rate.
    The device stops itself, even if your thread is busy, and calls
    ``stop_fn(event, message)`` on the USB thread.  ``stop_fn`` is also
    called when the device stops for any other reason, such as removal.
    Use ``contiguous_duration`` instead to require that the duration has
    no missing samples.
2.  **Blocking read**: ``device.read(duration)`` does the same as (1)
    and returns the samples, when the samples fit in the stream buffer.
3.  **Wall-clock loop**: ``device.start()``, then loop until
    ``time.time()`` reaches the end time, then ``device.stop()``.  The
    capture length then includes the USB startup delay and your thread's
    scheduling, so the sample count varies from run to run.  The loop also
    does not notice when the device stops on an error, unless you also
    give ``stop_fn``.

Use (1) or (2) for a fixed duration.  Only use a loop for open-ended
captures that end on Ctrl-C or on an external condition, and still give
``stop_fn``.  Here is the recommended pattern::

    import joulescope
    import threading

    with joulescope.scan_require_one(config='auto') as device:
        done = threading.Event()

        def stop_fn(event, message):  # runs on the USB thread
            done.set()

        device.start(stop_fn=stop_fn, duration=10.0)
        while not done.wait(0.1):  # a timeout keeps Ctrl-C responsive
            pass

For statistics, the equivalent of (1) is a count:
``device.statistics_iter(count=duration * reduction_frequency)``.
