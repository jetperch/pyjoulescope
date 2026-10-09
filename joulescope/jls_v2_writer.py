# Copyright 2020-2021 Jetperch LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from joulescope.units import str_to_number
from pyjls import Writer, SourceDef, SignalDef, SignalType, DataType
import numpy as np


"""This modules writes Joulescope streaming data to JLS v2 files.

The JLS v2 file format is very flexible with multiple options.  This module
provides a simple interface to connect Joulescope streaming data to
the JLS v2 file writer.
"""


SIGNALS = {
    'current': (1, 'A'),
    'voltage': (2, 'V'),
    'power': (3, 'W'),
}

UTC_INTERVAL = 60 * 2 ** 30  #: The UTC entry interval, in joulescope.time units.


def signals_validator(s):
    """Validate the JLS writer signal names.

    :param s: The signal names as a comma-separated string or a list,
        such as "current,voltage".  See :data:`SIGNALS`.
    :return: The list of lowercase signal names.
    :raise ValueError: On an unsupported signal name.
    """
    result = []
    if isinstance(s, str):
        p = s.split(',')
    else:
        p = s
    for signal_name in p:
        k = signal_name.strip().lower()
        if k not in SIGNALS:
            raise ValueError(f'unsupported signal {signal_name}')
        result.append(k)
    return result


def sampling_rate_validator(s):
    """Validate a sampling rate.

    :param s: The sampling rate as a number, or as a string such as
        "1 MHz", "500 kHz", or "2000000".
    :return: The integer sampling rate in Hz.
    :raise ValueError: If s is not a valid sampling rate.
    """
    return int(str_to_number(s))


# Deprecated private names, for backwards compatibility.
_signals_validator = signals_validator
_sampling_rate_validator = sampling_rate_validator


class JlsWriter:

    def __init__(self, device, filename, signals=None, info=None, sampling_frequency=None):
        """Create a new JLS file writer instance.

        :param device: The Joulescope device instance.
        :param filename: The output ".jls" filename.
        :param signals: The signals to record as either a list of string names
            or a comma-separated string.  The supported signals include
            ['current', 'voltage', 'power']
        :param info: The device information from :meth:`Device.info`.
            None (default) reads it from the device in :meth:`open`.
        :param sampling_frequency: The sampling frequency in Hz.
            None (default) reads it from the device in :meth:`open`.

        This class implements joulescope.driver.StreamProcessApi and may also
        be used as a context manager.

        The constructor does not access the device.  :meth:`open` reads
        the device information that was not provided, so call it from your
        thread, not from a stream or statistics callback.  To open from a
        callback, such as to start recording on a trigger, first read
        the information on your thread and pass it to the constructor::

            info = device.info()
            fs = device.parameter_get('sampling_frequency')
            wr = JlsWriter(device, 'trigger.jls', info=info, sampling_frequency=fs)
            # wr.open() and wr.close() are now safe from a callback
        """
        self._device = device
        self._filename = filename
        if signals is None:
            signals = ['current', 'voltage']
        signals = signals_validator(signals)
        self._signals = signals
        self._info = info
        self._sampling_frequency = None
        if sampling_frequency is not None:
            self._sampling_frequency = sampling_rate_validator(sampling_frequency)
        self._wr = None
        self._idx = 0
        self._utc_next = None
        self._utc_last = None
        self._utc_pending = None

    def _device_read(self):
        if self._info is None:
            self._info = self._device.info()
        if self._sampling_frequency is None:
            sampling_frequency = self._device.parameter_get('sampling_frequency')
            self._sampling_frequency = sampling_rate_validator(sampling_frequency)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open and configure the JLS writer file.

        This method reads the device information that the constructor
        did not receive, so see the constructor for which thread may
        call it.
        """
        self.close()
        self._device_read()
        info = self._info
        sampling_rate = self._sampling_frequency

        source = SourceDef(
            source_id=1,
            name=str(self._device),
            vendor='Jetperch',
            model=info['model'],
            version=info['hardware_version'],
            serial_number=info['serial_number'],
        )

        wr = Writer(self._filename)
        try:
            wr.source_def_from_struct(source)
            for s in self._signals:
                idx, units = SIGNALS[s]
                s_def = SignalDef(
                    signal_id=idx,
                    source_id=1,
                    signal_type=SignalType.FSR,
                    data_type=DataType.F32,
                    sample_rate=sampling_rate,
                    name=s,
                    units=units,
                )
                wr.signal_def_from_struct(s_def)

        except Exception:
            wr.close()
            raise

        self._wr = wr
        self._utc_next = None
        self._utc_last = None
        self._utc_pending = None
        return wr

    def close(self):
        """Finalize and close the JLS file."""
        wr, self._wr = self._wr, None
        if wr is not None:
            try:
                if self._utc_pending is not None:
                    self._utc_write(wr, *self._utc_pending)
            finally:
                self._utc_pending = None
                wr.close()

    def _utc_write(self, wr, sample_id, utc):
        self._utc_last = sample_id
        for s in self._signals:
            wr.utc(SIGNALS[s][0], sample_id, utc)

    def _utc_update(self, anchor):
        # Same cadence as pyjoulescope_driver Record: the first anchor,
        # one per UTC_INTERVAL, and the latest anchor at close.
        if anchor is None:
            return
        sample_id, utc = anchor
        if self._utc_last is not None and sample_id <= self._utc_last:
            return  # entries must advance
        if self._utc_next is None:
            self._utc_write(self._wr, sample_id, utc)
            self._utc_next = utc + UTC_INTERVAL
        elif utc >= self._utc_next:
            self._utc_write(self._wr, sample_id, utc)
            self._utc_next += UTC_INTERVAL
            self._utc_pending = None
        else:
            self._utc_pending = anchor

    def stream_notify(self, stream_buffer):
        """Handle incoming stream data.

        :param stream_buffer: The :class:`StreamBuffer` instance which contains
            the new data from the Joulescope.
        :return: False to continue streaming.
        """
        # called from USB thead, keep fast!
        # long-running operations will cause sample drops
        start_id, end_id = stream_buffer.sample_id_range
        start_id = max(self._idx, start_id)
        if start_id < end_id:
            # the v0 StreamBuffer has no UTC information
            self._utc_update(getattr(stream_buffer, 'utc_anchor', None))
            data = stream_buffer.samples_get(start_id, end_id, fields=self._signals)
            for s in self._signals:
                x = np.ascontiguousarray(data['signals'][s]['value'])
                idx = SIGNALS[s][0]
                self._wr.fsr_f32(idx, start_id, x)
            self._idx = end_id
        return False

    def fsr_f32(self, signal_name, sample_id, x, utc=None):
        """Write signal data directly.

        :param signal_name: The signal name string.
        :param sample_id: The starting sample id for x.
        :param x: The 1-d ndarray of float sample data
        :param utc: The optional :mod:`joulescope.time` UTC timestamp
            for sample_id.  Provide it for the first write, and then
            periodically, such as every minute, so that readers can
            display the sample time.
        """
        idx = SIGNALS[signal_name][0]
        sample_id = int(sample_id)
        if utc is not None:
            self._wr.utc(idx, sample_id, int(utc))
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self._wr.fsr_f32(idx, sample_id, np.ascontiguousarray(x))
