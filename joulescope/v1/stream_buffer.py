# Copyright 2018-2022 Jetperch LLC
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

"""
Stream Buffer implementation for v1 backend.
"""

from .sample_buffer import SampleBuffer
from .stats import compute_stats
import numpy as np
import logging


# statistics format for numpy structured data type
# https://docs.scipy.org/doc/numpy/user/basics.rec.html
_STATS_FIELDS = 6  # current, voltage, power, current_range, current_lsb, voltage_lsb
NP_STATS_FORMAT = ['u8', 'f8', 'f8', 'f8', 'f8']
NP_STATS_NAMES = ['length', 'mean', 'variance', 'min', 'max']
STATS_FIELD_NAMES = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb']
STATS_DTYPE = np.dtype({'names': NP_STATS_NAMES, 'formats': NP_STATS_FORMAT})
STATS_FIELD_COUNT = _STATS_FIELDS
FIELDS = {
    'current':       [(1, 0), 'A'],
    'voltage':       [(2, 0), 'V'],
    'power':         [(3, 0), 'W'],
    'current_range': [(4, 0), ''],
    'current_lsb':   [(5, 0), ''],
    'voltage_lsb':   [(5, 1), ''],
}

class StreamBuffer:
    """Efficient real-time Joulescope data buffering.

    :param duration: The total length of the buffering in seconds.
    """

    def __init__(self, duration, frequency, device, output_frequency=None):
        # current, voltage, power, current_range, gpi0, gpi1
        self._input_sampling_frequency = frequency
        if output_frequency is not None:
            self._sampling_frequency = output_frequency
        else:
            self._sampling_frequency = frequency
        self._buffer_duration = duration  #: total buffer duration in seconds
        self._device = device
        self._log = logging.getLogger(__name__)
        self._length = 0
        self.buffers = {}
        self._duration_max = 0
        self._contiguous_duration_max = 0
        self._sample_id_start = None
        self._callback = None
        self._update()

    def _update(self):
        self._length = int(self._buffer_duration * self._sampling_frequency)
        self.buffers.clear()
        self._sample_id_start = None
        self.buffers = {
            # (field_id, index): SampleBuffer
            (1, 0): SampleBuffer(self._length, dtype=np.float32, name='current'),
            (2, 0): SampleBuffer(self._length, dtype=np.float32, name='voltage'),
            (3, 0): SampleBuffer(self._length, dtype=np.float32, name='power'),
            (4, 0): SampleBuffer(self._length, dtype=np.uint8, name='current_range'),  # (converted u4->u8 by pyjoulescope_driver binding)
            (5, 0): SampleBuffer(self._length, dtype='u1', name='gpi0'),
            (5, 1): SampleBuffer(self._length, dtype='u1', name='gpi1'),
        }

    def __len__(self):
        return self._length

    def __str__(self):
        return f'StreamBuffer(length={self._length}, reductions=[])'

    @property
    def voltage_range(self):
        return 0

    @property
    def has_raw(self):
        return False

    @property
    def sample_id_range(self):
        """Get the range of sample ids currently available in the buffer.

        :return: Tuple of sample_id start, sample_id end.
            Start and stop follow normal python indexing:
            start is inclusive, end is exclusive
        """
        r = [b.range for b in self.buffers.values() if b.active]
        if len(r) == 0:
            return None, None
        start = max([b[0] for b in r])
        end = min([b[1] for b in r])
        if end < start:
            end = start
        return start, end

    @property
    def duration_max(self):
        return self._duration_max

    @duration_max.setter
    def duration_max(self, value):
        self._duration_max = value  # stop streaming when reach this sample

    @property
    def is_duration_max(self):
        if self._duration_max is None:
            return False
        e1, e2 = self.sample_id_range
        if e1 is None or self._sample_id_start is None:
            return False
        duration = (e2 - self._sample_id_start) / self._sampling_frequency
        if duration < self._duration_max:
            return False
        return True

    @property
    def contiguous_duration_max(self):
        return self._contiguous_duration_max

    @contiguous_duration_max.setter
    def contiguous_duration_max(self, value):
        self._contiguous_duration_max = value

    @property
    def is_contiguous_duration_max(self):
        if self._contiguous_duration_max is None:
            return False
        # todo: reset on sample drops
        e1, e2 = self.sample_id_range
        if e1 is None or self._sample_id_start is None:
            return False
        duration = (e2 - self._sample_id_start) / self._sampling_frequency
        if duration < self._contiguous_duration_max:
            return False
        return True

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    @property
    def input_sampling_frequency(self):
        return self._input_sampling_frequency

    @property
    def output_sampling_frequency(self):
        return self._sampling_frequency

    @output_sampling_frequency.setter
    def output_sampling_frequency(self, value):
        self._sampling_frequency = value
        self._update()

    @property
    def buffer_duration(self):
        return self._buffer_duration

    @buffer_duration.setter
    def buffer_duration(self, value):
        self._buffer_duration = value
        self._update()

    @property
    def limits_time(self):
        return 0.0, len(self) / self._sampling_frequency

    @property
    def limits_samples(self):
        _, s_max = self.sample_id_range
        return (s_max - len(self)), s_max

    def time_to_sample_id(self, t):
        idx_start, idx_end = self.limits_samples
        t_start, t_end = self.limits_time
        return int(round((float(t) - t_start) / (t_end - t_start) * (idx_end - idx_start) + idx_start))

    def sample_id_to_time(self, s):
        idx_start, idx_end = self.limits_samples
        t_start, t_end = self.limits_time
        return (int(s) - idx_start) / (idx_end - idx_start) * (t_end - t_start) + t_start

    def status(self):
        return {
            'device_sample_id': {'value': 0, 'units': 'samples'},
            # 'sample_id': {'value': self._sample_id_max, 'units': 'samples'},
            'sample_missing_count': {'value': 0, 'units': 'samples'},
            'skip_count': {'value': 0, 'units': ''},
            'sample_sync_count': {'value': 0, 'units': 'samples'},
            'contiguous_count': {'value': 0, 'units': 'samples'},
        }

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        pass

    def reset(self):
        for b in self.buffers.values():
            b.clear()
        self._sample_id_start = None

    def insert(self, topic, value):
        try:
            idx = (value['field_id'], value['index'])
            b = self.buffers[idx]
            sample_rate = value.get('sample_rate')
            decimate_factor = value.get('decimate_factor')
            local_decimate = 1
            if decimate_factor == 1 and self._sampling_frequency != sample_rate:
                local_decimate = sample_rate // self._sampling_frequency
            b.add(value['sample_id'], value['data'],
                  sample_rate=sample_rate,
                  decimate_factor=decimate_factor,
                  local_decimate_factor=local_decimate)
            if self._sample_id_start is None:
                e1, e2 = self.sample_id_range
                if e1 is not None and e1 != e2:
                    self._sample_id_start = e1
            # print(f'{topic} {value["field_id"]}.{value["index"]} {value["sample_id"]} {len(value["data"])} {sample_rate} {decimate_factor} {local_decimate}')
        except KeyError:
            pass  # buffer does not exist, skip
        return 0

    def statistics_get(self, start, stop, out=None):
        """Get exact statistics over the specified range.

        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :param out: The optional output np.ndarray(STATS_FIELD_COUNT, dtype=DTYPE).
            None (default) creates and outputs a new record.
        :return: The tuple of (np.ndarray(STATS_FIELD_COUNT, dtype=DTYPE), [start, stop]).
            The values of start and stop will be constrained to the
            available range.
        """
        self_start, self_stop = self.sample_id_range
        #self._log.debug(f'statistics_get({start}, {stop}) in ({self_start}, {self_stop})')
        start = max(start, self_start)
        stop = min(stop, self_stop)
        if out is None:
            out = np.zeros(_STATS_FIELDS, dtype=STATS_DTYPE)
        if stop >= self_start and start < self_stop:
            out[:]['length'] = stop - start
            for i, b in enumerate(self.buffers.values()):
                if not b.active:
                    out[i]['length'] = 0
                    out[i]['mean'] = np.nan
                    out[i]['variance'] = np.nan
                    out[i]['min'] = np.nan
                    out[i]['max'] = np.nan
                    continue
                d = b.get_range(start, stop)
                compute_stats(d, out[i])
        else:
            out[:]['length'] = 0
            out[:]['mean'] = np.nan
            out[:]['variance'] = np.nan
            out[:]['min'] = np.nan
            out[:]['max'] = np.nan
        return out, (start, stop)

    def data_get(self, start, stop, increment=None, out=None):
        """Get the samples with statistics.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :param out: The optional output np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE) to populate with
            the result.  None (default) will construct and return a new array.
        :return: The np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE) data.

        This method provides fast access to statistics data, primarily for
        graphical display.  Applications should prefer using
        :meth:`samples_get` which provides metadata along with the samples.
        """
        self_start, self_stop = self.sample_id_range
        # self._log.debug(f'data_get({start}, {stop}, {increment}) in ({self_start}, {self_stop})')
        expected_length = (stop - start) // increment
        n_total = (stop - start) // increment
        if out is not None:
            if len(out) < expected_length:
                raise ValueError('out too small')
        else:
            out = np.zeros((expected_length, _STATS_FIELDS), dtype=STATS_DTYPE)

        for n in range(n_total):
            k_start = start + n * increment
            k_end = k_start + increment
            if k_start < self_start or k_end > self_stop:
                out[n, :]['length'] = 0
                out[n, :]['mean'] = np.nan
                out[n, :]['variance'] = np.nan
                out[n, :]['min'] = np.nan
                out[n, :]['max'] = np.nan
                continue
            for i, b in enumerate(self.buffers.values()):
                if not b.active:
                    out[n, i]['length'] = increment
                    out[n, i]['mean'] = np.nan
                    out[n, i]['variance'] = np.nan
                    out[n, i]['min'] = np.nan
                    out[n, i]['max'] = np.nan
                try:
                    d = b.get_range(k_start, k_end)
                    compute_stats(d, out[n, i])
                except KeyError:
                    pass
        return out

    def samples_get(self, start, stop, fields=None):
        """Get exact sample data without any skips or reductions.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param fields: The single field or list of field names to return.
            None (default) is equivalent to
            ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb'].
            The available fields are:

            * raw: The raw u16 data from Joulescope.
              Equivalent to self.raw_get(start, stop)
            * raw_current: The raw 14-bit current data in LSBs.
            * raw_voltage: The raw 14-bit voltage data in LSBs.
            * current: The calibrated float32 current data array in amperes.
            * voltage: The calibrated float32 voltage data array in volts.
            * current_voltage: The calibrated float32 Nx2 array of current, voltage.
            * power: The calibrated float32 power data array in watts.
            * bits: The (voltage_lsb << 5) | (current_lsb << 4) | current_range
            * current_range: The current range. 0 = 10A, 6 = 18 uA, 7=off.
            * current_lsb: The current LSB, which can be assign to a general purpose input.
            * voltage_lsb: The voltage LSB, which can be assign to a general purpose input.

        :return: The dict containing top-level 'time' and 'signals' keys.
            The 'time' value is a dict contain the timing metadata for
            these samples.  The 'signals' value is a dict with one
            key for each field in fields.  Each field value is also
            a dict with entries 'value' and 'units'.
            However, if single field string is provided to fields, then just
            return that field's value.
        """
        if fields is None:
            fields = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb']
        self_start, self_stop = self.sample_id_range
        #self._log.debug(f'samples_get({start}, {stop}) in ({self_start}, {self_stop})')
        start = max(start, self_start)
        stop = min(stop, self_stop)
        t1 = self.sample_id_to_time(start)
        t2 = self.sample_id_to_time(stop)
        result = {
            'time': {
                'range': {'value': [t1, t2], 'units': 's'},
                'delta': {'value': t2 - t1, 'units': 's'},
                'sample_id_range': {'value': [start, stop], 'units': 'samples'},
                'samples': {'value': stop - start, 'units': 'samples'},
                'input_sampling_frequency': {'value': self.input_sampling_frequency, 'units': 'Hz'},
                'output_sampling_frequency': {'value': self.output_sampling_frequency, 'units': 'Hz'},
                'sampling_frequency': {'value': self.output_sampling_frequency, 'units': 'Hz'},
            },
            'signals': {},
        }
        for field in fields:
            try:
                idx, units = FIELDS[field]
                b = self.buffers.get(idx)
                if b.active:
                    out = b.get_range(start, stop)
                    result['signals'][field] = {'value': out, 'units': units}
                else:
                    d = np.empty(stop - start, dtype=np.float32)
                    d[:] = np.nan
                    result['signals'][field] = {'value': out, 'units': units}
            except KeyError:
                pass  # cannot include this signal
        return result
