# Copyright 2020 Jetperch LLC
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


from .decimators import DECIMATORS
from .filter_fir cimport FilterFir, filter_fir_cbk
STREAM_BUFFER_REDUCTIONS = [200, 100, 50]  # in samples in sample units of the previous reduction
STREAM_BUFFER_DURATION = 1.0  # seconds


DS_NP_VALUE_FORMAT = ['f4', 'f4', 'f4', 'u1', 'u1', 'u1', 'u1']
DS_VALUE_DTYPE = np.dtype({'names': STATS_FIELD_NAMES + ['rsv1'], 'formats': DS_NP_VALUE_FORMAT})
cdef uint32_t _DS_NP_LENGTH_C = _STATS_FIELDS + 1


cdef struct ds_value_s:
    float current
    float voltage
    float power
    uint8_t current_range
    uint8_t current_lsb
    uint8_t voltage_lsb
    uint8_t rsv1


cdef class DownsamplingStreamBuffer:

    cdef StreamBuffer _stream_buffer
    cdef double _input_sampling_frequency
    cdef double _output_sampling_frequency
    cdef uint64_t _length # in samples
    cdef uint64_t _processed_sample_id
    cdef uint64_t _process_idx
    cdef FilterFir _filter_fir
    cdef object _input_npy
    cdef double * _input_dbl
    cdef object _buffer_npy
    cdef ds_value_s * _buffer_ptr
    cdef uint64_t _accum[3]
    cdef uint64_t _downsample_m

    def __init__(self, duration, reductions, input_sampling_frequency, output_sampling_frequency):
        assert(sizeof(ds_value_s) == 16)
        if input_sampling_frequency is not None and input_sampling_frequency != 2000000:
            raise ValueError(f'Require 2000000 sps, provided {input_sampling_frequency}')
        if int(output_sampling_frequency) not in DECIMATORS:
            raise ValueError(f'Unsupported output frequency: {output_sampling_frequency}')
        self._input_sampling_frequency = input_sampling_frequency
        self._output_sampling_frequency = output_sampling_frequency
        self._downsample_m = input_sampling_frequency / self._output_sampling_frequency
        if input_sampling_frequency:
            self._stream_buffer = StreamBuffer(STREAM_BUFFER_DURATION, STREAM_BUFFER_REDUCTIONS, input_sampling_frequency)
            self._stream_buffer.process_stats_callback_set(<stream_buffer_process_fn> self._stream_buffer_process_cbk, <void *> self)
        reduction_step = int(np.prod(reductions))
        length = int(duration * output_sampling_frequency)
        length = ((length + reduction_step - 1) // reduction_step) * reduction_step
        self._length = length

        self._buffer_npy = np.full(length, 0, dtype=DS_VALUE_DTYPE)
        cdef ds_value_s [::1] _buffer_view = self._buffer_npy
        self._buffer_ptr = &_buffer_view[0]

        self._input_npy = np.zeros(_DS_NP_LENGTH_C, dtype=np.float64)
        cdef double [::1] _input_view = self._input_npy
        self._input_dbl = &_input_view[0]

        if input_sampling_frequency:
            decimator = DECIMATORS[self._output_sampling_frequency]
            self._filter_fir = FilterFir(decimator, width=3)
            self._filter_fir.c_callback_set(<filter_fir_cbk> self._filter_fir_cbk, <void *> self)
        self.reset()

    def __len__(self):
        return int(self._length)

    def __str__(self):
        return 'DownsamplingStreamBuffer(length=%d)' % (self._length)

    @property
    def has_raw(self):
        return False

    @property
    def sample_id_range(self):
        s_end = int(self._processed_sample_id)
        s_start = s_end - int(self._length)
        if s_start < 0:
            s_start = 0
        return s_start, s_end

    @property
    def sample_id_max(self):
        # in units of input samples
        return self._stream_buffer.sample_id_max

    @sample_id_max.setter
    def sample_id_max(self, value):
        # in units of input samples
        self._stream_buffer.sample_id_max = value  # stop streaming when reach this sample

    @property
    def contiguous_max(self):
        # in units of input samples
        return self._stream_buffer.contiguous_max

    @contiguous_max.setter
    def contiguous_max(self, value):
        # in units of input samples
        self._stream_buffer.contiguous_max = value

    @property
    def callback(self):
        return self._stream_buffer.callback

    @callback.setter
    def callback(self, value):
        self._stream_buffer.callback = value

    @property
    def voltage_range(self):
        return self._stream_buffer.voltage_range

    @voltage_range.setter
    def voltage_range(self, value):
        self._stream_buffer.voltage_range = value

    @property
    def suppress_mode(self):
        return self._stream_buffer.suppress_mode

    @suppress_mode.setter
    def suppress_mode(self, value):
        self._stream_buffer.suppress_mode = value

    @property
    def input_sampling_frequency(self):
        return self._input_sampling_frequency

    @property
    def output_sampling_frequency(self):
        return self._output_sampling_frequency

    @property
    def limits_time(self):
        return 0.0, len(self) / self._output_sampling_frequency

    @property
    def limits_samples(self):
        _, s_max = self.sample_id_range
        return (s_max - len(self)), s_max

    def time_to_sample_id(self, t):
        idx_start, idx_end = self.limits_samples
        t_start, t_end = self.limits_time
        return int(np.round((t - t_start) / (t_end - t_start) * (idx_end - idx_start) + idx_start))

    def sample_id_to_time(self, s):
        idx_start, idx_end = self.limits_samples
        t_start, t_end = self.limits_time
        return (s - idx_start) / (idx_end - idx_start) * (t_end - t_start) + t_start

    def status(self):
        return self._stream_buffer.status()

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        return self._stream_buffer.calibration_set(current_offset, current_gain, voltage_offset, voltage_gain)

    cdef _reset_accum(self):
        for idx in range(len(self._accum)):
            self._accum[idx] = 0

    def reset(self):
        self._processed_sample_id = 0
        self._process_idx = 0
        if self._stream_buffer:
            self._stream_buffer.reset()
        if self._filter_fir:
            self._filter_fir.reset()
        self._reset_accum()

    cpdef insert(self, data):
        return self._stream_buffer.insert(data)

    cpdef insert_raw(self, data):
        return self._stream_buffer.insert_raw(data)

    cpdef insert_downsampled_and_process(self, data):
        """Insert already downsampled data.
        
        :param data: The N x DS_VALUE_DTYPE numpy array to insert.
        
        Note: this function is not used for normal operation.
        """
        length = len(data)
        start_idx = self._process_idx
        if start_idx + length > self._length:
            split_idx = self._length - start_idx
            end_idx = length - split_idx
            self._buffer_npy[start_idx:] = data[:split_idx]
            self._buffer_npy[:end_idx] = data[split_idx:]
        else:
            end_idx = start_idx + length
            self._buffer_npy[start_idx:end_idx] = data
        self._processed_sample_id += length
        self._process_idx = end_idx

    @staticmethod
    cdef void _stream_buffer_process_cbk(void * user_data, float cal_i, float cal_v, uint8_t bits):
        cdef DownsamplingStreamBuffer self = <object> user_data
        self._input_dbl[0] = <double> cal_i
        self._input_dbl[1] = <double> cal_v
        self._input_dbl[2] = <double> cal_i * cal_v
        self._accum[0] += bits & 0x0f  # current_range
        self._accum[1] += (bits & 0x10) >> 4  # current_lsb
        self._accum[2] += (bits & 0x20) >> 5  # voltage_lsb
        self._filter_fir.c_process(self._input_dbl, 3)  # todo _DS_NP_LENGTH_C

    @staticmethod
    cdef void _filter_fir_cbk(void * user_data, const double * y, uint32_t y_length):
        cdef DownsamplingStreamBuffer self = <object> user_data
        cdef ds_value_s * v = &self._buffer_ptr[self._process_idx]
        v.current = <float> y[0]
        v.voltage = <float> y[1]
        v.power = <float> y[2]
        v.current_range = <uint8_t> ((self._accum[0] * 16) / self._downsample_m)
        v.current_lsb = <uint8_t> ((self._accum[1] * 255) / self._downsample_m)
        v.voltage_lsb = <uint8_t>((self._accum[2] * 255) / self._downsample_m)
        self._reset_accum()
        self._process_idx += 1
        self._processed_sample_id += 1
        if self._process_idx >= self._length:
            self._process_idx = 0

    def process(self):
        self._stream_buffer.process()

    cdef int _range_check(self, uint64_t start, uint64_t stop):
        if stop <= start:
            log.warning("js_stream_buffer_get stop <= start")
            return 0
        if start > self._processed_sample_id:
            log.warning("js_stream_buffer_get start newer that current")
            return 0
        if stop > self._processed_sample_id:
            log.warning("js_stream_buffer_get stop newer than current")
            return 0
        return 1

    cdef uint64_t _stats_get(self, int64_t start, int64_t stop, c_running_statistics.statistics_s * stats_accum):
        """Get exact statistics over the specified range.

        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :param stats_accum: The output computed statistics.
        :return: The number of sample used to compute the statistic
        """
        cdef uint64_t idx = (start % self._length)
        cdef uint64_t count = stop - start
        _stats_reset(stats_accum)
        for idx_inner in range(count):
            v = self._buffer_ptr + idx
            c_running_statistics.statistics_add(stats_accum + 0, v.current)
            c_running_statistics.statistics_add(stats_accum + 1, v.voltage)
            c_running_statistics.statistics_add(stats_accum + 2, v.power)
            c_running_statistics.statistics_add(stats_accum + 3, (<double> v.current_range) * (1.0 / 16))
            c_running_statistics.statistics_add(stats_accum + 4, (<double> v.current_lsb) * (1.0 / 255))
            c_running_statistics.statistics_add(stats_accum + 5, (<double> v.voltage_lsb) * (1.0 / 255))
            idx += 1
            if idx >= self._length:
                idx = 0
        return count

    def statistics_get(self, start, stop, out=None):
        cdef c_running_statistics.statistics_s * out_ptr
        if out is None:
            out = _stats_factory(NULL)
        out_ptr = _stats_ptr(out)
        if start < 0 or stop < 0 or not self._range_check(start, stop):
            log.warning('start, stop invalid: %d, %d', start, stop)
            _stats_invalidate(out_ptr)
            return out
        self._stats_get(start, stop, out_ptr)
        return out

    cdef uint64_t _data_get(self, c_running_statistics.statistics_s *buffer, uint64_t buffer_samples,
                            int64_t start, int64_t stop, uint64_t increment):
        """Get the summarized statistics over a range.
        
        :param buffer: The N x _STATS_FIELDS buffer to populate.
        :param buffer_samples: The value of N of the buffer_ptr (effective buffer length).
        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :return: The number of samples placed into buffer.
        """
        cdef uint64_t buffer_samples_orig = buffer_samples
        cdef uint8_t i
        cdef int64_t idx
        cdef int64_t idx_outer
        cdef int64_t idx_inner
        cdef int64_t data_offset
        cdef c_running_statistics.statistics_s stats[_STATS_FIELDS]
        cdef uint64_t fill_count = 0
        cdef uint64_t fill_count_tmp
        cdef uint64_t samples_per_step
        cdef uint64_t samples_per_step_next
        cdef uint64_t length
        cdef int64_t idx_start
        cdef int64_t end_gap
        cdef int64_t start_orig = start
        cdef uint64_t n
        cdef c_running_statistics.statistics_s * out_ptr
        cdef c_running_statistics.statistics_s * b
        cdef ds_value_s * v

        if (stop + self._length) < self._processed_sample_id:
            fill_count = buffer_samples_orig  # too old, no data
        elif start < 0:
            # round to floor, absolute value
            fill_count_tmp = ((-start + increment - 1) // increment)
            start += fill_count_tmp * increment
            #log.info('_data_get start < 0: %d [%d] => %d', start_orig, fill_count_tmp, start)
            fill_count += fill_count_tmp

        if not self._range_check(start, stop):
            return 0

        if (start + self._length) < self._processed_sample_id:
            fill_count_tmp = (self._processed_sample_id - (start + self._length)) // increment
            start += fill_count_tmp * increment
            #log.info('_data_get behind < 0: %d [%d] => %d', start_orig, fill_count_tmp, start)
            fill_count += fill_count_tmp

        # Fill in too old of data with NAN
        for n in range(fill_count):
            if buffer_samples == 0:
                log.warning('_data_get filled with NaN %d of %d', buffer_samples_orig, fill_count)
                return buffer_samples_orig
            _stats_invalidate(buffer)
            buffer_samples -= 1
            buffer += _STATS_FIELDS
        if buffer_samples != buffer_samples_orig:
            log.debug('_data_get filled %s', buffer_samples_orig - buffer_samples)

        # print(f'data_get({start}, {stop}, {increment}), {self._length}')
        if increment <= 1:
            # direct copy
            idx = start % self._length
            while start != stop and buffer_samples:
                v = self._buffer_ptr + idx
                for b in buffer[:_STATS_FIELDS]:
                    b.k = 1
                    b.s = 0.0
                    b.min = NAN
                    b.max = NAN
                buffer[0].m = v.current
                buffer[1].m = v.voltage
                buffer[2].m = v.power
                buffer[3].m = (<double> v.current_range) * (1.0 / 16)
                buffer[4].m = (<double> v.current_lsb) * (1.0 / 255)
                buffer[5].m = (<double> v.voltage_lsb) * (1.0 / 255)
                buffer_samples -= 1
                idx += 1
                start += 1
                buffer += _STATS_FIELDS
                if idx >= <int64_t> self._length:
                    idx = 0
        else:
            idx = (start % self._length)
            for idx_outer in range(start, stop, increment):
                if buffer_samples == 0:
                    break
                self._stats_get(idx_outer, idx_outer + increment, buffer)
                buffer_samples -= 1
                buffer += _STATS_FIELDS
        return buffer_samples_orig - buffer_samples

    def data_get(self, start, stop, increment=None, out=None):
        # The np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE) data.
        cdef c_running_statistics.statistics_s * out_ptr
        increment = 1 if increment is None else int(increment)
        if start >= stop:
            log.info('data_get: start >= stop')
            return np.zeros((0, _STATS_FIELDS), dtype=STATS_DTYPE)
        expected_length = (stop - start) // increment
        if out is None:
            out = _stats_array_factory(expected_length, NULL)
        out_ptr = _stats_array_ptr(out)
        length = self._data_get(out_ptr, len(out), start, stop, increment)
        if length != expected_length:
            log.warning('length mismatch: expected=%s, returned=%s', expected_length, length)
        return out[:length, :]

    def samples_get(self, start, stop, fields=None):
        """Get sample data without any skips or reductions.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param fields: The list of field names to return.  None (default) is
            equivalent to:
            ['current', 'voltage', 'power', 'current_range',
            'current_lsb', 'voltage_lsb'].
            The available fields are:
            * current: The calibrated float32 current data array in amperes.
            * voltage: The calibrated float32 voltage data array in volts.
            * power: The calibrated float32 Nx2 array of current, voltage.
            * current_range: The current range. 0 = 10A, 6 = 18 uA, 7=off.
            * current_lsb: The current LSB, which can be assign to a general purpose input.
            * voltage_lsb: The voltage LSB, which can be assign to a general purpose input.
        :return: See :method:`StreamBuffer.samples_get`.
        """
        is_single_result = False
        fields_orig = fields
        if fields is None:
            fields = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb']
        elif isinstance(fields, str):
            if fields == 'current_voltage':
                fields = ['current', 'voltage']
            else:
                fields = [fields]
            is_single_result = True
        self._range_check(start, stop)
        length = stop - start
        buffer_npy = np.empty(length, dtype=DS_VALUE_DTYPE)
        start_idx = start % self._length
        stop_idx = stop % self._length

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

        if stop_idx > start_idx:
            for field in fields:
                result['signals'][field] = {
                    'value': self._buffer_npy[start_idx:stop_idx][field].copy(),
                    'units': FIELD_UNITS.get(field, '')}
        else:
            idx = self._length - start_idx
            for field in fields:
                v = np.empty(length, dtype=self._buffer_npy[0][field].dtype)
                v[:idx][:] = self._buffer_npy[start_idx:self._length][field]
                v[idx:][:] = self._buffer_npy[:stop_idx][field]
                result['signals'][field] = {
                    'value': v,
                    'units': FIELD_UNITS.get(field, '')}
        if is_single_result:
            if fields_orig == 'current_voltage':
                i = result['signals']['current']['value']
                v = result['signals']['voltage']['value']
                return np.hstack((i.reshape(-1, 1), v.reshape(-1, 1)))
            else:
                return result['signals'][fields[0]]['value']
        return result
