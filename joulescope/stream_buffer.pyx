# Copyright 2018 Jetperch LLC
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
Optimized Cython native Joulescope code.
"""

# See https://cython.readthedocs.io/en/latest/index.html

# cython: boundscheck=True, wraparound=True, nonecheck=True, overflowcheck=True, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
from libc.float cimport DBL_MAX
from libc.math cimport isfinite, NAN

from libc.string cimport memset, memcpy
from joulescope.units import FIELD_UNITS
import logging
import numpy as np
import psutil
cimport numpy as np
from . cimport c_running_statistics
include "running_statistics.pxi"
include "raw_processor.pxi"


DEF PACKET_TOTAL_SIZE = 512
DEF PACKET_HEADER_SIZE = 8
DEF PACKET_PAYLOAD_SIZE = PACKET_TOTAL_SIZE - PACKET_HEADER_SIZE
DEF PACKET_INDEX_MASK = 0xffff
DEF PACKET_INDEX_WRAP = PACKET_INDEX_MASK + 1

DEF REDUCTION_MAX = 5
DEF SAMPLES_PER_PACKET = PACKET_PAYLOAD_SIZE // (2 * 2)
DEF MEMBYTES_PER_SAMPLE = 16

DEF RAW_SAMPLE_SZ = 2 * 2  # sizeof(uint16_t)
DEF CAL_SAMPLE_SZ = 2 * 4  # sizeof(float)

DEF _STATS_FIELDS = 6  # current, voltage, power, current_range, current_lsb, voltage_lsb


# statistics format for numpy structured data type
# https://docs.scipy.org/doc/numpy/user/basics.rec.html
NP_STATS_FORMAT = ['u8', 'f8', 'f8', 'f8', 'f8']
NP_STATS_NAMES = ['length', 'mean', 'variance', 'min', 'max']
STATS_FIELD_NAMES = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb']
STATS_DTYPE = np.dtype({'names': NP_STATS_NAMES, 'formats': NP_STATS_FORMAT})
STATS_FIELD_COUNT = _STATS_FIELDS
assert(len(STATS_FIELD_NAMES) == STATS_FIELD_COUNT)
log = logging.getLogger(__name__)
np.import_array()  # initialize numpy


NAME_TO_COLUMN = {
    'current': 0,
    'i': 0,
    'i_raw': 0,
    'voltage': 1,
    'v': 1,
    'v_raw': 1,
    'power': 2,
    'p': 2,
    'current_range': 3,
    'current_lsb': 4,
    'voltage_lsb': 5,
}

_SIGNALS = [
    # name, units, integral_units
    ('current', 'A', 'C'),
    ('voltage', 'V', None),
    ('power', 'W', 'J'),
    ('current_range', '', None),
    ('current_lsb', '', None),
    ('voltage_lsb', '', None),
]

ctypedef void (*js_stream_buffer_cbk)(void * user_data, c_running_statistics.statistics_s * stats)


cdef struct js_stream_buffer_reduction_s:
    int32_t enabled
    uint32_t samples_per_step
    uint32_t samples_per_reduction_sample
    uint32_t sample_counter
    uint32_t length
    js_stream_buffer_cbk cbk_fn
    void * cbk_user_data
    c_running_statistics.statistics_s * data  # [length * _STATS_FIELDS]


cdef void _stats_reset(c_running_statistics.statistics_s * s):
    cdef uint8_t i
    for i in range(_STATS_FIELDS):
        c_running_statistics.statistics_reset(&s[i])


cdef void _stats_invalidate(c_running_statistics.statistics_s * s):
    cdef uint8_t i
    for i in range(_STATS_FIELDS):
        c_running_statistics.statistics_invalid(&s[i])


cdef uint64_t _stats_length(c_running_statistics.statistics_s * s):
    return s[0].k


cdef void _stats_copy(c_running_statistics.statistics_s * dst, c_running_statistics.statistics_s * src):
    cdef uint8_t i
    for i in range(_STATS_FIELDS):
        c_running_statistics.statistics_copy(&dst[i], &src[i])


cdef void _stats_compute_one(c_running_statistics.statistics_s * s,
                             float current,
                             float voltage,
                             uint8_t bits):
    c_running_statistics.statistics_add(&s[0], current)
    c_running_statistics.statistics_add(&s[1], voltage)
    c_running_statistics.statistics_add(&s[2], current * voltage)
    c_running_statistics.statistics_add(&s[3], bits & 0x0f)
    c_running_statistics.statistics_add(&s[4], (bits & 0x10) >> 4)
    c_running_statistics.statistics_add(&s[5], (bits & 0x20) >> 5)


cdef uint64_t _stats_compute_run(
        c_running_statistics.statistics_s * s,
        float * data, uint8_t * bits, uint32_t data_length,
        uint64_t sample_id, uint64_t length):
    cdef uint64_t i
    cdef uint32_t idx = sample_id % data_length
    cdef uint32_t data_idx
    cdef uint64_t counter = 0
    _stats_reset(s)
    for i in range(length):
        data_idx = idx * 2
        if isfinite(data[data_idx]):
            _stats_compute_one(s, data[data_idx], data[data_idx + 1], bits[idx])
            counter += 1
        idx += 1
        if idx >= data_length:
            idx = 0
    return counter


cdef uint64_t _stats_combine(
        c_running_statistics.statistics_s * tgt,
        c_running_statistics.statistics_s * a,
        c_running_statistics.statistics_s * b):
    cdef uint8_t i
    for i in range(_STATS_FIELDS):
        c_running_statistics.statistics_combine(&tgt[i], &a[i], &b[i])
    return tgt[0].k


cdef c_running_statistics.statistics_s * _stats_ptr(c_running_statistics.statistics_s [::1] d):
    return &d[0]


cdef _stats_factory(c_running_statistics.statistics_s ** d_ptr):
    d = np.zeros(_STATS_FIELDS, dtype=STATS_DTYPE)
    _stats_reset(_stats_ptr(d))
    if d_ptr:
        d_ptr[0] = _stats_ptr(d)
    return d


def stats_factory():
    return np.zeros(_STATS_FIELDS, dtype=STATS_DTYPE)


def stats_invalidate(s):
    s[:]['length'] = 0
    s[:]['mean'] = NAN
    s[:]['variance'] = NAN
    s[:]['min'] = NAN
    s[:]['max'] = NAN


cdef c_running_statistics.statistics_s * _stats_array_ptr(c_running_statistics.statistics_s [:, ::1] d):
    return &d[0, 0]


cdef _stats_array_factory(length, c_running_statistics.statistics_s ** d_ptr):
    d = np.zeros((length, _STATS_FIELDS), dtype=STATS_DTYPE)
    d[:, :]['min'] = DBL_MAX
    d[:, :]['max'] = -DBL_MAX
    if d_ptr:
        d_ptr[0] = _stats_array_ptr(d)
    return d


def stats_array_factory(length):
    return np.zeros((length, _STATS_FIELDS), dtype=STATS_DTYPE)


def stats_array_clear(s):
    s[:, :]['length'] = 0
    s[:, :]['mean'] = 0.0
    s[:, :]['variance'] = 0.0
    s[:, :]['min'] = DBL_MAX
    s[:, :]['max'] = -DBL_MAX


def stats_array_invalidate(s):
    s[:, :]['length'] = 0
    s[:, :]['mean'] = NAN
    s[:, :]['variance'] = NAN
    s[:, :]['min'] = NAN
    s[:, :]['max'] = NAN


def stats_compute(data, out):
    cdef uint64_t length = len(data)
    cdef uint64_t idx
    cdef np.float32_t [::1] data_c = data
    cdef c_running_statistics.statistics_s * s = _stats_ptr(out)
    c_running_statistics.statistics_reset(s)
    for idx in range(length):
        if isfinite(data_c[idx]):
            c_running_statistics.statistics_add(s, data_c[idx])


cdef uint64_t reduction_stats(js_stream_buffer_reduction_s * r, c_running_statistics.statistics_s * tgt,
                              uint32_t idx_start, uint32_t length):
    cdef c_running_statistics.statistics_s * src
    cdef uint32_t count
    cdef uint64_t reduction_sample_count
    cdef uint32_t idx = idx_start
    cdef uint8_t i

    sample_count = 0
    _stats_reset(tgt)
    for count in range(length):
        # log.debug('reduction %d, %d', sample_count, r.samples_per_data[idx])
        src = &r.data[idx * _STATS_FIELDS]
        for i in range(_STATS_FIELDS):
            c_running_statistics.statistics_combine(&tgt[i], &tgt[i], &src[i])
        idx += 1
        if idx >= r.length:
            idx = 0

    if 0 == tgt[0].k:
        # log.warning('reduction_stats empty')
        _stats_invalidate(tgt)
    return tgt[0].k


cdef _reduction_downsample(js_stream_buffer_reduction_s * r,
                           c_running_statistics.statistics_s * tgt,
                           uint32_t idx_start, uint32_t idx_stop, uint32_t increment):
    cdef uint32_t idx = idx_start
    while idx + increment <= idx_stop:
        reduction_stats(r, tgt, idx, increment)
        idx += increment
        tgt += _STATS_FIELDS


def reduction_downsample(reduction, idx_start, idx_stop, increment):
    """Downsample a data reduction.

    :param reduction: The np.array((N, STATS_FIELD_COUNT), dtype=DTYPE)
    :param idx_start: The starting index (inclusive) in reduction.
    :param idx_stop: The stopping index (exclusive) in reduction.
    :param increment: The increment value
    :return: The downsampled reduction.

    The x-values can be constructed:

        x = np.arange(idx_start, idx_stop - increment + 1, increment, dtype=np.float64)
    """
    cdef c_running_statistics.statistics_s * out_ptr
    cdef js_stream_buffer_reduction_s r_inst
    r_inst.length = <uint32_t> len(reduction)
    length = (idx_stop - idx_start) // increment
    r_inst.data = _stats_array_ptr(reduction)
    r_inst.samples_per_step = 1  # does not matter, weight all equally

    out = _stats_array_factory(length, &out_ptr)
    _reduction_downsample(&r_inst, out_ptr, idx_start, idx_stop, increment)
    return out


cdef class Statistics:

    cdef object stats
    cdef c_running_statistics.statistics_s * stats_ptr

    def __cinit__(self):
        self.stats = _stats_factory(&self.stats_ptr)

    def __init__(self, stats=None):
        if stats is not None:
            self.stats[:] = stats

    def __len__(self):
        return _stats_length(self.stats_ptr)

    def combine(self, other: Statistics):
        _stats_combine(self.stats_ptr, self.stats_ptr, other.stats_ptr)
        return self

    cdef _value(self):
        cdef uint8_t i
        cdef uint8_t k
        cdef c_running_statistics.statistics_s * out_ptr
        out = _stats_factory(&out_ptr)
        _stats_copy(out_ptr, self.stats_ptr)
        return out

    @property
    def value(self):
        return self._value()



ctypedef void (*usb_bulk_data_processor_cbk_fn)(object user_data,
                                                const uint16_t *data, size_t length,
                                                uint8_t voltage_range)


cdef class UsbBulkProcessor:
    """Process bulk packets received over USB into sample data.

    Detect missing packets and insert samples as needed.
    """

    cdef uint8_t _voltage_range
    cdef uint64_t _packet_index
    # packet_index 64-bits is enough for 18 million years at 2 MSPS, 126 samples / packet
    #  2**63 / (2000000/126)/ (60 * 60 * 24 * 365)
    cdef uint64_t _packet_missing_count
    cdef uint64_t _packet_type_invalid_count
    cdef uint64_t _packet_data_invalid_count

    cdef object _missing_pkt
    cdef uint16_t *_missing_pkt_ptr
    cdef usb_bulk_data_processor_cbk_fn _cbk_fn
    cdef object _cbk_user_data

    def __cinit__(self):
        self._missing_pkt = np.full((SAMPLES_PER_PACKET * 2), 0xffff, dtype=np.uint16)
        assert(self._missing_pkt.flags['C_CONTIGUOUS'])
        cdef np.uint16_t [::1] missing_pkt_c = self._missing_pkt
        self._missing_pkt_ptr = &missing_pkt_c[0]

    def __init__(self):
        self.reset()

    def reset(self):
        self._voltage_range = 0
        self._packet_index = 0
        self._packet_missing_count = 0
        self._packet_type_invalid_count = 0
        self._packet_data_invalid_count = 0

    cdef callback_set(self, usb_bulk_data_processor_cbk_fn cbk, object user_data):
        self._cbk_fn = cbk
        self._cbk_user_data = user_data

    def status(self):
        return {
            'voltage_range': {'value': self._voltage_range, 'units': ''},
            'packet_index': {'value': self._packet_index, 'units': 'packets'},
            'packet_missing_count': {'value': self._packet_missing_count, 'units': 'packets'},
            'packet_type_invalid_count': {'value': self._packet_type_invalid_count, 'units': 'packets'},
            'packet_data_invalid_count': {'value': self._packet_data_invalid_count, 'units': 'packets'},
        }

    cdef void process(self, const uint8_t *data, size_t length):
        """Process one or more USB frames into samples.
        
        :param data: The pointer to the USB frame data.
        :param length: The length of data in bytes.
        
        This function invokes the callback for the samples contained
        in each USB frame that contains sample data.
        """
        cdef uint8_t buffer_type
        cdef uint8_t status
        cdef uint16_t pkt_length
        cdef uint8_t voltage_range
        cdef uint64_t pkt_index
        # cdef uint16_t usb_frame_index

        while length >= PACKET_TOTAL_SIZE:
            buffer_type = data[0]
            if 1 == buffer_type:
                status = data[1]
                pkt_length = (data[2] | ((<uint16_t> data[3] & 0x7f) << 8)) & 0x7fff
                voltage_range = <uint8_t> ((data[3] >> 7) & 0x01)
                pkt_index = <uint64_t> (data[4] | ((<uint16_t> data[5]) << 8))
                # usb_frame_index = data[6] | ((<uint16_t> data[7]) << 8)
                if (0 == status) and (PACKET_TOTAL_SIZE == pkt_length):
                    while (self._packet_index & 0xffff) != pkt_index:
                        self._cbk_fn(self._cbk_user_data,
                                     self._missing_pkt_ptr, SAMPLES_PER_PACKET * 2,
                                     self._voltage_range)
                        self._packet_index += 1
                        self._packet_missing_count += 1
                    self._voltage_range = voltage_range
                else:
                    self._packet_data_invalid_count += 1
                self._cbk_fn(self._cbk_user_data, <uint16_t *> &data[8], SAMPLES_PER_PACKET * 2, self._voltage_range)
                self._packet_index += 1
            else:
                self._packet_type_invalid_count += 1
            data += PACKET_TOTAL_SIZE
            length -= PACKET_TOTAL_SIZE


ctypedef void (*stream_buffer_process_fn)(void * user_data, float cal_i, float cal_v, uint8_t bits)


cdef class StreamBuffer:
    """Efficient real-time Joulescope data buffering.

    :param duration: The total length of the buffering in seconds.
    :param reductions: The list of reduction integers.  Each integer represents
        the reduction amount for each resuting sample in units of samples
        of the previous reduction.  Reduction 0 is in raw sample units.
    """
    cdef UsbBulkProcessor _usb_bulk_processor
    cdef RawProcessor _raw_processor
    cdef uint32_t reduction_step
    cdef uint32_t length # in samples
    cdef uint64_t device_sample_id      # exclusive (last received is device_sample_id - 1)
    cdef uint64_t _preprocessed_sample_id
    cdef uint64_t processed_sample_id   # exclusive (last processed is processed_sample_id - 1)
    cdef uint16_t *raw_ptr  # raw[length][2]   as i, v
    cdef float *data_ptr    # data[length][2]  as i, v
    cdef uint8_t * bits_ptr # data[length]  # packed bits: 7:6=0 , 5=voltage_lsb, 4=current_lsb, 3:0=i_range
    cdef js_stream_buffer_reduction_s reductions[REDUCTION_MAX]
    cdef uint32_t reduction_count
    cdef double _sampling_frequency
    cdef stream_buffer_process_fn _process_stats_cbk_fn
    cdef void * _process_stats_cbk_user_data

    cdef uint32_t stats_counter  # excludes NAN for mean
    cdef uint32_t stats_remaining
    cdef c_running_statistics.statistics_s stats[_STATS_FIELDS]

    cdef object raw
    cdef object data
    cdef object bits
    cdef object reductions_data
    cdef uint64_t _sample_id_max  # used to automatically stop streaming
    cdef uint64_t _contiguous_max  # used to automatically stop streaming
    cdef object _callback  # fn(dict), see _on_cbk
    cdef object _charge_picocoulomb  # python integer for infinite precision
    cdef object _energy_picojoules  # python integer for infinite precision

    def __cinit__(self, duration, reductions, sampling_frequency):
        length = int(sampling_frequency * duration)
        mem_free = psutil.virtual_memory().available
        if mem_free < length * MEMBYTES_PER_SAMPLE:
            duration_fit = mem_free // (sampling_frequency * MEMBYTES_PER_SAMPLE)
            length = int(sampling_frequency * duration_fit)
            log.warning('not enough memory: reducing duration from %s to %s', duration, duration_fit)
            duration = duration_fit
        self._usb_bulk_processor = UsbBulkProcessor()
        self._usb_bulk_processor.callback_set(<usb_bulk_data_processor_cbk_fn> self._process_samples, self)
        self._raw_processor = RawProcessor()
        self._raw_processor.callback_set(<raw_processor_cbk_fn> self._process_stats, self)
        self._process_stats_cbk_fn = NULL
        self._process_stats_cbk_user_data = NULL
        cdef uint32_t r_samples = 1
        self._sampling_frequency = sampling_frequency
        if length < SAMPLES_PER_PACKET:
            raise ValueError('length to small')
        if len(reductions) > REDUCTION_MAX:
            raise ValueError('too many reductions')

        self._charge_picocoulomb = 0
        self._energy_picojoules = 0
        memset(self.reductions, 0, sizeof(self.reductions))

        # round up length to multiple of reductions
        self.reduction_step = int(np.prod(reductions))
        length = int(np.ceil(length / self.reduction_step)) * self.reduction_step
        self.length = length

        self.raw = np.full((length * 2), 0, dtype=np.uint16)
        assert(self.raw.flags['C_CONTIGUOUS'])
        cdef np.uint16_t [::1] raw_c = self.raw
        self.raw_ptr = &raw_c[0]

        self.data = np.full((length * 2), 0.0, dtype=np.float32)
        assert(self.data.flags['C_CONTIGUOUS'])
        cdef np.float32_t [::1] data_c = self.data
        self.data_ptr = &data_c[0]

        self.bits = np.full(length, 0, dtype=np.uint8)
        assert(self.bits.flags['C_CONTIGUOUS'])
        cdef np.uint8_t [::1] bits_c = self.bits
        self.bits_ptr = &bits_c[0]

        self.reductions_data = []

        cdef js_stream_buffer_reduction_s * r
        sz = length

        for idx, rsamples in enumerate(reductions):
            sz = sz // rsamples
            r = &self.reductions[idx]
            r.enabled = 1
            r.samples_per_step = <uint32_t> rsamples
            r.length = sz
            r_samples *= rsamples
            r.samples_per_reduction_sample = r_samples
            d = _stats_array_factory(sz, &r.data)
            self.reductions_data.append(d)

        self.reduction_count = <uint32_t> len(reductions)
        if len(reductions):
            self.reductions[len(reductions) - 1].cbk_fn = self._on_cbk
            self.reductions[len(reductions) - 1].cbk_user_data = <void *> self

        self._sample_id_max = 0  # used to automatically stop streaming
        self._contiguous_max = 0  # used to automatically stop streaming
        self._callback = None
        self._charge_picocoulomb = 0
        self._energy_picojoules = 0  # integer for infinite precision

    def __init__(self, duration, reductions, sampling_frequency):
        self.reset()

    def __len__(self):
        return self.length

    def __str__(self):
        reductions = []
        for idx in range(REDUCTION_MAX):
            if self.reductions[idx].enabled:
                reductions.append(self.reductions.samples_per_step)
        return 'StreamBuffer(length=%d, reductions=%r)' % (self.length, reductions)

    @property
    def has_raw(self):
        """Query if this instance provides raw sample data.

        :return: True if samples_get supports 'raw', False otherwise.
        """
        return True

    @property
    def sample_id_range(self):
        """Get the range of sample ids currently available in the buffer.

        :return: Tuple of sample_id start, sample_id end.
            Start and stop follow normal python indexing:
            start is inclusive, end is exclusive
        """
        s_end = int(self.processed_sample_id)
        s_start = s_end - self.length
        if s_start < 0:
            s_start = 0
        return s_start, s_end

    @property
    def data_buffer(self):
        """Get the underlying data buffer.

        WARNING: Don't use this!  It  should only be used for unit testing.
        """
        return self.data  # the cdef np.ndarray

    @property
    def sample_id_max(self):
        return self._sample_id_max

    @sample_id_max.setter
    def sample_id_max(self, value):
        self._sample_id_max = value  # stop streaming when reach this sample

    @property
    def contiguous_max(self):
        return self._contiguous_max

    @contiguous_max.setter
    def contiguous_max(self, value):
        self._contiguous_max = value

    @property
    def callback(self):
        return self._callback

    @callback.setter
    def callback(self, value):
        self._callback = value

    cdef process_stats_callback_set(self, stream_buffer_process_fn fn, void * user_data):
        self._process_stats_cbk_fn = fn
        self._process_stats_cbk_user_data = user_data

    @property
    def voltage_range(self):
        return self._raw_processor.voltage_range

    @voltage_range.setter
    def voltage_range(self, value):
        """Set the voltage range for applying calibration.

        Note that the Joulescope device normally conveys the voltage range in
        along with the sample data.
        """
        self._raw_processor.voltage_range = value

    @property
    def suppress_mode(self):
        return self._raw_processor.suppress_mode

    @suppress_mode.setter
    def suppress_mode(self, value):
        self._raw_processor.suppress_mode = value

    @property
    def input_sampling_frequency(self):
        return self._sampling_frequency

    @property
    def output_sampling_frequency(self):
        return self._sampling_frequency

    @property
    def limits_time(self):
        """Get the time limits.

        :return: (start, stop).  Stop corresponds to the exclusive sample
            returned by b.limits_samples[1].
        """
        return 0.0, len(self) / self._sampling_frequency

    @property
    def limits_samples(self):
        """Get the sample limits.

        :return: (start, stop) where start is inclusive and stop is exclusive.

        In other words:
            self.limits_time == (b.sample_id_to_time(b.limits_samples[0]),
                                 b.sample_id_to_time(b.limits_samples[1]))
        """
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
            'device_sample_id': {'value': self.device_sample_id, 'units': 'samples'},
            'sample_id': {'value': self.processed_sample_id, 'units': 'samples'},
            'sample_missing_count': {'value': self._raw_processor.sample_missing_count, 'units': 'samples'},
            'skip_count': {'value': self._raw_processor.skip_count, 'units': ''},
            'sample_sync_count': {'value': self._raw_processor.sample_sync_count, 'units': 'samples'},
            'contiguous_count': {'value': self._raw_processor.contiguous_count, 'units': 'samples'},
        }

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        self._raw_processor.calibration_set(current_offset, current_gain, voltage_offset, voltage_gain)

    cdef _stats_reset(self):
        self.stats_counter = 0
        self.stats_remaining = 0
        if self.reductions[0].enabled:
            self.stats_remaining = self.reductions[0].samples_per_step
        _stats_reset(self.stats)

    def reset(self):
        self.device_sample_id = 0
        self._preprocessed_sample_id = 0
        self.processed_sample_id = 0
        self._sample_id_max = 1 << 63  # big enough
        self._contiguous_max = 1 << 63  # big enough
        self._charge_picocoulomb = 0
        self._energy_picojoules = 0
        self.stats_counter = 0
        for idx in range(REDUCTION_MAX):
            self.reductions[idx].sample_counter = 0
        self._stats_reset()
        self._usb_bulk_processor.reset()
        self._raw_processor.reset()

    cdef uint32_t reduction_index(self, js_stream_buffer_reduction_s * r, uint32_t parent_samples_per_step):
        cdef uint32_t idx = <uint32_t> (self.processed_sample_id % self.length)
        cdef uint32_t samples_per_step = parent_samples_per_step * r.samples_per_step;
        idx /= samples_per_step
        if 0 == idx:
            idx = r.length - 1
        else:
            idx -= 1
        return idx

    cdef void reduction_update_n(self, int n, uint32_t parent_samples_per_step):
        cdef js_stream_buffer_reduction_s * r = &self.reductions[n]
        cdef c_running_statistics.statistics_s stats[_STATS_FIELDS]
        cdef uint32_t samples_per_step
        cdef uint32_t idx_target
        cdef uint32_t idx_start

        if 0 == r.enabled:
            return
        r.sample_counter += 1
        if r.sample_counter >= r.samples_per_step:
            r.sample_counter = 0
            samples_per_step = parent_samples_per_step * r.samples_per_step
            idx_target = self.reduction_index(r, parent_samples_per_step)
            idx_start = idx_target * r.samples_per_step
            reduction_stats(&self.reductions[n - 1], stats, idx_start, r.samples_per_step)
            _stats_copy(&r.data[idx_target * _STATS_FIELDS], stats)
            if r.cbk_fn:
                r.cbk_fn(r.cbk_user_data, &r.data[idx_target * _STATS_FIELDS])
            self.reduction_update_n(n + 1, samples_per_step)

    cdef void reduction_update_0(self):
        cdef js_stream_buffer_reduction_s * r = &self.reductions[0]
        if 0 == r.enabled:
            log.warning('reduction_update_0 disabled')
            return
        if 0 == self.stats_counter:
            # log.debug('reduction_update_0 but stats_counter = 0')
            _stats_invalidate(self.stats)
        cdef uint32_t idx = self.reduction_index(r, 1)
        _stats_copy(&r.data[idx * _STATS_FIELDS], self.stats)
        self.reduction_update_n(1, self.reductions[0].samples_per_step)

    cdef _check_stop(self):
        cdef bint duration_stop = self.device_sample_id >= self._sample_id_max
        cdef bint contiguous_stop = self._raw_processor.contiguous_count >= self._contiguous_max
        if duration_stop:
            log.info('insert causing duration stop %d >= %d',
                     self.device_sample_id, self._sample_id_max)
            return True
        elif contiguous_stop:
            log.info('insert causing contiguous stop %d >= %d',
                     self._raw_processor.contiguous_count, self._contiguous_max)
            return True
        else:
            return False

    cpdef insert(self, data):
        """Insert new device USB data into the buffer.

        :param data: The new data to insert.
        :return: False to continue streaming, True to end.
        """
        cdef np.uint8_t [::1] data_c
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=np.uint8)
            data_c = data
            self._usb_bulk_processor.process(&data_c[0], len(data))
        elif data is not None:
            self._usb_bulk_processor.process(data, len(data))
        else:
            return True  # todo signal our data listener?
        return self._check_stop()

    cpdef insert_raw(self, data):
        """Insert raw data into the buffer
        
        :param data: The np.array of np.uint16 data to insert.
        :return: False to continue streaming, True to end.
        """
        if data.dtype != np.uint16:
            raise ValueError('raw data must np np.uint16 array')
        data = data.reshape((-1, ))
        sample_count = len(data)
        if sample_count % 2:
            raise ValueError('raw data must be multiples of 2 16-bit values')
        sample_count = sample_count // 2
        # log.debug('insert_raw %d', sample_count)
        idx = self.device_sample_id % self.length
        sample_count_remaining = sample_count
        while idx + sample_count_remaining > self.length:
            samples_to_end = self.length - idx
            self.raw[idx * 2:] = data[:samples_to_end * 2]
            data = data[samples_to_end * 2:]
            idx = 0
            sample_count_remaining -= samples_to_end
        if sample_count_remaining:
            self.raw[idx * 2: (idx + sample_count_remaining) * 2] = data
        self.device_sample_id += sample_count
        return self._check_stop()

    cdef void _process(self):
        cdef uint32_t idx_start
        cdef uint32_t idx
        if self._preprocessed_sample_id + self.length < self.device_sample_id:
            log.warning('process: stream_buffer is behind: %r + %r < %r',
                        self._preprocessed_sample_id, self.length, self.device_sample_id)
            self._preprocessed_sample_id = self.device_sample_id - self.length
            self.processed_sample_id = self._preprocessed_sample_id
        idx_start = <uint32_t> (self._preprocessed_sample_id % self.length)

        while self._preprocessed_sample_id < self.device_sample_id:
            idx = idx_start * 2
            raw_i = self.raw_ptr[idx + 0]
            raw_v = self.raw_ptr[idx + 1]
            self._raw_processor.process(raw_i, raw_v)
            idx_start += 1
            if idx_start >= self.length:
                idx_start = 0
            self._preprocessed_sample_id += 1

    cdef void _process_samples(self,
                               const uint16_t *data, size_t length,
                               uint8_t voltage_range):
        """Samples from UsbBulkProcessor.
        
        :param data: The data samples in i,v order.
        :param length: The length of data in uint16_t.
        :param voltage_range: The voltage range for these samples.
        """
        cdef uint64_t sample_count  = length // 2
        cdef uint64_t k
        cdef uint64_t idx
        self._raw_processor.voltage_range = voltage_range
        while sample_count:
            idx = self.device_sample_id % self.length
            k = sample_count if (idx + sample_count) <= self.length else self.length - idx
            memcpy(&self.raw_ptr[idx * 2], data, k * RAW_SAMPLE_SZ)
            data += k * 2
            self.device_sample_id += k
            sample_count -= k

    cdef void _process_stats(self, float cal_i, float cal_v, uint8_t bits):
        cdef uint32_t idx
        idx = <uint32_t> (self.processed_sample_id % self.length)
        self.bits_ptr[idx] = bits
        idx *= 2
        self.data_ptr[idx + 0] = cal_i
        self.data_ptr[idx + 1] = cal_v
        self.processed_sample_id += 1
        if self._process_stats_cbk_fn:
            self._process_stats_cbk_fn(self._process_stats_cbk_user_data, cal_i, cal_v, bits)
        if 0 == self.reductions[0].enabled:
            return
        if isfinite(cal_i):
            self.stats_counter += 1
            _stats_compute_one(self.stats, cal_i, cal_v, bits)
        if self.stats_remaining > self.reductions[0].samples_per_step:
            log.warning('Internal error stats_remaining: %d > %d',
                        self.stats_remaining,
                        self.reductions[0].samples_per_step)
            self._stats_reset()
        elif self.stats_remaining > 1:
            self.stats_remaining -= 1
        elif self.stats_remaining <= 1:
            self.reduction_update_0()
            self._stats_reset()

    def process(self):
        self._process()

    cdef int range_check(self, uint64_t start, uint64_t stop):
        if stop <= start:
            log.warning("js_stream_buffer_get stop <= start")
            return 0
        if start >= self.processed_sample_id:
            log.warning("js_stream_buffer_get start newer that current")
            return 0
        if stop > self.processed_sample_id:
            log.warning("js_stream_buffer_get stop newer than current")
            return 0
        return 1

    cdef _constrain_range(self, start, stop):
        xlim_start, xlim_stop = self.sample_id_range
        x_start = max(0, start, xlim_start)
        x_stop = min(stop, xlim_stop)
        x_stop = max(x_start, x_stop)
        if x_start != start or x_stop != stop:
            log.warning('range [%r, %r] constrained to [%r, %r]', start, stop, x_start, x_stop)
        return [x_start, x_stop]

    cdef _stats_get(self, int64_t start, int64_t stop, c_running_statistics.statistics_s * stats_accum):
        """Get exact statistics over the specified range.

        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :param stats_accum: The output computed statistics.
        :return: The number of sample used to compute the statistic
        """
        cdef uint64_t length
        cdef c_running_statistics.statistics_s stats_merge[_STATS_FIELDS]
        cdef uint32_t samples_per_step

        _stats_reset(stats_accum)
        start, stop = self._constrain_range(start, stop)
        length = stop - start
        ranges = [[start, stop], [None, None]]

        for n in range(self.reduction_count - 1, -1, -1):
            samples_per_step = self.reductions[n].samples_per_reduction_sample
            for idx, [r1, r2] in enumerate(ranges):
                if r1 is None:
                    continue
                k1 = r1 // samples_per_step * samples_per_step
                if k1 < r1:
                    k1 += samples_per_step
                k2 = (r2 // samples_per_step) * samples_per_step
                if k1 < k2:  # we can use this reduction!
                    # log.debug('reduction %d on %d: %s to %s', n, idx, k1, k2)
                    r_idx_start = <uint32_t> ((k1 % self.length) // samples_per_step)
                    r_sample_length = k2 - k1
                    r_idx_length = r_sample_length // samples_per_step
                    reduction_stats(&self.reductions[n], stats_merge, r_idx_start, r_idx_length)
                    _stats_combine(stats_accum, stats_accum, stats_merge)
                    if idx == 0:
                        if r1 == k1:
                            ranges[idx] = [None, None]
                        else:
                            ranges[idx] = [r1, k1]
                        if ranges[1][0] is None and k2 < r2:
                            ranges[1] = [k2, r2]
                    else:
                        if r2 == k2:
                            ranges[idx] = [None, None]
                        else:
                            ranges[idx] = [k2, r2]

        # log.debug('ranges = %r', ranges)
        for r1, r2 in ranges:
            if r1 is not None:
                r_sample_length = r2 - r1
                _stats_compute_run(stats_merge, self.data_ptr, self.bits_ptr, self.length, r1, r_sample_length)
                # log.debug('direct: %s to %s (%d + %d)', r1, r2, stats_accum[0].k, count)
                _stats_combine(stats_accum, stats_accum, stats_merge)

        if _stats_length(stats_accum) == 0:
            log.warning('_stats_get no samples')
            _stats_invalidate(stats_accum)
        return [start, stop], _stats_length(stats_accum)

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
        cdef c_running_statistics.statistics_s * out_ptr
        if out is None:
            out = _stats_factory(NULL)
        out_ptr = _stats_ptr(out)
        x_range, _ = self._stats_get(start, stop, out_ptr)
        return out, x_range

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

        if stop + self.length < (<int64_t> self.processed_sample_id):
            fill_count = buffer_samples_orig
        elif start < 0:
            # round to floor, absolute value
            fill_count_tmp = ((-start + increment - 1) // increment)
            start += fill_count_tmp * increment
            log.info('_data_get start < 0: %d [%d] => %d', start_orig, fill_count_tmp, start)
            fill_count += fill_count_tmp

        if not self.range_check(start, stop):
            return 0

        if (start + self.length) < (<int64_t> self.processed_sample_id):
            fill_count_tmp = (self.processed_sample_id - (start + self.length)) // increment
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
            buffer = &buffer[_STATS_FIELDS]
        if buffer_samples != buffer_samples_orig:
            log.debug('_data_get filled %s', buffer_samples_orig - buffer_samples)

        if increment <= 1:
            # direct copy
            idx = start % self.length
            while start != stop and buffer_samples:
                data_offset = idx * 2
                for i in range(_STATS_FIELDS):
                    buffer[i].k = 1
                    buffer[i].s = 0.0
                    buffer[i].min = NAN
                    buffer[i].max = NAN
                buffer[0].m = self.data_ptr[data_offset]
                buffer[1].m = self.data_ptr[data_offset + 1]
                buffer[2].m = self.data_ptr[data_offset] * self.data_ptr[data_offset + 1]
                buffer[3].m = <double> (self.bits_ptr[idx] & 0x0f)
                buffer[4].m = <double> ((self.bits_ptr[idx] & 0x10) >> 4)
                buffer[5].m = <double> ((self.bits_ptr[idx] & 0x20) >> 5)
                buffer_samples -= 1
                idx += 1
                start += 1
                buffer = &buffer[_STATS_FIELDS]
                if idx >= self.length:
                    idx = 0
        elif not self.reductions[0].enabled or (self.reductions[0].samples_per_step > increment):
            # compute over raw data.
            while start + <int64_t> increment <= stop and buffer_samples:
                _stats_compute_run(buffer, self.data_ptr, self.bits_ptr, self.length, start, increment)
                if buffer[0].k == 0:
                    _stats_invalidate(buffer)
                buffer = &buffer[_STATS_FIELDS]
                start += increment
                buffer_samples -= 1
        else:
            # use reductions through stats_get
            while start + <int64_t> increment <= stop and buffer_samples:
                next_start = start + increment
                self._stats_get(start, next_start, buffer)
                buffer = &buffer[_STATS_FIELDS]
                start += increment
                buffer_samples -= 1
        return buffer_samples_orig - buffer_samples

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
        cdef c_running_statistics.statistics_s * out_ptr
        increment = 1 if increment is None else int(increment)
        if start >= stop:
            log.info('data_get: start >= stop')
            return np.zeros((0, _STATS_FIELDS), dtype=STATS_DTYPE)
        expected_length = (stop - start) // increment
        if out is None:
            out = _stats_array_factory(expected_length, NULL)
        elif len(out) < expected_length:
            raise ValueError('out too small')
        out_ptr = _stats_array_ptr(out)
        length = self._data_get(out_ptr, len(out), start, stop, increment)
        if length != expected_length:
            log.warning('length mismatch: expected=%s, returned=%s', expected_length, length)
        return out[:length, :]

    def samples_get(self, start, stop, fields=None):
        """Get exact sample data without any skips or reductions.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param fields: The single field or list of field names to return.
            None (default) is equivalent to
            ['current', 'voltage', 'power', 'current_range', 'current_lsb',
            'voltage_lsb', 'raw'].
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
        bits = None
        is_single_result = False
        if fields is None:
            fields = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb', 'raw']
        elif isinstance(fields, str):
            fields = [fields]
            is_single_result = True

        if stop <= start:
            log.warning('samples_get %d <= %d', start, stop)
            start = stop

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

        total_length = self.length
        start_idx = start % total_length
        stop_idx = stop % total_length
        if 0 == stop_idx:
            stop_idx = total_length

        def populate_bits():
            nonlocal bits
            if bits is None:
                if stop_idx > start_idx:
                    bits = self.bits[start_idx:stop_idx]
                else:
                    bits = np.concatenate((self.bits[start_idx:], self.bits[:stop_idx]))

        for field in fields:
            value_dict = {'units': FIELD_UNITS.get(field, '')}
            if field == 'raw':
                if stop_idx > start_idx:
                    out = self.raw[(start_idx * 2):(stop_idx * 2)].reshape((-1, 2))
                else:
                    out = np.vstack((
                        self.raw[(start_idx * 2):].reshape((-1, 2)),
                        self.raw[:(stop_idx * 2)].reshape((-1, 2))))
                value_dict['voltage_range'] = self.voltage_range
            elif field == 'raw_current':
                if stop_idx > start_idx:
                    d = self.raw[(start_idx * 2):(stop_idx * 2):2]
                else:
                    d = np.concatenate((
                        self.raw[(start_idx * 2)::2],
                        self.raw[:(stop_idx * 2):2]))
                out = np.right_shift(d, 2)
            elif field == 'raw_voltage':
                if stop_idx > start_idx:
                    d = self.raw[(start_idx * 2 + 1):(stop_idx * 2):2]
                else:
                    d = np.concatenate((
                        self.raw[(start_idx * 2 + 1)::2],
                        self.raw[1:(stop_idx * 2):2]))
                out = np.right_shift(d, 2)
            elif field == 'bits':
                populate_bits()
                out = bits
            elif field == 'current':
                if stop_idx > start_idx:
                    out = self.data[(start_idx * 2):(stop_idx * 2):2]
                else:
                    out = np.concatenate((self.data[(start_idx * 2)::2],
                                          self.data[:(stop_idx * 2):2]))
            elif field == 'voltage':
                if stop_idx > start_idx:
                    out = self.data[(start_idx * 2 + 1):(stop_idx * 2):2]
                else:
                    out = np.concatenate((self.data[(start_idx * 2 + 1)::2],
                                          self.data[1:(stop_idx * 2):2]))
            elif field == 'current_voltage':
                if stop_idx > start_idx:
                    out = self.data[(start_idx * 2):(stop_idx * 2)].reshape((-1, 2))
                else:
                    out = np.vstack((self.data[(start_idx * 2):].reshape((-1, 2)),
                                     self.data[:(stop_idx * 2)].reshape((-1, 2))))
            elif field == 'power':
                if stop_idx > start_idx:
                    current = self.data[(start_idx * 2):(stop_idx * 2):2]
                    voltage = self.data[(start_idx * 2 + 1):(stop_idx * 2):2]
                    out = current * voltage
                else:
                    i1 = self.data[(start_idx * 2)::2]
                    i2 = self.data[:(stop_idx * 2):2]
                    v1 = self.data[(start_idx * 2 + 1)::2]
                    v2 = self.data[1:(stop_idx * 2):2]
                    out = np.concatenate(((i1 * v1), (i2 * v2)))
            elif field == 'current_range':
                populate_bits()
                out = np.bitwise_and(bits, 0x0f)
            elif field == 'current_lsb':
                populate_bits()
                out = np.bitwise_and(np.right_shift(bits, 4), 1)
            elif field == 'voltage_lsb':
                populate_bits()
                out = np.bitwise_and(np.right_shift(bits, 5), 1)
            else:
                raise ValueError(f'Unsupported field {field}')
            value_dict['value'] = out
            result['signals'][field] = value_dict
        if is_single_result:
            return result['signals'][fields[0]]['value']
        return result

    def get_reduction(self, idx, start, stop):
        """Get reduction data directly (for testing only).

        :param idx: The reduction index.
        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :return: The The np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE)
            reduction data which normally is memory mapped to the underlying
            data, but will be copied on rollover.

        This method should not be used by production code.  Use
        :meth:`data_get` or :meth:`samples_get`.
        """
        total_length = self.length
        if stop - start > total_length:
            raise ValueError('requested size is too large')
        reduction = 1
        for i in range(idx + 1):
            reduction *= self.reductions[i].samples_per_step
        r_len = self.reductions[idx].length
        start = (start % total_length) // reduction
        stop = (stop % total_length) // reduction
        k = stop - start
        r = self.reductions_data[idx]
        if k == 0:
            return _stats_array_factory(0, NULL)
        elif k < 0:  # copy on wrap
            k += r_len
            d = _stats_array_factory(k, NULL)
            d[:(r_len - start)] = r[start:]
            d[r_len - start:] = r[:stop]
            return d
        else:
            return r[start:stop]

    @staticmethod
    cdef void _on_cbk(void * user_data, c_running_statistics.statistics_s * stats):
        cdef StreamBuffer self = <object> user_data
        if callable(self._callback):
            sample_count = self.reductions[self.reduction_count - 1].samples_per_reduction_sample
            end_id = self.processed_sample_id
            start_id = end_id - sample_count
            duration = sample_count / self._sampling_frequency  # seconds
            start_time = start_id / self._sampling_frequency
            end_time = start_time + duration
            charge_picocoulomb = (stats[0].m * 1e12)  * duration
            energy_picojoules = (stats[2].m * 1e12) * duration
            if isfinite(charge_picocoulomb) and isfinite(energy_picojoules):
                self._charge_picocoulomb += int(charge_picocoulomb)
                self._energy_picojoules += int(energy_picojoules)
            charge = self._charge_picocoulomb * 1e-12
            energy = self._energy_picojoules * 1e-12
            data = _stats_to_api(stats, start_time, end_time)
            data['time']['sample_range'] = {'value': [start_id, end_id], 'units': 'input_samples'}
            data['accumulators'] = {
                'charge' : {
                    'value': charge,
                    'units': 'C',
                },
                'energy' : {
                    'value': energy,
                    'units': 'J',
                },
            }
            self._callback(data)


include "downsampling_stream_buffer.pxi"


def usb_packet_factory(packet_index, count=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :return: The bytes containing the packet data
    """
    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    frame = np.empty(512 * count, dtype=np.uint8)
    for i in range(count):
        idx = packet_index + i
        k = i * 512
        frame[k + 0] = 1     # packet type raw
        frame[k + 1] = 0     # status = 0
        frame[k + 2] = 0x00  # length LSB
        frame[k + 3] = 0x02  # length MSB
        frame[k + 4] = idx & 0xff
        frame[k + 5] = (idx >> 8) & 0xff
        frame[k + 6] = 0
        frame[k + 7] = 0
        k += 8
        for j in range(126 * 2):
            v = (idx * 126 * 2 + j) << 2
            if j & 1:
                v |= j & 0x0002
            frame[k + j * 2] = v & 0xff
            frame[k + j * 2 + 1] = (v >> 8) & 0xff
    return frame


cpdef usb_packet_factory_signal(packet_index, count=None, samples_total=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :param samples_total: The total number samples in the signal.  This value
        is used to unsure uniqueness.
    :return: The bytes containing the packet data
    """
    cdef uint16_t ij
    cdef uint16_t vj
    cdef float slope
    cdef uint64_t sample_offset = 0
    cdef int i
    cdef int j
    cdef int k
    cdef int z


    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    sample_rate = 2000000
    samples_total = sample_rate * 100 if samples_total is None else int(samples_total)
    slope = (2 ** 14 - 1) / samples_total
    stream_buffer = StreamBuffer(0.1, [100], sample_rate)

    cdef frame = np.empty(512 * count, dtype=np.uint8)
    for i in range(count):
        packet_idx = packet_index + i
        k = i * 512
        frame[k + 0] = 1     # packet type raw
        frame[k + 1] = 0     # status = 0
        frame[k + 2] = 0x00  # length LSB
        frame[k + 3] = 0x02  # length MSB
        frame[k + 4] = packet_idx & 0xff
        frame[k + 5] = (packet_idx >> 8) & 0xff
        frame[k + 6] = 0
        frame[k + 7] = 0
        k += 8
        for j in range(SAMPLES_PER_PACKET):
            ij = <uint16_t> ((<float> sample_offset) * slope)
            vj = 16383 - ij
            ij = (ij << 2)
            vj = (vj << 2) | 0x02
            z = k + j * 4
            frame[z + 0] = ij & 0xff
            frame[z + 1] = (ij >> 8) & 0xff
            frame[z + 2] = vj & 0xff
            frame[z + 3] = (vj >> 8) & 0xff
    return frame


cpdef single_stat_to_api(v_mean, v_var, v_min, v_max, units):
    """Create statistics for a single field.
    
    :param v_mean: The mean.
    :param v_var: The variance.
    :param v_min: The minimum value.
    :param v_max: The maximum value.
    :param units: The units for v_mean, v_var, v_min, and v_max.
    :return: The dict suitable for use in the statistics data structure.
    """
    return {
        'µ': {'value': v_mean, 'units': units},
        'σ2': {'value': v_var, 'units': units},
        'min': {'value': v_min, 'units': units},
        'max': {'value': v_max, 'units': units},
        'p2p': {'value': v_max - v_min, 'units': units},
    }


cdef _stats_to_api(c_running_statistics.statistics_s * stats, t_start, t_stop):
    t_start = float(t_start)
    t_stop = float(t_stop)
    dt = t_stop - t_start
    if stats is NULL:
        k = 0
    else:
        k = int(stats[0].k)
    data = {
        'time': {
            'range': {'value': [t_start, t_stop], 'units': 's'},
            'delta': {'value': dt, 'units': 's'},
            'samples': {'value': k, 'units': 'samples'},
        },
        'signals': {},
        'source': 'stream_buffer',
    }
    if stats is not NULL:
        for i, (signal, units, integral_units) in enumerate(_SIGNALS):
            if k == 0:
                v_mean, v_var, v_min, v_max = NAN, NAN, NAN, NAN
            else:
                v_mean = float(stats[i].m)
                v_var = float(c_running_statistics.statistics_var(&stats[i]))
                v_min, v_max = float(stats[i].min), float(stats[i].max)
            data['signals'][signal] = single_stat_to_api(v_mean, v_var, v_min, v_max, units)
            if integral_units is not None:
                data['signals'][signal]['∫'] = {'value': v_mean * dt, 'units': integral_units}
    return data

def stats_to_api(stats, t_start, t_stop):
    """Convert StreamBuffer statistics to API statistics.

    :param stats: The np.ndarray(FIELD_COUNT, dtype=DTYPE), such as returned
        by :method:`StreamBuffer.stats_get`.
    :param t_start: The start time in seconds.
    :param t_stop: The stop time in seconds.
    :return: The API statistics data structure.

    The statistics data structure looks like:

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
          }
        }
    """
    cdef c_running_statistics.statistics_s [::1] s = stats
    if stats is None:
        return _stats_to_api(NULL, t_start, t_stop)
    else:
        return _stats_to_api(&s[0], t_start, t_stop)
