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

# cython: boundscheck=False, wraparound=False, nonecheck=False, overflowcheck=False, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t
from libc.float cimport FLT_MAX, FLT_MIN
from libc.math cimport isfinite, NAN

from libc.string cimport memset, memcpy
import logging
import numpy as np
cimport numpy as np


DEF PACKET_TOTAL_SIZE = 512
DEF PACKET_HEADER_SIZE = 8
DEF PACKET_PAYLOAD_SIZE = PACKET_TOTAL_SIZE - PACKET_HEADER_SIZE
DEF PACKET_INDEX_MASK = 0xffff
DEF PACKET_INDEX_WRAP = PACKET_INDEX_MASK + 1

DEF REDUCTION_MAX = 20
DEF SAMPLES_PER_PACKET = PACKET_PAYLOAD_SIZE / (2 * 2)

DEF RAW_SAMPLE_SZ = 2 * 2  # sizeof(uint16_t)
DEF CAL_SAMPLE_SZ = 2 * 4  # sizeof(float)

DEF STATS_FIELDS = 3  # current, voltage, power
DEF STATS_VALUES = 4  # mean, variance, min, max
DEF STATS_FLOATS_PER_SAMPLE = STATS_FIELDS * STATS_VALUES

log = logging.getLogger(__name__)


NAME_TO_COLUMN = {
    'current': 0,
    'i': 0,
    'i_raw': 0,
    'voltage': 1,
    'v': 1,
    'v_raw': 1,
    'power': 1,
    'p': 1,
}


ctypedef void (*js_stream_buffer_cbk)(void * user_data, float * stats)


cdef struct js_stream_buffer_calibration_s:
    float current_offset[8]
    float current_gain[8]
    float voltage_offset[2]
    float voltage_gain[2]


cdef struct js_stream_buffer_reduction_s:
    int32_t enabled
    uint32_t samples_per_step
    uint32_t sample_counter
    uint32_t length
    js_stream_buffer_cbk cbk_fn
    void * cbk_user_data
    float *data    # data[length][3][4]  as [sample][i, v, power][mean, var, min, max]


cdef void stats_compute_reset(float stats[STATS_FIELDS][STATS_VALUES]):
    for i in range(STATS_FIELDS):
        stats[i][0] = 0.0  # mean
        stats[i][1] = 0.0  # variance
        stats[i][2] = FLT_MAX  # min
        stats[i][3] = FLT_MIN  # max


cdef void stats_compute_one(float stats[STATS_FIELDS][STATS_VALUES],
                            float current,
                            float voltage):
    stats[0][0] += current
    if current < stats[0][2]:
        stats[0][2] = current
    if current > stats[0][3]:
        stats[0][3] = current

    stats[1][0] += voltage
    if voltage < stats[1][2]:
        stats[1][2] = voltage
    if voltage > stats[1][3]:
        stats[1][3] = voltage

    cdef float power = current * voltage
    stats[2][0] += power
    if power < stats[2][2]:
        stats[2][2] = power
    if power > stats[2][3]:
        stats[2][3] = power


cdef void stats_compute_end(float stats[STATS_FIELDS][STATS_VALUES],
                            float * data, uint32_t data_length,
                            uint64_t sample_id,
                            uint32_t length, uint32_t valid_length):
    cdef uint32_t k
    cdef uint32_t idx = sample_id % data_length
    # compute mean
    cdef float scale = (<float> 1.0) / (<float> valid_length)
    stats[0][0] *= scale
    stats[1][0] *= scale
    stats[2][0] *= scale

    # compute variance
    cdef float i_mean = stats[0][0]
    cdef float v_mean = stats[1][0]
    cdef float p_mean = stats[2][0]
    cdef float i_var = 0.0
    cdef float v_var = 0.0
    cdef float p_var = 0.0
    cdef float t
    for count in range(length):
        k = 2 * idx
        if isfinite(data[k]):
            t = data[k] - i_mean
            i_var += t * t
            t = data[k + 1] - v_mean
            v_var += t * t
            t = data[k] * data[k + 1] - p_mean
            p_var += t * t
        idx += 1
        if idx >= data_length:
            idx = 0
    stats[0][1] = i_var * scale
    stats[1][1] = v_var * scale
    stats[2][1] = p_var * scale


cdef uint32_t stats_compute_run(
        float stats[STATS_FIELDS][STATS_VALUES],
        float * data, uint32_t data_length,
        uint64_t sample_id, uint32_t length):
    cdef uint32_t idx = sample_id % data_length
    cdef uint32_t data_idx
    cdef uint32_t counter = 0
    stats_compute_reset(stats)
    for i in range(length):
        data_idx = idx * 2
        if isfinite(data[data_idx]):
            stats_compute_one(stats, data[data_idx], data[data_idx + 1])
            counter += 1
        idx += 1
        if idx >= data_length:
            idx = 0
    stats_compute_end(stats, data, data_length, sample_id, length, counter)
    return counter


cdef uint32_t reduction_stats(js_stream_buffer_reduction_s * r,
        float stats[STATS_FIELDS][STATS_VALUES], uint32_t idx_start, uint32_t length):
    cdef uint32_t count
    cdef uint32_t j
    cdef uint32_t valid = 0
    cdef uint32_t idx = idx_start
    cdef float * f
    cdef float scale
    cdef float i_mean
    cdef float v_mean
    cdef float p_mean
    cdef float i_var
    cdef float v_var
    cdef float p_var
    cdef float dv

    stats_compute_reset(stats)
    for count in range(length):
        f = r.data + idx * STATS_FLOATS_PER_SAMPLE
        if isfinite(f[0]):
            valid += 1
            for j in range(STATS_FIELDS):
                stats[j][0] += f[0]
                if f[2] < stats[j][2]:
                    stats[j][2] = f[2]
                if f[3] > stats[j][3]:
                    stats[j][3] = f[3]
                f += STATS_VALUES
        idx += 1
        if idx >= r.length:
            idx = 0

    if 0 == valid:
        for j in range(STATS_FIELDS):
            stats[j][0] = NAN
            stats[j][1] = NAN
            stats[j][2] = NAN
            stats[j][3] = NAN
    else:
        scale = (<float> 1.0) / (<float> valid)
        stats[0][0] *= scale
        stats[1][0] *= scale
        stats[2][0] *= scale

        idx = idx_start
        i_mean = stats[0][0]
        v_mean = stats[1][0]
        p_mean = stats[2][0]
        i_var = 0.0
        v_var = 0.0
        p_var = 0.0
        for count in range(length):
            f = r.data + idx * STATS_FLOATS_PER_SAMPLE
            if isfinite(f[0]):
                dv = f[0] - i_mean
                i_var += f[1] + dv * dv
                dv = f[4] - v_mean
                v_var += f[5] + dv * dv
                dv = f[8] - p_mean
                p_var += f[9] + dv * dv
            idx += 1
            if idx >= r.length:
                idx = 0
        stats[0][1] = i_var * scale
        stats[1][1] = v_var * scale
        stats[2][1] = p_var * scale
    return valid


cdef void cal_init(js_stream_buffer_calibration_s * self):
    for i in range(8):
        self.current_offset[i] = <float> 0.0
        self.current_gain[i] = <float> 1.0
    self.current_gain[7] = NAN  # compute NAN on invalid
    for i in range(2):
        self.voltage_offset[i] = <float> 0.0
        self.voltage_gain[i] = <float> 1.0


cdef class StreamBuffer:
    """Efficient real-time Joulescope data buffering.

    :param length: The total length of the buffering in samples.
    :param reductions: The list of reduction integers.  Each integer represents
        the reduction amount for each resuting sample in units of samples
        of the previous reduction.  Reduction 0 is in raw sample units.
    """

    cdef uint32_t reduction_step
    cdef uint32_t length # in samples
    cdef uint64_t packet_index
    cdef uint64_t packet_index_offset
    cdef uint64_t device_sample_id
    cdef uint64_t processed_sample_id
    cdef uint64_t sample_missing_count  # based upon sample_id
    cdef uint64_t skip_count            # number of sample skips
    cdef uint64_t sample_sync_count     # based upon alternating 0/1 pattern
    cdef uint64_t contiguous_count      #
    cdef uint16_t *raw_ptr  # raw[length][2]   as i, v
    cdef float *data_ptr    # data[length][2]  as i, v
    cdef js_stream_buffer_calibration_s cal
    cdef js_stream_buffer_reduction_s reductions[REDUCTION_MAX]

    cdef uint32_t stats_counter  # excludes NAN for mean
    cdef uint32_t stats_remaining
    cdef float stats[STATS_FIELDS][STATS_VALUES]  # [i, v, power][mean, var, min, max]

    cdef uint16_t sample_toggle_last
    cdef uint16_t sample_toggle_mask
    cdef uint8_t voltage_range

    cdef object raw
    cdef object data
    cdef object reductions_data
    cdef uint64_t _sample_id_max  # used to automatically stop streaming
    cdef uint64_t _contiguous_max  # used to automatically stop streaming
    cdef object _callback  # fn(np.array [3][4] of statistics, energy)
    cdef object _energy_picojoules  # python integer for infinite precision


    def __cinit__(self, length, reductions):
        if length < SAMPLES_PER_PACKET:
            raise ValueError('length to small')
        if len(reductions) > REDUCTION_MAX:
            raise ValueError('too many reductions')

        self._energy_picojoules = 0
        memset(self.reductions, 0, sizeof(self.reductions))

        # round up length to multiple of reductions
        self.reduction_step = int(np.prod(reductions))
        length = int(np.ceil(length / self.reduction_step)) * self.reduction_step
        self.length = length
        cal_init(&self.cal)

        self.raw = np.empty((length * 2), dtype=np.uint16)
        self.raw = np.ascontiguousarray(self.raw, dtype=np.uint16)
        cdef np.ndarray[np.uint16_t, ndim=1, mode = 'c'] raw_c = self.raw
        self.raw_ptr = <uint16_t *> raw_c.data

        self.data = np.empty((length * 2), dtype=np.float32)
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] data_c = self.data
        self.data_ptr = <float *> data_c.data

        self.reductions_data = []
        cdef js_stream_buffer_reduction_s * r
        cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] reduction_data_c
        sz = length

        for idx, rsamples in enumerate(reductions):
            sz = sz // rsamples
            r = &self.reductions[idx]
            r.enabled = 1
            r.samples_per_step = <uint32_t> rsamples
            r.length = sz

            d = np.empty((sz, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
            d = np.ascontiguousarray(d, dtype=np.float32)
            reduction_data_c = d
            r.data = <float *> reduction_data_c.data
            self.reductions_data.append(d)

        if len(reductions):
            self.reductions[len(reductions) - 1].cbk_fn = _on_cbk
            self.reductions[len(reductions) - 1].cbk_user_data = <void *> self

        self._sample_id_max = 0  # used to automatically stop streaming
        self._contiguous_max = 0  # used to automatically stop streaming
        self._callback = None  # fn(np.array [3][4] of statistics, energy)
        self._energy_picojoules = 0  # integer for infinite precision

    def __init__(self, length, reductions):
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
    def sample_id_range(self):
        """Get the range of sample ids currently available in the buffer.

        :return: Tuple of sample_id start, sample_id end.
        """
        s_end = int(self.processed_sample_id)
        s_start = s_end - self.length
        if s_start < 0:
            s_start = 0
        return s_start, s_end

    @property
    def data_buffer(self):
        return self.data  # the cdef np.ndarray

    @property
    def sample_id_max(self):
        return self._sample_id_max

    @sample_id_max.setter
    def sample_id_max(self, value):
        self._sample_id_max = value

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

    def status(self):
        return {
            'device_sample_id': {'value': self.device_sample_id, 'units': 'samples'},
            'sample_id': {'value': self.processed_sample_id, 'units': 'samples'},
            'sample_missing_count': {'value': self.sample_missing_count, 'units': 'samples'},
            'skip_count': {'value': self.skip_count, 'units': ''},
            'sample_sync_count': {'value': self.sample_sync_count, 'units': 'samples'},
            'contiguous_count': {'value': self.contiguous_count, 'units': 'samples'},
        }

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        cdef js_stream_buffer_calibration_s * cal = &self.cal
        for i in range(7):
            cal.current_offset[i] = current_offset[i]
            cal.current_gain[i] = current_gain[i]
        cal.current_offset[7] = NAN
        cal.current_gain[7] = NAN
        for i in range(2):
            cal.voltage_offset[i] = voltage_offset[i]
            cal.voltage_gain[i] = voltage_gain[i]

    cdef _stats_reset(self):
        self.stats_counter = 0
        self.stats_remaining = 0
        if self.reductions[0].enabled:
            self.stats_remaining = self.reductions[0].samples_per_step
        stats_compute_reset(self.stats)

    def reset(self):
        self.packet_index = 0
        self.packet_index_offset = 0
        self.device_sample_id = 0
        self.processed_sample_id = 0
        self.sample_missing_count = 0
        self.skip_count = 0
        self.sample_sync_count = 0
        self.contiguous_count = 0
        self.sample_toggle_last = 0
        self.sample_toggle_mask = 0
        self.voltage_range = 0
        self._sample_id_max = 1 << 63  # big enough
        self._contiguous_max = 1 << 63  # big enough
        self._energy_picojoules = 0
        self.stats_counter = 0
        for idx in range(REDUCTION_MAX):
            self.reductions[idx].sample_counter = 0
        self._stats_reset()

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
        cdef float stats[STATS_FIELDS][STATS_VALUES]
        cdef uint32_t samples_per_step
        cdef uint32_t idx_target
        cdef uint32_t idx_start
        cdef uint32_t data_idx

        if 0 == r.enabled:
            return
        r.sample_counter += 1
        if r.sample_counter >= r.samples_per_step:
            r.sample_counter = 0
            samples_per_step = parent_samples_per_step * r.samples_per_step
            idx_target = self.reduction_index(r, parent_samples_per_step)
            idx_start = idx_target * r.samples_per_step
            data_idx = idx_target * <uint32_t> (sizeof(stats) / sizeof(float))
            reduction_stats(&self.reductions[n - 1], stats, idx_start, r.samples_per_step)
            memcpy(r.data + data_idx, stats, sizeof(stats))
            if r.cbk_fn:
                r.cbk_fn(r.cbk_user_data, r.data + data_idx)
            self.reduction_update_n(n + 1, samples_per_step)

    cdef void reduction_update_0(self):
        cdef js_stream_buffer_reduction_s * r = &self.reductions[0]
        if 0 == r.enabled:
            return
        cdef uint32_t idx = self.reduction_index(r, 1)
        memcpy(r.data + idx * sizeof(self.stats) / sizeof(float), self.stats, sizeof(self.stats))
        self.reduction_update_n(1, self.reductions[0].samples_per_step)

    cdef void stats_finalize(self):
        if 0 == self.stats_counter:
            for i in range(STATS_FIELDS):
                self.stats[i][0] = NAN
                self.stats[i][1] = NAN
                self.stats[i][2] = NAN
                self.stats[i][3] = NAN
            return
        cdef uint32_t length = self.reductions[0].samples_per_step
        cdef uint32_t idx = self.reduction_index(&self.reductions[0], 1)
        idx *= self.reductions[0].samples_per_step
        stats_compute_end(self.stats, self.data_ptr, self.length, idx, length, self.stats_counter)

    cdef void _insert_usb_bulk(self, const uint8_t *data, size_t length):
        cdef uint8_t buffer_type
        cdef uint8_t status
        cdef uint16_t pkt_length
        cdef uint64_t pkt_index
        cdef uint64_t sample_id
        cdef uint32_t idx
        cdef uint32_t idx2
        cdef uint32_t samples
        cdef uint32_t samples_to_end

        while length >= PACKET_TOTAL_SIZE:
            buffer_type = data[0]
            status = data[1]
            pkt_length = data[2] | ((<uint16_t> data[3] & 0x7f) << 8)
            self.voltage_range = <uint8_t> ((data[3] >> 7) & 0x01)
            pkt_index = <uint64_t> (data[4] | ((<uint16_t> data[5]) << 8))
            # uint16_t usb_frame_index = data[6] | ((<uint16_t> data[7]) << 8)
            length -= PACKET_TOTAL_SIZE

            if (1 != buffer_type) or (0 != status) or (PACKET_TOTAL_SIZE != pkt_length):
                data += PACKET_TOTAL_SIZE
                continue
            pkt_index += self.packet_index_offset
            while pkt_index < self.packet_index:
                pkt_index += PACKET_INDEX_WRAP
                self.packet_index_offset += PACKET_INDEX_WRAP
            sample_id = pkt_index * SAMPLES_PER_PACKET
            idx = self.device_sample_id % self.length

            if sample_id < self.device_sample_id:
                log.warning("WARNING: duplicate data")
            elif self.device_sample_id < sample_id:
                log.info("Fill missing samples")
                self.skip_count += 1
                self.contiguous_count = 0
                while self.device_sample_id < sample_id:
                    idx2 = idx * 2
                    self.raw[idx2 + 0] = 0x3
                    self.raw[idx2 + 1] = 0x3
                    idx += 1
                    if idx >= self.length:
                        idx = 0
                    self.device_sample_id += 1
                    self.sample_missing_count += 1

            samples = SAMPLES_PER_PACKET
            self.contiguous_count += samples
            data += PACKET_HEADER_SIZE  # skip header
            if (idx + SAMPLES_PER_PACKET) > self.length:
                samples_to_end = self.length - idx
                memcpy(&self.raw_ptr[idx * 2], data, samples_to_end * RAW_SAMPLE_SZ)
                idx = 0
                data += samples_to_end * RAW_SAMPLE_SZ
                samples -= samples_to_end
            memcpy(&self.raw_ptr[idx * 2], data, samples * RAW_SAMPLE_SZ)
            data += samples * RAW_SAMPLE_SZ
            self.device_sample_id = sample_id + SAMPLES_PER_PACKET
            self.packet_index += 1

    cpdef insert(self, data):
        """Insert new data into the buffer.

        :param data: The new data to insert.
        :return: False to continue streaming, True to end.
        """
        cdef np.ndarray[np.uint8_t, ndim=1, mode = 'c'] data_c
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=np.uint8)
            data_c = data
            data_ptr = <uint8_t *> data_c.data
            self._insert_usb_bulk(data_ptr, len(data))
        else:
            self._insert_usb_bulk(data, len(data))
        cdef bint duration_stop = self.device_sample_id >= self._sample_id_max
        cdef bint contiguous_stop = self.contiguous_count >= self._contiguous_max
        rv = duration_stop or contiguous_stop
        if rv:
            if duration_stop:
                log.info('insert causing duration stop %d >= %d',
                         self.device_sample_id, self._sample_id_max)
            elif duration_stop:
                log.info('insert causing contiguous stop %d >= %d',
                         self.contiguous_count, self._contiguous_max)
        return rv

    cdef void _process(self):
        cdef uint32_t idx_start
        cdef uint32_t idx
        cdef uint16_t raw_i
        cdef uint16_t raw_v
        cdef uint8_t i_range
        cdef uint16_t sample_toggle_current
        cdef uint64_t sample_sync_count
        cdef float cal_i
        cdef float cal_v

        if self.processed_sample_id + self.length < self.device_sample_id:
            log.warning('process: stream_buffer is behind')
            self.processed_sample_id = self.device_sample_id - self.length
        idx_start = <uint32_t> (self.processed_sample_id % self.length)

        while self.processed_sample_id < self.device_sample_id:
            idx = idx_start * 2
            raw_i = self.raw_ptr[idx + 0]
            raw_v = self.raw_ptr[idx + 1]
            i_range = <uint8_t> ((raw_i & 0x0003) | ((raw_v & 0x0001) << 2))
            sample_toggle_current = (raw_v >> 1) & 0x1
            sample_sync_count = (sample_toggle_current ^ self.sample_toggle_last ^ 1) & \
                    self.sample_toggle_mask
            if sample_sync_count:
                self.contiguous_count = 0
            self.sample_sync_count += sample_sync_count
            self.sample_toggle_last = sample_toggle_current
            self.sample_toggle_mask = 0x1
            cal_i = <float> (raw_i >> 2)
            cal_i += self.cal.current_offset[i_range]
            cal_i *= self.cal.current_gain[i_range]

            cal_v = <float> (raw_v >> 2)
            cal_v += self.cal.voltage_offset[self.voltage_range]
            cal_v *= self.cal.voltage_gain[self.voltage_range]
            if not isfinite(cal_i):
                cal_v = NAN
            self.data_ptr[idx + 0] = cal_i
            self.data_ptr[idx + 1] = cal_v

            self.stats_counter += 1
            stats_compute_one(self.stats, cal_i, cal_v)

            self.processed_sample_id += 1
            idx_start += 1
            if idx_start >= self.length:
                idx_start = 0

            if self.stats_remaining > 1:
                self.stats_remaining -= 1
            elif self.stats_remaining == 1:
                self.stats_finalize()
                self.reduction_update_0()
                self._stats_reset()

    def process(self):
        self._process()

    cdef int range_check(self, uint64_t start, uint64_t stop):
        if stop <= start:
            log.warning("js_stream_buffer_get stop <= start")
            return 0
        if start > self.processed_sample_id:
            log.warning("js_stream_buffer_get start newer that current")
            return 0
        if stop > self.processed_sample_id:
            log.warning("js_stream_buffer_get stop newer than current")
            return 0
        return 1

    cdef uint32_t _data_get(self, float * buffer, uint32_t buffer_samples,
                            uint64_t start, uint64_t stop, uint32_t increment):
        """Get the summarized statistics over a range.
        
        :param buffer: The Nx3x4 buffer to populate.
        :param buffer_samples: The size N of the buffer in units of 3x4 float samples.
        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :return: The number of samples placed into buffer.
        """
        cdef uint32_t buffer_samples_orig = buffer_samples
        cdef uint32_t idx
        cdef uint32_t data_offset
        cdef float stats[STATS_FIELDS][STATS_VALUES]
        cdef uint32_t count
        cdef uint32_t samples_per_step
        cdef uint32_t samples_per_step_next
        cdef uint32_t length
        cdef uint32_t idx_start
        cdef int n

        start = (start / increment) * increment
        stop = (stop / increment) * increment
        if not self.range_check(start, stop):
            return 0

        # Fill in too old of data with NAN
        while (start + self.length) < self.device_sample_id:
            if buffer_samples == 0:
                log.warning('_data_get filled with NaN')
                return buffer_samples_orig
            for j in range(STATS_FLOATS_PER_SAMPLE):
                buffer[j] = NAN
            buffer += STATS_FLOATS_PER_SAMPLE
            buffer_samples -= 1
            start += increment
        if buffer_samples != buffer_samples_orig:
            log.warning('_data_get filled %s', buffer_samples_orig - buffer_samples)

        if increment <= 1:
            # direct copy
            idx = start % self.length
            while start != stop and buffer_samples:
                data_offset = idx * 2
                buffer[0] = self.data_ptr[data_offset]
                buffer[1] = <float> 0.0
                buffer[2] = NAN
                buffer[3] = NAN
                buffer[4] = self.data_ptr[data_offset + 1]
                buffer[5] = <float> 0.0
                buffer[6] = NAN
                buffer[7] = NAN
                buffer[8] = self.data_ptr[data_offset] * self.data_ptr[data_offset + 1]
                buffer[9] = <float> 0.0
                buffer[10] = NAN
                buffer[11] = NAN
                buffer_samples -= 1
                idx += 1
                start += 1
                buffer += STATS_FLOATS_PER_SAMPLE
                if idx >= self.length:
                    idx = 0
        elif not self.reductions[0].enabled or (self.reductions[0].samples_per_step > increment):
            # compute over raw data.
            while start + increment <= stop and buffer_samples:
                count = stats_compute_run(stats, self.data_ptr, self.length, start, increment)
                memcpy(buffer, stats, sizeof(stats))
                buffer += STATS_FLOATS_PER_SAMPLE
                start += increment
                buffer_samples -= 1
        else:
            # use reductions
            samples_per_step = 1
            for n in range(REDUCTION_MAX):
                samples_per_step_next = samples_per_step * self.reductions[n].samples_per_step
                if not self.reductions[n].enabled or samples_per_step_next > increment:
                    break
                samples_per_step = samples_per_step_next
            if n < 1:
                raise RuntimeError('could not find reduction')
            n = n - 1
            start = (start / samples_per_step) * samples_per_step
            while start + increment <= stop and buffer_samples:
                length = <uint32_t> ((start + increment) / samples_per_step - start / samples_per_step)
                idx_start = <uint32_t> ((start % self.length) / samples_per_step)
                reduction_stats(&self.reductions[n], stats, idx_start, length)
                memcpy(buffer, stats, sizeof(stats))
                buffer += STATS_FLOATS_PER_SAMPLE
                start += increment
                buffer_samples -= 1
        return buffer_samples_orig - buffer_samples

    def data_get(self, start, stop, increment=None, out=None):
        """Get the samples with statistics.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :param out: The optional output np.ndarray(N, 3, 4) to populate with
            the result.  None (default) will construct and return a new array.
        :return: The np.ndarray((N, 3, 4), dtype=np.float32) data of
            (length, fields, values) with
            fields (current, voltage, power) and
            values (mean, variance, min, max).
        """
        increment = 1 if increment is None else int(increment)
        if start >= stop:
            log.info('data_get: start >= stop')
            return np.empty((0, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        expected_length = (stop - start) // increment
        if out is None:
            out = np.empty((expected_length, STATS_FIELDS, STATS_VALUES), dtype=np.float32)

        #out = np.ascontiguousarray(out, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=3, mode = 'c'] out_c = out
        out_ptr = <float *> out_c.data

        length = self._data_get(out_ptr, len(out), start, stop, increment)
        if length != expected_length:
            log.warning('length mismatch: expected=%s, returned=%s', expected_length, length)
        return out[:length, :, :]

    def raw_get(self, start, stop):
        """Get the raw data from Joulescope.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :return: The np.ndarray((2 * N), dtype=np.uint16) data of
            interleaved current, voltage left-shifted 14-bit samples.
            The least significant 2 bits contain current range select
            information.
        """
        if stop <= start:
            log.warning('raw %d <= %d', start, stop)
            return np.empty((0, 2), dtype=np.uint16)
        total_length = self.length
        start_idx = start % total_length
        stop_idx = stop % total_length
        if 0 == stop_idx:
            stop_idx = total_length
        if stop_idx > start_idx:
            return self.raw[(start_idx * 2):(stop_idx * 2)].reshape((-1, 2))
        # on wrap, have to copy
        expected_length = stop - start
        samples1 = self.length - start_idx
        samples2 = expected_length - samples1
        out = np.empty((expected_length, 2), dtype=np.uint16)
        raw = self.raw.reshape((-1, 2))
        out[:samples1, :] = raw[start_idx:, :]
        out[samples1:, :] = raw[:samples2, :]
        return out

    def get_reduction(self, idx, start, stop):
        """Get reduction data directly.

        :param idx: The reduction index.
        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :return: The reduction data which normally is memory mapped to the
            underlying data, but will be copied on rollover.
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
            return np.empty((0, 3, 4), dtype=np.float32)
        elif k < 0:  # copy on wrap
            k += r_len
            d = np.empty((k, 3, 4), dtype=np.float32)
            d[:(r_len - start), :, :] = r[start:, :, :]
            d[r_len - start:, :, :] = r[:stop, :, :]
            return d
        else:
            return r[start:stop, :, :]

    cdef _on_cbk(self, float * stats):
        if callable(self._callback):
            b = np.empty(12, dtype=np.float32)
            for i in range(12):
                b[i] = stats[i]
            b = b.reshape((3, 4))
            # todo handle variable sampling frequencies and reductions
            time_interval = 0.5  # seconds
            power_picowatts = b[2][0] * 1e12
            energy_picojoules = power_picowatts * time_interval
            if isfinite(energy_picojoules):
                self._energy_picojoules += int(energy_picojoules)
            energy = self._energy_picojoules * 1e-12
            self._callback(b, energy)


cdef void _on_cbk(void * user_data, float * stats):
    cdef StreamBuffer self = <object> user_data
    self._on_cbk(stats)


def usb_packet_factory(packet_index, count=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :return: The bytes containing the packet data
    """
    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    frame = np.empty((packet_index + 1) * 512 * count, dtype=np.uint8)
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