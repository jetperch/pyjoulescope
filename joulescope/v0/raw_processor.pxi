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


DEF _SUPPRESS_SAMPLES_MAX = 64  # must be power of 2
DEF _SUPPRESS_SAMPLES_MASK = (_SUPPRESS_SAMPLES_MAX - 1)
DEF SUPPRESS_WINDOW_MAX = 12
DEF SUPPRESS_PRE_MAX = 8
DEF SUPPRESS_POST_MAX = 8
DEF _I_RANGE_MISSING = 8

DEF SUPPRESS_MODE_OFF = 0    # disabled, force zero delay
DEF SUPPRESS_MODE_MEAN = 1
DEF SUPPRESS_MODE_INTERP = 2
DEF SUPPRESS_MODE_NAN = 3


SUPPRESS_SAMPLES_MAX = _SUPPRESS_SAMPLES_MAX
I_RANGE_MISSING = _I_RANGE_MISSING

# experimentally determined charge coupling durations in samples at 2 MSPS
# These values are aggressive and result in min/max distortion
cdef uint8_t[9][9] SUPPRESS_MATRIX_M = [   # [to][from]
    #0, 1, 2, 3, 4, 5, 6, 7, 8    # from this current select
    [0, 5, 5, 5, 5, 5, 6, 6, 0],  # to 0
    [3, 0, 5, 5, 5, 6, 7, 8, 0],  # to 1
    [4, 4, 0, 6, 6, 7, 7, 8, 0],  # to 2
    [4, 4, 4, 0, 6, 6, 7, 7, 0],  # to 3
    [4, 4, 4, 4, 0, 6, 7, 6, 0],  # to 4
    [4, 4, 4, 4, 4, 0, 7, 6, 0],  # to 5
    [4, 4, 4, 4, 4, 4, 0, 6, 0],  # to 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # to 7 (off)
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # to 8 (missing)
]

# experimentally determined charge coupling durations in samples at 2 MSPS
# These values are more conservative for less min/max distortion.
cdef uint8_t[9][9] SUPPRESS_MATRIX_N = [   # [to][from]
    #0, 1, 2, 3, 4, 5, 6, 7, 8    # from this current select
    [0, 5, 7, 7, 7, 7, 7, 8, 0],  # to 0
    [3, 0, 7, 7, 7, 7, 7, 8, 0],  # to 1
    [5, 5, 0, 7, 7, 7, 7, 8, 0],  # to 2
    [5, 5, 5, 0, 7, 7, 7, 8, 0],  # to 3
    [5, 5, 5, 5, 0, 7, 7, 8, 0],  # to 4
    [5, 5, 5, 5, 5, 0, 7, 8, 0],  # to 5
    [5, 5, 5, 5, 5, 5, 0, 8, 0],  # to 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # to 7 (off)
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # to 8 (missing)
]


cdef struct js_stream_buffer_calibration_s:
    float current_offset[8]
    float current_gain[8]
    float voltage_offset[2]
    float voltage_gain[2]



cdef void cal_init(js_stream_buffer_calibration_s * self):
    cdef uint8_t i
    for i in range(8):
        self.current_offset[i] = <float> 0.0
        self.current_gain[i] = <float> 1.0
    self.current_gain[7] = 0.0  # always compute zero current when off
    for i in range(2):
        self.voltage_offset[i] = <float> 0.0
        self.voltage_gain[i] = <float> 1.0


ctypedef void (*raw_processor_cbk_fn)(object user_data, float cal_i, float cal_v, uint8_t bits)


cdef class RawProcessor:
    # FIFO circular buffer for current range change filtering
    cdef float d_cal[_SUPPRESS_SAMPLES_MAX][2]  # as i, v
    cdef uint8_t d_bits[_SUPPRESS_SAMPLES_MAX]   # packed bits: 7:6=0 , 5=voltage_lsb, 4=current_lsb, 3:0=i_range
    cdef js_stream_buffer_calibration_s _cal
    cdef float cal_i_pre
    cdef raw_processor_cbk_fn _cbk_fn
    cdef object _cbk_user_data

    cdef int32_t is_skipping
    cdef int32_t _idx_suppress_start
    cdef int32_t _idx_out
    cdef uint64_t sample_missing_count  # based upon sample_id
    cdef uint64_t skip_count            # number of sample skips
    cdef uint64_t sample_sync_count     # based upon alternating 0/1 pattern
    cdef uint64_t contiguous_count      #

    cdef uint8_t _i_range_last
    cdef int32_t _suppress_samples_pre     # the number of samples to use before range change
    cdef int32_t _suppress_samples_window  # the total number of samples to suppress after range change
    cdef int32_t _suppress_samples_post
    cdef uint8_t (*_suppress_matrix)[9][9]

    cdef int32_t _suppress_samples_remaining  # the suppress counter, 0 = idle, 1 = take action, >1 = decrement
    cdef int32_t _suppress_samples_counter
    cdef uint8_t _suppress_mode

    cdef uint16_t sample_toggle_last
    cdef uint16_t sample_toggle_mask
    cdef uint8_t _voltage_range

    cdef uint16_t * bulk_raw
    cdef float * bulk_cal
    cdef uint8_t * bulk_bits
    cdef uint32_t bulk_index
    cdef uint32_t bulk_length  # in samples

    def __cinit__(self):
        cal_init(&self._cal)
        self._suppress_samples_pre = 1
        self._suppress_samples_window = 0  # use N
        self._suppress_samples_post = 1
        self._suppress_mode = SUPPRESS_MODE_INTERP
        self._suppress_matrix = &SUPPRESS_MATRIX_N

    def __init__(self):
        self.reset()

    cdef callback_set(self, raw_processor_cbk_fn cbk, object user_data):
        self._cbk_fn = cbk
        self._cbk_user_data = user_data

    @property
    def voltage_range(self):
        return self._voltage_range

    @voltage_range.setter
    def voltage_range(self, value):
        """Set the voltage range for applying calibration.

        Note that the Joulescope device normally conveys the voltage range in
        along with the sample data.
        """
        self._voltage_range = value

    @property
    def suppress_mode(self):
        if self._suppress_mode == SUPPRESS_MODE_OFF:
            return 'off'
        if self._suppress_mode == SUPPRESS_MODE_NAN:
            name = 'nan'
        elif self._suppress_mode == SUPPRESS_MODE_INTERP:
            name = 'interp'
        else:
            name = 'mean'
        if self._suppress_matrix == &SUPPRESS_MATRIX_M:
            window = 'm'
        elif self._suppress_matrix == &SUPPRESS_MATRIX_N:
            window = 'n'
        else:
            window = self._suppress_samples_window
        return f'{name}_{self._suppress_samples_pre}_{window}_{self._suppress_samples_post}'

    @suppress_mode.setter
    def suppress_mode(self, value):
        """Set the suppression mode filter.

        :param value: The specification string, which is normally in the format
            'mode_pre_window_post' where:
            * mode:
              * 'off': Use the data as provided with no filtering
              * 'nan': Insert NaNs into the window.
              * 'mean': Compute the mean over pre and post, then insert into window.
              * 'interp': Compute the mean over pre and post, then interpolate.
            * pre: The number of samples before the current range change used to
              compute the value for window.
            * window: The number of samples to modify after the current range change.
              'n' uses the characterized duration based upon the actual range switch.
            * post: The number of additional samples to use to compute
              the value for window.
        """
        self._suppress_mode = SUPPRESS_MODE_OFF
        self._suppress_samples_pre = 0
        self._suppress_samples_window = 0
        self._suppress_samples_post = 0
        self._suppress_matrix = <uint8_t (*)[9][9]> NULL

        if isinstance(value, str):
            value = value.lower()
        if value is None or not bool(value) or value == 'off':
            return
        if value == 'normal':
            value = 'interp_1_n_1'
        parts = value.split('_')
        if len(parts) != 4:
            raise ValueError(f'Invalid suppress_mode: {value}')

        pre = max(0, int(parts[1]))
        post = max(0, int(parts[3]))

        if parts[2] == 'm':
            self._suppress_matrix = &SUPPRESS_MATRIX_M
        elif parts[2] == 'n':
            self._suppress_matrix = &SUPPRESS_MATRIX_N
        else:
            window = int(parts[2])
            if window > SUPPRESS_WINDOW_MAX:
                raise ValueError(f'suppress_samples_window must be < {SUPPRESS_WINDOW_MAX}, was {window}')
            self._suppress_samples_window = window

        if parts[0] == 'nan':
            self._suppress_mode = SUPPRESS_MODE_NAN
        elif parts[0] in ['interp', 'interpolate']:
            self._suppress_mode = SUPPRESS_MODE_INTERP
            self._suppress_samples_pre = 1
            self._suppress_samples_post = 1
        elif parts[0] == 'mean':
            self._suppress_mode = SUPPRESS_MODE_MEAN
            if pre > SUPPRESS_PRE_MAX:
                raise ValueError(f'suppress_samples_pre must be < {SUPPRESS_PRE_MAX}, was {pre}')
            self._suppress_samples_pre = pre
            if post > SUPPRESS_POST_MAX:
                raise ValueError(f'suppress_samples_post must be < {SUPPRESS_POST_MAX}, was {post}')
            self._suppress_samples_post = post
            if (self._suppress_samples_pre + self._suppress_samples_post) <= 0:
                log.warning('pre and post are both 0, force to 1')
                self._suppress_samples_pre = 1
                self._suppress_samples_post = 1
        elif parts[0] == 'off':
            # allow algorithm testing
            self._suppress_mode = SUPPRESS_MODE_OFF
        else:
            raise ValueError(f'Invalid suppress_mode: {value}')

    def reset(self):
        cdef uint32_t idx
        self.sample_missing_count = 0
        self.is_skipping = 1
        self.skip_count = 0
        self.sample_sync_count = 0
        self.contiguous_count = 0

        self._suppress_samples_remaining = 0
        self._suppress_samples_counter = 0
        self._i_range_last = 7  # off

        self.sample_toggle_last = 0
        self.sample_toggle_mask = 0
        self._voltage_range = 0
        self._idx_out = 0
        self._idx_suppress_start = 0

        for idx in range(SUPPRESS_SAMPLES_MAX):
            self.d_cal[idx][0] = NAN
            self.d_cal[idx][1] = NAN
            self.d_bits[idx] = 0xff
        self.cal_i_pre = NAN

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        cdef js_stream_buffer_calibration_s * cal = &self._cal
        if len(current_offset) < 7 or len(current_gain) < 7:
            raise ValueError('current calibration vector too small')
        if len(voltage_offset) < 2 or len(voltage_gain) < 2:
            raise ValueError('voltage calibration vector too small')
        for i in range(7):
            cal.current_offset[i] = current_offset[i]
            cal.current_gain[i] = current_gain[i]
        cal.current_offset[7] = 0.0
        cal.current_gain[7] = 0.0
        for i in range(2):
            cal.voltage_offset[i] = voltage_offset[i]
            cal.voltage_gain[i] = voltage_gain[i]

    cdef void process(self, uint16_t raw_i, uint16_t raw_v):
        cdef uint32_t is_missing
        cdef uint8_t suppress_window
        cdef int32_t suppress_idx
        cdef int32_t idx
        cdef int32_t idx_wrap
        cdef uint8_t bits
        cdef uint8_t i_range
        cdef uint16_t sample_toggle_current
        cdef uint64_t sample_sync_count
        cdef float cal_i
        cdef float cal_i_step
        cdef float cal_v

        is_missing = 0
        if 0xffff == raw_i and 0xffff == raw_v:
            is_missing = 1
            i_range = _I_RANGE_MISSING  # missing sample
            self.sample_missing_count += 1
            self.contiguous_count = 0
            if self.is_skipping == 0:
                self.skip_count += 1
                self.is_skipping = 1
        else:
            i_range = <uint8_t> ((raw_i & 0x0003) | ((raw_v & 0x0001) << 2))
            self.is_skipping = 0
            self.contiguous_count += 1
        bits = (i_range & 0x0f) | ((raw_i & 0x0004) << 2) | ((raw_v & 0x0004) << 3)

        sample_toggle_current = (raw_v >> 1) & 0x1
        raw_i = raw_i >> 2
        raw_v = raw_v >> 2
        sample_sync_count = (sample_toggle_current ^ self.sample_toggle_last ^ 1) & \
                self.sample_toggle_mask
        if sample_sync_count and is_missing == 0:
            self.skip_count += 1
            self.is_skipping = 1
            self.sample_sync_count += 1
        self.sample_toggle_last = sample_toggle_current
        self.sample_toggle_mask = 0x1

        if i_range > 7:  # missing sample
            cal_i = NAN
            cal_v = NAN
            i_range = _I_RANGE_MISSING
        else:
            cal_i = <float> raw_i
            cal_i += self._cal.current_offset[i_range]
            cal_i *= self._cal.current_gain[i_range]
            cal_v = <float> raw_v
            cal_v += self._cal.voltage_offset[self._voltage_range]
            cal_v *= self._cal.voltage_gain[self._voltage_range]

        self.d_bits[self._idx_out] = bits
        self.d_cal[self._idx_out][0] = cal_i
        self.d_cal[self._idx_out][1] = cal_v

        # process i_range for glitch suppression
        if i_range != self._i_range_last:
            if SUPPRESS_MODE_OFF == self._suppress_mode:
                suppress_window = 0
            elif self._suppress_matrix is not NULL:
                suppress_window = self._suppress_matrix[0][i_range][self._i_range_last]
            else:
                suppress_window = self._suppress_samples_window
            if suppress_window:
                idx = suppress_window + self._suppress_samples_post
                if idx > self._suppress_samples_remaining:
                    self._suppress_samples_remaining = idx
            if suppress_window and self._suppress_samples_counter == 0:
                self._idx_suppress_start = self._idx_out
                if SUPPRESS_MODE_MEAN == self._suppress_mode:
                    # sum samples over pre for mean computation
                    self.cal_i_pre = 0
                    for idx in range(self._suppress_samples_pre):
                        idx_wrap = (self._idx_out - self._suppress_samples_pre + idx) & _SUPPRESS_SAMPLES_MASK
                        self.cal_i_pre += self.d_cal[idx_wrap][0]
                else:
                    idx_wrap = (self._idx_out - 1) & _SUPPRESS_SAMPLES_MASK
                    self.cal_i_pre = self.d_cal[idx_wrap][0]

        # Suppress Joulescope range switching glitch (at least for now).
        if self._suppress_samples_remaining > 0:  # defer output until suppress computed
            self._suppress_samples_counter += 1
            if self._suppress_samples_remaining == 1:  # last sample, take action

                if self._suppress_samples_counter >= _SUPPRESS_SAMPLES_MAX:
                    log.warning('Suppression filter too long for actual data: %s > %s',
                                self._suppress_samples_counter, _SUPPRESS_SAMPLES_MAX)
                    while self._suppress_samples_counter >= _SUPPRESS_SAMPLES_MAX:
                        self._cbk_fn(self._cbk_user_data, NAN, NAN, 0xff)
                        self._suppress_samples_counter -= 1
                    self._cbk_fn(self._cbk_user_data, NAN, NAN, 0xff)
                    self._idx_suppress_start = (self._idx_out + 1) & _SUPPRESS_SAMPLES_MASK
                    self.cal_i_pre = NAN

                if SUPPRESS_MODE_INTERP == self._suppress_mode:
                    if not isfinite(self.cal_i_pre):
                        self.cal_i_pre = cal_i
                    if self._suppress_samples_counter <= 0:
                        log.warning('self._suppress_samples_counter <= 0')
                        cal_i_step = 0.0
                    else:
                        cal_i_step = (cal_i - self.cal_i_pre) / self._suppress_samples_counter
                    for idx in range(self._suppress_samples_counter - 1):  # exclude the 1 post sample
                        self.cal_i_pre += cal_i_step
                        idx_wrap = (self._idx_suppress_start + idx) & _SUPPRESS_SAMPLES_MASK
                        self._cbk_fn(self._cbk_user_data,
                                     self.cal_i_pre,
                                     self.d_cal[idx_wrap][1],
                                     self.d_bits[idx_wrap])

                elif SUPPRESS_MODE_MEAN == self._suppress_mode:
                    # sum samples over post for mean computation
                    suppress_idx = self._suppress_samples_pre
                    if not isfinite(self.cal_i_pre):
                        suppress_idx = 0
                        self.cal_i_pre = 0
                    for idx in range(self._suppress_samples_post):
                        idx_wrap = (self._idx_out - idx) & _SUPPRESS_SAMPLES_MASK
                        self.cal_i_pre += self.d_cal[idx_wrap][0]
                        suppress_idx += 1
                    if suppress_idx:
                        cal_i = self.cal_i_pre / suppress_idx
                    else:
                        cal_i = NAN
                    self.cal_i_pre = cal_i

                    # update suppressed samples
                    for idx in range(self._suppress_samples_counter - self._suppress_samples_post):
                        idx_wrap = (self._idx_suppress_start + idx) & _SUPPRESS_SAMPLES_MASK
                        self._cbk_fn(self._cbk_user_data,
                                     cal_i,
                                     self.d_cal[idx_wrap][1],
                                     self.d_bits[idx_wrap])

                elif SUPPRESS_MODE_NAN == self._suppress_mode:
                    for idx in range(self._suppress_samples_counter):
                        idx_wrap = (self._idx_suppress_start + idx) & _SUPPRESS_SAMPLES_MASK
                        self._cbk_fn(self._cbk_user_data, NAN, NAN, self.d_bits[idx_wrap])

                else:
                    # SUPPRESS_MODE_OFF should never get here
                    raise RuntimeError('unsupported suppress_mode')

                # update post samples
                for idx in range(self._suppress_samples_post):
                    idx_wrap = (self._idx_out + 1 - self._suppress_samples_post + idx) & _SUPPRESS_SAMPLES_MASK
                    self._cbk_fn(self._cbk_user_data,
                                 self.d_cal[idx_wrap][0],
                                 self.d_cal[idx_wrap][1],
                                 self.d_bits[idx_wrap])
                self._suppress_samples_counter = 0
            self._suppress_samples_remaining -= 1

        else:
            self.cal_i_pre = cal_i
            self._cbk_fn(self._cbk_user_data, cal_i, cal_v, bits)
            self._suppress_samples_counter = 0
        self._idx_out = (self._idx_out + 1) & _SUPPRESS_SAMPLES_MASK
        self._i_range_last = i_range

    def process_bulk(self, raw):
        """Process a group of raw samples in bulk.

        :param raw: The np.ndarray of uint16 raw data samples.
        :return: tuple (cal, bits).  cal[0::2] is current,
            cal[1::2] is voltage.
        """
        cdef int32_t idx
        cdef uint16_t raw_i
        cdef uint16_t raw_v

        tmp_cbk_fn, tmp_user_data = self._cbk_fn, self._cbk_user_data
        self.callback_set(<raw_processor_cbk_fn> self._process_bulk_cbk, self)

        self.bulk_index = 0
        self.bulk_length = <uint32_t> (len(raw) // 2)
        raw = np.ascontiguousarray(raw, dtype=np.uint16)
        cdef np.ndarray[np.uint16_t, ndim=1, mode = 'c'] raw_c = raw
        self.bulk_raw = <uint16_t *> raw_c.data

        d_cal = np.full((self.bulk_length, 2), 0, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] d_cal_c = d_cal
        self.bulk_cal = <float *> d_cal_c.data

        d_bits = np.full(self.bulk_length, 0, dtype=np.uint8)
        cdef np.ndarray[np.uint8_t, ndim=1, mode = 'c'] d_bits_c = d_bits
        self.bulk_bits = <uint8_t *> d_bits_c.data

        for idx in range(0, self.bulk_length * 2, 2):
            raw_i = self.bulk_raw[idx + 0]
            raw_v = self.bulk_raw[idx + 1]
            self.process(raw_i, raw_v)

        while self.bulk_index < self.bulk_length:
            self.bulk_cal[self.bulk_index * 2 + 0] = NAN
            self.bulk_cal[self.bulk_index * 2 + 1] = NAN
            self.bulk_bits[self.bulk_index] = I_RANGE_MISSING
            self.bulk_index += 1

        self.callback_set(tmp_cbk_fn, tmp_user_data)

        self.bulk_raw = <uint16_t *> 0
        self.bulk_cal = <float *> 0
        self.bulk_bits = <uint8_t *> 0
        self.bulk_index = 0
        self.bulk_length = 0

        return d_cal, d_bits

    cdef void _process_bulk_cbk(self, float cal_i, float cal_v, uint8_t bits):
        self.bulk_cal[self.bulk_index * 2 + 0] = cal_i
        self.bulk_cal[self.bulk_index * 2 + 1] = cal_v
        self.bulk_bits[self.bulk_index] = bits
        self.bulk_index += 1
