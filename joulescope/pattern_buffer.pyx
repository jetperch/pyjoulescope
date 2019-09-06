# Copyright 2019 Jetperch LLC
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
Optimized Cython native Joulescope code for digital pattern testing.
"""

# See https://cython.readthedocs.io/en/latest/index.html

# cython: boundscheck=False, wraparound=False, nonecheck=False, overflowcheck=False, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t, int64_t
import numpy as np
cimport numpy as np
import logging


DEF PACKET_TOTAL_SIZE = 512
DEF PACKET_TOTAL_SIZE_U32 = PACKET_TOTAL_SIZE // 4
log = logging.getLogger(__name__)


cdef class PatternBuffer:

    cdef uint16_t _pkt_index
    cdef uint32_t _bit_pattern
    cdef uint16_t _counter_pattern
    cdef uint64_t _sample_id
    cdef uint64_t _sample_id_max  # used to automatically stop streaming
    cdef uint64_t _contiguous_max  # used to automatically stop streaming
    cdef uint8_t _pattern_state

    cdef uint64_t _header_error
    cdef uint64_t _pkt_index_error
    cdef uint64_t _pattern_error

    def __cinit__(self):
        self._reset()

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

    def close(self):
        pass

    def status(self):
        return {
            'sample_id':self. _sample_id,
            'header_error': self._header_error,
            'pkt_index_error': self._pkt_index_error,
            'pattern_error': self._pattern_error,
        }

    def _reset(self):
        self._pkt_index = 0
        self._bit_pattern = 0
        self._counter_pattern = 0
        self._sample_id = 0
        self._sample_id_max = 0
        self._contiguous_max = 0
        self._pattern_state = 0
        self._header_error = 0
        self._pkt_index_error = 0
        self._pattern_error = 0

    def reset(self):
        self._reset()

    def calibration_set(self, *args, **kwargs):
        pass

    cdef _insert_u32(self, uint32_t p):
        # print('%d: %08x' % (self._sample_id, p))
        if self._pattern_state == 2:  # expect next counter
            self._counter_pattern += 1
            p_lower = p & 0xffff
            p_upper = (~(p >> 16)) & 0xffff
            if p_lower != self._counter_pattern or p_upper != self._counter_pattern:
                log.info('counter resync: sampled_id=%d, counter=0x%04x, value=0x%08x',
                         self._sample_id, self._counter_pattern, p)
                self._pattern_error += 1
                self._pattern_state = 0
            else:
                self._pattern_state = 3
        elif self._pattern_state == 3:  # expect next shift
            if self._bit_pattern == 0:
                self._bit_pattern = 1
            else:
                self._bit_pattern = self._bit_pattern << 1
            if p != self._bit_pattern:
                log.info('bit_pattern resync: sampled_id=%d, bit_pattern=0x%08x, value=0x%08x',
                         self._sample_id, self._bit_pattern, p)
                self._pattern_error += 1
                self._pattern_state = 0
            else:
                self._pattern_state = 2
        elif self._pattern_state == 0:  # unsync, look for counter
            log.info('pattern_state resync %d', self._sample_id)
            if (p & 0xffff) == (~(p >> 16)) & 0xffff:
                # detected counter!
                self._counter_pattern = p & 0xffff
                self._pattern_state = 1
        elif self._pattern_state == 1:  # resync shift
            self._bit_pattern = p
            self._pattern_state = 2
        else:
            log.warning('invalid pattern state')
            self._pattern_state = 0
            self._pattern_error += 1

    cdef _insert_usb_bulk(self, const uint32_t *data_u32, size_t length_u32):
        cdef uint32_t * pkt = data_u32
        cdef uint8_t buffer_type
        cdef uint8_t status
        cdef uint16_t pkt_length
        cdef uint8_t voltage_range
        cdef uint32_t p

        while length_u32 >= PACKET_TOTAL_SIZE_U32:
            buffer_type = <uint8_t> (pkt[0] & 0xff)
            status = <uint8_t> ((pkt[0] & 0xff00) >> 8)
            pkt_length = <uint16_t> ((pkt[0] & 0x7fff0000) >> 16)
            voltage_range = <uint8_t> ((pkt[0] & 0x80000000) >> 31)
            if buffer_type != 0x80:
                self._header_error += 1
                log.warning('invalid packet type %d', buffer_type)
            elif status != 0x00:
                self._header_error += 1
                log.warning('invalid packet status %d', status)
            elif pkt_length != 0x0200:
                self._header_error += 1
                log.warning('invalid packet length %d', pkt_length)
            else:
                if self._pkt_index != pkt[1]:
                    self._pkt_index_error += 1
                    log.warning('pkt_index mismatch: expected %d, received %d', self._pkt_index, pkt[1])
                self._pkt_index = pkt[1] + 1
                for idx in range(2, PACKET_TOTAL_SIZE_U32):
                    self._insert_u32(pkt[idx])
                    self._sample_id += 1

            pkt += PACKET_TOTAL_SIZE_U32
            length_u32 -= PACKET_TOTAL_SIZE_U32
            if self._sample_id_max and self._sample_id >= self._sample_id_max:
                return True
        return False

    cpdef insert(self, data):
        """Insert new device USB data into the buffer.

        :param data: The new data to insert.
        :return: False to continue streaming, True to end.
        """
        cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] data_c
        if data is None:
            return True
        if len(data) % PACKET_TOTAL_SIZE != 0:
            raise ValueError('invalid data length')
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data, dtype=np.uint8)
            data_u32 = data.view(dtype=np.uint32)
            data_c = data_u32
            data_ptr = <uint32_t *> data_c.data
            return self._insert_usb_bulk(data_ptr, len(data) // 4)
        else:
            raise ValueError('invalid data')

    def process(self):
        return False

    @property
    def sample_id_range(self):
        return 0, self._sample_id

    def raw_get(self, start, stop):
        return np.empty((0, 2), dtype=np.uint16)
