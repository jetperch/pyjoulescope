# Copyright 2022 Jetperch LLC
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

"""A circular sample buffer for storing samples."""

import numpy as np
import logging


class SampleBuffer:

    def __init__(self, size, dtype, name=None):
        """Construct a new sample buffer.

        :param size: The size in decimated samples.
        :param dtype: The numpy data type, 'u4' or 'u1' for the incoming sample data type.
        :param name: The optional name for this buffer.
        """
        self._name = '__none__' if name is None else str(name)
        self._dtype = dtype
        if dtype == 'u1':
            dtype = np.uint8
            size = ((size + 7) // 8) * 8
        elif dtype == 'u4':
            dtype = np.uint8
            if size & 1:
                size += 1
        self._log = logging.getLogger(f'{__name__}.{self._name}')
        self._sample_rate = None
        self._incoming_decimate = 1
        self._local_decimate = 1
        self._local_decimate_offset = 0
        #print(f'SampleBuffer({size}, {dtype}, {self._decimate})')
        self._size = size
        self._first = None   # first sample_id
        self._head = None    # head sample_id
        self._buffer = np.empty(size, dtype)
        self.clear()
        self.active = True

    def __str__(self):
        return f'SampleBuffer(size={self._size}, dtype={self._dtype}, name={self._name})'

    def _wrap(self, ptr):
        return ptr % self._size

    @property
    def len_max(self):
        return self._size - 1

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def decimate(self):
        return self._incoming_decimate * self._local_decimate

    def __len__(self):
        if self._first is None:
            return 0
        return min(self._head - self._first, self.len_max)

    def clear(self):
        self._first = None
        self._head = None
        self._local_decimate_offset = 0
        if self._buffer.dtype in [np.float32, np.float64]:
            self._buffer[:] = np.nan
        else:
            self._buffer[:] = 0

    @property
    def range(self):
        if self._first is None:
            return 0, 0
        h = self._head
        start = max(h - self._size + 1, self._first)
        return start, self._head

    def add(self, sample_id, data, sample_rate=None, decimate_factor=None, local_decimate_factor=None):
        if sample_id is None or data is None:
            self._log.warning('Invalid add')
            return
        clr = False
        if sample_rate is not None and self._sample_rate != sample_rate:
            self.clear()
            self._sample_rate = sample_rate
            clr = True
        if decimate_factor is not None and self._incoming_decimate != decimate_factor:
            self.clear()
            self._incoming_decimate = decimate_factor
            clr = True
        if local_decimate_factor is not None and self._local_decimate != local_decimate_factor:
            self.clear()
            self._local_decimate = local_decimate_factor
            clr = True
        if clr:
            self._log.info('add -> auto clear')

        sample_id_orig = sample_id
        sample_id //= (self._incoming_decimate * self._local_decimate)
        if self._dtype == 'u1':
            data = np.unpackbits(data)
        elif self._dtype == 'u4':
            d = np.empty(len(data) * 2, dtype=np.uint8)
            d[0::2] = np.logical_and(data, 0x0f)
            d[1::2] = np.logical_and(np.right_shift(data, 4), 0x0f)
            data = d

        sz_max = self._size - 1
        if self._head != sample_id:
            self._log.info('skip head=%s, sample_id=%s, sample_id_orig=%s, sz=%s',
                           self._head, sample_id, sample_id_orig, len(data))
            if self._dtype in [np.float32, np.float64]:
                skip_fill = np.nan
            else:
                skip_fill = 0
            if self._head is None:
                # first sample, set empty
                offset = sample_id_orig - sample_id * self._incoming_decimate * self._local_decimate
                if self._local_decimate > 1 and offset != 0:
                    sample_id += 1
                    self._local_decimate_offset = self._local_decimate - offset // self._incoming_decimate
                self._first = sample_id
            else:
                skip_sz = sample_id - self._head
                if skip_sz >= sz_max:
                    self._buffer[:] = skip_fill
                else:
                    ptr1 = self._wrap(self._head)
                    ptr2 = self._wrap(self._head + skip_sz)
                    if ptr2 > ptr1:
                        self._buffer[ptr1:ptr2] = skip_fill
                    else:
                        self._buffer[ptr1:] = skip_fill
                        self._buffer[:ptr2] = skip_fill
            self._head = sample_id

        if self._local_decimate > 1:
            offset = (self._local_decimate_offset + len(data)) % self._local_decimate
            data = data[self._local_decimate_offset::self._local_decimate]
            self._local_decimate_offset = offset
        sz = len(data)
        ptr1 = self._wrap(self._head)
        ptr2 = self._wrap(self._head + sz)
        if sz == 0:
            pass
        elif ptr2 > ptr1:
            self._buffer[ptr1:ptr2] = data
        else:
            k = self._size - ptr1
            self._buffer[ptr1:] = data[:k]
            self._buffer[:ptr2] = data[k:]
        self._head += sz

    def get_range(self, start, end):
        s_start, s_end = self.range
        if start < s_start or end > s_end:
            raise ValueError(f'get {start}:{end} out of available range {s_start}:{s_end}')
        ptr1 = self._wrap(start)
        ptr2 = self._wrap(end)
        if ptr2 > ptr1:
            return self._buffer[ptr1:ptr2]
        else:
            return np.concatenate((self._buffer[ptr1:], self._buffer[:ptr2]))
