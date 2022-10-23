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


class SampleBuffer:

    def __init__(self, size, dtype, decimate=None):
        """Construct a new sample buffer.

        :param size: The size in samples.
        :param dtype: The numpy data type, 'u4' or 'u1' for the incoming sample data type.
        :param decimate: The sample_id decimation factor.  None is 1.
        """
        self._dtype = dtype
        if dtype == 'u1':
            dtype = np.uint8
            size = ((size + 7) // 8) * 8
        elif dtype == 'u4':
            dtype = np.uint8
            if size & 1:
                size += 1
        self._decimate = 1 if decimate is None else max(1, int(decimate))
        #print(f'SampleBuffer({size}, {dtype}, {self._decimate})')
        self._size = size
        self._first = None   # first sample_id
        self._head = None    # head sample_id
        self._buffer = np.empty(size, dtype)
        self.clear()
        self.active = True

    def _wrap(self, ptr):
        return ptr % self._size

    @property
    def len_max(self):
        return self._size - 1

    def __len__(self):
        if self._first is None:
            return 0
        return min(self._head - self._first, self.len_max)

    def clear(self):
        self._first = None
        self._head = None
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

    def add(self, sample_id, data):
        sample_id //= self._decimate
        if self._dtype == 'u1':
            data = np.unpackbits(data)
            if self._decimate > 1:
                data = data[::self._decimate]
        elif self._dtype == 'u4':
            if self._decimate == 2:
                data = np.logical_and(data, 0x0f)
            else:
                d = np.empty(len(data) * 2, dtype=np.uint8)
                d[0::2] = np.logical_and(data, 0x0f)
                d[1::2] = np.logical_and(np.right_shift(data, 4), 0x0f)
                data = d[::self._decimate]
        elif self._dtype in ['u8', np.uint8]:
            l1 = len(data)
            data = data[::self._decimate]
        sz = len(data)
        sz_max = self._size - 1
        if self._head != sample_id:
            if self._dtype in [np.float32, np.float64]:
                skip_fill = np.nan
            else:
                skip_fill = 0
            if self._head is None:
                # first sample, set empty
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

        ptr1 = self._wrap(self._head)
        ptr2 = self._wrap(self._head + sz)
        if ptr2 > ptr1:
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
