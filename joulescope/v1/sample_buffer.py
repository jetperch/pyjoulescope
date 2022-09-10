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
        self._decimate = 1 if decimate is None else int(decimate)
        self._size = size
        self._first = None   # first sample_id
        self._head = None    # head sample_id
        self._buffer = np.empty(size, dtype)
        self._buffer[:] = np.nan

    def _wrap(self, ptr):
        return (ptr // self._decimate) % self._size

    @property
    def len_max(self):
        return self._size - 1

    def __len__(self):
        if self._first is None:
            return 0
        return min((self._head - self._first) // self._decimate, self.len_max)

    def clear(self):
        self._first = None
        self._head = None
        self._buffer[:] = np.nan

    @property
    def range(self):
        if self._first is None:
            return 0, 0
        h = self._head
        start = max(h - self._size + 1, self._first)
        return start, self._head

    @property
    def range_decimated(self):
        start, end = self.range
        return start // self._decimate, end // self._decimate

    def add(self, sample_id, data):
        sz = len(data)
        sz_max = self._size - 1
        if self._head != sample_id:
            if self._head is None:
                # first sample, set empty
                self._first = sample_id
            else:
                skip_sz = (sample_id - self._head) // self._decimate
                if skip_sz >= sz_max:
                    self._buffer[:] = np.nan
                else:
                    ptr1 = self._wrap(self._head)
                    ptr2 = self._wrap(self._head + skip_sz)
                    if ptr2 > ptr1:
                        self._buffer[ptr1:ptr2] = np.nan
                    else:
                        self._buffer[ptr1:] = np.nan
                        self._buffer[:ptr2] = np.nan
            self._head = sample_id
        ptr1 = self._wrap(self._head)
        ptr2 = self._wrap(self._head + sz * self._decimate)
        if ptr2 > ptr1:
            self._buffer[ptr1:ptr2] = data
        else:
            k = self._size - ptr1
            self._buffer[ptr1:] = data[:k]
            self._buffer[:ptr2] = data[k:]
        self._head += sz * self._decimate

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
