# Copyright 2022-2025 Jetperch LLC
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


def _decimate_u1(x):
    v = np.mean(x)
    return 0 if v < 0.5 else 1


def _decimate_float(x):
    return float(np.mean(x, dtype=np.float64))



class SampleBuffer:

    def __init__(self, size, dtype, name=None):
        """Construct a new sample buffer.

        :param size: The size in decimated samples.
        :param dtype: The numpy data type, 'u4' or 'u1' for the incoming sample data type.
        :param name: The optional name for this buffer.
        """
        self._name = '__none__' if name is None else str(name)
        self._dtype = dtype
        self._skip_fill = np.nan if self._dtype in [np.float32, np.float64] else 0
        self._decimate_fn = _decimate_float
        if dtype == 'u1':
            dtype = np.uint8
            size = ((size + 7) // 8) * 8
            self._decimate_fn = _decimate_u1
        elif dtype == 'u4':
            dtype = np.uint8
            if size & 1:
                size += 1
        self._log = logging.getLogger(f'{__name__}.{self._name}')
        self._sample_rate = None
        self._incoming_decimate = 1
        self._local_decimate = 1
        self._local_decimate_data = None
        #print(f'SampleBuffer({size}, {dtype}, {self._decimate})')
        self._size = size
        self._sample_id = None  # next expected sample_id, in full-rate samples
        self._first = None   # first sample_id, in decimated samples
        self._head = None    # head sample_id, in decimated samples
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
        if self._head is not None:
            self._log.info('clear')
        self._sample_id = None
        self._first = None
        self._head = None
        self._local_decimate_data = np.array([], dtype=self._buffer.dtype)
        self._buffer[:] = self._skip_fill

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
        if sample_rate is not None and self._sample_rate != sample_rate:
            self.clear()
            self._sample_rate = sample_rate
        if decimate_factor is not None and self._incoming_decimate != decimate_factor:
            self.clear()
            self._incoming_decimate = decimate_factor
        if local_decimate_factor is not None and self._local_decimate != local_decimate_factor:
            self.clear()
            self._local_decimate = local_decimate_factor
        decimate_k = self._incoming_decimate * self._local_decimate

        if self._dtype == 'u1':
            data = np.unpackbits(data, bitorder='little')
        elif self._dtype == 'u4':
            d = np.empty(len(data) * 2, dtype=np.uint8)
            d[0::2] = np.logical_and(data, 0x0f)
            d[1::2] = np.logical_and(np.right_shift(data, 4), 0x0f)
            data = d

        head_this = (sample_id + decimate_k - 1) // decimate_k
        if self._head is None:
            self._head = head_this
            self._first = self._head
            self._sample_id = sample_id

        if self._sample_id == sample_id:
            pass  # normal operation
        elif sample_id < self._sample_id:
            k = len(data) * decimate_factor
            if (sample_id + k) < self._sample_id:
                return  # complete duplication
            else:  # partial overlap
                k = (self._sample_id - sample_id) // decimate_factor
                data = data[k:]
                sample_id = self._sample_id
        else:  # sample_id > self._sample_id
            skip_sz = sample_id // decimate_k - self._head
            if skip_sz >= self.len_max:
                self._buffer[:] = self._skip_fill
                self._head = head_this
            elif skip_sz > 0:
                ptr1 = self._wrap(self._head)
                ptr2 = self._wrap(self._head + skip_sz)
                self._head += skip_sz
                if ptr2 > ptr1:
                    self._buffer[ptr1:ptr2] = self._skip_fill
                else:
                    self._buffer[ptr1:] = self._skip_fill
                    self._buffer[:ptr2] = self._skip_fill

        if self._local_decimate > 1:
            ptr1 = self._wrap(self._head)
            d_idx1 = sample_id // self._incoming_decimate
            d_idx2 = ((d_idx1 + self._local_decimate - 1) // self._local_decimate) * self._local_decimate
            d_len = d_idx2 - d_idx1
            src_data = data
            dst_data_len = len(src_data) // self._local_decimate + 1
            dst_data = np.empty(dst_data_len, dtype=np.float64)
            dst_data_idx = 0

            if d_len:
                data_first = src_data[:d_len]
                src_data = src_data[d_len:]
                data_first = np.concatenate((data_first, self._local_decimate_data))
                dst_data[0] = np.mean(data_first, dtype=np.float64)
                dst_data_idx = 1
            k = len(src_data) // self._local_decimate
            src_data_end = k * self._local_decimate
            self._local_decimate_data = src_data[src_data_end:]  # for next time
            src_data_view = src_data[:src_data_end].reshape((k, self._local_decimate))
            dst_data[dst_data_idx:(dst_data_idx + k)] = np.mean(src_data_view, axis=1, dtype=np.float64)
            dst_data = dst_data[:(dst_data_idx + k)]
            if self._dtype == 'u1':
                data = (dst_data > 0.5).astype(np.uint8)
            else:
                data = dst_data.astype(self._buffer.dtype)

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
            raise ValueError(f'get_range {start}:{end} out of available range {s_start}:{s_end}')
        elif start >= end:
            return []
        ptr1 = self._wrap(start)
        ptr2 = self._wrap(end)
        if ptr2 > ptr1:
            return self._buffer[ptr1:ptr2]
        else:
            return np.concatenate((self._buffer[ptr1:], self._buffer[:ptr2]))
