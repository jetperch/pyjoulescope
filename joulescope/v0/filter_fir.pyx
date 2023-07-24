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
Optimized Cython native Joulescope code for FIR filtering.
"""

# See https://cython.readthedocs.io/en/latest/index.html

# cython: boundscheck=True, wraparound=True, nonecheck=True, overflowcheck=True, cdivision=True

from libc.stdint cimport uint32_t, uint64_t
from . cimport c_filter_fir
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
from cython cimport view
import numpy as np
cimport numpy as np


cdef class FilterFir:
    # See filter_fir.pxd for C class members

    def __cinit__(self):
        self._filters = NULL
        self._filters_length = 0
        self._py_callback = None

    def __init__(self, filters, width=None):
        cdef np.float64_t [::1] taps_c
        width = 1 if width is None else int(width)
        self._filters = <c_filter_fir.filter_fir_s **> PyMem_Malloc(len(filters) * sizeof(void *))
        self._data_np = np.full(width, 0, dtype=np.float64)
        cdef np.float64_t [::1] data_c = self._data_np
        self._data = &data_c[0]
        self._filters_length = <uint32_t> len(filters)
        self._filters_width = width
        self._filter_taps = []
        for idx, filter_def in enumerate(filters):
            taps = np.array(filter_def['taps'], dtype=np.float64)
            self._filter_taps.append(taps)
            taps_c = taps
            M = filter_def.get('M', 1)
            self._filters[idx] = c_filter_fir.filter_fir_alloc(&taps_c[0], <uint32_t> len(taps), M, width)

        for idx in range(self._filters_length - 1):
            c_filter_fir.filter_fir_callback_set(self._filters[idx],
                                                 <c_filter_fir.filter_fir_cbk> c_filter_fir.filter_fir_single, self._filters[idx + 1])

    def __dealloc__(self):
        PyMem_Free(self._filters)

    def reset(self):
        for idx in range(self._filters_length):
            c_filter_fir.filter_fir_reset(self._filters[idx])

    def callback_set(self, fn):
        self._py_callback = fn
        self.c_callback_set(self._callback, <void *> self)

    @staticmethod
    cdef void _callback(void * user_data, const double * y, uint32_t y_length) noexcept:
        cdef FilterFir self = <object> user_data
        cdef np.float64_t [::1] data_c
        if y_length == 1:
            self._py_callback(float(y[0]))
        else:
            data = np.empty(y_length, dtype=np.float64)
            for idx in range(y_length):
                data[idx] = y[idx]
            self._py_callback(data)

    cdef void c_callback_set(self, c_filter_fir.filter_fir_cbk fn, void * user_data):
        c_filter_fir.filter_fir_callback_set(self._filters[self._filters_length - 1], fn, user_data)

    cdef void c_process(self, const double * x, uint32_t x_length):
        if x_length != self._filters_width:
            assert(x_length == self._filters_width)
            return
        c_filter_fir.filter_fir_single(self._filters[0], x, x_length)

    def process(self, x):
        """Process the new data.

        :param x: The new data as either:
            * a single numerical value
            * a 1D np.ndarray with dtype=np.float64.  The length must match width.
            * a NxW 2D np.ndarray with dtype=np.float64.  W must match width 
        """
        cdef double [::1] x1d_view
        cdef double [:, ::1] x2d_view
        cdef ssize_t idx

        if isinstance(x, (float, int)):
            self._data[0] = x
            self.c_process(&self._data[0], 1)
        elif len(x.shape) == 1:
            if len(x) != self._filters_width:
                raise RuntimeError(f'Invalid width: {len(x)} != {self._filters_width}')
            x1d_view = x
            self.c_process(&x1d_view[0], self._filters_width)
        elif len(x.shape) == 2:
            if x.shape[1] != self._filters_width:
                raise RuntimeError(f'Invalid width: {len(x)} != {self._filters_width}')
            x2d_view = x
            for idx in range(len(x)):
                c_filter_fir.filter_fir_single(self._filters[0], &x2d_view[idx, 0], self._filters_width)
