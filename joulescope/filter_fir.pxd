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


from .c_filter_fir cimport filter_fir_s, filter_fir_cbk
from libc.stdint cimport uint32_t


cdef class FilterFir:

    cdef filter_fir_s ** _filters
    cdef uint32_t _filters_length
    cdef uint32_t _filters_width
    cdef object _data_np
    cdef double * _data
    cdef object _py_callback
    cdef object _filter_taps

    cdef void c_callback_set(self, filter_fir_cbk fn, void * user_data)
    cdef void c_process(self, const double * x, uint32_t x_length)

    @staticmethod
    cdef void _callback(void * user_data, const double * y, uint32_t y_length)
