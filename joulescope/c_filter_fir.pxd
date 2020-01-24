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


from libc.stdint cimport uint32_t

cdef extern from "native/filter_fir.h":
    struct filter_fir_s
    ctypedef void (*filter_fir_cbk)(void * user_data, const double * y, uint32_t y_length)
    filter_fir_s * filter_fir_alloc(const double * taps, uint32_t taps_length,
        uint32_t M, uint32_t width)
    void filter_fir_free(filter_fir_s * self)
    void filter_fir_reset(filter_fir_s * self)
    void filter_fir_callback_set(filter_fir_s * self, filter_fir_cbk fn, void * user_data)
    void filter_fir_single(filter_fir_s * self, const double * x, uint32_t x_length)
