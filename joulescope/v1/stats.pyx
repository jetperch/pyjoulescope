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

# See https://cython.readthedocs.io/en/latest/index.html

# cython: boundscheck=True, wraparound=True, nonecheck=True, overflowcheck=True, cdivision=True

from libc.stdint cimport uint8_t, uint64_t
import numpy as np
cimport numpy as np


np.import_array()  # initialize numpy


cdef compute_stats_u8(d, out):
    cdef uint64_t sz = len(d)
    cdef uint64_t i
    cdef uint8_t [:] d_u8
    cdef uint8_t v_u8
    cdef uint64_t v_accum = 0
    cdef uint8_t v_min = 255
    cdef uint8_t v_max = 0
    cdef double v_f64
    cdef double v_mean
    cdef double v_var = 0.0
    d_u8 = d
    for i in range(sz):
        v_u8 = d_u8[i]
        v_accum += v_u8
        if v_u8 < v_min:
            v_min = v_u8
        if v_u8 > v_max:
            v_max = v_u8
    v_mean = v_accum
    v_mean /= sz
    for i in range(sz):
        v_u8 = d_u8[i]
        v_f64 = v_u8 - v_mean
        v_f64 *= v_f64
        v_var += v_f64
    out['length'] = sz
    out['mean'] = v_mean
    out['variance'] = v_var
    out['min'] = v_min
    out['max'] = v_max


cdef compute_stats_f32(d, out):
    cdef uint64_t sz = len(d)
    cdef uint64_t i
    cdef float [:] d_f32
    cdef float v_f32
    cdef double v_accum = 0.0
    cdef float v_min = 255
    cdef float v_max = 0
    cdef double v_f64
    cdef double v_mean
    cdef double v_var = 0.0
    d_f32 = d
    for i in range(sz):
        v_f32 = d_f32[i]
        v_accum += v_f32
        if v_f32 < v_min:
            v_min = v_f32
        if v_f32 > v_max:
            v_max = v_f32
    v_mean = v_accum
    v_mean /= sz
    for i in range(sz):
        v_f32 = d_f32[i]
        v_f64 = v_f32 - v_mean
        v_f64 *= v_f64
        v_var += v_f64
    out['length'] = sz
    out['mean'] = v_mean
    out['variance'] = v_var
    out['min'] = v_min
    out['max'] = v_max


def compute_stats(d, out):
    dtype = d.dtype
    if dtype == np.uint8:
        compute_stats_u8(d, out)
    elif dtype == np.float32:
        compute_stats_f32(d, out)
    else:
        raise ValueError(f'unsupported type {d.dtype}')
