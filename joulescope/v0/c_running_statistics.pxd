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

from libc.stdint cimport uint64_t

cdef extern from "native/running_statistics.h":
    struct statistics_s:  # add packed?
        uint64_t k
        double m
        double s
        double min
        double max
    void statistics_reset(statistics_s * s)
    void statistics_invalid(statistics_s * s)
    void statistics_add(statistics_s * s, double x)
    double statistics_var(statistics_s * s)
    void statistics_copy(statistics_s * tgt, statistics_s * src)
    void statistics_combine(statistics_s * tgt, statistics_s * a, statistics_s * b)
