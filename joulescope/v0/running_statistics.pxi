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


# pxi includes are not really recommended, but this class is just for test code.


cdef class RunningStatistics:
    """For unit testing only."""
    cdef c_running_statistics.statistics_s stats

    def __cinit__(self):
        c_running_statistics.statistics_reset(&self.stats)

    def __init__(self):
        pass

    def clear(self):
        c_running_statistics.statistics_reset(&self.stats)

    def invalid(self):
        c_running_statistics.statistics_invalid(&self.stats)

    def __len__(self):
        return self.stats.k

    def add(self, x):
        c_running_statistics.statistics_add(&self.stats, x)

    @property
    def mean(self):
        return self.stats.m

    @property
    def var(self):
        return c_running_statistics.statistics_var(&self.stats)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def min(self):
        return self.stats.min

    @property
    def max(self):
        return self.stats.max

    def copy(self, other: RunningStatistics):
        c_running_statistics.statistics_copy(&self.stats, &other.stats)
        return self

    def combine(self, other: RunningStatistics):
        r = RunningStatistics()
        c_running_statistics.statistics_combine(&r.stats, &self.stats, &other.stats)
        return r
