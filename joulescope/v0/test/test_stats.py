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


import unittest
import numpy as np
from joulescope.v0.stream_buffer import RunningStatistics as Statistics
from joulescope.v0.stream_buffer import stats_compute, stats_array_factory



class TestStatistics(unittest.TestCase):

    def test_initialize_empty(self):
        s = Statistics()
        self.assertEqual(0, len(s))
        self.assertEqual(0.0, s.mean)
        self.assertEqual(0.0, s.var)
        self.assertEqual(0.0, s.std)

    def test_add_zero_once(self):
        s = Statistics()
        s.add(0.0)
        self.assertEqual(1, len(s))
        self.assertEqual(0.0, s.mean)
        self.assertEqual(0.0, s.var)
        self.assertEqual(0.0, s.std)
        self.assertEqual(0.0, s.min)
        self.assertEqual(0.0, s.max)

    def test_add_zero_twice(self):
        s = Statistics()
        s.add(0.0)
        s.add(0.0)
        self.assertEqual(2, len(s))
        self.assertEqual(0.0, s.mean)
        self.assertEqual(0.0, s.var)
        self.assertEqual(0.0, s.std)
        self.assertEqual(0.0, s.min)
        self.assertEqual(0.0, s.max)

    def test_add_multiple(self):
        s = Statistics()
        s.add(0.0)
        s.add(1.0)
        s.add(2.0)
        self.assertEqual(3, len(s))
        self.assertEqual(1.0, s.mean)
        # note that np.var returns the variance, not sample variance
        sample_variance = 3.0 / 2.0 * np.var([0.0, 1.0, 2.0])
        self.assertEqual(1.0, sample_variance)
        self.assertEqual(sample_variance, s.var)
        self.assertEqual(np.sqrt(sample_variance), s.std)
        self.assertEqual(0.0, s.min)
        self.assertEqual(2.0, s.max)

    def test_combine_both_empty(self):
        s1 = Statistics()
        s2 = Statistics()
        s3 = s1.combine(s2)
        self.assertEqual(0, len(s3))

    def test_combine_self_empty(self):
        s1 = Statistics()
        s2 = Statistics()
        s2.add(1.0)
        s3 = s1.combine(s2)
        self.assertEqual(1, len(s3))
        self.assertEqual(1.0, s3.mean)

    def test_combine_other_empty(self):
        s1 = Statistics()
        s2 = Statistics()
        s1.add(1.0)
        s3 = s1.combine(s2)
        self.assertEqual(1, len(s3))
        self.assertEqual(1.0, s3.mean)

    def test_combine(self):
        s1 = Statistics()
        for x in np.arange(0, 11):
            s1.add(x)
        s2 = Statistics()
        for x in np.arange(11, 21):
            s2.add(x)
        s3 = Statistics()
        for x in np.arange(0, 21):
            s3.add(x)
        sc = s1.combine(s2)
        self.assertEqual(len(s3), len(sc))
        self.assertEqual(s3.mean, sc.mean)
        self.assertEqual(s3.var, sc.var)
        self.assertEqual(s3.std, sc.std)
        self.assertEqual(s3.min, sc.min)
        self.assertEqual(s3.max, sc.max)

    def test_stats_compute(self):
        v = np.arange(0, 11, dtype=np.float32)
        s_array = stats_array_factory(1)
        stats_compute(v, s_array[0, 0:1])
        self.assertEqual(np.mean(v), s_array[0, 0]['mean'])
