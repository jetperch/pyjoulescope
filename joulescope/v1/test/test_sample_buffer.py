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

"""
Test the circular sample buffer.
"""

import unittest
from joulescope.v1.sample_buffer import SampleBuffer
import numpy as np


class TestSampleBuffer(unittest.TestCase):

    def test_new(self):
        c = SampleBuffer(10, dtype=np.float32)
        self.assertEqual(0, len(c))
        self.assertEqual((0, 0), c.range)
        with self.assertRaises(ValueError):
            c.get_range(0, 1)

    def test_add_one(self):
        c = SampleBuffer(10, dtype=np.float32)
        a = np.array([2.0], dtype=np.float32)
        c.add(0, a)
        self.assertEqual(1, len(c))
        self.assertEqual((0, 1), c.range)
        np.testing.assert_equal(a, c.get_range(*c.range))
        with self.assertRaises(ValueError):
            c.get_range(-1, 0)
        with self.assertRaises(ValueError):
            c.get_range(1, 2)

    def test_add_one_offset(self):
        c = SampleBuffer(10, dtype=np.float32)
        a = np.array([2.0], dtype=np.float32)
        c.add(111, a)
        self.assertEqual(1, len(c))
        self.assertEqual((111, 112), c.range)
        np.testing.assert_equal(a, c.get_range(*c.range))
        with self.assertRaises(ValueError):
            c.get_range(110, 111)
        with self.assertRaises(ValueError):
            c.get_range(112, 113)

    def test_add_one_overfill(self):
        c = SampleBuffer(10, dtype=np.float32)
        for i in range(20):
            c.add(i, np.array([i], dtype=np.float32))
        self.assertEqual(9, len(c))
        self.assertEqual((11, 20), c.range)
        np.testing.assert_equal(np.arange(11, 20, 1, dtype=np.float32), c.get_range(*c.range))

    def test_add_multiple_overflow_initial(self):
        c = SampleBuffer(10, dtype=np.float32)
        a = np.arange(9, 15, dtype=np.float32)
        c.add(9, a)
        self.assertEqual(len(a), len(c))
        self.assertEqual((9, 15), c.range)
        np.testing.assert_equal(a, c.get_range(*c.range))
        np.testing.assert_equal(a[1:], c.get_range(10, 15))
        np.testing.assert_equal(a[:2], c.get_range(9, 11))

    def test_skip(self):
        c = SampleBuffer(10, dtype=np.float32)
        c.add(1, [1.0])
        c.add(3, [3.0])
        self.assertEqual(3, len(c))
        self.assertEqual((1, 4), c.range)
        s = c.get_range(*c.range)
        self.assertEqual(1.0, s[0])
        self.assertTrue(np.isnan(s[1]))
        self.assertEqual(3.0, s[2])
