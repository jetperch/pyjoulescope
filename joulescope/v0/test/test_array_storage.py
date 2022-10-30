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

"""
Test the array_starage
"""

import unittest
from joulescope.v0 import array_storage
import numpy as np


class TestArrayStorage(unittest.TestCase):

    def test_empty(self):
        b = array_storage.pack({}, sample_count=0)
        y, c = array_storage.unpack(b)
        self.assertEqual(c, 0)
        self.assertEqual(len(y), 0)

    def test_single_u8(self):
        x = np.arange(256, dtype=np.uint8)
        b = array_storage.pack({2: x}, sample_count=42)
        y, c = array_storage.unpack(b)
        self.assertEqual(c, 42)
        self.assertIn(2, y)
        np.testing.assert_equal(y[2], x)

    def test_invalid_array_id(self):
        x = np.arange(256, dtype=np.uint8)
        with self.assertRaises(ValueError):
            array_storage.pack({-1: x}, sample_count=42)

    def test_invalid_sample_count(self):
        with self.assertRaises(ValueError):
            array_storage.pack({}, sample_count=-1)
        array_storage.pack({}, sample_count=0)
        array_storage.pack({}, sample_count=2 ** 24 - 1)
        with self.assertRaises(ValueError):
            array_storage.pack({}, sample_count=2 ** 24)

    def test_single_float32(self):
        x = np.arange(0, 10, 0.25, dtype=np.float32)
        b = array_storage.pack({3: x}, sample_count=43)
        y, c = array_storage.unpack(b)
        self.assertEqual(c, 43)
        self.assertIn(3, y)
        np.testing.assert_equal(y[3], x)

    def test_multiple(self):
        x_u8 = np.arange(256, dtype=np.uint8)
        x_f32 = np.arange(0, 10, 0.25, dtype=np.float32)
        x = {9: x_u8, 5: x_f32}
        b = array_storage.pack(x, sample_count=44)
        y, c = array_storage.unpack(b)
        self.assertEqual(c, 44)
        self.assertEqual(len(y), 2)
        np.testing.assert_equal(y[9], x_u8)
        np.testing.assert_equal(y[5], x_f32)

    def test_unpack_empty(self):
        with self.assertRaises(ValueError):
            array_storage.unpack(np.zeros(0, dtype=np.uint8))

    def test_unpack_invalid(self):
        with self.assertRaises(ValueError):
            array_storage.unpack(np.zeros(100, dtype=np.uint8))

    def test_truncated(self):
        x = np.arange(0, 10, 0.25, dtype=np.float32)
        b = array_storage.pack({3: x}, sample_count=43)
        with self.assertRaises(ValueError):
            array_storage.unpack(b[:-16])
