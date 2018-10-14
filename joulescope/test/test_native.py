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
Test the native Joulescope code.
"""

import unittest
from joulescope import native
from joulescope.native import usb_packet_factory
import numpy as np


class TestStreamBuffer(unittest.TestCase):

    def test_init(self):
        b = native.StreamBuffer(1000000, [])
        self.assertIsNotNone(b)
        del b

    def test_insert_process(self):
        b = native.StreamBuffer(1000000, [100, 100, 100])
        frame = usb_packet_factory(0, 1)
        self.assertEqual(0, b.status()['device_sample_id']['value'])
        b.insert(frame)
        self.assertEqual(126, b.status()['device_sample_id']['value'])
        self.assertEqual(0, b.status()['sample_id']['value'])
        b.process()
        self.assertEqual(126, b.status()['sample_id']['value'])

    def test_get(self):
        b = native.StreamBuffer(2000, [10, 10])
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.get(0, 40, 20)
        self.assertEqual((2, 3, 4), data.shape)
        np.testing.assert_allclose([19.0, 133.0, 0.0, 38.0], data[0, 0, :])
        np.testing.assert_allclose([20.0, 133.0, 1.0, 39.0], data[0, 1, :])
        np.testing.assert_allclose([59.0, 133.0, 40.0, 78.0], data[1, 0, :])

    def test_calibration(self):
        b = native.StreamBuffer(2000, [10, 10])
        b.calibration_set([-10.0] * 7, [2.0] * 7, [-2.0, 1.0], [4.0, 1.0])
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.get(0, 1, 1)
        self.assertEqual((1, 3, 4), data.shape)
        np.testing.assert_allclose([-20.0, -4.0, 80.0], data[0, :, 0])
