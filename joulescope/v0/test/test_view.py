# Copyright 2019 Jetperch LLC
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
Test the view
"""

import unittest
from joulescope.view import View
from joulescope.v0.stream_buffer import StreamBuffer, usb_packet_factory
import numpy as np


class TestViewOpenClose(unittest.TestCase):

    def test_open_close(self):
        v = View(stream_buffer=None, calibration=None)
        v.open()
        self.assertEqual((('hello', ), {'one': 1}), v.ping('hello', one=1))
        v.close()


class TestView(unittest.TestCase):

    def setUp(self):
        self.b = StreamBuffer(2.0, [10, 10], sampling_frequency=1000)
        self.v = View(stream_buffer=self.b, calibration=None)
        self.b.insert(usb_packet_factory(0, 2))
        self.b.process()
        self.v.open()

    def tearDown(self):
        self.v.close()

    def test_sample_conversion(self):
        self.assertEqual((('hello', ), {'one': 1}), self.v.ping('hello', one=1))
        self.assertEqual(252, self.v.time_to_sample_id(2.0))
        self.assertEqual(2.0, self.v.sample_id_to_time(252))
        self.assertEqual(152, self.v.time_to_sample_id(1.9))

    def test_statistics_get(self):
        s1 = self.v.statistics_get(-2, -1, units='samples')
        # todo  test
