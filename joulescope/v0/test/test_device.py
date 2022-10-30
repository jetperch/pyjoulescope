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
Test the JouleScope device

This test requires that JouleScope hardware be attached to this PC!
"""

import unittest
from joulescope.v0.driver import scan, UsbdRequest, LOOPBACK_BUFFER_SIZE, Device
from joulescope.v0.usb import hw_tests
from joulescope.v0.pattern_buffer import PatternBuffer


class TestPattern(unittest.TestCase):

    def setUp(self):
        self.device = None
        self.devices = scan(name='joulescope')
        if not len(self.devices):
            raise unittest.SkipTest("no devices found")
        if len(self.devices) > 1:
            print("multiple devices found")
        self.device = self.devices[0]
        self.device.open()

    def tearDown(self):
        if self.device is not None:
            self.device.close()

    def test_control_loopback_wvalue(self):
        usb = self.device.usb_device
        hw_tests.control_loopback_wvalue(usb, UsbdRequest.LOOPBACK_WVALUE, 17)

    def test_control_loopback_buffer(self):
        usb = self.device.usb_device
        hw_tests.control_loopback_buffer(usb, UsbdRequest.LOOPBACK_BUFFER, LOOPBACK_BUFFER_SIZE, 4)

    def _pattern(self, duration=None):
        duration = int(duration) if duration is not None else 1.0
        buffer = PatternBuffer()
        self.device.stream_buffer = buffer
        self.device.read(duration=duration, out_format='raw')
        s = buffer.status()
        self.assertGreater(s['sample_id'], 1000000)
        self.assertEqual(s['header_error'], 0)
        self.assertLessEqual(s['pkt_index_error'], 1)
        self.assertLessEqual(s['pattern_error'], 1)

    def test_datapath_usb(self):
        self.device.parameter_set('control_test_mode', 'usb')
        self.device.parameter_set('source', 'pattern_usb')
        self._pattern(1.0)

    def test_datapath_sensor(self):
        self.device.parameter_set('control_test_mode', 'normal')
        self.device.parameter_set('source', 'pattern_sensor')
        self._pattern(1.0)

    def test_read(self):
        self.device.parameter_set('source', 'raw')
        for i in range(10):
            v = self.device.read(contiguous_duration=0.125, out_format='raw')
            self.assertEqual((self.device.sampling_frequency // 8, 2), v.shape)


class TestAttributes(unittest.TestCase):

    def test_defaults(self):
        d = Device(None)
        self.assertEqual(2000000, d.sampling_frequency)
        self.assertEqual('30 seconds', d.parameter_get('buffer_duration'))
        self.assertEqual(30, d.parameter_get('buffer_duration', dtype='actual'))
        self.assertEqual('2 Hz', d.parameter_get('reduction_frequency'))
        self.assertIsNone(d.stream_buffer)
        self.assertIsNone(d.calibration)
        self.assertEqual('i_range', d.parameters(name='i_range').name)
        self.assertEqual('off', d.parameter_get('i_range'))
        with self.assertRaises(Exception):
            d.serial_number
        with self.assertRaises(Exception):
            d.info()
        self.assertFalse(d.is_streaming)
        with self.assertRaises(Exception):
            d.status()
        with self.assertRaises(Exception):
            d.extio_status()

    def test_parameter_arbitrary(self):
        d = Device(None)
        d.parameter_set('buffer_duration', 2)

    def test_reduction_frequency(self):
        d = Device(None)
        for frequency in [1, 2, 4, 10, 20, 50, 100]:
            with self.subTest(frequency=frequency):
                d.reduction_frequency = frequency
                self.assertEqual(frequency, d.reduction_frequency)

    def test_info(self):
        d = Device(None)
        self.assertEqual('JS110', d.parameter_get('model'))
        self.assertEqual(None, d.parameter_get('device_serial_number'))
        self.assertEqual(None, d.parameter_get('hardware_serial_number'))
