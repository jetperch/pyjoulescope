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
Test the JLS v2 writer.
"""

import unittest
from joulescope import JlsWriter
from joulescope.jls_v2_writer import signals_validator, sampling_rate_validator
from joulescope.v0.stream_buffer import StreamBuffer, usb_packet_factory
from pyjls import Reader
import tempfile
import numpy as np
import os
import shutil

PACKET_HEADER_SIZE = 8
SAMPLES_PER_PACKET = (512 - PACKET_HEADER_SIZE) // 4


class FakeJS110:

    def __init__(self):
        self._sampling_frequency = 2000000
        self.info_calls = 0

    def __str__(self):
        return 'Joulescope:000001'

    def parameter_get(self, name):
        if name == 'sampling_frequency':
            return self._sampling_frequency
        else:
            raise ValueError(r'invalid parameter {name}')

    def parameter_set(self, name, value):
        if name == 'sampling_frequency':
            self._sampling_frequency = int(value)
        else:
            raise ValueError(r'invalid parameter {name}')

    def info(self):
        self.info_calls += 1
        return {
            'type': 'info',
            'ver': 1,
            'model': 'JS110',
            'hardware_version': 'H',
            'serial_number': '000001',
            'ctl': {
                'mfg': {
                    'country': 'USA',
                    'location': 'MD_01',
                    'lot': '201927_00'},
                'hw': {
                    'rev': 'H',
                    'sn_mcu': '0123456789abcdef0123456789abcdef',
                    'sn_mfg': '000001'},
                'fw': {'ver': '1.3.2'},
                'fpga': {'ver': '0.2.0', 'prod_id': '0x9314acf2'}},
            'sensor': {
                'fw': {'ver': '1.3.2'},
                'fpga': {'ver': '1.2.1'}},
        }


class TestJlsWriter(unittest.TestCase):

    def setUp(self):
        self._tempdir = tempfile.mkdtemp()
        self._filename1 = os.path.join(self._tempdir, 'f1.jls')

    def tearDown(self):
        shutil.rmtree(self._tempdir)

    def test_empty_file(self):
        self.assertFalse(os.path.isfile(self._filename1))
        d = FakeJS110()
        with JlsWriter(d, self._filename1) as wr:
            pass  # empty file
        self.assertTrue(os.path.isfile(self._filename1))

    def _create_file(self, packet_index, count=None, signals=None):
        d = FakeJS110()
        stream_buffer = StreamBuffer(60.0, [10], 1000.0)
        stream_buffer.suppress_mode = 'off'
        if packet_index > 0:
            data = usb_packet_factory(0, packet_index - 1)
            stream_buffer.insert(data)
            stream_buffer.process()
        with JlsWriter(d, self._filename1, signals=signals) as wr:
            wr.stream_notify(stream_buffer)
            data = usb_packet_factory(packet_index, count)
            stream_buffer.insert(data)
            stream_buffer.process()
            wr.stream_notify(stream_buffer)
        self.assertTrue(os.path.isfile(self._filename1))

    def test_simple_file(self):
        self._create_file(0, 2)
        with Reader(self._filename1) as r:
            self.assertEqual([0, 1], sorted(r.sources.keys()))
            self.assertEqual(3, len(r.signals))
            self.assertEqual([0, 1, 2], sorted(r.signals.keys()))
            i = r.signals[1]
            self.assertEqual('current', i.name)
            self.assertEqual('A', i.units)
            self.assertEqual(252, i.length)
            v = r.signals[2]
            self.assertEqual('voltage', v.name)
            self.assertEqual('V', v.units)
            self.assertEqual(252, v.length)

            i_data = r.fsr(1, 0, i.length)
            np.testing.assert_allclose(np.arange(0, i.length * 2, 2), i_data)

    def _source_and_rate(self):
        with Reader(self._filename1) as r:
            return r.sources[1], r.signals[1].sample_rate

    def test_info_args(self):
        d = FakeJS110()
        info = {'model': 'JS320', 'hardware_version': '1.0.0', 'serial_number': '8W2A'}
        with JlsWriter(d, self._filename1, info=info, sampling_frequency='1 MHz'):
            pass
        self.assertEqual(0, d.info_calls)
        source, sample_rate = self._source_and_rate()
        self.assertEqual('JS320', source.model)
        self.assertEqual('8W2A', source.serial_number)
        self.assertEqual(1000000, sample_rate)

    def test_constructor_does_not_access_open_device(self):
        d = FakeJS110()
        d.is_open = True
        wr = JlsWriter(d, self._filename1)
        self.assertEqual(0, d.info_calls)
        d.parameter_set('sampling_frequency', 100000)  # after construction, before open
        with wr:
            pass
        self.assertEqual(1, d.info_calls)
        source, sample_rate = self._source_and_rate()
        self.assertEqual('000001', source.serial_number)
        self.assertEqual(100000, sample_rate)

    def test_args_allow_open_from_callback(self):
        d = FakeJS110()
        wr = JlsWriter(d, self._filename1, info=d.info(), sampling_frequency=d.parameter_get('sampling_frequency'))
        d.info = None  # open() must not access the device
        d.parameter_get = None
        with wr:
            pass
        source, sample_rate = self._source_and_rate()
        self.assertEqual('000001', source.serial_number)
        self.assertEqual(2000000, sample_rate)

    def test_reopen_reuses_device_read(self):
        d = FakeJS110()
        wr = JlsWriter(d, self._filename1)
        wr.open()
        wr.close()
        wr.open()
        wr.close()
        self.assertEqual(1, d.info_calls)

    def test_closed_device_read_at_open(self):
        d = FakeJS110()
        wr = JlsWriter(d, self._filename1)
        self.assertEqual(0, d.info_calls)
        with wr:
            pass
        self.assertEqual(1, d.info_calls)


class TestValidators(unittest.TestCase):

    def test_signals(self):
        self.assertEqual(['current', 'voltage'], signals_validator('current, Voltage'))
        self.assertEqual(['power'], signals_validator(['power']))
        with self.assertRaises(ValueError):
            signals_validator('gpi0')

    def test_sampling_rate(self):
        for value, expect in [(2000000, 2000000), ('2000000', 2000000), ('1 MHz', 1000000),
                              ('500 kHz', 500000), ('10 Hz', 10), ('1MHz', 1000000)]:
            with self.subTest(value=value):
                self.assertEqual(expect, sampling_rate_validator(value))
        with self.assertRaises(ValueError):
            sampling_rate_validator('fast')
