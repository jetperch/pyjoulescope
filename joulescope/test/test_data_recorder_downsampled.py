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
Test the data recorder
"""

import unittest
from joulescope.data_recorder import DataRecorder, DataReader
from joulescope.v0.stream_buffer import DownsamplingStreamBuffer
from joulescope.v0.calibration import Calibration
import io
import tempfile
import numpy as np
import os
import shutil

PACKET_HEADER_SIZE = 8
SAMPLES_PER_PACKET = (512 - PACKET_HEADER_SIZE) // 4


class TestDataRecorderDownsampled(unittest.TestCase):

    def setUp(self):
        self._tempdir = tempfile.mkdtemp()
        self._filename1 = os.path.join(self._tempdir, 'f1.joulescope')

    def tearDown(self):
        shutil.rmtree(self._tempdir)

    def create_sinusoid_data(self, sample_rate, samples):
        x = np.arange(samples, dtype=np.float32)
        x *= (1 / sample_rate)
        data = np.empty(samples * 2, dtype=np.uint16)
        data[0::2] = (2000 * np.sin(2 * np.pi * 1000 * x) + 5000).astype(np.uint16)
        data[1::2] = (2000 * np.cos(2 * np.pi * 42 * x) + 5000).astype(np.uint16)
        np.left_shift(data, 2, out=data)
        data_view = data[1::4]
        np.bitwise_or(data_view, 0x20, out=data_view)
        return data

    def create_sinusoid_file(self, file_duration, input_sample_rate, output_sample_rate,
                             stream_buffer_duration=None, chunk_size=None):
        stream_buffer_duration = 1.0 if stream_buffer_duration is None else float(stream_buffer_duration)
        min_duration = 400000 / output_sample_rate
        stream_buffer_duration = max(stream_buffer_duration, min_duration)
        chunk_size = 1024 if chunk_size is None else int(chunk_size)
        cal = Calibration()
        cal.current_offset[:7] = -3000
        cal.current_gain[:7] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        cal.voltage_offset[:2] = -3000
        cal.voltage_gain[:2] = [1e-3, 1e-4]
        cal.data = cal.save(bytes([0] * 32))

        fh = io.BytesIO()
        d = DataRecorder(fh, calibration=cal)

        buffer = DownsamplingStreamBuffer(stream_buffer_duration, [100], input_sample_rate, output_sample_rate)
        buffer.calibration_set(cal.current_offset, cal.current_gain, cal.voltage_offset, cal.voltage_gain)
        d.stream_notify(buffer)
        input_samples = int(file_duration * input_sample_rate)
        data = self.create_sinusoid_data(input_sample_rate, input_samples)

        i = 0
        while i < input_samples:
            i_next = min(i + chunk_size, input_samples)
            buffer.insert_raw(data[i:i_next])
            buffer.process()
            d.stream_notify(buffer)
            i = i_next

        d.close()
        fh.seek(0)
        return fh

    def test_samples_get(self):
        #fh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_recording_01.jls')
        fh = self.create_sinusoid_file(2.0, 2000000, 100000)
        r = DataReader().open(fh)
        k = r.samples_get(0, 1000, units='samples', fields=['current'])
        self.assertIn('time', k)
        self.assertIn('signals', k)
        self.assertIn('current', k['signals'])
        self.assertIn('value', k['signals']['current'])
        self.assertEqual(k['signals']['current']['units'], 'A')
        i = k['signals']['current']['value']
        self.assertEqual(1000, len(i))
        # todo how do we know this is right?

    def test_statistics_get(self):
        #fh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_recording_01.jls')
        fh = self.create_sinusoid_file(2.0, 2000000, 100000)
        r = DataReader().open(fh)
        ranges = [
            (0, 1000, 'samples'),         # trivial, direct
            (0, 20000, 'samples'),        # trivial, single reduction
            (100000, 101000, 'samples'),  # offset, direct
            (100000, 120000, 'samples'),  # offset, ex
            (99000, 120000, 'samples'),
            (100000, 121000, 'samples'),
            (99000, 121000, 'samples'),
            (0.066780, 0.069004, 'seconds'),
        ]

        for k_start, k_stop, units in ranges[:1]:
            # print(f'range {k_start}:{k_stop}')
            s1 = r.statistics_get(k_start, k_stop, units=units)
            k = r.samples_get(k_start, k_stop, units=units, fields=['current'])
            i_mean = np.mean(k['signals']['current']['value'])
            np.testing.assert_allclose(s1['signals']['current']['µ']['value'], i_mean, rtol=0.0005)
        r.close()

    @unittest.SkipTest
    def test_regression_01(self):
        fname = 'C:/Users/Matth/Documents/joulescope/20201009_185143.jls'
        r = DataReader().open(fname)
        s_start, s_stop = r.sample_id_range
        increment = (s_stop - s_start) // 1000
        r.data_get(s_start, s_stop, increment, units='samples')
