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
from joulescope.stream_buffer import StreamBuffer, usb_packet_factory, usb_packet_factory_signal
import io
import tempfile
import numpy as np
import os
import shutil

PACKET_HEADER_SIZE = 8
SAMPLES_PER_PACKET = (512 - PACKET_HEADER_SIZE) // 4


class TestDataRecorder(unittest.TestCase):

    def setUp(self):
        self._tempdir = tempfile.mkdtemp()
        self._filename1 = os.path.join(self._tempdir, 'f1.joulescope')

    def tearDown(self):
        shutil.rmtree(self._tempdir)

    def test_init_with_file_handle(self):
        fh = io.BytesIO()
        d = DataRecorder(fh, 2000)
        d.close()
        self.assertGreater(len(fh.getbuffer()), 0)

    def test_init_with_filename(self):
        self.assertFalse(os.path.isfile(self._filename1))
        d = DataRecorder(self._filename1, 2000)
        self.assertTrue(os.path.isfile(self._filename1))
        d.close()

    def _create_file(self, packet_index, count=None):
        stream_buffer = StreamBuffer(2000, [10])
        stream_buffer.suppress_mode = 'off'
        if packet_index > 0:
            data = usb_packet_factory(0, packet_index - 1)
            stream_buffer.insert(data)
            stream_buffer.process()

        fh = io.BytesIO()
        d = DataRecorder(fh, sampling_frequency=1000)
        d.process(stream_buffer)
        data = usb_packet_factory(packet_index, count)
        stream_buffer.insert(data)
        stream_buffer.process()
        d.process(stream_buffer)
        d.close()
        fh.seek(0)

        # from joulescope import datafile
        # dfr = datafile.DataFileReader(fh)
        # dfr.pretty_print()
        # fh.seek(0)
        return fh

    def test_write_read_direct(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get(0, 10, 1)
        np.testing.assert_allclose(np.arange(0, 20, 2), data[:, 0, 0])

    def test_write_read_direct_with_offset(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        # d = np.right_shift(r.raw(5, 10), 2)
        data = r.get(5, 10, 1)
        np.testing.assert_allclose(np.arange(10, 20, 2), data[:, 0, 0])

    def test_write_read_direct_with_sample_overscan_before(self):
        fh = self._create_file(1, 3)  # will be samples 120 to 250 (not 126 to 252)
        r = DataReader().open(fh)
        data = r.get(0, 140, 1)
        np.testing.assert_allclose(np.arange(252, 532, 2), data[:, 0, 0])

    def test_write_read_stats_over_samples(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get(0, 50, 5)
        np.testing.assert_allclose(np.arange(4, 100, 10), data[:, 0, 0])

    def test_write_read_stats_over_samples_offset(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get(5, 50, 10)
        np.testing.assert_allclose(np.arange(9, 70, 20), data[:, 0, 0])

    def test_write_read_get_reduction(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get_reduction(0, 100)
        np.testing.assert_allclose(np.arange(9, 200, 20), data[:, 0, 0])

    def test_write_read_get_reduction_offset(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get_reduction(30, 95)
        np.testing.assert_allclose(np.arange(69, 180, 20), data[:, 0, 0])

    def test_write_read_reduction_direct(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get(0, 100, 10)
        np.testing.assert_allclose(np.arange(9, 200, 20), data[:, 0, 0])

    def test_write_read_reduction_indirect(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        data = r.get(0, 200, 20)
        np.testing.assert_allclose(np.arange(19, 400, 40), data[:, 0, 0])

    def _create_large_file(self, samples=None):
        """Create a large file.

        :param samples: The total number of samples which will be rounded
            to a full USB packet.
        """
        sample_rate = 2000000
        samples_total = sample_rate * 2
        packets_per_burst = 128
        bursts = int(np.ceil(samples / (SAMPLES_PER_PACKET * packets_per_burst)))
        stream_buffer = StreamBuffer(sample_rate, [100])
        fh = io.BytesIO()
        d = DataRecorder(fh, sampling_frequency=sample_rate)
        d.process(stream_buffer)
        for burst_index in range(bursts):
            packet_index = burst_index * packets_per_burst
            frames = usb_packet_factory_signal(packet_index, count=packets_per_burst, samples_total=samples_total)
            stream_buffer.insert(frames)
            stream_buffer.process()
            d.process(stream_buffer)
        d.close()
        fh.seek(0)

        # dfr = datafile.DataFileReader(fh)
        # dfr.pretty_print()
        # fh.seek(0)

        return fh

    def test_large_file(self):
        sample_count = 2000000 * 2
        fh = self._create_large_file(sample_count)
        r = DataReader().open(fh)
        self.assertEqual([0, sample_count], r.sample_id_range)
        reduction = r.get_reduction()
        self.assertEqual(sample_count / 20000, len(reduction))
