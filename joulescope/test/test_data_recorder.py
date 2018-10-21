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
from joulescope.data_recorder import DataRecorder, DataReader, DataRecorderConfiguration, datafile
from joulescope.stream_buffer import StreamBuffer, usb_packet_factory
import io
import tempfile
import numpy as np
import os
import shutil


class TestDataRecorder(unittest.TestCase):

    def setUp(self):
        self._tempdir = tempfile.mkdtemp()
        self._filename1 = os.path.join(self._tempdir, 'f1.joulescope')

    def tearDown(self):
        shutil.rmtree(self._tempdir)

    def test_init_with_file_handle(self):
        fh = io.BytesIO()
        d = DataRecorder(fh)
        d.close()
        self.assertGreater(len(fh.getbuffer()), 0)

    def test_init_with_filename(self):
        self.assertFalse(os.path.isfile(self._filename1))
        d = DataRecorder(self._filename1)
        self.assertTrue(os.path.isfile(self._filename1))
        d.close()

    def test_configuration_default(self):
        c = DataRecorderConfiguration()
        c.validate()

    def _create_file(self, packet_index, count=None):
        stream_buffer = StreamBuffer(2000, [10])
        if packet_index > 0:
            data = usb_packet_factory(0, packet_index - 1)
            stream_buffer.insert(data)
            stream_buffer.process()

        config = DataRecorderConfiguration()
        config.samples_per_block = 10
        config.reductions = [10]
        config.blocks_per_reduction = [10]
        config.sample_id_offset = packet_index * 126

        fh = io.BytesIO()
        d = DataRecorder(fh, configuration=config)
        data = usb_packet_factory(packet_index, count)
        stream_buffer.insert(data)
        stream_buffer.process()
        d.process(stream_buffer)
        d.close()
        fh.seek(0)

        # dfr = datafile.DataFileReader(fh)
        # dfr.pretty_print()
        # fh.seek(0)

        return fh

    def test_write_read_direct(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        x, data = r.get(0, 10, 1)
        np.testing.assert_allclose(np.arange(0, 10), x)
        np.testing.assert_allclose(np.arange(0, 20, 2), data[:, 0, 0])

    def test_write_read_direct_with_offset(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        d = np.right_shift(r.raw(5, 10), 2)
        x, data = r.get(5, 10, 1)
        np.testing.assert_allclose(np.arange(5, 10), x)
        np.testing.assert_allclose(np.arange(10, 20, 2), data[:, 0, 0])

    def test_write_read_direct_with_sample_overscan_before(self):
        fh = self._create_file(1, 1)  # will be samples 120 to 250 (not 126 to 252)
        r = DataReader().open(fh)
        x, data = r.get(0, 140, 1)
        np.testing.assert_allclose(np.arange(0, 140), x)
        np.testing.assert_allclose(np.arange(240, 500, 2), data[:130, 0, 0])
        np.testing.assert_allclose(np.full(10, np.nan), data[130:, 0, 0])

    def test_write_read_stats_over_samples(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        _, data = r.get(0, 50, 5)
        np.testing.assert_allclose(np.arange(4, 100, 10), data[:, 0, 0])

    def test_write_read_reduction_direct(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        _, data = r.get(0, 100, 10)
        np.testing.assert_allclose(np.arange(9, 200, 20), data[:, 0, 0])

    def test_write_read_reduction_indirect(self):
        fh = self._create_file(0, 2)
        r = DataReader().open(fh)
        _, data = r.get(0, 200, 20)
        np.testing.assert_allclose(np.arange(19, 400, 40), data[:, 0, 0])
