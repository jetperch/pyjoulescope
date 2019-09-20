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
Test stream buffer
"""

import unittest
import numpy as np
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
from joulescope.stream_buffer import Statistics, StreamBuffer, usb_packet_factory


SAMPLES_PER = 126


class TestStatistics(unittest.TestCase):

    def test_initialize_empty(self):
        s = Statistics()
        self.assertEqual(0, len(s))
        self.assertEqual((3, 4), s.value.shape)
        np.testing.assert_allclose(0, s.value[:, 0])
        np.testing.assert_allclose(0, s.value[:, 1])

    def test_initialize(self):
        d = np.arange(3*4, dtype=np.float32).reshape((3, 4))
        s = Statistics(length=10, stats=d)
        self.assertEqual(10, len(s))
        np.testing.assert_allclose(d, s.value)

    def test_combine(self):
        d1 = np.array([[1, 0, 1, 1], [2, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float32)
        d2 = np.array([[3, 0, 3, 3], [4, 0, 4, 4], [5, 0, 5, 5]], dtype=np.float32)
        e = np.array([[2, 1, 1, 3], [3, 1, 2, 4], [4, 1, 3, 5]], dtype=np.float32)
        s1 = Statistics(length=1, stats=d1)
        s2 = Statistics(length=1, stats=d2)
        s1.combine(s2)
        self.assertEqual(2, len(s1))
        np.testing.assert_allclose(e, s1.value)

    def test_combine_other_empty(self):
        d = np.array([[1, 0, 1, 1], [2, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float32)
        s1 = Statistics(length=10, stats=d)
        s2 = Statistics()
        s1.combine(s2)
        self.assertEqual(10, len(s1))
        np.testing.assert_allclose(d, s1.value)

    def test_combine_self_empty(self):
        d = np.array([[1, 0, 1, 1], [2, 0, 2, 2], [3, 0, 3, 3]], dtype=np.float32)
        s1 = Statistics()
        s2 = Statistics(length=10, stats=d)
        s1.combine(s2)
        self.assertEqual(10, len(s1))
        np.testing.assert_allclose(d, s1.value)

    def test_combine_both_empty(self):
        s1 = Statistics()
        s2 = Statistics()
        s1.combine(s2)
        self.assertEqual(0, len(s1))


class TestStreamBuffer(unittest.TestCase):

    def test_init(self):
        b = StreamBuffer(1000, [10, 10], 1000.0)
        self.assertIsNotNone(b)
        self.assertEqual(1000, len(b))
        self.assertEqual(1000.0, b.sampling_frequency)
        self.assertEqual((0, 0), b.sample_id_range)
        self.assertEqual((0.0, 1.0), b.limits_time)
        self.assertEqual((-1000, 0), b.limits_samples)
        self.assertEqual(0, b.time_to_sample_id(1.0))
        self.assertEqual(1.0, b.sample_id_to_time(0))
        del b

    def test_insert_process(self):
        b = StreamBuffer(1000, [100, 10], 1000.0)
        b.suppress_mode = 'off'
        frame = usb_packet_factory(0, 1)
        self.assertEqual(0, b.status()['device_sample_id']['value'])
        b.insert(frame)
        self.assertEqual(126, b.status()['device_sample_id']['value'])
        self.assertEqual(0, b.status()['sample_id']['value'])
        b.process()
        self.assertEqual((0, 126), b.sample_id_range)
        self.assertEqual((0.0, 1.0), b.limits_time)
        self.assertEqual((-874, 126), b.limits_samples)
        self.assertEqual(126, b.time_to_sample_id(1.0))
        self.assertEqual(1.0, b.sample_id_to_time(126))
        self.assertEqual(0.999, b.sample_id_to_time(125))
        self.assertEqual(b.limits_time[0], b.sample_id_to_time(b.limits_samples[0]))
        self.assertEqual(b.limits_time[1], b.sample_id_to_time(b.limits_samples[1]))
        self.assertEqual(126, b.status()['sample_id']['value'])
        data = b.raw_get(0, 126)
        expect = np.arange(126*2, dtype=np.uint16).reshape((126, 2))
        np.testing.assert_allclose(expect, np.right_shift(data, 2))
        np.testing.assert_allclose(expect, b.data_buffer[0:126*2].reshape((126, 2)))
        data = b.data_get(0, 126)
        np.testing.assert_allclose(expect[:, 0], data[:, 0, 0])

    def test_wrap_aligned(self):
        frame = usb_packet_factory(0, 4)
        b = StreamBuffer(2 * SAMPLES_PER, [], 1000.0)
        b.suppress_mode = 'off'
        b.insert(frame)
        b.process()
        self.assertEqual((SAMPLES_PER * 2, SAMPLES_PER * 4), b.sample_id_range)
        data = b.raw_get(SAMPLES_PER * 2, SAMPLES_PER * 4)
        data = np.right_shift(data, 2)
        expect = np.arange(SAMPLES_PER * 4, SAMPLES_PER * 8, dtype=np.uint16).reshape((SAMPLES_PER * 2, 2))
        np.testing.assert_allclose(expect, data)
        np.testing.assert_allclose(expect[:, 0], b.data_buffer[::2])
        data = b.data_get(SAMPLES_PER * 2, SAMPLES_PER * 4)
        np.testing.assert_allclose(expect[:, 0], data[:, 0, 0])
        np.testing.assert_allclose(expect[:, 1], data[:, 1, 0])

    def test_wrap_unaligned(self):
        frame = usb_packet_factory(0, 4)
        b = StreamBuffer(2 * SAMPLES_PER + 2, [], 1000.0)
        b.insert(frame)
        b.process()
        self.assertEqual(SAMPLES_PER * 4, b.sample_id_range[1])
        data = np.right_shift(b.raw_get(SAMPLES_PER * 2, SAMPLES_PER * 4), 2)
        expect = np.arange(SAMPLES_PER * 4, SAMPLES_PER * 8, dtype=np.uint16).reshape((SAMPLES_PER * 2, 2))
        np.testing.assert_allclose(expect, data)

    def test_get_over_samples(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.data_get(0, 21, 5)
        self.assertEqual((4, 3, 4), data.shape)
        np.testing.assert_allclose(np.arange(10), b.data_buffer[0:10])  # processed correctly
        np.testing.assert_allclose([4.0, 14.0, 24.0, 34.0], data[:, 0, 0])
        np.testing.assert_allclose([4.0, 8.0, 0.0, 8.0], data[0, 0, :])
        np.testing.assert_allclose([5.0, 15.0, 25.0, 35.0], data[:, 1, 0])

    def test_get_over_reduction_direct(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.data_get(0, 20, 10)
        self.assertEqual((2, 3, 4), data.shape)
        np.testing.assert_allclose([9.0, 33.0, 0.0, 18.0], data[0, 0, :])
        np.testing.assert_allclose([29.0, 33.0, 20.0, 38.0], data[1, 0, :])
        np.testing.assert_allclose([10.0, 33.0, 1.0, 19.0], data[0, 1, :])
        np.testing.assert_allclose([30.0, 33.0, 21.0, 39.0], data[1, 1, :])

    def test_get_over_reduction(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.data_get(0, 40, 20)
        self.assertEqual((2, 3, 4), data.shape)
        np.testing.assert_allclose([19.0, 133.0, 0.0, 38.0], data[0, 0, :])
        np.testing.assert_allclose([20.0, 133.0, 1.0, 39.0], data[0, 1, :])
        np.testing.assert_allclose([59.0, 133.0, 40.0, 78.0], data[1, 0, :])

    def test_get_over_reduction_direct_with_raw_nan(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        frame = usb_packet_factory(0, 1)
        frame[8+11*4:8+15*4:] = 0xff
        b.insert(frame)
        b.process()
        r = b.get_reduction(0, 0, 126)
        self.assertTrue(all(np.isfinite(r[:, 0, 0])))

    def test_get_over_reduction_direct_with_reduction0_nan(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        frame = usb_packet_factory(0, 1)
        frame[8+10*4:8+22*4:] = 0xff
        b.insert(frame)
        b.process()
        r0 = b.get_reduction(0, 0, 126)
        self.assertFalse(np.isfinite(r0[1, 0, 0]))
        r1 = b.get_reduction(1, 0, 126)
        self.assertTrue(np.isfinite(r0[0, 0, 0]))

    def test_calibration(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        self.assertEqual('off', b.suppress_mode)
        b.calibration_set([-10.0] * 7, [2.0] * 7, [-2.0, 1.0], [4.0, 1.0])
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        data = b.data_get(0, 10, 1)
        self.assertEqual((10, 3, 4), data.shape)
        np.testing.assert_allclose([-20.0, -4.0, 80.0], data[0, :, 0])
        np.testing.assert_allclose([12.,  60., 720.], data[8, :, 0])

    def stream_buffer_01(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        frame = usb_packet_factory(0, 1)
        b.insert(frame)
        b.process()
        return b

    def stream_buffer_02(self):
        b = StreamBuffer(2000, [10, 10], 1000.0)
        b.suppress_mode = 'off'
        b.insert(usb_packet_factory(0))
        b.insert(usb_packet_factory(2))
        b.process()
        return b

    def test_stats_direct(self):
        b = self.stream_buffer_01()
        s = b.stats_get(13, 13)
        self.assertIsNone(s)
        np.testing.assert_allclose(b.data_get(13, 14, 1)[0, 0, 0], b.stats_get(13, 14)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(13, 29, 1)[:, 0, 0]), b.stats_get(13, 29)[0, 0])

    def test_stats_direct_nan(self):
        b = self.stream_buffer_02()
        self.assertEqual((0, 126 * 3), b.sample_id_range)
        d = b.data_get(0, 126 * 3, 1)[:, 0, 0]
        self.assertTrue(np.all(np.logical_not(np.isnan(d[0:126]))))
        self.assertTrue(np.all(np.isnan(d[126:252])))
        self.assertTrue(np.all(np.logical_not(np.isnan(d[252:]))))

        self.assertFalse(np.isnan(b.stats_get(121, 126)[0, 0]))
        self.assertFalse(np.isnan(b.stats_get(124, 129)[0, 0]))
        self.assertTrue(np.isnan(b.stats_get(130, 135)[0, 0]))
        self.assertTrue(np.isnan(b.stats_get(247, 252)[0, 0]))
        self.assertFalse(np.isnan(b.stats_get(249, 254)[0, 0]))

    def test_stats_over_single_reduction_exact(self):
        b = self.stream_buffer_01()
        np.testing.assert_allclose(np.mean(b.data_get(10, 20, 1)[:, 0, 0]), b.stats_get(10, 20)[0, 0])

    def test_stats_over_single_reduction_leading(self):
        b = self.stream_buffer_01()
        np.testing.assert_allclose(np.mean(b.data_get(9, 20, 1)[:, 0, 0]), b.stats_get(9, 20)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(8, 20, 1)[:, 0, 0]), b.stats_get(8, 20)[0, 0])

    def test_stats_over_single_reduction_trailing(self):
        b = self.stream_buffer_01()
        np.testing.assert_allclose(np.mean(b.data_get(10, 21, 1)[:, 0, 0]), b.stats_get(10, 21)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(10, 22, 1)[:, 0, 0]), b.stats_get(10, 22)[0, 0])

    def test_stats_over_single_reduction_extended(self):
        b = self.stream_buffer_01()
        np.testing.assert_allclose(np.mean(b.data_get(9, 21, 1)[:, 0, 0]), b.stats_get(9, 21)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(5, 25, 1)[:, 0, 0]), b.stats_get(5, 25)[0, 0])

    def test_stats_over_single_reductions(self):
        b = self.stream_buffer_01()
        np.testing.assert_allclose(np.mean(b.data_get(9, 20, 1)[:, 0, 0]), b.stats_get(9, 20)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(10, 21, 1)[:, 0, 0]), b.stats_get(10, 21)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(9, 21, 1)[:, 0, 0]), b.stats_get(9, 21)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(9, 101, 1)[:, 0, 0]), b.stats_get(9, 101)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(5, 105, 1)[:, 0, 0]), b.stats_get(5, 105)[0, 0])

    def test_stats_over_single_reduction_nan(self):
        b = self.stream_buffer_02()
        np.testing.assert_allclose(np.mean(b.data_get(120, 126, 1)[:, 0, 0]), b.stats_get(120, 130)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(120, 126, 1)[:, 0, 0]), b.stats_get(120, 140)[0, 0])
        np.testing.assert_allclose(np.mean(b.data_get(110, 126, 1)[:, 0, 0]), b.stats_get(110, 130)[0, 0])

    def test_stats_over_reductions_nan(self):
        b = self.stream_buffer_02()
        np.testing.assert_allclose(np.mean(b.data_get(0, 126, 1)[:, 0, 0]), b.stats_get(0, 200)[0, 0])

    def test_insert_raw_simple(self):
        b = StreamBuffer(1000, [100, 100, 100], 1000.0)
        b.suppress_mode = 'off'
        expect = np.arange(126 * 2, dtype=np.uint16).reshape((126, 2))
        b.insert_raw(np.left_shift(expect, 2))
        b.process()
        data = b.data_get(0, 126)
        np.testing.assert_allclose(expect[:, 0], data[:, 0, 0])

    def test_insert_raw_wrap(self):
        b = StreamBuffer(200, [], 1000.0)
        expect = np.arange(250 * 2, dtype=np.uint16).reshape((250, 2))
        b.insert_raw(np.left_shift(expect[:100], 2))
        b.process()
        b.insert_raw(np.left_shift(expect[100:], 2))
        b.process()
        data = b.data_get(50, 250)
        np.testing.assert_allclose(expect[50:, 0], data[:, 0, 0])

    def test_voltage_range(self):
        b = StreamBuffer(200, [], 1000.0)
        self.assertEqual(0, b.voltage_range)
        b.voltage_range = 1
        self.assertEqual(1, b.voltage_range)
