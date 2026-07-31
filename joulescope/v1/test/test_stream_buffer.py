# Copyright 2026 Jetperch LLC
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
Test the v1 StreamBuffer using canned pyjoulescope_driver !data messages.

These tests form the backwards-compatibility regression baseline for the
joulescope package v1 backend.  The message shapes replicate what the
pyjoulescope_driver binding produces for each device:

* JS110: sample_rate=2 MHz, decimate_factor=1, all streams full rate.
* JS220: sample_rate=2 MHz, decimate_factor=2 (h/fs=1 MHz).
* JS320: sample_rate=16 MHz, decimate_factor=16 (h/fs=1 MHz) where
  i/v/p follow h/fs but current_range and GPI remain at 1 Msps.

Data dtypes match the binding: float32 for i/v/p, unpacked uint8 for
current_range (u4 -> u8 conversion in the binding), and packed u1
(np.packbits little bit order) for GPI signals.
"""

import unittest
from joulescope.v1.stream_buffer import StreamBuffer, STATS_FIELD_NAMES
import numpy as np


LEGACY_FIELDS = ['current', 'voltage', 'power', 'current_range',
                 'current_lsb', 'voltage_lsb']


def _msg(field_id, index, sample_id, data, sample_rate, decimate_factor):
    """Replicate the pyjoulescope_driver binding stream message dict."""
    return {
        'sample_id': sample_id,
        'utc': 0,
        'field_id': field_id,
        'index': index,
        'sample_rate': sample_rate,
        'decimate_factor': decimate_factor,
        'time_map': {
            'offset_time': 0,
            'offset_counter': 0,
            'counter_rate': sample_rate,
        },
        'data': data,
    }


class DeviceSim:
    """Generate canned device stream messages into a StreamBuffer.

    :param buffer: The StreamBuffer under test.
    :param sample_rate: The native sample rate reported in messages.
    :param ivp_decimate: The decimate_factor for i, v, p streams.
    :param aux_decimate: The decimate_factor for current_range and GPI.
    """

    def __init__(self, buffer, sample_rate, ivp_decimate, aux_decimate):
        self._b = buffer
        self._sample_rate = sample_rate
        self._ivp_decimate = ivp_decimate
        self._aux_decimate = aux_decimate
        self.sample_id = 0  # full-rate (native) sample id

    def feed(self, duration_ids, current=1.0, voltage=2.0, current_range=3,
             gpi0=0, gpi1=1):
        """Feed one message per stream covering duration_ids native ids."""
        sample_id = self.sample_id
        n_ivp = duration_ids // self._ivp_decimate
        n_aux = duration_ids // self._aux_decimate
        i = np.full(n_ivp, current, dtype=np.float32)
        v = np.full(n_ivp, voltage, dtype=np.float32)
        p = np.full(n_ivp, current * voltage, dtype=np.float32)
        r = np.full(n_aux, current_range, dtype=np.uint8)
        g0 = np.packbits(np.full(n_aux, gpi0, dtype=np.uint8), bitorder='little')
        g1 = np.packbits(np.full(n_aux, gpi1, dtype=np.uint8), bitorder='little')
        for (field_id, index), data, decimate in [
                ((1, 0), i, self._ivp_decimate),
                ((2, 0), v, self._ivp_decimate),
                ((3, 0), p, self._ivp_decimate),
                ((4, 0), r, self._aux_decimate),
                ((5, 0), g0, self._aux_decimate),
                ((5, 1), g1, self._aux_decimate)]:
            self._b.insert('s/x/!data', _msg(
                field_id, index, sample_id, data,
                self._sample_rate, decimate))
        self.sample_id += duration_ids


class TestStreamBufferJs110(unittest.TestCase):
    """JS110 shape: 2 Msps native, no decimation."""

    def setUp(self):
        self.b = StreamBuffer(0.01, frequency=2000000, device='js110',
                              output_frequency=2000000)
        self.sim = DeviceSim(self.b, 2000000, 1, 1)

    def test_empty(self):
        self.assertEqual(len(self.b), 20000)
        self.assertEqual((0, 0), self.b.sample_id_range)

    def test_insert_and_ranges(self):
        for _ in range(4):
            self.sim.feed(1000)
        self.assertEqual((0, 4000), self.b.sample_id_range)

    def test_samples_get_legacy_fields_default(self):
        for _ in range(4):
            self.sim.feed(1000)
        s = self.b.samples_get(0, 4000)
        self.assertEqual(LEGACY_FIELDS, list(s['signals'].keys()))
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])
        np.testing.assert_allclose(2.0, s['signals']['voltage']['value'])
        np.testing.assert_allclose(2.0, s['signals']['power']['value'])
        np.testing.assert_equal(3, s['signals']['current_range']['value'])
        np.testing.assert_equal(0, s['signals']['current_lsb']['value'])
        np.testing.assert_equal(1, s['signals']['voltage_lsb']['value'])
        self.assertEqual('A', s['signals']['current']['units'])
        self.assertEqual('V', s['signals']['voltage']['units'])
        self.assertEqual('W', s['signals']['power']['units'])
        self.assertEqual(4000, s['time']['samples']['value'])
        self.assertEqual([0, 4000], s['time']['sample_id_range']['value'])

    def test_samples_get_single_field_str(self):
        self.sim.feed(1000)
        v = self.b.samples_get(0, 1000, fields='voltage')
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(np.float32, v.dtype)
        np.testing.assert_allclose(2.0, v)

    def test_samples_get_clamps_range(self):
        self.sim.feed(1000)
        s = self.b.samples_get(0, 100000)
        self.assertEqual([0, 1000], s['time']['sample_id_range']['value'])

    def test_statistics_get(self):
        for k in range(4):
            self.sim.feed(1000, current=float(k))
        out, (start, stop) = self.b.statistics_get(0, 4000)
        self.assertEqual((0, 4000), (start, stop))
        self.assertEqual(len(STATS_FIELD_NAMES), len(out))
        self.assertEqual(4000, out[0]['length'])
        self.assertAlmostEqual(1.5, out[0]['mean'], places=6)
        self.assertAlmostEqual(0.0, out[0]['min'], places=6)
        self.assertAlmostEqual(3.0, out[0]['max'], places=6)
        self.assertAlmostEqual(2.0, out[1]['mean'], places=6)

    def test_data_get(self):
        for k in range(4):
            self.sim.feed(1000, current=float(k))
        out = self.b.data_get(0, 4000, 1000)
        self.assertEqual((4, len(STATS_FIELD_NAMES)), out.shape)
        for k in range(4):
            self.assertAlmostEqual(float(k), out[k, 0]['mean'], places=6)
            self.assertAlmostEqual(2.0, out[k, 1]['mean'], places=6)

    def test_insert_unknown_field_ignored(self):
        data = np.zeros(100, dtype=np.float32)
        self.b.insert('s/x/!data', _msg(9, 0, 0, data, 2000000, 1))
        self.assertEqual((0, 0), self.b.sample_id_range)

    def test_insert_gap_fills_nan(self):
        self.sim.feed(1000)
        self.sim.sample_id += 1000  # skip 1000 samples
        self.sim.feed(1000)
        s = self.b.samples_get(0, 3000)
        i = s['signals']['current']['value']
        self.assertTrue(np.all(np.isfinite(i[:1000])))
        self.assertTrue(np.all(np.isnan(i[1000:2000])))
        self.assertTrue(np.all(np.isfinite(i[2000:])))


class TestStreamBufferJs220(unittest.TestCase):
    """JS220 shape: 2 Msps native, decimate_factor=2 (h/fs=1 MHz)."""

    def setUp(self):
        self.b = StreamBuffer(0.01, frequency=2000000, device='js220',
                              output_frequency=1000000)
        self.sim = DeviceSim(self.b, 2000000, 2, 2)

    def test_insert_and_samples_get(self):
        for _ in range(4):
            self.sim.feed(2000)  # 2000 native ids -> 1000 output samples
        self.assertEqual((0, 4000), self.b.sample_id_range)
        s = self.b.samples_get(0, 4000)
        self.assertEqual(LEGACY_FIELDS, list(s['signals'].keys()))
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])
        np.testing.assert_equal(3, s['signals']['current_range']['value'])
        np.testing.assert_equal(1, s['signals']['voltage_lsb']['value'])


class TestStreamBufferJs320(unittest.TestCase):
    """JS320 shape: 16 Msps native, i/v/p follow h/fs, aux at 1 Msps."""

    def setUp(self):
        self.b = StreamBuffer(0.01, frequency=16000000, device='js220',
                              output_frequency=1000000)
        self.sim = DeviceSim(self.b, 16000000, 16, 16)

    def test_full_rate(self):
        for _ in range(4):
            self.sim.feed(16000)  # 16000 native ids -> 1000 output samples
        self.assertEqual((0, 4000), self.b.sample_id_range)
        s = self.b.samples_get(0, 4000)
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])
        np.testing.assert_allclose(2.0, s['signals']['voltage']['value'])
        np.testing.assert_equal(3, s['signals']['current_range']['value'])
        np.testing.assert_equal(0, s['signals']['current_lsb']['value'])
        np.testing.assert_equal(1, s['signals']['voltage_lsb']['value'])

    def test_downsampled_aux_host_decimate(self):
        # h/fs = 10 kHz: i/v/p decimate on-instrument (factor 1600),
        # current_range and GPI remain at 1 Msps (factor 16) and must be
        # decimated on the host by 100x.
        b = StreamBuffer(0.1, frequency=16000000, device='js220',
                         output_frequency=10000)
        sim = DeviceSim(b, 16000000, 1600, 16)
        for _ in range(4):
            sim.feed(160000)  # -> 100 output samples per feed
        self.assertEqual((0, 400), b.sample_id_range)
        s = b.samples_get(0, 400)
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])
        np.testing.assert_allclose(2.0, s['signals']['voltage']['value'])
        np.testing.assert_equal(3, s['signals']['current_range']['value'])
        np.testing.assert_equal(0, s['signals']['current_lsb']['value'])
        np.testing.assert_equal(1, s['signals']['voltage_lsb']['value'])

    def test_statistics(self):
        for k in range(4):
            self.sim.feed(16000, current=float(k))
        out, (start, stop) = self.b.statistics_get(0, 4000)
        self.assertEqual((0, 4000), (start, stop))
        self.assertAlmostEqual(1.5, out[0]['mean'], places=6)


if __name__ == '__main__':
    unittest.main()


class TestStreamBufferInactive(unittest.TestCase):

    def test_samples_get_inactive_field_returns_nan(self):
        b = StreamBuffer(0.01, frequency=2000000, device='js110',
                         output_frequency=2000000)
        for buf in b.buffers.values():
            if buf._name in ['gpi0', 'gpi1']:
                buf.active = False
        sim = DeviceSim(b, 2000000, 1, 1)
        n = 1000
        sample_id = 0
        i = np.full(n, 1.0, dtype=np.float32)
        r = np.full(n, 3, dtype=np.uint8)
        for (field_id, index), data in [((1, 0), i), ((2, 0), i),
                                        ((3, 0), i), ((4, 0), r)]:
            b.insert('s/x/!data', _msg(field_id, index, sample_id, data,
                                       2000000, 1))
        self.assertEqual((0, 1000), b.sample_id_range)
        s = b.samples_get(0, 1000)
        v = s['signals']['current_lsb']['value']
        self.assertEqual(1000, len(v))
        self.assertTrue(np.all(np.isnan(v)))
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])

    def test_duplicate_stream_message(self):
        b = StreamBuffer(0.01, frequency=2000000, device='js110',
                         output_frequency=2000000)
        sim = DeviceSim(b, 2000000, 1, 1)
        sim.feed(1000)
        sim.sample_id = 0  # retransmit everything
        sim.feed(1000)
        self.assertEqual((0, 1000), b.sample_id_range)
        s = b.samples_get(0, 1000)
        np.testing.assert_allclose(1.0, s['signals']['current']['value'])


class TestStreamBufferExtendedSignals(unittest.TestCase):

    def _buffer(self, extras):
        b = StreamBuffer(0.01, frequency=1000000, device='js220',
                         output_frequency=1000000)
        b.extra_signals = extras
        return b

    def test_extra_signals_allocation(self):
        b = self._buffer([])
        for idx in [(5, 2), (5, 3), (5, 7)]:
            self.assertNotIn(idx, b.buffers)
        b.extra_signals = ['2', 'gpi3', 'T']
        self.assertEqual(['gpi2', 'gpi3', 'trigger_in'], b.extra_signals)
        for idx in [(5, 2), (5, 3), (5, 7)]:
            self.assertIn(idx, b.buffers)
        b.extra_signals = None
        for idx in [(5, 2), (5, 3), (5, 7)]:
            self.assertNotIn(idx, b.buffers)

    def test_extra_signals_invalid(self):
        b = self._buffer([])
        with self.assertRaises(KeyError):
            b.extra_signals = ['__invalid__']
        with self.assertRaises(ValueError):
            b.extra_signals = ['current']  # valid field, not extended

    def test_extended_stream_and_samples_get(self):
        b = self._buffer(['gpi2', 'gpi3', 'trigger_in'])
        sim = DeviceSim(b, 1000000, 1, 1)
        n = 1000
        ones = np.packbits(np.ones(n, dtype=np.uint8), bitorder='little')
        zeros = np.packbits(np.zeros(n, dtype=np.uint8), bitorder='little')
        for _ in range(2):
            sim.feed(n)
            for index, data in [(2, ones), (3, zeros), (7, ones)]:
                b.insert('s/x/!data', _msg(5, index, sim.sample_id - n,
                                           data, 1000000, 1))
        s = b.samples_get(0, 2 * n, fields=['0', '1', '2', '3', 'T'])
        self.assertEqual(['0', '1', '2', '3', 'T'],
                         list(s['signals'].keys()))
        np.testing.assert_equal(0, s['signals']['0']['value'])
        np.testing.assert_equal(1, s['signals']['1']['value'])
        np.testing.assert_equal(1, s['signals']['2']['value'])
        np.testing.assert_equal(0, s['signals']['3']['value'])
        np.testing.assert_equal(1, s['signals']['T']['value'])
        # canonical names also work
        s2 = b.samples_get(0, 2 * n, fields=['gpi2', 'trigger_in'])
        np.testing.assert_equal(1, s2['signals']['gpi2']['value'])
        # legacy statistics remain 6 columns
        out, _ = b.statistics_get(0, 2 * n)
        self.assertEqual(len(STATS_FIELD_NAMES), len(out))

    def test_extended_absent_returns_nan(self):
        b = self._buffer([])
        sim = DeviceSim(b, 1000000, 1, 1)
        sim.feed(1000)
        s = b.samples_get(0, 1000, fields=['current', 'gpi2'])
        v = s['signals']['gpi2']['value']
        self.assertEqual(1000, len(v))
        self.assertTrue(np.all(np.isnan(v)))

    def test_legacy_default_fields_unchanged(self):
        b = self._buffer(['gpi2'])
        sim = DeviceSim(b, 1000000, 1, 1)
        sim.feed(1000)
        b.insert('s/x/!data', _msg(
            5, 2, 0, np.packbits(np.ones(1000, dtype=np.uint8),
                                 bitorder='little'), 1000000, 1))
        s = b.samples_get(0, 1000)
        self.assertEqual(LEGACY_FIELDS, list(s['signals'].keys()))

    def test_data_get_inactive_buffers(self):
        # signals subset: unselected buffers are inactive; data_get must
        # fill NaN, not raise (audit fix: missing continue)
        b = self._buffer([])
        for idx in [(3, 0), (4, 0), (5, 0), (5, 1)]:
            b.buffers[idx].active = False
        sim = DeviceSim(b, 1000000, 1, 1)
        n = 1000
        i = np.full(n, 1.0, dtype=np.float32)
        b.insert('s/x/!data', _msg(1, 0, 0, i, 1000000, 1))
        b.insert('s/x/!data', _msg(2, 0, 0, i, 1000000, 1))
        out = b.data_get(0, n, 100)
        self.assertEqual((10, len(STATS_FIELD_NAMES)), out.shape)
        self.assertAlmostEqual(1.0, out[0, 0]['mean'], places=6)
        self.assertTrue(np.isnan(out[0, 2]['mean']))  # power inactive
        self.assertTrue(np.isnan(out[0, 4]['mean']))  # gpi0 inactive
