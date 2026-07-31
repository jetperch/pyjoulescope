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

"""Test the v1 Device using a stubbed joulescope_driver."""

import unittest
from joulescope.v1.device import Device


class FakeDriver:

    def __init__(self):
        self.published = []
        self.subscribed = []
        self.unsubscribed = []

    def open(self, path, mode=None, timeout=None):
        return 0

    def close(self, path, timeout=None):
        return 0

    def publish(self, topic, value, timeout=None):
        self.published.append((topic, value))

    def subscribe(self, topic, flags, fn, timeout=None):
        self.subscribed.append(topic)

    def unsubscribe(self, topic, fn, timeout=None):
        self.unsubscribed.append(topic)


class StreamProcess:
    """Minimal StreamProcessApi instance."""

    def __init__(self):
        self.closed = 0
        self.driver_active = False

    def close(self):
        self.closed += 1


class TestDeviceStreamProcess(unittest.TestCase):

    def test_close_notifies_and_unregisters(self):
        d = Device(FakeDriver(), 'u/js220/000000')
        obj = StreamProcess()
        d.open()
        d.stream_process_register(obj)
        d.close()
        self.assertEqual(1, obj.closed)
        d.open()
        d.close()
        self.assertEqual(1, obj.closed)  # not called again: unregistered


if __name__ == '__main__':
    unittest.main()


class TestSignalsParameter(unittest.TestCase):

    def _device(self, cls=None, path='u/js220/000000'):
        from joulescope.v1.js220 import DeviceJs220
        cls = DeviceJs220 if cls is None else cls
        driver = FakeDriver()
        return cls(driver, path), driver

    def _data_topics(self, driver, path='u/js220/000000'):
        prefix = path + '/'
        return [t[len(prefix):] for t in driver.subscribed
                if t.endswith('!data')]

    def test_default_streams_legacy_six(self):
        d, driver = self._device()
        d.open()
        d.start()
        expect = ['s/i/!data', 's/v/!data', 's/p/!data', 's/i/range/!data',
                  's/gpi/0/!data', 's/gpi/1/!data']
        self.assertEqual(expect, self._data_topics(driver))
        for topic in ['s/i/ctrl', 's/v/ctrl', 's/p/ctrl', 's/i/range/ctrl',
                      's/gpi/0/ctrl', 's/gpi/1/ctrl']:
            self.assertIn((f'u/js220/000000/{topic}', 1), driver.published)
        d.stop()
        self.assertEqual(expect, self._data_topics(driver))
        for topic in ['s/i/ctrl', 's/gpi/1/ctrl']:
            self.assertIn((f'u/js220/000000/{topic}', 0), driver.published)
        d.close()

    def test_signals_select_all(self):
        d, driver = self._device()
        d.open()
        d.parameter_set('signals', 'i,v,p,0,1,2,3,T')
        self.assertEqual('i,v,p,0,1,2,3,T', d.parameter_get('signals'))
        d.start()
        topics = self._data_topics(driver)
        for topic in ['s/gpi/2/!data', 's/gpi/3/!data', 's/gpi/7/!data']:
            self.assertIn(topic, topics)
        for idx in [(5, 2), (5, 3), (5, 7)]:
            self.assertIn(idx, d.stream_buffer.buffers)
            self.assertTrue(d.stream_buffer.buffers[idx].active)
        d.stop()
        d.close()

    def test_signals_long_names_canonicalized(self):
        d, _ = self._device()
        d.parameter_set('signals', 'current,voltage,gpi2,trigger_in')
        self.assertEqual('i,v,2,T', d.parameter_get('signals'))

    def test_signals_subset(self):
        d, driver = self._device()
        d.open()
        d.parameter_set('signals', 'i,v')
        d.start()
        self.assertEqual(['s/i/!data', 's/v/!data'],
                         self._data_topics(driver))
        self.assertFalse(d.stream_buffer.buffers[(3, 0)].active)
        self.assertFalse(d.stream_buffer.buffers[(5, 0)].active)
        d.stop()
        d.close()

    def test_signals_invalid_name_raises(self):
        d, _ = self._device()
        with self.assertRaises(ValueError):
            d.parameter_set('signals', 'i,bogus')

    def test_signals_empty_raises(self):
        d, _ = self._device()
        with self.assertRaises(ValueError):
            d.parameter_set('signals', '')

    def test_js110_rejects_extended_when_closed(self):
        from joulescope.v1.js110 import DeviceJs110
        d, _ = self._device(DeviceJs110, 'u/js110/000000')
        with self.assertRaises(ValueError):
            d.parameter_set('signals', 'i,v,2')

    def test_js110_accepts_legacy_signals(self):
        from joulescope.v1.js110 import DeviceJs110
        d, driver = self._device(DeviceJs110, 'u/js110/000000')
        d.parameter_set('signals', 'i,v,p,r,0,1')
        d.open()
        d.start()
        self.assertEqual(
            ['s/i/!data', 's/v/!data', 's/p/!data', 's/i/range/!data',
             's/gpi/0/!data', 's/gpi/1/!data'],
            self._data_topics(driver, 'u/js110/000000'))
        d.stop()
        d.close()

    def test_js320_supports_extended(self):
        from joulescope.v1.js320 import DeviceJs320
        d, driver = self._device(DeviceJs320, 'u/js320/000000')
        d.parameter_set('signals', '0,1,2,3,T')
        d.open()
        d.start()
        topics = self._data_topics(driver, 'u/js320/000000')
        self.assertEqual(['s/gpi/0/!data', 's/gpi/1/!data', 's/gpi/2/!data',
                          's/gpi/3/!data', 's/gpi/7/!data'], topics)
        d.stop()
        d.close()

    def test_js220_r_with_8_streams_rejected(self):
        d, _ = self._device()
        with self.assertRaises(ValueError):
            d.parameter_set('signals', 'i,v,p,r,0,1,2,3')
        d.parameter_set('signals', 'i,v,p,r,0,1,T')  # 7 with r is OK

    def test_js320_all_signals_accepted(self):
        from joulescope.v1.js320 import DeviceJs320
        d, _ = self._device(DeviceJs320, 'u/js320/000000')
        d.parameter_set('signals', 'i,v,p,r,0,1,2,3,T')
        self.assertEqual('i,v,p,r,0,1,2,3,T', d.parameter_get('signals'))

class TestParametersOverride(unittest.TestCase):

    def _options(self, device, name):
        return [o[0] for o in device.parameters(name).options]

    def test_js220_sampling_frequency_options(self):
        from joulescope.v1.js220 import DeviceJs220
        d = DeviceJs220(FakeDriver(), 'u/js220/000000')
        options = self._options(d, 'sampling_frequency')
        self.assertIn('1 MHz', options)
        self.assertIn('500 kHz', options)
        self.assertNotIn('2 MHz', options)
        self.assertEqual('1 MHz', d.parameters('sampling_frequency').default)

    def test_js320_sampling_frequency_options(self):
        from joulescope.v1.js320 import DeviceJs320
        d = DeviceJs320(FakeDriver(), 'u/js320/000000')
        options = self._options(d, 'sampling_frequency')
        self.assertIn('1 MHz', options)
        self.assertNotIn('500 kHz', options)
        self.assertNotIn('2 MHz', options)

    def test_js220_v_range_options(self):
        from joulescope.v1.js220 import DeviceJs220
        d = DeviceJs220(FakeDriver(), 'u/js220/000000')
        self.assertEqual(['15V', '2V'], self._options(d, 'v_range'))

    def test_js110_unchanged(self):
        from joulescope.v1.js110 import DeviceJs110
        d = DeviceJs110(FakeDriver(), 'u/js110/000000')
        options = self._options(d, 'sampling_frequency')
        self.assertIn('2 MHz', options)
        self.assertEqual(['15V', '5V'], self._options(d, 'v_range'))

    def test_parameters_list_includes_override(self):
        from joulescope.v1.js320 import DeviceJs320
        d = DeviceJs320(FakeDriver(), 'u/js320/000000')
        params = {p.name: p for p in d.parameters()}
        self.assertNotIn('500 kHz',
                         [o[0] for o in params['sampling_frequency'].options])
        self.assertIn('signals', params)

class TestVRangeCompat(unittest.TestCase):
    """JS110 v_range values map to safe JS220/JS320 equivalents."""

    def _select_published(self, driver, path):
        topic = f'{path}/s/v/range/select'
        return [v for t, v in driver.published if t == topic]

    def _device_open(self, cls, path):
        driver = FakeDriver()
        d = cls(driver, path)
        d.open()
        return d, driver

    def test_5v_variants_select_15v(self):
        from joulescope.v1.js220 import DeviceJs220
        from joulescope.v1.js320 import DeviceJs320
        for cls, path in [(DeviceJs220, 'u/js220/000000'),
                          (DeviceJs320, 'u/js320/000000')]:
            for value in ['5V', '5 V', 'high', 1]:
                d, driver = self._device_open(cls, path)
                d.parameter_set('v_range', value)
                selects = self._select_published(driver, path)
                self.assertEqual(['15 V'], selects[-1:],
                                 f'{path} v_range={value!r}')
                d.close()

    def test_5v_readback_preserved(self):
        from joulescope.v1.js220 import DeviceJs220
        d, _ = self._device_open(DeviceJs220, 'u/js220/000000')
        d.parameter_set('v_range', '5V')
        self.assertEqual('5V', d.parameter_get('v_range'))
        d.close()

    def test_15v_and_2v_unchanged(self):
        from joulescope.v1.js220 import DeviceJs220
        d, driver = self._device_open(DeviceJs220, 'u/js220/000000')
        d.parameter_set('v_range', '15V')
        self.assertEqual('15 V',
                         self._select_published(driver, 'u/js220/000000')[-1])
        d.parameter_set('v_range', '2V')
        self.assertEqual('2 V',
                         self._select_published(driver, 'u/js220/000000')[-1])
        d.close()

    def test_all_advertised_sampling_frequencies_settable(self):
        from joulescope.v1.js220 import DeviceJs220
        from joulescope.v1.js320 import DeviceJs320
        for cls, path in [(DeviceJs220, 'u/js220/000000'),
                          (DeviceJs320, 'u/js320/000000')]:
            d = cls(FakeDriver(), path)
            d.open()
            p = d.parameters('sampling_frequency')
            for name, value, aliases in p.options:
                d.parameter_set('sampling_frequency', value)
                self.assertEqual(value, d.parameter_get(
                    'sampling_frequency', dtype='actual'), f'{path} {name}')
            d.close()
