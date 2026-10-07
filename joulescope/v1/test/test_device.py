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

import threading
import unittest
import warnings
from unittest import mock
from pyjoulescope_driver import DeviceContext, SubscribeContext
from joulescope import deprecation
from joulescope.v1.device import Device
from joulescope.v1.js110 import DeviceJs110
from joulescope.v1.js220 import DeviceJs220
from joulescope.v1.js320 import DeviceJs320


class FakeDriver:

    def __init__(self):
        self.published = []
        self.subscribed = []
        self.unsubscribed = []
        self.fns = {}  # topic -> fn, for active subscriptions
        self.queries = {}

    def open(self, path, mode=None, timeout=None):
        return DeviceContext(self, path)

    def close(self, path, timeout=None):
        return 0

    def publish(self, topic, value, timeout=None):
        self.published.append((topic, value))

    def subscribe(self, topic, flags, fn, timeout=None):
        self.subscribed.append(topic)
        self.fns[topic] = fn
        return SubscribeContext(self, [(topic, fn)])

    def unsubscribe(self, topic, fn, timeout=None):
        self.unsubscribed.append(topic)
        self.fns.pop(topic, None)

    def unsubscribe_all(self, fn, timeout=None):
        self.unsubscribed.append(fn)

    def query(self, topic, timeout=None):
        return self.queries[topic]

    def publish_and_wait(self, publish_topic, publish_value, response_topic,
                         timeout=None, match=None):
        self.published.append((publish_topic, publish_value))
        if response_topic not in self.queries:
            raise TimeoutError('publish_and_wait timed out')
        return self.queries[response_topic]


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


def _stats_value(sample_start=0, sample_stop=1_000_000, charge=1.0, energy=2.0):
    """Construct a minimal driver statistics value."""
    signal = {'avg': {'value': 0.5, 'units': 'A'}, 'std': {'value': 0.1, 'units': 'A'}}
    return {
        'time': {'samples': {'value': [sample_start, sample_stop], 'units': 'samples'}},
        'signals': {'current': dict(signal), 'voltage': dict(signal)},
        'accumulators': {
            'charge': {'value': charge, 'units': 'C'},
            'energy': {'value': energy, 'units': 'J'},
        },
    }


class DeprecationTestCase(unittest.TestCase):

    def setUp(self):
        p = mock.patch.object(deprecation, '_warned', set())
        p.start()
        self.addCleanup(p.stop)


class TestStatistics(DeprecationTestCase):

    def _open(self, cls=DeviceJs220, path='u/js220/000000', config=None):
        driver = FakeDriver()
        d = cls(driver, path)
        d.config = config
        d.open()
        return d, driver

    def _publish(self, driver, d, **kwargs):
        topic = f'{d.device_path}/{d._statistics_topics()[1]}'
        driver.fns[topic](topic, _stats_value(**kwargs))

    def test_callback_source_added(self):
        d, driver = self._open()
        values = []
        d.statistics_callback_register(values.append)
        self.assertIn(('u/js220/000000/s/stats/ctrl', 1), driver.published)
        self._publish(driver, d)
        self.assertEqual('sensor', values[0]['source'])
        self.assertEqual(0.5, values[0]['signals']['current']['µ']['value'])

    def test_source_deprecated_warns_once(self):
        d, _ = self._open()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            d.statistics_callback_register(print, 'sensor')
            d.statistics_callback_unregister(print, 'sensor')
            d.statistics_callback_register(print)
        self.assertEqual(1, len(w))
        self.assertIs(DeprecationWarning, w[0].category)

    def test_statistics_source_by_model_and_config(self):
        for cls, path, config, source, topic in [
                (DeviceJs110, 'u/js110/1', None, 'host', 's/stats/value'),
                (DeviceJs110, 'u/js110/1', 'auto', 'host', 's/stats/value'),
                (DeviceJs110, 'u/js110/1', 'off', 'sensor', 's/sstats/value'),
                (DeviceJs220, 'u/js220/1', 'off', 'sensor', 's/stats/value'),
                (DeviceJs320, 'u/js320/1', 'auto', 'sensor', 's/stats/value')]:
            with self.subTest(path=path, config=config):
                d, driver = self._open(cls, path, config)
                values = []
                d.statistics_callback_register(values.append)
                self.assertEqual(source, d.statistics_source)
                self.assertIn(f'{path}/{topic}', driver.fns)
                self._publish(driver, d)
                self.assertEqual(source, values[0]['source'])
                d.statistics_callback_unregister(values.append)
                self.assertNotIn(f'{path}/{topic}', driver.fns)

    def test_get(self):
        d, driver = self._open()
        t = threading.Timer(0.01, lambda: self._publish(driver, d))
        t.start()
        value = d.statistics_get(timeout=1.0)
        t.join()
        self.assertEqual(0.5, value['signals']['current']['µ']['value'])

    def test_get_buffers_consecutive_values(self):
        d, driver = self._open()
        with self.assertRaises(TimeoutError):
            d.statistics_get(timeout=0.01)  # registers
        for k in range(3):
            self._publish(driver, d, sample_start=k, sample_stop=k + 1)
        samples = [d.statistics_get(0)['time']['samples']['value'][0] for _ in range(3)]
        self.assertEqual([0, 1, 2], samples)

    def test_get_queue_overflow_drops_oldest(self):
        d, driver = self._open()
        with self.assertRaises(TimeoutError):
            d.statistics_get(timeout=0)
        for k in range(105):
            self._publish(driver, d, sample_start=k, sample_stop=k + 1)
        self.assertEqual(5, d.statistics_get(0)['time']['samples']['value'][0])

    def test_get_requires_open(self):
        d = DeviceJs220(FakeDriver(), 'u/js220/1')
        with self.assertRaises(RuntimeError):
            d.statistics_get()
        with self.assertRaises(RuntimeError):
            d.statistics_iter()

    def test_iter_count(self):
        d, driver = self._open()
        it = d.statistics_iter(count=2, timeout=1.0)
        threading.Timer(0.01, lambda: [self._publish(driver, d) for _ in range(3)]).start()
        self.assertEqual(2, len(list(it)))

    def test_iter_timeout(self):
        d, _ = self._open()
        with self.assertRaises(TimeoutError):
            next(d.statistics_iter(timeout=0.01))

    def test_iter_stops_on_close(self):
        d, driver = self._open()
        values = []

        def run():
            for value in d.statistics_iter(timeout=2.0):
                values.append(value)

        def wait_for(predicate):
            for _ in range(2000):
                if predicate():
                    return
                threading.Event().wait(0.001)
            self.fail('wait_for timed out')

        thread = threading.Thread(target=run)
        thread.start()
        wait_for(lambda: len(d._statistics_callbacks))
        self._publish(driver, d)
        wait_for(lambda: len(values))
        d.close()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(1, len(values))

    def test_close_unregisters_queue(self):
        d, driver = self._open()
        with self.assertRaises(TimeoutError):
            d.statistics_get(timeout=0)
        d.close()
        self.assertEqual([], d._statistics_callbacks)
        self.assertNotIn('u/js220/000000/s/stats/value', driver.fns)
        self.assertIn(('u/js220/000000/s/stats/ctrl', 0), driver.published)


class TestDeviceMisc(DeprecationTestCase):

    def test_status_deprecated_warns_once(self):
        for cls in [Device, DeviceJs110, DeviceJs220, DeviceJs320]:
            d = cls(FakeDriver(), 'u/js220/1')
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.assertEqual(0, d.status()['driver']['return_code']['value'])
            self.assertEqual(1 if cls is Device else 0, len(w))

    def test_model_and_serial_number(self):
        d = Device(FakeDriver(), 'u/&js220/000415/')
        self.assertEqual('u/&js220/000415', d.device_path)
        self.assertEqual('js220', d.model)
        self.assertEqual('000415', d.serial_number)
        self.assertEqual('&JS220-000415', str(d))

    def test_topics_relative_to_device(self):
        driver = FakeDriver()
        d = Device(driver, 'u/js220/1')
        d.open()
        d.publish('/s/i/range/mode', 'auto')
        d.publish('s/v/range/mode', 'auto')
        self.assertEqual([('u/js220/1/h/fs', 2000000), ('u/js220/1/s/i/range/mode', 'auto'),
                          ('u/js220/1/s/v/range/mode', 'auto')], driver.published)

    def test_close_unsubscribes_remaining(self):
        driver = FakeDriver()
        d = Device(driver, 'u/js220/1')
        d.open()
        d.subscribe('s/gpi/+/!value', 'pub', print)
        d.close()
        self.assertEqual({}, driver.fns)

    def test_unsubscribe_all(self):
        driver = FakeDriver()
        d = Device(driver, 'u/js220/1')
        d.unsubscribe_all(print)
        self.assertEqual([print], driver.unsubscribed)

    def test_query_gpi_value(self):
        driver = FakeDriver()
        d = Device(driver, 'u/js220/1')
        with self.assertRaises(RuntimeError):
            d._query_gpi_value()
        driver.queries['u/js220/1/s/gpi/+/!value'] = 3
        self.assertEqual(3, d._query_gpi_value())
        self.assertEqual(('u/js220/1/s/gpi/+/!req', 0), driver.published[-1])

    def test_js220_info_versions(self):
        driver = FakeDriver()
        driver.queries = {'u/js220/1/c/hw/version': 0x01020003,
                          'u/js220/1/c/fw/version': 0x01030004,
                          'u/js220/1/s/fpga/version': 0x01040005}
        info = DeviceJs220(driver, 'u/js220/1').info()
        self.assertEqual('1.2.3', info['hardware_version'])
        self.assertEqual('1.2.3', info['ctl']['hw']['rev'])
        self.assertEqual('1.3.4', info['ctl']['fw']['ver'])
        self.assertEqual('1.4.5', info['sensor']['fpga']['ver'])


if __name__ == '__main__':
    unittest.main()
