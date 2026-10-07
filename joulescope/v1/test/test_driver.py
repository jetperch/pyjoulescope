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

"""Test the v1 DriverWrapper and DeviceNotify using a stubbed joulescope_driver."""

import unittest
import warnings
from unittest import mock
from joulescope import deprecation
from pyjoulescope_driver import DevicePath
from joulescope.v1 import driver as v1_driver
from joulescope.v1.js110 import DeviceJs110
from joulescope.v1.js220 import DeviceJs220
from joulescope.v1.js320 import DeviceJs320


class FakeWatch:

    def __init__(self, driver, on_add, on_remove):
        self.driver = driver
        self.on_add = on_add
        self.on_remove = on_remove

    def unsubscribe(self):
        self.driver.watches.remove(self)


class FakeDriver:
    """Emulate Driver.device_watch, including the initial on_add calls."""

    connected = []

    def __init__(self):
        self.log_level = None
        self.watches = []
        self.finalized = False

    def device_watch(self, on_add, on_remove):
        watch = FakeWatch(self, on_add, on_remove)
        self.watches.append(watch)
        for device_path in self.connected:
            on_add(DevicePath(device_path))
        return watch

    def add(self, device_path):
        for watch in list(self.watches):
            watch.on_add(DevicePath(device_path))

    def remove(self, device_path):
        for watch in list(self.watches):
            watch.on_remove(DevicePath(device_path))

    def open(self, path, mode=None, timeout=None):
        return 0

    def close(self, path, timeout=None):
        return 0

    def finalize(self):
        self.finalized = True


class TestDriverWrapper(unittest.TestCase):

    def setUp(self):
        patches = [
            mock.patch.object(v1_driver, 'Driver', FakeDriver),
            mock.patch.object(v1_driver.atexit, 'register'),
            mock.patch.object(v1_driver.DriverWrapper, '__singleton__', None),
            mock.patch.object(FakeDriver, 'connected', ['u/js320/8W2A', 'u/mb/93NP']),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.w = v1_driver.DriverWrapper()
        self.d = self.w.driver

    def test_connected(self):
        self.assertEqual(['u/js320/8W2A'], list(self.w.devices))
        self.assertIsInstance(self.w.devices['u/js320/8W2A'], DeviceJs320)

    def test_add_remove(self):
        for device_path, cls in [('u/js110/1', DeviceJs110), ('u/js220/2', DeviceJs220),
                                 ('u/&js220/3', DeviceJs220)]:
            self.d.add(device_path)
            self.assertIsInstance(self.w.devices[device_path], cls, device_path)
        self.d.add('u/mb/1')
        self.assertNotIn('u/mb/1', self.w.devices)
        device = self.w.devices['u/js220/2']
        with mock.patch.object(device, 'close') as close:
            self.d.remove('u/js220/2')
        close.assert_called_once_with()
        self.assertNotIn('u/js220/2', self.w.devices)
        self.d.remove('u/js220/2')  # not present

    def test_scan(self):
        self.d.add('u/&js220/3')
        self.assertEqual(['u/js320/8W2A'], [x.device_path for x in self.w.scan()])
        self.assertEqual(['u/&js220/3'], [x.device_path for x in self.w.scan('bootloader')])

    def test_finalize(self):
        self.w._finalize()
        self.assertEqual([], self.d.watches)
        self.assertEqual({}, self.w.devices)
        self.assertTrue(self.d.finalized)


class TestScan(unittest.TestCase):

    def setUp(self):
        patches = [
            mock.patch.object(v1_driver, 'Driver', FakeDriver),
            mock.patch.object(v1_driver.atexit, 'register'),
            mock.patch.object(v1_driver.DriverWrapper, '__singleton__', None),
            mock.patch.object(FakeDriver, 'connected', [
                'u/js320/8W2A', 'u/js220/000415', 'u/js220/001707', 'u/&js110/000578',
                'u/mb/93NP']),
            mock.patch.object(deprecation, '_warned', set()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def paths(self, *args, **kwargs):
        return [d.device_path for d in v1_driver.scan(*args, **kwargs)]

    def test_all(self):
        expect = ['u/js220/000415', 'u/js220/001707', 'u/js320/8W2A']
        self.assertEqual(expect, self.paths())
        self.assertEqual(expect, self.paths('Joulescope'))
        self.assertEqual(expect, self.paths(''))

    def test_specs(self):
        self.assertEqual(['u/js220/000415', 'u/js220/001707'], self.paths('js220'))
        self.assertEqual(['u/js220/000415', 'u/js320/8W2A'], self.paths('8w2a, 000415'))
        self.assertEqual(['u/js220/001707'], self.paths(['js220/001707']))
        self.assertEqual(['u/js320/8W2A'], self.paths('u/js320/'))
        self.assertEqual([], self.paths('mb'))  # not a Joulescope
        self.assertEqual([], self.paths('93NP'))
        self.assertEqual([], self.paths('js110'))  # bootloader mode only

    def test_bootloader(self):
        self.assertEqual(['u/&js110/000578'], self.paths('bootloader'))
        self.assertEqual(['u/&js110/000578'], self.paths('&js110'))

    def test_invalid(self):
        with self.assertRaises(TypeError):
            v1_driver.scan(3)

    def test_config(self):
        devices = v1_driver.scan('js220', config='off')
        self.assertEqual(['off', 'off'], [d.config for d in devices])
        with self.assertRaises(v1_driver.ScanError):
            v1_driver.scan_require_one('31NB')
        self.assertEqual(['off', 'off'], [d.config for d in devices])  # unchanged

    def test_name_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(['u/js320/8W2A'], self.paths(name='js320'))
            self.assertEqual(['u/js320/8W2A'], self.paths(name='js320'))
        self.assertEqual(1, len(w))
        self.assertIs(DeprecationWarning, w[0].category)

    def test_require_one(self):
        d = v1_driver.scan_require_one('8W2A', config='auto')
        self.assertEqual('u/js320/8W2A', d.device_path)
        self.assertEqual('auto', d.config)
        with self.assertWarns(DeprecationWarning):
            d = v1_driver.scan_require_one(name='001707')
        self.assertEqual('u/js220/001707', d.device_path)

    def test_require_one_errors(self):
        for specs, msg in [(None, 'Multiple Joulescopes found'),
                           ('js220', 'matched multiple'),
                           ('31NB', 'not found')]:
            with self.subTest(specs=specs):
                with self.assertRaises(RuntimeError) as cm:  # backwards compatible
                    v1_driver.scan_require_one(specs)
                ex = cm.exception
                self.assertIsInstance(ex, v1_driver.ScanError)
                self.assertIsInstance(ex, ValueError)
                self.assertIn(msg, str(ex))
                self.assertIn('u/js320/8W2A', [str(p) for p in ex.available])

    def test_scan_for_changes(self):
        now, added, removed = v1_driver.scan_for_changes('js220')
        self.assertEqual(2, len(added))
        now, added, removed = v1_driver.scan_for_changes('000415', now)
        self.assertEqual((1, 0, 1), (len(now), len(added), len(removed)))


class TestDeviceNotify(unittest.TestCase):

    def setUp(self):
        self.d = FakeDriver()
        self.d.connected = ['u/js320/8W2A']
        wrapper = mock.Mock(driver=self.d)
        p = mock.patch.object(v1_driver, 'DriverWrapper', return_value=wrapper)
        p.start()
        self.addCleanup(p.stop)
        self.events = []

    def cbk(self, inserted, info):
        self.events.append((inserted, info))

    def test_notify(self):
        n = v1_driver.DeviceNotify(self.cbk)
        self.assertEqual([(True, None)], self.events)  # connected devices not repeated
        self.d.add('u/js220/1')
        self.d.remove('u/js320/8W2A')
        self.assertEqual([(True, None), (True, 'u/js220/1'), (False, 'u/js320/8W2A')],
                         self.events)
        n.close()
        n.close()
        self.assertEqual([], self.d.watches)
        self.d.add('u/js110/1')
        self.assertEqual(3, len(self.events))
