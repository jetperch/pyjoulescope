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

    def open(self, path, mode=None, timeout=None):
        return 0

    def close(self, path, timeout=None):
        return 0

    def publish(self, topic, value, timeout=None):
        self.published.append((topic, value))

    def subscribe(self, topic, flags, fn, timeout=None):
        pass

    def unsubscribe(self, topic, fn, timeout=None):
        pass


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
