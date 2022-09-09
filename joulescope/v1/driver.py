# Copyright 2022 Jetperch LLC
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


"""The pyjoulescope_driver wrapper to implement the v0 API."""


from pyjoulescope_driver import Driver
from .device import Device
from .js220 import DeviceJs220
from .js110 import DeviceJs110
import atexit
import logging
from typing import List


_log = logging.getLogger(__name__)


class DriverWrapper:
    """Singleton to wrap pyjoulescope_driver.Driver"""

    def __new__(cls, *args, **kwds):
        s = cls.__dict__.get("__singleton__")
        if s is not None:
            return s
        s = object.__new__(cls)
        cls.__singleton__ = s
        s._initialize(*args, **kwds)
        return s

    def _initialize(self):
        self.driver = Driver()
        self.driver.log_level = 'INFO'
        atexit.register(self._finalize)
        self.devices = {}
        self.driver.subscribe('@/!add', 'pub', self._on_device_add)
        self.driver.subscribe('@/!remove', 'pub', self._on_device_remove)
        for d in self.driver.device_paths():
            self._on_device_add('@/!add', d)

    def _finalize(self):
        while len(self.devices):
            _, device = self.devices.popitem()
            device.close()
        d, self.driver = self.driver, None
        d.finalize()

    def _on_device_add(self, topic, value):
        if value in self.devices:
            return
        if 'js220' in value:
            cls = DeviceJs220
        elif 'js110' in value:
            cls = DeviceJs110
        else:
            _log.info('Unsupported device: %s', value)
            return
        self.devices[value] = cls(self.driver, value)

    def _on_device_remove(self, topic, value):
        d = self.devices.pop(value, None)
        if d is not None:
            d.close()

    def scan(self):
        return sorted(self.devices.values(), key=lambda x: str(x))


_device_wrapper = DriverWrapper()


def scan(name: str = None, config=None) -> List[Device]:
    """Scan for connected devices.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param config: The configuration for the :class:`Device`.
    :return: The list of :class:`Device` instances.  A new instance is created
        for each detected device.  Use :func:`scan_for_changes` to preserved
        existing instances.
    :raises: None - guaranteed not to raise an exception
    """
    d = DriverWrapper()
    return d.scan()


def scan_require_one(name: str = None, config=None) -> Device:
    """Scan for one and only one device.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param config: The configuration for the :class:`Device`.
    :return: The :class:`Device` found.
    :raise RuntimeError: If no devices or more than one device was found.
    """
    devices = scan(name, config=config)
    if not len(devices):
        raise RuntimeError("no devices found")
    if len(devices) > 1:
        raise RuntimeError("multiple devices found")
    return devices[0]


def scan_for_changes(name: str = None, devices=None, config=None):
    """Scan for device changes.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param devices: The list of existing :class:`Device` instances returned
        by a previous scan.  Pass None or [] if no scan has yet been performed.
    :param config: The configuration for the :class:`Device`.
    :return: The tuple of lists (devices_now, devices_added, devices_removed).
        "devices_now" is the list of all currently connected devices.  If the
        device was in "devices", then return the :class:`Device` instance from
        "devices".
        "devices_added" is the list of connected devices not in "devices".
        "devices_removed" is the list of devices in "devices" but not "devices_now".
    """
    devices_prev = [] if devices is None else devices
    devices_next = scan(name, config=config)
    devices_added = []
    devices_removed = []
    devices_now = []

    for d in devices_next:
        matches = [x for x in devices_prev if str(x) == str(d)]
        if len(matches):
            devices_now.append(matches[0])
        else:
            devices_added.append(d)
            devices_now.append(d)

    for d in devices_prev:
        matches = [x for x in devices_next if str(x) == str(d)]
        if not len(matches):
            devices_removed.append(d)

    _log.info('scan_for_changes %d devices: %d added, %d removed',
              len(devices_now), len(devices_added), len(devices_removed))
    return devices_now, devices_added, devices_removed
