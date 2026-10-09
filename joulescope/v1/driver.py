# Copyright 2022-2023 Jetperch LLC
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


from pyjoulescope_driver import Driver, device_filter
from pyjoulescope_driver.device_filter import DeviceFilterError
from joulescope.deprecation import warn_once
from .device import Device
from .js320 import DeviceJs320
from .js220 import DeviceJs220
from .js110 import DeviceJs110
import atexit
import logging
from typing import List


_log = logging.getLogger(__name__)
_DEVICE_CLASSES = {
    'js320': DeviceJs320,
    'js220': DeviceJs220,
    'js110': DeviceJs110,
}


class DriverWrapper:
    """Singleton to wrap pyjoulescope_driver.Driver"""
    __singleton__ = None

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
        self._notify_fns = []  # DeviceNotify callbacks fn(is_add, device_path)
        self._watch = self.driver.device_watch(self._on_device_add, self._on_device_remove)

    def _finalize(self):
        self._notify_fns.clear()
        self._watch.unsubscribe()
        while len(self.devices):
            _, device = self.devices.popitem()
            try:
                device.close()
            except Exception:
                _log.exception('device close failed during finalize: %s', device)
        d, self.driver = self.driver, None
        d.finalize()

    def _on_device_add(self, device_path):
        # DevicePath.model omits the bootloader "&", as in "u/&js220/000415".
        cls = _DEVICE_CLASSES.get(device_path.model)
        if cls is None:
            _log.info('Unsupported device: %s', device_path)
        else:
            self.devices[device_path] = cls(self.driver, device_path)
        self._notify(True, device_path)

    def _on_device_remove(self, device_path):
        # on the driver thread: the driver already closed the device
        d = self.devices.pop(device_path, None)
        if d is not None:
            d._on_remove()
        self._notify(False, device_path)

    def notify_register(self, fn):
        """Register a device notification callback.

        :param fn: The callable(is_add, device_path) called from the
            driver thread for every device addition and removal
            processed after this call, in order.  Reading
            :attr:`devices` then shows the same device set that the
            notifications continue from.
        """
        self._notify_fns.append(fn)

    def notify_unregister(self, fn):
        """Unregister a :meth:`notify_register` callback."""
        try:
            self._notify_fns.remove(fn)
        except ValueError:
            pass

    def _notify(self, is_add, device_path):
        for fn in list(self._notify_fns):
            try:
                fn(is_add, device_path)
            except Exception:
                _log.exception('device notify callback')

    def paths(self, specs=None):
        """Find the matching device paths.  See :func:`scan`."""
        specs = _specs_legacy(specs)
        paths = list(self.devices)  # snapshot: the driver thread adds and removes
        if specs == _BOOTLOADER:
            return [p for p in paths if p.is_bootloader]
        paths = device_filter.find(paths, specs, _BRAND)
        if not _specs_select_bootloader(specs):
            paths = [p for p in paths if not p.is_bootloader]
        return paths

    def scan(self, specs=None, config=None, name=None):
        """Scan for connected devices.  See :func:`scan`."""
        devices = [self.devices.get(p) for p in self.paths(_specs_name(specs, name))]
        devices = sorted([d for d in devices if d is not None], key=str)
        for d in devices:
            d.config = config
        return devices


# The joulescope package only supports Joulescope instruments.
_BRAND = 'Joulescope'

# The legacy scan names: "Joulescope" selects all devices in application
# mode, and "bootloader" selects all devices in bootloader mode.
_BOOTLOADER = 'bootloader'
_LEGACY_NAMES = {'joulescope': None, _BOOTLOADER: _BOOTLOADER}


def _specs_legacy(specs):
    if isinstance(specs, str):
        return _LEGACY_NAMES.get(specs.strip().lower(), specs)
    return specs


def _specs_name(specs, name):
    if name is None:
        return specs
    warn_once('scan_name', 'scan name is deprecated, use specs')
    return name if specs is None else specs


def _specs_select_bootloader(specs):
    """Check if specs explicitly select bootloader mode, as in "&js220"."""
    if specs is None:
        return False
    if isinstance(specs, str):
        specs = [specs]
    return any('&' in spec for spec in specs)


def scan(specs=None, config=None, name=None) -> List[Device]:
    """Scan for connected devices.

    :param specs: The device specifications, which is one of:

        * None (default) to select all devices.
        * a string containing one or more comma-separated device
          specifications, such as "js320" or "31NB, u/js220/000415".
        * a list of device specification strings.

        See :meth:`pyjoulescope_driver.DevicePath.match` for the
        specification format.  For backwards compatibility,
        "Joulescope" selects all devices, and "bootloader" selects
        all devices in bootloader mode.
    :param config: The configuration for the :class:`Device`.
    :param name: Deprecated alias for specs, for backwards compatibility.
    :return: The list of :class:`Device` instances, sorted by name.
        Devices in bootloader mode are only included when requested by
        "bootloader" or a "&" model specification, such as "&js220".
    :raise TypeError: If specs has an invalid type.
    """
    specs = _specs_name(specs, name)
    return DriverWrapper().scan(specs, config)


class ScanError(DeviceFilterError, RuntimeError):
    """The scan did not find exactly one device.

    This exception is both a
    :class:`pyjoulescope_driver.device_filter.DeviceFilterError`
    (a ValueError) and, for backwards compatibility, a RuntimeError.
    """


def scan_require_one(specs=None, config=None, name=None) -> Device:
    """Scan for one and only one device.

    :param specs: The device specifications.  See :func:`scan`.
    :param config: The configuration for the :class:`Device`.
    :param name: Deprecated alias for specs.
    :return: The :class:`Device` found.
    :raise ScanError: If zero or multiple devices match.
    """
    specs = _specs_name(specs, name)
    devices = scan(specs, config=config)
    if len(devices) != 1:
        specs_list = [specs] if isinstance(specs, str) else specs
        raise ScanError(specs_list, _BRAND, [d.device_path for d in devices],
                        DriverWrapper().paths())
    return devices[0]


def scan_for_changes(specs=None, devices=None, config=None, name=None):
    """Scan for device changes.

    :param specs: The device specifications.  See :func:`scan`.
    :param devices: The list of existing :class:`Device` instances returned
        by a previous scan.  Pass None or [] if no scan has yet been performed.
    :param config: The configuration for the :class:`Device` which is one of
        ['auto', 'ignore', 'off'].  None is equivalent to 'auto'.
    :param name: Deprecated alias for specs.
    :return: The tuple of lists (devices_now, devices_added, devices_removed).
        "devices_now" is the list of all currently connected devices.  If the
        device was in "devices", then return the :class:`Device` instance from
        "devices".
        "devices_added" is the list of connected devices not in "devices".
        "devices_removed" is the list of devices in "devices" but not "devices_now".
    """
    devices_prev = [] if devices is None else devices
    devices_next = scan(_specs_name(specs, name), config=config)
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


class DeviceNotify:

    def __init__(self, cbk):
        """Start device insertion/removal notification.

        :param cbk: The function called on device insertion or removal.  The
            arguments are (inserted, info).  "inserted" is True on insertion
            and False on removal.  "info" contains the device path for the
            device.  The device path is also None on the one initial call,
            which happens once the notifications are active.  In general,
            the application should rescan for relevant devices.
            Insertion and removal callbacks run on the driver thread.
        """
        self._cbk = cbk
        self._wrapper = None
        self.open()
        self._cbk(True, None)

    def open(self):
        self.close()
        self._wrapper = DriverWrapper()
        self._wrapper.notify_register(self._cbk)

    def close(self):
        wrapper, self._wrapper = self._wrapper, None
        if wrapper is not None:
            wrapper.notify_unregister(self._cbk)
