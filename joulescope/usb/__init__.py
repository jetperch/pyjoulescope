# Copyright 2017 Jetperch LLC

import sys
if sys.platform.startswith('win'):
    from .winusb.device import WinUsbDevice as Device
    from .winusb.device import scan
    from .winusb.win32_device_notify import DeviceNotify
elif sys.platform in ['linux', 'darwin']:
    from .libusb.device import LibUsbDevice as Device
    from .libusb.device import scan
    from .libusb.device import DeviceNotify
else:
    raise RuntimeError('Platform not supported: %s' % sys.platform)


BULK_IN_LENGTH = 512
