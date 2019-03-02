# Copyright 2017 Jetperch LLC

import sys
if sys.platform.startswith('win'):
    from .winusb.device import WinUsbDevice as Device
    from .winusb.device import scan
    from .winusb.win32_device_notify import DeviceNotify
    from .winusb.kernel32 import get_error_str
elif sys.platform in ['linux', 'darwin']:
    from .libusb.device import LibUsbDevice as Device
    from .libusb.device import scan
    from .libusb.device import DeviceNotify

    def get_error_str(rv):  # todo
        return str(rv)
else:
    raise RuntimeError('Platform not supported: %s' % sys.platform)


BULK_IN_LENGTH = 512
