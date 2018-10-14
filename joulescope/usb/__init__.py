# Copyright 2017 Jetperch LLC

import sys
if sys.platform.startswith('win'):
    from .winusb.device import WinUsbDevice as Device
    from .winusb.device import scan
else:
    raise RuntimeError('Platform not supported: %s' % sys.platform)


BULK_IN_LENGTH = 512
