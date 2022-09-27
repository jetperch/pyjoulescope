# Copyright 2018 Jetperch LLC
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

import os
import sys
import platform
from .version import __version__, __title__, __description__, __url__, \
    __author__, __author_email__, __license__, __copyright__


if os.environ.get('JOULESCOPE_BACKEND', '1').lower() in ['1', 'v1', 'true']:
    from joulescope.v1 import scan, scan_require_one, scan_for_changes, DeviceNotify
    from joulescope.jls_v2_writer import JlsWriter
else:
    from joulescope.v0.driver import scan, scan_require_one, scan_for_changes, \
        bootloaders_run_application, bootloader_go
    from joulescope.v0.usb import DeviceNotify
    from joulescope.jls_v2_writer import JlsWriter


VERSION = __version__  # for backwards compatibility
__all__ = ['scan', 'scan_require_one', 'scan_for_changes', 'bootloaders_run_application',
           'bootloader_go', 'JlsWriter', 'DeviceNotify',
           '__version__', '__title__', '__description__', '__url__',
           '__author__', '__author_email__', '__license__', '__copyright__']


if sys.hexversion < 0x030700:
    raise RuntimeError('joulescope requires Python 3.7+')


# Although only 64-bit OS/Python is supported, may be able to run on 32bit Python / 32bit Windows.
p_sz, _ = platform.architecture()
is_32bit = sys.maxsize < (1 << 32)

if (is_32bit and '32' not in p_sz) or (not is_32bit and '64' not in p_sz):
    raise RuntimeError('joulescope Python bits must match platform bits')
