# Copyright 2020 Jetperch LLC
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

"""
This executable displays information about the joulescope package and
connected devices.
"""

import joulescope
import logging
import platform
import sys


def parser_config(p):
    """Display platform and connected device details."""
    return on_cmd


def on_cmd(args):
    frozen = getattr(sys, 'frozen', False)
    if frozen:
        frozen = getattr(sys, '_MEIPASS', frozen)
    print('System information')
    print(f'    Python: {sys.version}')
    print(f'    Platform: {platform.platform()} ({sys.platform})')
    print(f'    Processor: {platform.processor()}')
    print(f'    executable: {sys.executable}')
    print(f'    frozen: {frozen}')

    print('')
    print(f'joulescope version: {joulescope.__version__}')
    devices = joulescope.scan()
    device_count = len(devices)
    if device_count == 0:
        print('Found 0 connected Joulescopes.')
    elif device_count == 1:
        print('Found 1 connected Joulescope:')
    else:
        print(f'Found {device_count} connected Joulescopes:')
    logging.getLogger().setLevel(logging.WARNING)
    for device in devices:
        try:
            with device:
                info = device.info()
            ctl_fw = info.get('ctl', {}).get('fw', {}).get('ver', '')
            sensor_fw = info.get('sensor', {}).get('fw', {}).get('ver', '')
            info = f'  ctl={ctl_fw:<15}  sensor={sensor_fw}'
        except Exception:
            info = ''
        print(f'    {device} {info}')

