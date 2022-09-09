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


from .device import Device


def _version_u32_to_str(v):
    v = int(v)
    major = (v >> 24) & 0xff
    minor = (v >> 16) & 0xff
    patch = v & 0xffff
    return f'{major}.{minor}.{patch}'


class DeviceJs220(Device):

    def __init__(self, driver, device_path):
        super().__init__(driver, device_path)

    def info(self):
        info = {
            'type': 'info',
            'ver': 2,
            'model': self.model,
            'serial_number': self.serial_number,
            'ctl': {
                'hw': {
                    'rev': _version_u32_to_str(self.query('c/hw/version')),
                    'sn_mcu': self.serial_number,
                    'sn_mfg': self.serial_number,
                    'ver': _version_u32_to_str(self.query('c/hw/version')),
                },
                'fw': {
                    'ver': _version_u32_to_str(self.query('c/fw/version')),
                }
            },
            'sensor': {
                'fw': {
                    'ver': _version_u32_to_str(self.query('s/fpga/version')),
                },
                'fpga': {
                    'ver': _version_u32_to_str(self.query('s/fpga/version')),
                },
            },
        }
        return info
