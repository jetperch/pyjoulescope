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

"""
This module contains a registry of information for each target scan device.
USB implementations use this information the find the correct attached devices.
"""


# import uuid; str(uuid.uuid4())


INFO = {
    'joulescope': [
        {
            'name': 'alpha',
            'DeviceInterfaceGUID': '{99a06894-3518-41a5-a207-8519746da89f}',
            'vendor_id': 0x1FC9,  # NXP
            'product_id': 0xFC93,  # Random
        },
        {
            'name': 'beta',
            'DeviceInterfaceGUID': '{576d606f-f3de-4e4e-8a87-065b9fd21eb0}',
            'vendor_id': 0x16D0,  # MCS Electronics
            'product_id': 0x0E88,  # Assigned but not suitable for USB certification
        },
    ],
    'bootloader': [
        {
            'name': 'alpha',
            'DeviceInterfaceGUID': '{76ab0534-eeda-48b5-b0c9-617fc6ce8296}',
            'vendor_id': 0x1FC9,  # NXP
            'product_id': 0xFC94,  # Random
        },
        {
            'name': 'beta',
            'DeviceInterfaceGUID': '{09f5f2f2-9725-4bce-9079-5e8184f9d587}',
            'vendor_id': 0x16D0,  # MCS Electronics
            'product_id': 0x0E87,  # Assigned but not suitable for USB certification
        },
    ],
    'joulescope-mfg': [
        {
            'name': 'normal',
            'DeviceInterfaceGUID': '{ddc8888c-fcb3-455b-9c3b-ee45e56fd741}',
            'vendor_id': 0x1FC9,  # NXP
            'product_id': 0xFC95,  # Random
        },
    ],
}
"""The dict of mapping lower-case device name to metadata.  The metadata 
contains the keys:
* name: The variant name.
* DeviceInterfaceGUID: The GUID for the Microsoft WinUSB descriptor.
* vendor_id: The 2-byte USB vendor identifier integer.
* product_id: The 2-byte USB product identifier integer.
"""
