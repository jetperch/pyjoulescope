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
Force all Joulescope bootloaders to run their applications.

This command is normally not necessary.  However, if a user forces
a Joulescope into bootloader mode. then the Joulescope bootloader will
no longer automatically launch the application.  This command forces
all connected Joulescopes running the bootloder to start the application.
"""

from joulescope import scan


def parser_config(p):
    """Force all bootloaders to run the application."""
    return on_cmd


def on_cmd(args):
    for d in scan(name='bootloader'):
        d.open()
        print(f'Found {d}')
        try:
            d.go()
        except Exception:
            print(f'Could not start application on {d}')
    return 0
