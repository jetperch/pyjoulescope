# Copyright 2019 Jetperch LLC
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
This executable drives the general purpose outputs on a Joulescope."""

import time
from joulescope import scan_require_one


def parser_config(p):
    """Demonstrate the general purpose output functionality."""
    p.add_argument('--voltage',
                   choices=['1.8V', '2.1V', '2.5V', '2.7V', '3.0V', '3.3V', '5.0V'],
                   default='3.3V',
                   help='The GPI/O voltage reference.')
    return on_cmd


def on_cmd(args):
    d = scan_require_one(name='Joulescope')
    d.open()
    try:
        d.parameter_set('sensor_power', 'on')
        d.parameter_set('io_voltage', args.voltage)
        d.parameter_set('gpo0', '0')
        d.parameter_set('gpo1', '0')
        for count in range(17):
            d.parameter_set('gpo0', str(count & 1))
            d.parameter_set('gpo1', str((count & 2) >> 1))
            time.sleep(0.25)

    finally:
        d.close()


