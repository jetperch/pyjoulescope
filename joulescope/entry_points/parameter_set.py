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
This executable configures Joulescope parameters.
"""

from joulescope import scan
import logging


def parser_config(p):
    """Configure Joulescope parameters."""
    p.add_argument('--config',
                   choices=['auto', 'off', 'ignore'],
                   default='ignore',
                   help='The configuration defaults.')
    p.add_argument('--select',
                   help='Select a single Joulescope by serial number.')
    p.add_argument('assigns',
                   nargs='*',
                   help='A parameter name=value to assign.  See parameters_v1.py for details.')
    return on_cmd


def on_cmd(args):
    rv = 0
    devices = scan(name='Joulescope', config=args.config)
    for device in devices:
        try:
            if args.select and int(device.device_serial_number) != int(args.select):
                continue
            print(f'{device}')
            with device:
                for name_value in args.assigns:
                    try:
                        name, value = name_value.split('=')
                        device.parameter_set(name, value)
                    except Exception:
                        logging.exception(f'{device} parameter {name_value} failed')
                        print(f'{device} parameter {name_value} failed')
                pass
        except Exception:
            logging.exception(f'{device}: failed')
            print(f'{device}: failed')
            rv = 1
    return rv
