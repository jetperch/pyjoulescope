# Copyright 2018 Jetperch LLC.  All rights reserved.
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

import time
from joulescope.driver import scan_require_one


def parser_config(p):
    """Program Joulescope firmware and calibration."""
    p.add_argument('target',
                   choices=['controller', 'sensor', 'calibration_factory', 'calibration_active'],
                   help='The firmware program target')
    p.add_argument('filename',
                   help='The source filename containing the firmware image.')
    return on_run


def _controller_program_from_bootloader(d, data):
    try:
        d.open()
        rc = d.firmware_program(data)
    finally:
        d.close()
    return rc


def _controller_program_from_app(d, data):
    try:
        d.open()
        rc = d.controller_firmware_program(data)
    finally:
        d.close()
    return rc


def controller_program(data):
    try:
        d = scan_require_one(name='bootloader')
        fn = _controller_program_from_bootloader
    except RuntimeError:
        d = scan_require_one(name='joulescope')
        fn = _controller_program_from_app
    return fn(d, data)


def sensor_program(data):
    d = scan_require_one(name='joulescope')
    try:
        d.open()
        start_time = time.time()
        d.sensor_firmware_program(data)
        stop_time = time.time()
        print('Firmware program took %.2f seconds' % (stop_time - start_time, ))
    finally:
        d.close()
    return 0


def calibration_program(data, is_factory):
    d = scan_require_one(name='joulescope')
    try:
        d.open()
        start_time = time.time()
        d.calibration_program(data, is_factory=is_factory)
        stop_time = time.time()
        print('Firmware program took %.2f seconds' % (stop_time - start_time, ))
    finally:
        d.close()
    return 0


def on_run(args):
    if args.filename == '!':
        data = b''
    else:
        with open(args.filename, 'rb') as f:
            data = f.read()
    start_time = time.time()
    if args.target == 'controller':
        rc = controller_program(data)
    elif args.target == 'sensor':
        rc = sensor_program(data)
    elif args.target == 'calibration_factory':
        rc = calibration_program(data, True)
    elif args.target == 'calibration_active':
        rc = calibration_program(data, False)
    else:
        raise ValueError(f'invalid target: {args.target}')
    stop_time = time.time()
    if rc:
        print('***FAILED*** program with error code %r' % rc)
    else:
        print('Firmware program took %.2f seconds' % (stop_time - start_time, ))
    return rc

