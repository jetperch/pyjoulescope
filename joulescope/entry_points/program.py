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
from joulescope import scan_require_one
import sys


def parser_config(p):
    """Program Joulescope firmware and calibration."""
    p.add_argument('target',
                   choices=['upgrade', 'controller', 'sensor',
                            'calibration_factory', 'calibration_active'],
                   help='The firmware program target')
    p.add_argument('filename',
                   help='The source filename containing the firmware image.')
    return on_run


def _progress(fract):
    # The MIT License (MIT)
    # Copyright (c) 2016 Vladimir Ignatev
    #
    # Permission is hereby granted, free of charge, to any person obtaining
    # a copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation
    # the rights to use, copy, modify, merge, publish, distribute, sublicense,
    # and/or sell copies of the Software, and to permit persons to whom the Software
    # is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included
    # in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    # PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    # FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
    # OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    # OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    fract = min(max(float(fract), 0.0), 1.0)
    bar_len = 50
    filled_len = int(round(bar_len * fract))
    percents = round(100.0 * fract, 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    sys.stdout.flush()


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
        d.sensor_firmware_program(data, progress_cbk=_progress)
        stop_time = time.time()
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
    finally:
        d.close()
    return 0


def _upgrade(filename):
    from joulescope.v0.firmware_manager import upgrade
    try:
        d = scan_require_one(name='bootloader')
    except RuntimeError:
        d = scan_require_one(name='joulescope')
    d.open()
    upgrade(d, filename, progress_cbk=_progress)
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
    elif args.target == 'upgrade':
        rc = _upgrade(args.filename)
    else:
        raise ValueError(f'invalid target: {args.target}')
    stop_time = time.time()
    if rc:
        print('\n***FAILED*** program with error code %r' % rc)
    else:
        print('\nFirmware program took %.2f seconds' % (stop_time - start_time, ))
    return rc

