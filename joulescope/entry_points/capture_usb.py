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
This executable captures the raw USB stream from Joulescope devices
and saves the raw stream data to a file.  This executable is a
development tool and is not intended for customer use.
"""

import signal
import time
from joulescope.pattern_buffer import PatternBuffer
import logging
from joulescope import scan


def parser_config(p):
    """Capture raw USB data from Joulescope."""
    p.add_argument('--duration',
                   type=float,
                   help='The capture duration.')
    p.add_argument('--filename',
                   help='The filename for output data.')
    return on_cmd


def on_cmd(args):
    d = scan(name='Joulescope')
    if not(len(d)):
        print('No devices found')
        return 1
    elif len(d) != 1:
        print('More than on device found)')
        return 2
    device = d[0]
    return run(device, filename=args.filename, duration=args.duration)


class RecordingBuffer:

    def __init__(self, filename):
        self.fh = open(filename, 'wb')
        self._sample_id_count = 0
        self.sample_id_max = None
        self.contiguous_max = None

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh is None

    def status(self):
        return {}

    def reset(self):
        self._sample_id_count = 0

    def calibration_set(self, *args, **kwargs):
        pass

    def insert(self, data):
        if self.sample_id_max is not None and self._sample_id_count >= self.sample_id_max or data is None:
            if self.fh is not None:
                self.fh.close()
            return True
        elif self.fh is not None:
            self.fh.write(data)
        self._sample_id_count += (len(data) // 512) * 126
        return False

    def process(self):
        return False


def run(device, filename, duration):
    logging.basicConfig(level=logging.DEBUG)
    time_last = time.time()
    quit = False
    if filename:
        buffer = RecordingBuffer(filename=filename)
    else:
        buffer = PatternBuffer()

    def do_quit(*args, **kwargs):
        nonlocal quit
        quit = 'quit from SIGINT'

    def do_stop(*args, **kwargs):
        nonlocal quit
        quit = 'quit from done'

    signal.signal(signal.SIGINT, do_quit)
    device.open()
    try:
        device.stream_buffer = buffer
        device.parameter_set('sensor_power', 'on')
        # device.parameter_set('control_test_mode', 'normal')
        device.parameter_set('source', 'pattern_sensor')
        device.start(stop_fn=do_stop, duration=duration)
        print('Press CTRL-C to stop data collection')
        while not quit:
            time.sleep(0.01)
            time_now = time.time()
            if time_now - time_last > 1.0:
                print(device.status())
                time_last = time_now
        print(device.status())
    finally:
        device.close()
    print('done capturing data: %s' % quit)

    return 0
