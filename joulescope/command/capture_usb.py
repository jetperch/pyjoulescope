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
import struct
import time
import logging
from joulescope.usb import scan
from joulescope.usb.device_thread import DeviceThread


def parser_config(p):
    """Capture raw USB data  from Joulescope"""
    p.add_argument('--duration',
                   type=float,
                   help='The capture duration.')
    p.add_argument('--endpoint_id',
                   type=int,
                   default=2,
                   help='The endpoint identifier.')
    p.add_argument('--threaded', '-t',
                   default=0,
                   action='count',
                   help='Use the threaded wrapper')
    p.add_argument('filename',
                   help='The filename for output data.')
    return on_cmd


def on_cmd(args):
    d = scan(guid='{99a06894-3518-41a5-a207-8519746da89f}')
    if not(len(d)):
        print('No devices found')
        return 1
    elif len(d) != 1:
        print('More than on device found)')
        return 2
    device = d[0]
    return run(device, filename=args.filename, 
               duration=args.duration,
               endpoint_id=args.endpoint_id,
               threaded=args.threaded)


def stream_settings(device):
    """Configure the Joulescope stream settings

    :param device: The USB device instance.
    """
    version = 1
    length = 16
    msg = struct.pack('<BBBBIBBBBBBBB',
                      version,
                      length,
                      0x01,  # PktType settings
                      0x00,  # reserved
                      0x00,  # reserved
                      0x01,  # sensor_power,
                      0x00,  # i_range,
                      0xC0,  # source raw,
                      0x00,  # options
                      0x03,  # streaming normal
                      0, 0, 0)
    rv = device.control_transfer_out(
        'device', 'vendor', request=3,
        value=0, index=0, data=msg)
    return rv


def run(device, filename, duration, endpoint_id, threaded):
    logging.basicConfig(level=logging.DEBUG)
    quit = False
    d = device
    for _ in range(threaded):
        d = DeviceThread(d)
    time_start = time.time()
    time_last = time_start

    with open(filename, 'wb') as fh:
        def do_quit(*args, **kwargs):
            nonlocal quit
            quit = 'quit from SIGINT'

        def on_data(data, length=None):
            nonlocal quit
            if data is None:
                if not quit:
                    quit = 'quit for on_data'
            else:
                fh.write(bytes(data)[:length])
            if duration is not None:
                if time.time() - time_start > duration:
                    return True
            return False

        def on_process():
            return False
            
        signal.signal(signal.SIGINT, do_quit)
        print('Press CTRL-C to stop data collection')
        try:
            d.open()
            rv = stream_settings(d)
            if 0 != rv.result:
                print('warning: %s', rv)
            d.read_stream_start(
                endpoint_id=endpoint_id,
                transfers=8,
                block_size=256 * 512,
                data_fn=on_data,
                process_fn=on_process)
            while not quit:
                d.process(timeout=0.01)
                time_now = time.time()
                if time_now - time_last > 1.0:
                    print(d.status())
                    time_last = time_now
            d.read_stream_stop(endpoint_id)
        finally:
            d.close()
        print('done capturing data: %s' % quit)

    return 0
