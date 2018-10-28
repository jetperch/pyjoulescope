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

import signal
import time
import logging
from joulescope.driver import scan_require_one


def parser_config(p):
    """Capture data from Joulescope"""
    p.add_argument('--duration',
                   type=float,
                   help='The capture duration in seconds.')
    p.add_argument('--contiguous',
                   type=float,
                   help='The contiguous capture duration (no missing samples) in seconds.')
    p.add_argument('filename',
                   help='The filename for output data.')
    p.add_argument('--profile',
                   action='store_true',
                   help='Profile the capture')
    return on_cmd


def on_cmd(args):
    device = scan_require_one(name='Joulescope')
    f = lambda: run(device, filename=args.filename,
                    duration=args.duration,
                    contiguous_duration=args.contiguous)
    if args.profile:
        import cProfile
        import pstats
        cProfile.runctx('f()', globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        return f()


def run(device, filename, duration=None, contiguous_duration=None):
    quit_ = False

    def do_quit(*args, **kwargs):
        nonlocal quit_
        quit_ = 'quit from SIGINT'

    def on_stop():
        nonlocal quit_
        quit_ = 'quit from stop duration'

    signal.signal(signal.SIGINT, do_quit)
    try:
        device.open()
        device.parameter_set('source', 'raw')
        device.start(stop_fn=on_stop, duration=duration, 
                     contiguous_duration=contiguous_duration)
        device.recording_start(filename)
        time_last = time.time()
        sample_id_last = 0
        sample_id_incr = 1000000
        sample_id_next = sample_id_last + sample_id_incr
        while not quit_:
            time.sleep(0.01)
            time_now = time.time()
            if time_now - time_last > 1.0:
                logging.getLogger().info(device.status())
                time_last = time_now
            while device.stream_buffer.sample_id_range[-1] >= sample_id_next:
                # todo save
                sample_id_next += sample_id_incr
        device.stop()
    finally:
        device.close()
    print('done capturing data: %s' % quit_)

    return 0
