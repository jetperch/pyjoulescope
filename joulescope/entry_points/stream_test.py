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

import signal
import time
import logging
from joulescope import scan_require_one
import numpy as np
from queue import Queue, Empty


def parser_config(p):
    """Test Joulescope streaming."""
    return on_cmd


def on_cmd(args):
    device = scan_require_one(name='Joulescope', config='auto')
    quit_ = False
    data_queue = Queue()

    def do_quit(*args, **kwargs):
        nonlocal quit_
        quit_ = 'quit from SIGINT'

    def on_stop(event, message):
        nonlocal quit_
        quit_ = 'quit from device stop'

    def on_update_fn(data):
        data_queue.put(data)

    signal.signal(signal.SIGINT, do_quit)
    view = None
    frame_count = 0
    time_start = time.time()
    try:
        device.open()
        view = device.view_factory()
        view.on_update_fn = on_update_fn
        view.open()
        view.on_x_change('resize', {'pixels': 5801})  # prime number
        device.start(stop_fn=on_stop)
        while not quit_:
            try:
                data = data_queue.get(timeout=0.01)
                view.refresh()
                frame_count += 1
                i = data['signals']['current']['µ']['value']
                idx_mask = np.isfinite(i)
                if np.count_nonzero(idx_mask):
                    # access the data to ensure memory validity
                    stats = {
                        'mean': np.mean(i[idx_mask]),
                        'var': np.mean(data['signals']['current']['σ2']['value']),  # not really
                        'p2p': data['signals']['current']['max']['value'] - data['signals']['current']['min']['value']
                    }
            except Empty:
                time.sleep(0.001)
        device.stop()
    except Exception as ex:
        logging.getLogger().exception('while streaming data')
        print('Data streaming failed')
        return 1
    finally:
        if view is not None:
            view.close()
        device.close()
    time_stop = time.time()
    duration = time_stop - time_start
    fps = frame_count / duration
    print(f'done streaming data: {frame_count} frames in {duration:.3f} seconds = {fps:.1f} fps')
    return 0
