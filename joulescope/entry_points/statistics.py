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


def parser_config(p):
    """Display Joulescope statistics."""
    p.add_argument('--compare',
                   action='store_true',
                   help='Compare sensor statistics to host stream_buffer statistics.')
    return on_cmd


def on_cmd(args):
    quit_ = False
    device = scan_require_one(name='Joulescope', config='off')

    def do_quit(*args, **kwargs):
        nonlocal quit_
        quit_ = 'quit from SIGINT'

    def statistics_cbk(s, indicator=None):
        indicator = '' if indicator is None else indicator
        t = s['time']['range']['value'][0]
        i = s['signals']['current']['µ']
        v = s['signals']['voltage']['µ']
        p = s['signals']['power']['µ']
        c = s['accumulators']['charge']
        e = s['accumulators']['energy']

        fmts = ['{x:.9f}', '{x:.3f}', '{x:.9f}', '{x:.9f}', '{x:.9f}']
        values = []
        for k, fmt in zip([i, v, p, c, e], fmts):
            value = fmt.format(x=k['value'])
            value = f'{value} {k["units"]}'
            values.append(value)
        ', '.join(values)
        print(f"{indicator}{t:.1f}: " + ', '.join(values))

    def on_stop(event=0, message=''):
        print(f'on_stop {event}: {message}')

    signal.signal(signal.SIGINT, do_quit)
    try:
        if args.compare:
            device.statistics_callback_register(lambda s: statistics_cbk(s, '< '), 'sensor')
            device.statistics_callback_register(lambda s: statistics_cbk(s, '> '), 'stream_buffer')
        else:
            device.statistics_callback_register(statistics_cbk, 'sensor')
        device.parameter_set('buffer_duration', 1)
        device.open()
        device.parameter_set('i_range', 'auto')
        device.parameter_set('v_range', '15V')
        if args.compare:
            device.parameter_set('source', 'raw')
            device.start(stop_fn=on_stop)
        while not quit_:
            device.status()
            time.sleep(0.100)
    except Exception as ex:
        logging.getLogger().exception('While getting statistics')
        print('Data streaming failed')
        return 1
    finally:
        device.close()
    return 0
