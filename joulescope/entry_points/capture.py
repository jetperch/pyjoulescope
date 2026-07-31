# Copyright 2018-2026 Jetperch LLC
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
    """Capture data from Joulescope."""
    p.add_argument('--duration',
                   type=float,
                   help='The capture duration in seconds.')
    p.add_argument('--contiguous',
                   type=float,
                   help='The contiguous capture duration (no missing samples) in seconds.')
    p.add_argument('--signals',
                   default='i,v',
                   help='The comma-separated signals to record: '
                        'i, v, p, r, 0, 1, 2, 3, T.  Defaults to "i,v".  '
                        'Only i, v, p apply to --format jls1.')
    p.add_argument('--format',
                   dest='out_format',
                   choices=['jls2', 'jls1'],
                   default=None,
                   help='The output file format.  '
                        'jls2 records a JLS v2 file (pyjls); '
                        'jls1 records the legacy JLS v1 file.  '
                        'Defaults to jls2 (jls1 on the v0 backend).')
    p.add_argument('filename',
                   help='The filename for output data.')
    p.add_argument('--profile',
                   choices=['cProfile', 'yappi'],
                   help='Profile the capture')
    return on_cmd


def on_cmd(args):
    device = scan_require_one(name='Joulescope', config='auto')
    f = lambda: run(device, filename=args.filename,
                    duration=args.duration,
                    contiguous_duration=args.contiguous,
                    signals=args.signals,
                    out_format=args.out_format)
    if args.profile is None:
        return f()
    elif args.profile == 'cProfile':
        import cProfile
        import pstats
        cProfile.runctx('f()', globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    elif args.profile == 'yappi':
        import yappi
        yappi.start()
        rv = f()
        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()
        return rv
    else:
        raise ValueError('bad profile argument')


def run(device, filename, duration=None, contiguous_duration=None,
        signals=None, out_format=None):
    """Capture streaming data to a file.

    :param device: The Joulescope device instance from scan.
    :param filename: The output filename.
    :param duration: The capture duration in seconds.
    :param contiguous_duration: The contiguous capture duration in seconds.
    :param signals: The comma-separated signals to record for JLS v2.
        None (default) is equivalent to 'i,v'.
    :param out_format: The output format, which is one of:
        * jls2: The JLS v2 format written by pyjls.
        * jls1: The legacy JLS v1 format written by DataRecorder.
        * None: (default) equivalent to 'jls2' on the v1 backend and
          'jls1' on the v0 backend.
    :return: 0 on success, error code on failure.
    """
    is_v1 = hasattr(device, 'publish')
    if out_format is None:
        out_format = 'jls2' if is_v1 else 'jls1'
    if out_format not in ['jls1', 'jls2']:
        raise ValueError(f'invalid out_format {out_format}')
    if out_format == 'jls2' and not is_v1:
        raise ValueError('out_format jls2 requires the v1 backend')
    if out_format == 'jls1':
        return _run_jls1(device, filename, duration, contiguous_duration)
    return _run_jls2(device, filename, duration, contiguous_duration, signals)


def _run_loop(device, quit_fn):
    """Poll device status until the capture completes.

    :param device: The open, streaming device.
    :param quit_fn: The callable() that returns the quit status.
    """
    time_last = time.time()
    status_failures = 0
    while not quit_fn():
        time.sleep(0.01)
        time_now = time.time()
        if time_now - time_last > 1.0:
            s = device.status()
            if s.get('driver', {}).get('return_code', {}).get('value', 1):
                status_failures += 1
                if status_failures >= 3:
                    raise RuntimeError(f'status_failures = {status_failures}')
            logging.getLogger().info(s)
            time_last = time_now


def _run_jls2(device, filename, duration=None, contiguous_duration=None,
              signals=None):
    """Capture to a JLS v2 file using pyjoulescope_driver.record.Record."""
    from pyjoulescope_driver.record import Record
    signals = 'i,v' if signals is None else signals
    quit_ = False

    def do_quit(*args, **kwargs):
        nonlocal quit_
        quit_ = 'quit from SIGINT'

    def on_stop(event, message):
        nonlocal quit_
        quit_ = 'quit from stop duration'

    recorder = None
    signals_prev = None
    signal.signal(signal.SIGINT, do_quit)
    try:
        device.open()
        signals_prev = device.parameter_get('signals')
        device.parameter_set('signals', signals)
        signals = device.parameter_get('signals')  # canonical form
        recorder = Record(device.driver, device.device_path,
                          signals=signals, auto=[])
        recorder.open(filename)
        device.start(stop_fn=on_stop, duration=duration,
                     contiguous_duration=contiguous_duration)
        _run_loop(device, lambda: quit_)
        device.stop()
    except Exception:
        logging.getLogger().exception('while capturing data')
        print('Data capture failed')
        return 1
    finally:
        # each step must run even if an earlier one fails (device removal)
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                logging.getLogger().exception('recorder close failed')
        if signals_prev is not None:
            try:
                device.parameter_set('signals', signals_prev)
            except Exception:
                logging.getLogger().exception('signals restore failed')
        device.close()
    print('done capturing data: %s' % quit_)
    return 0


def _run_jls1(device, filename, duration=None, contiguous_duration=None):
    """Capture to a legacy JLS v1 file using DataRecorder."""
    from joulescope.data_recorder import DataRecorder
    quit_ = False

    def do_quit(*args, **kwargs):
        nonlocal quit_
        quit_ = 'quit from SIGINT'

    def on_stop(event, message):
        nonlocal quit_
        quit_ = 'quit from stop duration'

    recorder = None
    signal.signal(signal.SIGINT, do_quit)
    try:
        device.open()
        recorder = DataRecorder(filename,
                                calibration=device.calibration)
        device.stream_process_register(recorder)
        device.start(stop_fn=on_stop, duration=duration,
                     contiguous_duration=contiguous_duration)
        _run_loop(device, lambda: quit_)
        device.stop()
    except Exception:
        logging.getLogger().exception('while capturing data')
        print('Data capture failed')
        return 1
    finally:
        if recorder is not None:
            recorder.close()
        device.close()
    print('done capturing data: %s' % quit_)
    return 0
