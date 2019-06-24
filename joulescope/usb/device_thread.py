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
Threaded wrapper for the device API.
"""

import threading
import queue
import logging
from joulescope.usb.api import DeviceDriverApi


log = logging.getLogger(__name__)
log.setLevel(level=logging.INFO)
TIMEOUT = 3.0
TIMEOUT_OPEN = 6.0


def default_callback(*args, **kwargs):
    return None


class DeviceThread:
    """Wrap a :class:`Device` in a thread.

    This class implements the Device API and simply wraps a Device
    implementation so that it runs in its own thread.
    """

    def __init__(self, usb_device: DeviceDriverApi):
        self._device = usb_device
        self._cmd_queue = queue.Queue()  # tuples of (command, args, callback)
        self._signal_queue = queue.Queue()
        self._thread = None
        self.counter = 0

    def _cmd_process(self, cmd, args, cbk):
        # all exceptions handled by _cmd_process_all
        log.debug('_cmd_process %s', cmd)
        if cmd == 'status':
            cbk(self._device.status())
        elif cmd == 'open':
            event_callback_fn = args
            cbk(self._device.open(event_callback_fn))
        elif cmd == 'close':
            cbk(self._device.close())
        elif cmd == 'control_transfer_out':
            args, kwargs = args
            self._device.control_transfer_out(cbk, *args, **kwargs)
        elif cmd == 'control_transfer_in':
            args, kwargs = args
            self._device.control_transfer_in(cbk, *args, **kwargs)
        elif cmd == 'read_stream_start':
            args, kwargs = args
            cbk(self._device.read_stream_start(*args, **kwargs))
        elif cmd == 'read_stream_stop':
            args, kwargs = args
            cbk(self._device.read_stream_stop(*args, **kwargs))
        elif cmd == '__str__':
            cbk(str(self._device))
        else:
            log.warning('unsupported command %s', cmd)

    def _cmd_process_all(self):
        _quit = False
        try:
            while not _quit:
                cmd, args, cbk = self._cmd_queue.get(timeout=0.0)
                self.counter += 1
                if not callable(cbk):
                    cbk = default_callback
                try:
                    self._cmd_process(cmd, args, cbk)
                except Exception as ex:
                    log.exception('DeviceThread._cmd_process_all')
                    cbk(ex)
                if cmd in ['close']:
                    log.info('DeviceThread._cmd_process_all close')
                    _quit = True
        except queue.Empty:
            pass
        except Exception:
            log.exception('DeviceThread._cmd_process_all unhandled')
        return _quit

    def _cmd_flush(self):
        while True:
            try:
                cmd, args, cbk = self._cmd_queue.get(timeout=0.0)
                self.counter += 1
                if not callable(cbk):
                    continue
                cbk(ConnectionError('device closed'))
            except queue.Empty:
                break
            except Exception:
                continue

    def run(self):
        _quit = False
        log.info('DeviceThread.run start')
        while not _quit:
            try:
                self._device.process(timeout=1.0)
            except Exception:
                log.exception('In device thread')
            _quit = self._cmd_process_all()
        self._cmd_flush()
        log.info('DeviceThread.run done')

    def _post(self, command, args, cbk):
        # log.debug('DeviceThread %s', command)
        if self._thread is None:
            log.info('DeviceThread.post(%s) when thread not running', command)
        else:
            self._cmd_queue.put((command, args, cbk))
            self._device.signal()

    def _post_block(self, command, args, timeout=None):
        timeout = TIMEOUT if timeout is None else float(timeout)
        q = queue.Queue()
        log.debug('_post_block %s start', command)
        self._post(command, args, lambda rv_=None: q.put(rv_))
        if self._thread is None:
            raise IOError('DeviceThread not running')
        else:
            try:
                rv = q.get(timeout=timeout)
            except queue.Empty as ex:
                log.error('device thread hung: %s', command)
                self.close()
                rv = ex
            except Exception as ex:
                rv = ex
        if isinstance(rv, Exception):
            raise IOError(rv)
        log.debug('_post_block %s done', command)  # rv
        return rv

    def __str__(self):
        if self._thread is not None:
            return self._post_block('__str__', None)
        else:
            return str(self._device)

    def open(self, event_callback_fn=None):
        self.close()
        log.info('open')
        self._thread = threading.Thread(name='usb_device', target=self.run)
        self._thread.start()
        return self._post_block('open', event_callback_fn, timeout=TIMEOUT_OPEN)

    def close(self):
        if self._thread is not None:
            log.info('close')
            try:
                self._post_block('close', None)
            except Exception:
                log.exception('while attempting to close')
            self._thread.join(timeout=TIMEOUT)
            self._thread = None

    def control_transfer_out(self, *args, **kwargs):
        return self._post_block('control_transfer_out', (args, kwargs))

    def control_transfer_in(self, *args, **kwargs):
        return self._post_block('control_transfer_in', (args, kwargs))

    def read_stream_start(self, *args, **kwargs):
        return self._post_block('read_stream_start', (args, kwargs))

    def read_stream_stop(self, *args, **kwargs):
        return self._post_block('read_stream_stop', (args, kwargs))

    def status(self):
        return self._post_block('status', None)

    def signal(self):
        self._signal_queue.put(None)

    def process(self, timeout=None):
        try:
            self._signal_queue.get(timeout=timeout)
        except queue.Empty:
            pass
