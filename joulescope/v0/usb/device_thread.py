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
from joulescope.v0.usb.api import DeviceDriverApi


log = logging.getLogger(__name__)
TIMEOUT = 3.0
TIMEOUT_OPEN = 10.0


def _queue_empty(q):
    while True:
        try:
            q.get(timeout=0.0)
        except queue.Empty:
            break


class DeviceThread:
    """Wrap a :class:`Device` in a thread.

    This class implements the Device API and simply wraps a Device
    implementation so that it runs in its own thread.
    """

    def __init__(self, usb_device: DeviceDriverApi):
        self._device = usb_device
        self._cmd_queue = queue.Queue()  # tuples of (command, args, callback)
        self._signal_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._thread = None
        self._closing = False
        self._str = None
        self.counter = 0

    def _cmd_process(self, cmd, args, cbk):
        delegate_cbk = False
        rv = None
        try:
            log.debug('_cmd_process %s - start', cmd)
            if cmd == 'status':
                rv = self._device.status()
            elif cmd == 'open':
                event_callback_fn = args
                rv = self._device.open(event_callback_fn)
            elif cmd == 'close':
                rv = self._device.close()
            elif cmd == 'control_transfer_out':
                delegate_cbk = True
                args, kwargs = args
                self._device.control_transfer_out(cbk, *args, **kwargs)
            elif cmd == 'control_transfer_in':
                delegate_cbk = True
                args, kwargs = args
                self._device.control_transfer_in(cbk, *args, **kwargs)
            elif cmd == 'read_stream_start':
                args, kwargs = args
                rv = self._device.read_stream_start(*args, **kwargs)
            elif cmd == 'read_stream_stop':
                args, kwargs = args
                rv = self._device.read_stream_stop(*args, **kwargs)
            elif cmd == '__str__':
                rv = str(self._device)
            else:
                log.warning('unsupported command %s', cmd)
        except Exception:
            log.exception('While running command')
        if not delegate_cbk and callable(cbk):
            try:
                cbk(rv)
            except Exception:
                log.exception('in callback')
        log.debug('_cmd_process %s - done', cmd)

    def _cmd_process_all(self):
        _quit = False
        try:
            while not _quit:
                cmd, args, cbk = self._cmd_queue.get(timeout=0.0)
                self.counter += 1
                self._cmd_process(cmd, args, cbk)
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
                log.exception('_cmd_flush')

    def run(self):
        _quit = False
        log.info('DeviceThread.run start')
        while not _quit:
            try:
                self._device.process(timeout=0.05)
            except Exception:
                log.exception('In device thread')
            _quit = self._cmd_process_all()
        log.info('DeviceThread.run flush')
        self._cmd_flush()
        log.info('DeviceThread.run done')

    def _post(self, command, args, cbk):
        # log.debug('DeviceThread %s', command)
        if self._thread is None:
            log.info('DeviceThread.post(%s) when thread not running', command)
        else:
            self._cmd_queue.put((command, args, cbk))
            self._device.signal()

    def _join(self, timeout=None):
        timeout = TIMEOUT if timeout is None else timeout
        if not self._closing:
            self._closing = True
            self._post('close', None, None)
        if self._thread:
            # thread can safely join() multiple times
            self._thread.join(timeout=timeout)
            self._thread = None

    def _post_block(self, command, args, timeout=None):
        timeout = TIMEOUT if timeout is None else float(timeout)
        log.debug('_post_block %s start', command)
        while not self._response_queue.empty():
            log.warning('response queue not empty')
            try:
                self._response_queue.get(timeout=0.0)
            except queue.Empty:
                pass
        self._post(command, args, lambda rv_=None: self._response_queue.put(rv_))
        if self._thread is None:
            raise IOError('DeviceThread not running')
        else:
            try:
                rv = self._response_queue.get(timeout=timeout)
            except queue.Empty as ex:
                log.error('device thread hung: %s - FORCE CLOSE', command)
                self._join(timeout=TIMEOUT)
                rv = ex
            except Exception as ex:
                rv = ex
        if isinstance(rv, Exception):
            raise IOError(rv)
        log.debug('_post_block %s done', command)  # rv
        return rv

    def __str__(self):
        if self._str is not None:
            pass
        elif self._thread is not None:
            self._str = self._post_block('__str__', None)
        else:
            self._str = str(self._device)
        return self._str

    @property
    def serial_number(self):
        if self._device is None:
            return None
        return self._device.serial_number

    def open(self, event_callback_fn=None):
        self.close()
        log.info('open')
        self._thread = threading.Thread(name='usb_device', target=self.run)
        self._thread.start()
        self._closing = False
        try:
            return self._post_block('open', event_callback_fn, timeout=TIMEOUT_OPEN)
        except Exception:
            self.close()
            raise

    def close(self):
        log.info('close')
        self._join(timeout=TIMEOUT)
        _queue_empty(self._cmd_queue)
        _queue_empty(self._response_queue)
        _queue_empty(self._signal_queue)

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
