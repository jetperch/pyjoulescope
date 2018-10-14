# Copyright 2018 Jetperch LLC.  All rights reserved.

"""
Handle the USB data IN stream.

This module contains a separate thread that is used to process the incoming
data stream.  This stream contains the raw samples from Joulescope.
"""

import threading
import queue
import time
from joulescope.native import StreamBuffer
import logging
log = logging.getLogger(__name__)


class StreamThread:
    """Stream thread to process data received on USB IN.

    This thread ensures that the main USB device thread remains responsive
    to ensure that the PC keeps up.
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(name='stream', target=self.run)
        self._thread.start()
        self.buffer = StreamBuffer(2000000 * 60, [100, 100, 100])
        self.return_value = False  # handle stop by callback
        self.counter = 0
        self.congestion = 0

    def insert(self, data):
        """Handle raw USB data.

        :param data: The ctypes.c_uint8 * N data to process.
        """
        self.buffer.insert(data)
        self._queue.put(('process', None))

    def process(self, cmd, args, congestion):
        if cmd == 'process':
            self.buffer.process()
        elif cmd == 'data':
            fn, data = args
            try:
                rv = fn(data, congestion)
                if bool(rv):
                    log.info('stop requested by data callback')
                    self.return_value = True
            except Exception:
                log.exception('while calling data function')
                self.return_value = True  # abort capture
        elif cmd == 'status':
            fn, status = args
            try:
                status['host_congestion'] = {'value': self.congestion, 'units': 'cmds'}
                status['host_data_thread'] = {'value': self.counter, 'units': 'cmds'}
                self.congestion = 0
                self.counter = 0
                rv = fn(status)
                if bool(rv):
                    log.info('stop requested by status callback')
                    self.return_value = True
            except Exception:
                log.exception('while calling status function')
                self.return_value = True  # abort capture
        elif cmd == 'user':
            fn, _ = args
            try:
                fn(congestion)
            except Exception:
                log.exception('while calling function %s' % fn.__name__)
                self.return_value = True  # abort capture
        elif cmd == 'reset':
            self.return_value = False
        elif cmd == 'quit':
            return True
        return False

    def run(self):
        _quit = False
        time_last = time.time()
        while not _quit:
            try:
                cmd, args = self._queue.get(timeout=1.0)
                self.counter += 1
                sz = self._queue.qsize()
                self.congestion = max(self.congestion, sz)
                _quit = self.process(cmd, args, sz)
            except queue.Empty:
                pass
            self.buffer.process()
            time_now = time.time()
            if time_now - time_last > 1.0:
                time_last = time_now

    def get_endpoint_callback(self, fn, cmd):
        """Get the callback for use by the endpoint.

        :param fn: The callback provided to
            :meth:`WinUsbDevice.read_stream_start`.  This callback will be
            posted to this data thread and then processed from the data thread.
        :param cmd: The command to issue which is one of
            ['data', 'status', 'user'].
        """
        def cbk(data):
            if self._thread:
                self._queue.put((cmd, (fn, data)))
            return self.return_value
        return cbk

    def post(self, fn):
        """Post a callable to executed from the data thread.

        :param fn: The callable(congestion).
            Congestion is the approximate number of commands
            current pending for the thread.
            Return True to halt data processing.
            Return None or False to continue data processing.
        """
        if self._thread:
            self._queue.put(('user', (fn, None)))

    def reset(self):
        if self._thread:
            self._queue.put(('reset', None))

    def close(self):
        if self._thread:
            self._queue.put(('quit', None))
            self._thread.join()
            self._thread = None
