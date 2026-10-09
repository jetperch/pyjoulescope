# Copyright 2022-2025 Jetperch LLC
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


from joulescope.deprecation import warn_once
from joulescope.parameters_v1 import PARAMETERS, PARAMETERS_DICT, name_to_value, value_to_name
from .stream_buffer import StreamBuffer
from joulescope.view import View
from pyjoulescope_driver import DeviceContext, DevicePath
import collections
import copy
import functools
import logging
import numpy as np
import queue
import threading


# The streaming signals common to all v1 devices, keyed by the canonical
# short name used by the 'signals' parameter.  Each topic prefix provides
# '!data' and 'ctrl' subtopics.  idx is the StreamBuffer buffer key.
_SIGNALS_BASE = {
    'i': {'topic': 's/i/', 'idx': (1, 0)},
    'v': {'topic': 's/v/', 'idx': (2, 0)},
    'p': {'topic': 's/p/', 'idx': (3, 0)},
    'r': {'topic': 's/i/range/', 'idx': (4, 0)},
    '0': {'topic': 's/gpi/0/', 'idx': (5, 0)},
    '1': {'topic': 's/gpi/1/', 'idx': (5, 1)},
}

# The extended signals available on the JS220 and JS320.
_SIGNALS_EXTENDED = {
    '2': {'topic': 's/gpi/2/', 'idx': (5, 2)},
    '3': {'topic': 's/gpi/3/', 'idx': (5, 3)},
    'T': {'topic': 's/gpi/7/', 'idx': (5, 7)},
}

# Map the short signal names to StreamBuffer extended signal names.
_SIGNALS_SHORT_TO_EXTENDED = {'2': 'gpi2', '3': 'gpi3', 'T': 'trigger_in'}

# The default timeout for statistics_get and statistics_iter, in seconds.
_STATISTICS_TIMEOUT = 2.0

# The maximum number of statistics values that statistics_get and
# statistics_iter buffer for the caller: 50 seconds at the default 2 Hz.
_STATISTICS_QUEUE_LENGTH = 100


# The statistics source names and the legacy v0 alias.
_STATISTICS_SOURCE_NAMES = {'sensor', 'host'}
_STATISTICS_SOURCE_ALIASES = {'stream_buffer': 'host'}

# The stop_fn event for device removal: v0 DeviceEvent.COMMUNICATION_ERROR.
_EVENT_DEVICE_REMOVED = 1


class Device:

    def __init__(self, driver, device_path):
        self.config = None
        self._driver = driver
        device_path = DevicePath(device_path.rstrip('/'))
        self._log = logging.getLogger(__name__ + '.' + device_path.replace('/', '.'))
        self._path = device_path
        self._ctx = DeviceContext(driver, device_path)  # replaced by open()
        self.is_open = False
        self._stream_cbk_objs = []
        self._stream_cbk_objs_add = []
        self._stop_fn = None
        self._input_sampling_frequency = 0
        self._output_sampling_frequency = 0
        self._h_fs = 2000000  # value published to h/fs on open: the maximum streaming rate
        self._statistics_callbacks = {}  # source -> [cbk]
        self._statistics_offsets = {}  # source -> [duration, charge, energy]
        self._statistics_queue = None  # deque for statistics_get, when active
        self._statistics_queue_cond = threading.Condition()
        self._statistics_active = {}  # source -> (ctrl, topic, fn) while subscribed
        self._is_streaming = False
        self._signals_map = dict(_SIGNALS_BASE)
        self._streaming_topics = []
        self._buffer_duration = 30
        self.stream_buffer = None
        self._on_stream_cbk = self._on_stream  # hold reference for unsub
        self._parameters = {}
        self._parameters_override = {}  # name -> device-specific Parameter
        self._parameter_set_queue = []
        for p in PARAMETERS:
            if p.default is not None:
                try:
                    self._parameters[p.name] = name_to_value(p.name, p.default)
                except KeyError:
                    if p.validator is not None:
                        self._parameters[p.name] = p.validator(p.default)
                    else:
                        self._parameters[p.name] = p.default

    def __str__(self):
        _, model, serial_number = self._path.split('/')  # keep "&" in bootloader mode
        return f'{model.upper()}-{serial_number}'

    @property
    def device_path(self):
        return self._path

    @property
    def driver(self):
        """The underlying pyjoulescope_driver.Driver instance."""
        return self._driver

    @property
    def usb_device(self):
        return self._path

    @property
    def input_sampling_frequency(self):
        """The original input sampling frequency."""
        return self._input_sampling_frequency

    @property
    def output_sampling_frequency(self):
        """The output sampling frequency."""
        return self._output_sampling_frequency

    def _output_sampling_frequency_set(self, value):
        """Protected setter for output_sampling_frequency.

        Applications should use: `parameter_set('sampling_frequency', value)`.
        """
        self._output_sampling_frequency = value
        if self.stream_buffer is not None:
            self.stream_buffer.output_sampling_frequency = self._output_sampling_frequency

    @property
    def sampling_frequency(self):
        """The output sampling frequency."""
        return self.output_sampling_frequency

    @property
    def buffer_duration(self):
        """The stream buffer duration."""
        return self._buffer_duration

    @buffer_duration.setter
    def buffer_duration(self, value):
        self._buffer_duration = value
        if self.stream_buffer is not None:
            self.stream_buffer.buffer_duration = self._buffer_duration

    @property
    def statistics_callback(self):
        """Get the first registered statistics callback."""
        for cbks in self._statistics_callbacks.values():
            if len(cbks):
                return cbks[0]
        return None

    @statistics_callback.setter
    def statistics_callback(self, cbk):
        """Set the statistics callback.

        :param cbk: The callable(data) where data is a statistics data
            structure.  See the `statistics documentation <statistics.html>`_
            for details on the data format.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
        """
        for source, cbks in list(self._statistics_callbacks.items()):
            for unregister_cbk in list(cbks):
                self.statistics_callback_unregister(unregister_cbk, source)
        self.statistics_callback_register(cbk)

    def statistics_callback_register(self, cbk, source=None):
        """Register a statistics callback.

        :param cbk: The callable(data) where data is a statistics data
            structure.  See the `statistics documentation <statistics.html>`_
            for details on the data format.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
            See :meth:`statistics_get` to receive statistics on the
            caller's thread instead.
        :param source: The statistics source, which is one of:

            * None (default): the model and scan config default,
              see :attr:`statistics_source`.
            * 'sensor': computed on the instrument.
            * 'host': computed by the host driver from the full-rate
              sample stream, so it requires streaming.  'stream_buffer'
              is the legacy v0 name.

            The JS110 supports both sources, even at the same time.
            The JS220 and JS320 only compute statistics on the
            instrument: they ignore 'host', with a DeprecationWarning.
        :raise ValueError: If source is not a statistics source name.

        WARNING: calling :meth:`statistics_callback` after calling this method
        may result in unusual behavior.  Do not mix these API calls.
        """
        if cbk is None:
            return
        if not callable(cbk):
            self._log.warning('Requested callback is not callable')
            return
        source = self._statistics_source_resolve(source)
        cbks = self._statistics_callbacks.setdefault(source, [])
        if not len(cbks):
            self._statistics_start(source)
        cbks.append(cbk)

    def statistics_callback_unregister(self, cbk, source=None):
        """Unregister a statistics callback.

        :param cbk: The callback previously provided to
            :meth:`statistics_callback_register`.
        :param source: The source provided to
            :meth:`statistics_callback_register`.  None (default)
            searches all sources.
        """
        if source is None:
            sources = [s for s, cbks in self._statistics_callbacks.items() if cbk in cbks]
        else:
            sources = [self._statistics_source_resolve(source)]
        for source in sources:
            cbks = self._statistics_callbacks.get(source, [])
            if cbk in cbks:
                cbks.remove(cbk)
                if not len(cbks):
                    self._statistics_stop(source)
                return
        self._log.warning('statistics_callback_unregister: callback not registered')

    @property
    def statistics_source(self):
        """The default statistics source, which is one of:

        * 'sensor': computed on the instrument.
        * 'host': computed by the host driver from the full-rate
          sample stream.

        The JS220 and JS320 always use 'sensor'.  The JS110 uses 'sensor'
        when scanned with config='off', and 'host' otherwise.
        :meth:`statistics_callback_register` selects the source for each
        callback.
        """
        return 'sensor'

    def _statistics_sources(self):
        """The supported statistics sources.

        :return: The dict mapping each source name to its
            (ctrl topic, value topic).  The ctrl topic is None when the
            value topic is always enabled.
        """
        return {'sensor': ('s/stats/ctrl', 's/stats/value')}

    def _statistics_source_resolve(self, source):
        """Resolve a caller-provided source to one this device supports."""
        default = self.statistics_source
        if source is None:
            return default
        name = str(source).lower()
        name = _STATISTICS_SOURCE_ALIASES.get(name, name)
        if name not in _STATISTICS_SOURCE_NAMES:
            raise ValueError(f'invalid statistics source {source!r}: '
                             f'use one of {sorted(_STATISTICS_SOURCE_NAMES)}')
        if name not in self._statistics_sources():
            model = self.model.upper()
            warn_once(f'statistics_source_{self.model}',
                      f'The {model} computes statistics on the instrument: '
                      f'statistics source {source!r} is deprecated and ignored.')
            return default
        return name

    def _on_stats(self, source, topic, value):
        # The JS320 reports 16 Msps sample ids; the JS110 and JS220 report 2 Msps.
        period = 1 / value['time'].get('sample_freq', {}).get('value', 2e6)
        s_start, s_stop = [x * period for x in value['time']['samples']['value']]

        offsets = self._statistics_offsets.get(source)
        if offsets is None:
            duration = s_start
            charge = value['accumulators']['charge']['value']
            energy = value['accumulators']['energy']['value']
            offsets = self._statistics_offsets[source] = [duration, charge, energy]
        duration, charge, energy = offsets
        value['time']['range'] = {
            'value': [s_start - duration, s_stop - duration],
            'units': 's'
        }
        value['time']['delta'] = {'value': s_stop - s_start, 'units': 's'}
        value['accumulators']['charge']['value'] -= charge
        value['accumulators']['energy']['value'] -= energy
        for k in value['signals'].values():
            k['µ'] = k['avg']
            k['σ2'] = {'value': k['std']['value'] ** 2, 'units': k['std']['units']}
            if 'integral' in k:
                k['∫'] = k['integral']
        value['source'] = source
        for cbk in list(self._statistics_callbacks.get(source, [])):
            cbk(value)

    def _statistics_start(self, source):
        if self.is_open and source not in self._statistics_active:
            ctrl, topic = self._statistics_sources()[source]
            fn = functools.partial(self._on_stats, source)
            self._statistics_active[source] = (ctrl, topic, fn)
            if ctrl is not None:
                self.publish(ctrl, 1)
            self.subscribe(topic, 'pub', fn)

    def _statistics_stop(self, source):
        active = self._statistics_active.pop(source, None)
        if self.is_open and active is not None:
            ctrl, topic, fn = active
            self.unsubscribe(topic, fn)
            if ctrl is not None:
                self.publish(ctrl, 0)

    def _on_statistics_queue(self, value):
        with self._statistics_queue_cond:
            q = self._statistics_queue
            if q is None:
                return
            if len(q) == q.maxlen:
                self._log.warning('statistics_get queue full: dropping oldest value')
            q.append(value)
            self._statistics_queue_cond.notify_all()

    def _statistics_queue_stop(self, unregister=True):
        """Stop statistics_get buffering and wake any waiting caller.

        :param unregister: False to leave the driver untouched, such as
            after device removal.
        """
        with self._statistics_queue_cond:
            q, self._statistics_queue = self._statistics_queue, None
            self._statistics_queue_cond.notify_all()
        if q is None:
            return
        if unregister:
            self.statistics_callback_unregister(self._on_statistics_queue)
        else:
            for cbks in self._statistics_callbacks.values():
                if self._on_statistics_queue in cbks:
                    cbks.remove(self._on_statistics_queue)

    def statistics_get(self, timeout=None):
        """Get the next statistics value on the caller's thread.

        :param timeout: The maximum time to wait in float seconds.
            None (default) waits 2 seconds.
        :return: The statistics data structure, the same as provided to
            :meth:`statistics_callback_register` callbacks.
        :raise RuntimeError: If the device is not open.
        :raise TimeoutError: If no statistics value arrives in time.

        The first call starts buffering statistics values until
        :meth:`close`, so consecutive calls return consecutive values
        without gaps.  The buffer holds the 100 most recent values,
        which is 50 seconds at the default reduction_frequency of 2 Hz.
        """
        if not self.is_open:
            raise RuntimeError('statistics_get requires an open device')
        timeout = _STATISTICS_TIMEOUT if timeout is None else float(timeout)
        cond = self._statistics_queue_cond
        with cond:
            register = self._statistics_queue is None
            if register:
                self._statistics_queue = collections.deque(maxlen=_STATISTICS_QUEUE_LENGTH)
        if register:  # outside the lock: the driver thread calls _on_statistics_queue
            self.statistics_callback_register(self._on_statistics_queue)
        with cond:
            ready = cond.wait_for(self._statistics_queue_ready, timeout)
            if self._statistics_queue is None:
                raise RuntimeError('device closed during statistics_get')
            if not ready:
                raise TimeoutError(f'statistics_get timed out after {timeout} seconds')
            return self._statistics_queue.popleft()

    def _statistics_queue_ready(self):
        q = self._statistics_queue
        return q is None or len(q) > 0

    def statistics_iter(self, count=None, timeout=None):
        """Iterate over statistics values on the caller's thread.

        :param count: The number of statistics values.  None (default)
            iterates until the device closes or the caller stops.
        :param timeout: The maximum time to wait for each value in
            float seconds.  None (default) waits 2 seconds.
        :return: The iterator over statistics data structures.
        :raise RuntimeError: If the device is not open.
        :raise TimeoutError: If a statistics value does not arrive in time.

        Example::

            for stats in device.statistics_iter(count=10):
                print(stats['signals']['current']['µ']['value'])

        See :meth:`statistics_get`.
        """
        if not self.is_open:
            raise RuntimeError('statistics_iter requires an open device')

        def generate():
            n = 0
            while count is None or n < count:
                try:
                    value = self.statistics_get(timeout)
                except RuntimeError:
                    if not self.is_open:
                        return  # device closed
                    raise
                yield value
                n += 1

        return generate()

    def statistics_accumulators_clear(self):
        """Clear the charge and energy accumulators."""
        self._statistics_offsets.clear()

    def view_factory(self):
        """Construct a new View into the device's data.

        :return: A View-compatible instance.
        """
        if self.stream_buffer is None:
            raise RuntimeError('view_factory, but no stream buffer')
        view = View(self.stream_buffer, self.calibration)
        view.on_close = lambda: self.stream_process_unregister(view)
        self.stream_process_register(view)
        return view

    def parameters(self, name=None):
        """Get the list of :class:`joulescope.parameter.Parameter` instances.

        :param name: The optional name of the parameter to retrieve.
            None (default) returns a list of all parameters.
        :return: The list of all parameters.  If name is provided, then just
            return that single parameters.

        The parameter options reflect the values supported by this device.
        For backwards compatibility, :meth:`parameter_set` also accepts the
        legacy values of other Joulescope models where possible.
        """
        params = [self._parameters_override.get(p.name, p) for p in PARAMETERS]
        if name is not None:
            for p in params:
                if p.name == name:
                    return copy.deepcopy(p)
            return None
        return copy.deepcopy(params)

    def parameter_set(self, name, value):
        """Set a parameter value.

        :param name: The parameter name
        :param value: The new parameter value
        :raise KeyError: if name not found.
        :raise ValueError: if value is not allowed
        """
        raise NotImplementedError()

    def parameter_get(self, name, dtype=None):
        """Get a parameter value.

        :param name: The parameter name.
        :param dtype: The data type for the parameter.  None (default)
            attempts to convert the value to the enum string.
            'actual' returns the value in its actual type used by the driver.
        :raise KeyError: if name not found.
        """
        if name == 'current_ranging':
            pnames = ['type', 'samples_pre', 'samples_window', 'samples_post']
            values = [str(self.parameter_get('current_ranging_' + p)) for p in pnames]
            return '_'.join(values)
        p = PARAMETERS_DICT[name]
        if p.path == 'info':
            return self._parameter_get_info(name)
        value = self._parameters[name]
        if dtype == 'actual':
            return value
        try:
            return value_to_name(name, value)
        except KeyError:
            return value

    def _topic_make(self, topic):
        """Get the absolute topic, for backwards compatibility."""
        return self._ctx.topic(topic.lstrip('/'))

    def publish(self, topic, value, timeout=None):
        """Publish to the underlying joulescope_driver instance.

        :param topic: The publish topic.
        :param value: The publish value.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._ctx.publish(topic.lstrip('/'), value, timeout)

    def query(self, topic, timeout=None):
        """Query the underlying joulescope_driver instance.

        :param topic: The publish topic to query.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        :return: The value associated with topic.
        """
        return self._ctx.query(topic.lstrip('/'), timeout)

    def subscribe(self, topic, flags, fn, timeout=None):
        """Subscribe to receive topic updates.

        :param self: The driver instance.
        :param topic: Subscribe to this topic string.
        :param flags: The flags or list of flags for this subscription.
            The flags can be int32 jsdrv_subscribe_flag_e or string
            mnemonics, which are:

            - pub: Subscribe to normal values
            - pub_retain: Subscribe to normal values and immediately publish
              all matching retained values.  With timeout, this function does
              not return successfully until all retained values have been
              published.
            - metadata_req: Subscribe to metadata requests (not normally useful).
            - metadata_rsp: Subscribe to metadata updates.
            - metadata_rsp_retain: Subscribe to metadata updates and immediately
              publish all matching retained metadata values.
            - query_req: Subscribe to all query requests (not normally useful).
            - query_rsp: Subscribe to all query responses.
            - return_code: Subscribe to all return code responses.

        :param fn: The function to call on each publish.  Note that python
            dynamically constructs bound methods.  To unsubscribe a method,
            provide the exact same bound method instance to unsubscribe.
            This constrain usually means that the caller needs to hold onto
            the instance.method value passed to this function.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        :raise RuntimeError: on subscribe failure.
        """
        return self._ctx.subscribe(topic.lstrip('/'), flags, fn, timeout)

    def unsubscribe(self, topic, fn, timeout=None):
        """Unsubscribe a callback to a topic.

        :param topic: Unsubscribe from this topic string.
        :param fn: The callback function to unsubscribe.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._ctx.unsubscribe(topic.lstrip('/'), fn, timeout)

    def unsubscribe_all(self, fn, timeout=None):
        """Unsubscribe a callback from all topics.

        :param fn: The callback function to unsubscribe.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._driver.unsubscribe_all(fn, timeout)

    def _config_apply(self, config=None):
        """Apply a configuration set by scan.

        :param config: The configuration string.
        """
        pass

    @property
    def _signals_selected(self):
        """The list of selected signal short names."""
        return self._parameters['signals'].split(',')

    def _on_signals(self, value):
        """Validate the 'signals' parameter selection for this device.

        :param value: The canonical comma-separated short-name string.
        :raise ValueError: If this device does not support a selected signal.

        The selection takes effect when streaming (re)starts.
        """
        selected = value.split(',')
        unsupported = [s for s in selected if s not in self._signals_map]
        if unsupported:
            raise ValueError(
                f'signals not supported by {self.model}: {unsupported}')
        if self._is_streaming:
            self._log.warning(
                'signals changed while streaming; takes effect on next start')

    def open(self, event_callback_fn=None, mode=None, timeout=None):
        """Open this device.

        :param event_callback_fn: The function(event, message) to call on
            asynchronous events, mostly to allow robust handling of device
            errors.  "event" is one of the :class:`DeviceEvent` values,
            and the message is a more detailed description of the event.
        :param mode: The open mode which is one of:
            * 'defaults': Reconfigure the device for default operation.
            * 'restore': Update our state with the current device state.
            * 'raw': Open the device in raw mode for development or firmware update.
            * None: equivalent to 'defaults'.
        :param timeout: The timeout in seconds.  None uses the default timeout.
        """
        self._ctx = self._driver.open(self._path, mode, timeout)
        self.is_open = True
        self._statistics_offsets.clear()  # the sample ids and accumulators may restart
        self.publish('h/fs', self._h_fs)
        while len(self._parameter_set_queue):
            name, value = self._parameter_set_queue.pop(0)
            self.parameter_set(name, value)
        for source, cbks in self._statistics_callbacks.items():
            if len(cbks):
                self._statistics_start(source)
        device = 'js110' if 'js110' in self._path.lower() else 'js220'
        self.stream_buffer = StreamBuffer(self._buffer_duration,
                                          frequency=self._input_sampling_frequency,
                                          device=device,
                                          output_frequency=self._output_sampling_frequency)
        self._config_apply(self.config)
        return self._ctx

    def close(self, timeout=None):
        """Close this device and release resources.

        :param timeout: The timeout in seconds.  None uses the default timeout.
        """
        if not self.is_open:
            return
        self._statistics_queue_stop()
        for source in list(self._statistics_active):
            self._statistics_stop(source)
        self.stop()
        self.is_open = False
        # notify and unregister the stream process objects (v0 compatible)
        self._stream_process_call('close')
        self._stream_cbk_objs.clear()
        self._stream_cbk_objs_add.clear()
        self.stream_buffer = None
        return self._ctx.close(timeout)  # also unsubscribes any remaining subscriptions

    def _on_remove(self):
        """Handle device removal, on the driver thread.

        The driver has already closed the device, so this only updates
        the host-side state and must not block on the driver.  Any
        stop_fn receives event 1 (v0 DeviceEvent.COMMUNICATION_ERROR).
        """
        if not self.is_open:
            return
        self._log.info('device removed')
        self.is_open = False
        self._statistics_queue_stop(unregister=False)
        self._statistics_active.clear()
        if self._is_streaming:
            self._is_streaming = False
            self._streaming_topics = []
            fn, self._stop_fn = self._stop_fn, None
            if callable(fn):
                fn(_EVENT_DEVICE_REMOVED, 'device removed')
            self._stream_process_call('stop')
        self._stream_process_call('close')
        self._stream_cbk_objs.clear()
        self._stream_cbk_objs_add.clear()
        self.stream_buffer = None
        self._ctx.close(timeout=0)  # release the subscriptions without waiting

    @property
    def firmware(self):
        """The running firmware image, which is one of:

        * 'app': the application firmware.
        * 'bootloader': the JS110 or JS220 bootloader, which the device
          path shows with the "&" model prefix.
        * 'recovery': the JS320 recovery image.  The driver does not yet
          report it, so a JS320 is always 'app'.
        """
        return 'bootloader' if self._path.is_bootloader else 'app'

    @property
    def model(self):
        """The lowercase model, such as "js220"."""
        return self._path.model

    @property
    def serial_number(self):
        return self._path.serial_number

    @property
    def device_serial_number(self):
        return self.serial_number

    @property
    def calibration(self):
        return None

    def info(self):
        """Get the device information structure.

        :return: The device information structure.
        """
        raise NotImplementedError()

    def _stream_process_call(self, method, *args, **kwargs):
        rv = False
        b, self._stream_cbk_objs, self._stream_cbk_objs_add = self._stream_cbk_objs + self._stream_cbk_objs_add, [], []
        for obj in b:
            fn = getattr(obj, method, None)
            if not callable(fn):
                self._stream_cbk_objs.append(obj)
                continue
            if obj.driver_active:
                try:
                    rv |= bool(fn(*args, **kwargs))
                    self._stream_cbk_objs.append(obj)
                except Exception:
                    self._log.exception('%s %s() exception', obj, method)
                    obj.driver_active = False
            if not obj.driver_active:
                try:
                    if hasattr(obj, 'close'):
                        obj.close()
                except Exception:
                    self._log.exception('%s close() exception', obj)
        return rv

    def _on_stream(self, topic, value):
        # runs on the driver thread; use a local reference since close()
        # may set self.stream_buffer to None concurrently
        b = self.stream_buffer
        if b is None:
            return False
        _, e1 = b.sample_id_range
        b.insert(topic, value)
        e0, e2 = b.sample_id_range

        if e1 == e2:
            return False
        if e0 == e2:
            return False
        rv = self._stream_process_call('stream_notify', b)
        if rv:
            self.stop()
        if b.is_duration_max or b.is_contiguous_duration_max:
            self.stop()

    def start(self, stop_fn=None, duration=None, contiguous_duration=None):
        """Start data streaming.

        :param stop_fn: The function(event, message) called when the device stops.
            The device can stop "automatically" on errors.
            Call :meth:`stop` to stop from the caller.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
        :param duration: The duration in seconds for the capture.
        :param contiguous_duration: The contiguous duration in seconds for
            the capture.  As opposed to duration, this ensures that the
            duration has no missing samples.  Missing samples usually
            occur when the device first starts.

        If streaming was already in progress, it will be restarted.
        """
        self.stop()
        # unwind any partial subscriptions left by a prior start() that
        # failed mid-loop (e.g. device removal); no-op normally
        topics, self._streaming_topics = self._streaming_topics, []
        for topic in topics:
            self.unsubscribe(topic + '!data', self._on_stream_cbk, timeout=0)
            self.publish(topic + 'ctrl', 0, timeout=0)
        selected = self._signals_selected
        extras = [_SIGNALS_SHORT_TO_EXTENDED[s] for s in selected
                  if s in _SIGNALS_SHORT_TO_EXTENDED]
        self.stream_buffer.extra_signals = extras  # no-op when unchanged
        self.stream_buffer.reset()
        self.stream_buffer.duration_max = duration
        self.stream_buffer.contiguous_duration_max = contiguous_duration
        self._stop_fn = stop_fn
        for name, info in self._signals_map.items():
            b = self.stream_buffer.buffers.get(info['idx'])
            if name in selected:
                topic = info['topic']
                if b is not None:
                    b.active = True
                self.subscribe(topic + '!data', 'pub', self._on_stream_cbk)
                self.publish(topic + 'ctrl', 1)
                self._streaming_topics.append(topic)
            elif b is not None:
                b.active = False
        self._is_streaming = True
        self._stream_process_call('start', self.stream_buffer)

    def stop(self):
        """Stop data streaming.

        :return: True if stopped.  False if was already stopped.

        This method is always safe to call, even after the device has been
        stopped or removed.
        """
        if self._is_streaming:
            self._is_streaming = False
            topics, self._streaming_topics = self._streaming_topics, []
            for topic in topics:
                self.unsubscribe(topic + '!data', self._on_stream_cbk, timeout=0)
                self.publish(topic + 'ctrl', 0, timeout=0)
            fn, self._stop_fn = self._stop_fn, None
            if callable(fn):
                fn(0, '')  # status, message
            self._stream_process_call('stop')

    def read(self, duration=None, contiguous_duration=None, out_format=None, fields=None):
        """Read data from the device.

        :param duration: The duration in seconds for the capture.
            The duration must fit within the stream_buffer.
        :param contiguous_duration: The contiguous duration in seconds for
            the capture.  As opposed to duration, this ensures that the
            duration has no missing samples.  Missing samples usually
            occur when the device first starts.
            The duration must fit within the stream_buffer.
        :param out_format: The output format which is one of:

            * calibrated: The Nx2 np.ndarray(float32) with columns current and voltage.
            * samples_get: The StreamBuffer samples get format.  Use the fields
              parameter to optionally specify the signals to include.
            * None: equivalent to 'calibrated'.

        :param fields: The fields for samples_get when out_format=samples_get.

        If streaming was already in progress, it will be restarted.
        If neither duration or contiguous duration is specified, the capture
        will only be stopped by callbacks registered through
        :meth:`stream_process_register`.
        """
        self._log.info('read(duration=%s, contiguous_duration=%s, out_format=%s)',
                 duration, contiguous_duration, out_format)
        if out_format not in ['calibrated', 'samples_get', None]:
            raise ValueError(f'Invalid out_format {out_format}')
        if duration is None and contiguous_duration is None:
            raise ValueError('Must specify duration or contiguous_duration')
        duration_max = len(self.stream_buffer) / self._output_sampling_frequency
        if contiguous_duration is not None and contiguous_duration > duration_max:
            raise ValueError(f'contiguous_duration {contiguous_duration} > {duration_max} max seconds')
        if duration is not None and duration > duration_max:
            raise ValueError(f'duration {duration} > {duration_max} max seconds')
        q = queue.Queue()

        def on_stop(*args, **kwargs):
            self._log.info('received stop callback: pending stop')
            q.put(None)

        self.start(on_stop, duration=duration, contiguous_duration=contiguous_duration)
        q.get()
        self.stop()
        start_id, end_id = self.stream_buffer.sample_id_range
        self._log.info('read available range %s, %s', start_id, end_id)
        if contiguous_duration is not None:
            start_id = end_id - int(contiguous_duration * self._output_sampling_frequency)
        elif duration is not None:
            start_id = end_id - int(duration * self._output_sampling_frequency)
        if start_id < 0:
            start_id = 0
        self._log.info('read actual %s, %s', start_id, end_id)

        if out_format in ['calibrated', None]:
            data = self.stream_buffer.samples_get(start_id, end_id, fields=['current', 'voltage'])
            i = data['signals']['current']['value']
            v = data['signals']['voltage']['value']
            return np.hstack([np.reshape(i, (-1, 1)), np.reshape(v, (-1, 1))])
        else:
            return self.stream_buffer.samples_get(start_id, end_id, fields=fields)

    @property
    def is_streaming(self):
        """Check if the device is streaming.

        :return: True if streaming.  False if not streaming.
        """
        return self._is_streaming

    def stream_process_register(self, obj):
        """Register a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi`.
            The instance must remain valid until its :meth:`close` is
            called.

        Call :meth:`stream_process_unregister` to disconnect the instance.
        """
        if self._is_streaming and hasattr(obj, 'start'):
            obj.start(self.stream_buffer)
        obj.driver_active = True
        self._stream_cbk_objs_add.append(obj)

    def stream_process_unregister(self, obj):
        """Unregister a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi` that was
            previously registered using :meth:`stream_process_register`.
        """
        obj.driver_active = False

    def status(self):
        """Get the current device status.

        :return: A dict containing constant status information.

        Deprecated: the v1 backend has no status to report, so this
        always returns the same values.
        """
        warn_once('status', 'Device.status() is deprecated: it returns constant values.')
        return {
            'driver': {
                'settings_result': {
                    'value': 0,
                    'units': ''},
                'fpga_frame_counter': {
                    'value': 0,
                    'units': 'frames'},
                'fpga_discard_counter': {
                    'value': 0,
                    'units': 'frames'},
                'sensor_flags': {
                    'value': 0,
                    'format': '0x{:02x}',
                    'units': ''},
                'sensor_i_range': {
                    'value': 0,
                    'format': '0x{:02x}',
                    'units': ''},
                'sensor_source': {
                    'value': 0,
                    'format': '0x{:02x}',
                    'units': ''},
                'return_code': {
                    'value': 0,
                    'format': '{}',
                    'units': '',
                },
            }
        }

    def __enter__(self):
        """Device context manager, automatically open."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Device context manager, automatically close."""
        self.close()

    def _query_gpi_value(self):
        try:
            return self._ctx.publish_and_wait('s/gpi/+/!req', 0, 's/gpi/+/!value', timeout=1.0)
        except TimeoutError as ex:
            raise RuntimeError('_query_gpi_value timed out') from ex

    def extio_status(self):
        """Read the EXTIO GPI value.

        :return: A dict containing the extio status.  Each key is the status
            item name.  The value is itself a dict with the following keys:

            * name: The status name, which is the same as the top-level key.
            * value: The actual value
            * units: The units, if applicable.
            * format: The recommended formatting string (optional).
            
        The most interesting key is "gpi_value" which returns the present
        general purpose input signal values.  The remaining keys simply
        copy parameter settings for convenience.
        """
        gpi_value = self._query_gpi_value()
        status = {
            'flags': {
                'value': 0,
                'units': ''},
            'trigger_source': {
                'value': self.parameter_get('trigger_source'),
                'units': ''},
            'current_lsb': {
                'value': self.parameter_get('current_lsb'),
                'units': ''},
            'voltage_lsb': {
                'value': self.parameter_get('voltage_lsb'),
                'units': ''},
            'gpo0': {
                'value': self.parameter_get('gpo0'),
                'units': ''},
            'gpo1': {
                'value': self.parameter_get('gpo1'),
                'units': ''},
            'gpi_value': {
                'value': gpi_value,
                'units': '',
            },
            'io_voltage': {
                'value': self.parameter_get('io_voltage'),
                'units': 'mV',
            },
        }
        for key, value in status.items():
            value['name'] = key
        return status
