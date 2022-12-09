# Copyright 2022 Jetperch LLC
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


from joulescope.parameters_v1 import PARAMETERS, PARAMETERS_DICT, name_to_value, value_to_name
from .stream_buffer import StreamBuffer
from joulescope.view import View
import copy
import logging
import numpy as np
import queue


class Device:

    def __init__(self, driver, device_path):
        self.config = None
        self._driver = driver
        while device_path.endswith('/'):
            device_path = device_path[:-1]
        self._log = logging.getLogger(__name__ + '.' + device_path.replace('/', '.'))
        self._path = device_path
        self.is_open = False
        self._stream_cbk_objs = []
        self._stream_cbk_objs_add = []
        self._stop_fn = None
        self._input_sampling_frequency = 0
        self._output_sampling_frequency = 0
        self._statistics_callbacks = []
        self._statistics_offsets = []
        self._is_streaming = False
        self._stream_topics = []
        self._buffer_duration = 30
        self.stream_buffer = None
        self._on_stats_cbk = self._on_stats  # hold reference for unsub
        self._on_stream_cbk = self._on_stream  # hold reference for unsub
        self._parameters = {}
        self._parameter_set_queue = []
        for p in PARAMETERS:
            if p.default is not None:
                self._parameters[p.name] = name_to_value(p.name, p.default)

    def __str__(self):
        _, model, serial_number = self._path.split('/')
        return f'{model.upper()}-{serial_number}'

    @property
    def device_path(self):
        return self._path

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

    @output_sampling_frequency.setter
    def output_sampling_frequency(self, value):
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
        """Get the registered statistics callback."""
        cbks = self._statistics_callbacks
        if len(cbks):
            return cbks[0]
        else:
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
        for unregister_cbk in list(self._statistics_callbacks):
            self.statistics_callback_unregister(unregister_cbk)
        self.statistics_callback_register(cbk)

    def statistics_callback_register(self, cbk, source=None):
        """Register a statistics callback.

        :param cbk: The callable(data) where data is a statistics data
            structure.  See the `statistics documentation <statistics.html>`_
            for details on the data format.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
        :param source: The statistics source where the computation is performed.
            Ignored, always use sensor-side statistics for the JS220.

        WARNING: calling :meth:`statistics_callback` after calling this method
        may result in unusual behavior.  Do not mix these API calls.
        """
        if cbk is None:
            return
        if not callable(cbk):
            self._log.warning('Requested callback is not callable')
            return
        if not len(self._statistics_callbacks):
            self._statistics_start()
        self._statistics_callbacks.append(cbk)

    def statistics_callback_unregister(self, cbk, source=None):
        """Unregister a statistics callback.

        :param cbk: The callback previously provided to
            :meth:`statistics_callback_register`.
        :param source: The callback source.
        """
        try:
            self._statistics_callbacks.remove(cbk)
        except ValueError:
            self._log.warning('statistics_callback_unregister but callback not registered.')
        if not len(self._statistics_callbacks):
            self._statistics_stop()

    def _on_stats(self, topic, value):
        period = 1 / 2e6
        s_start, s_stop = [x * period for x in value['time']['samples']['value']]

        if not len(self._statistics_offsets):
            duration = s_start
            charge = value['accumulators']['charge']['value']
            energy = value['accumulators']['energy']['value']
            offsets = [duration, charge, energy]
            self._statistics_offsets = [duration, charge, energy]
        duration, charge, energy = self._statistics_offsets
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
        for cbk in self._statistics_callbacks:
            cbk(value)

    def _statistics_start(self):
        if self.is_open:
            self.publish('s/stats/ctrl', 1)
            self.subscribe('s/stats/value', 'pub', self._on_stats_cbk)

    def _statistics_stop(self):
        if self.is_open:
            self.unsubscribe('s/stats/value', self._on_stats_cbk)
            self.publish('s/stats/ctrl', 0)

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
        """
        if name is not None:
            for p in PARAMETERS:
                if p.name == name:
                    return copy.deepcopy(p)
            return None
        return copy.deepcopy(PARAMETERS)

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
        if topic[0] != '/':
            topic = '/' + topic
        return self._path + topic

    def publish(self, topic, value, timeout=None):
        """Publish to the underlying joulescope_driver instance.

        :param topic: The publish topic.
        :param value: The publish value.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._driver.publish(self._topic_make(topic), value, timeout)

    def query(self, topic, timeout=None):
        """Query the underlying joulescope_driver instance.

        :param topic: The publish topic to query.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        :return: The value associated with topic.
        """
        return self._driver.query(self._topic_make(topic), timeout)

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
        return self._driver.subscribe(self._topic_make(topic), flags, fn, timeout)

    def unsubscribe(self, topic, fn, timeout=None):
        """Unsubscribe a callback to a topic.

        :param topic: Unsubscribe from this topic string.
        :param fn: The callback function to unsubscribe.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._driver.unsubscribe(self._topic_make(topic), fn, timeout)

    def unsubscribe_all(self, fn, timeout=None):
        """Unsubscribe a callback from all topics.

        :param fn: The callback function to unsubscribe.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        """
        return self._driver.unsubscribe(fn, timeout)

    def _config_apply(self, config=None):
        """Apply a configuration set by scan.

        :param config: The configuration string.
        """
        pass

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
        rc = self._driver.open(self._path, mode, timeout)
        self.is_open = True
        self.publish('h/fs', 2000000)
        while len(self._parameter_set_queue):
            name, value = self._parameter_set_queue.pop(0)
            self.parameter_set(name, value)
        if len(self._statistics_callbacks):
            self._statistics_start()
        device = 'js110' if 'js110' in self._path.lower() else 'js220'
        self.stream_buffer = StreamBuffer(self._buffer_duration,
                                          frequency=self._input_sampling_frequency,
                                          device=device,
                                          output_frequency=self._output_sampling_frequency)
        self._config_apply(self.config)
        return rc

    def close(self, timeout=None):
        """Close this device and release resources.

        :param timeout: The timeout in seconds.  None uses the default timeout.
        """
        if len(self._statistics_callbacks):
            self._statistics_stop()
        self.stop()
        self.is_open = False
        self.stream_buffer = None
        return self._driver.close(self._path, timeout)

    @property
    def model(self):
        return self._path.split('/')[1]

    @property
    def serial_number(self):
        return self._path.split('/')[-1]

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
        b = self.stream_buffer
        _, e1 = b.sample_id_range
        b.insert(topic, value)
        e0, e2 = b.sample_id_range

        if e1 == e2:
            return False
        if e0 == e2:
            return False
        rv = self._stream_process_call('stream_notify', self.stream_buffer)
        if rv:
            self.stop()
        if self.stream_buffer.is_duration_max or self.stream_buffer.is_contiguous_duration_max:
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
        self.stream_buffer.reset()
        self.stream_buffer.duration_max = duration
        self.stream_buffer.contiguous_duration_max = contiguous_duration
        self._stop_fn = stop_fn
        for topic, b in zip(self._stream_topics, self.stream_buffer.buffers.values()):
            if topic is None:
                b.active = False
                continue
            b.active = True
            self.subscribe(topic + '!data', 'pub', self._on_stream_cbk)
            self.publish(topic + 'ctrl', 1)
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
            for topic in self._stream_topics:
                if topic is not None:
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

        :return: A dict containing status information.
        """
        return {
            'driver': {
                'return_code': {
                    'value': 0,
                }
            }
        }

    def extio_status(self):
        """Read the EXTIO GPI value.

        :return: A dict containing the extio status.  Each key is the status
            item name.  The value is itself a dict with the following keys:

            * name: The status name, which is the same as the top-level key.
            * value: The actual value
            * units: The units, if applicable.
            * format: The recommended formatting string (optional).
        """
        return {}

    def __enter__(self):
        """Device context manager, automatically open."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Device context manager, automatically close."""
        self.close()
