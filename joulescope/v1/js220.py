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


from .device import Device
from .stream_buffer import StreamBuffer
from joulescope.view import View
from joulescope.parameters_v1 import PARAMETERS, PARAMETERS_DICT, name_to_value
import copy
import logging


_log = logging.getLogger(__name__)
_I_RANGE_LOOKUP = {
    0x01: '10 A',
    0x02: '10 A',
    0x04: '180 mA',
    0x08: '18 mA',
    0x10: '1.8 mA',
    0x20: '180 µA',
    0x40: '18 µA',
}
_SAMPLING_FREQUENCIES = [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000, 1_000_000,
]
_STREAM_TOPICS = ['s/i/', 's/v/']  #, 's/i/range/', 's/gpi/0/', 's/gpi/1/']  # todo p when not full rate


def _version_u32_to_str(v):
    v = int(v)
    major = (v >> 24) & 0xff
    minor = (v >> 16) & 0xff
    patch = v & 0xffff
    return f'{major}.{minor}.{patch}'


class DeviceJs220(Device):

    def __init__(self, driver, device_path):
        super().__init__(driver, device_path)
        self._param_map = {
            'i_range': self._on_i_range,
            'v_range': self._on_v_range,
            'buffer_duration': self._on_buffer_duration,
            'reduction_frequency': self._on_reduction_frequency,
            'sampling_frequency': self._on_sampling_frequency,
        }
        self._stream_cbk_objs = []
        self._stream_cbk_objs_add = []
        self._stop_fn = None
        self._input_sampling_frequency = 1000000
        self._output_sampling_frequency = 1000000
        self._buffer_duration = 30
        self._parameter_set_queue = []
        self._statistics_callbacks = []
        self._on_stats_cbk = self._on_stats  # hold reference for unsub
        self._on_stream_cbk = self._on_stream  # hold reference for unsub
        self._statistics_offsets = []
        self.stream_buffer = None
        self._parameters = {}
        self._is_streaming = False
        for p in PARAMETERS:
            if p.default is not None:
                self._parameters[p.name] = name_to_value(p.name, p.default)

    @property
    def input_sampling_frequency(self):
        """The original input sampling frequency."""
        return self._input_sampling_frequency

    @property
    def output_sampling_frequency(self):
        """The output sampling frequency."""
        return self._output_sampling_frequency

    @property
    def sampling_frequency(self):
        """The output sampling frequency."""
        return self._output_sampling_frequency

    @property
    def statistics_callback(self):
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
        for cbk in list(self._statistics_callbacks):
            self.statistics_callback_unregister(cbk)
        self.statistics_callback_register(cbk)

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
        if not len(self._statistics_callbacks):
            self._statistics_start()
        self._statistics_callbacks.append(cbk)

    def statistics_callback_unregister(self, cbk, source=None):
        """Unregister a statistics callback.

        :param cbk: The callback previously provided to
            :meth:`statistics_callback_register`.
        :param source: The callback source.
        """
        self._statistics_callbacks.remove(cbk)
        if not len(self._statistics_callbacks):
            self._statistics_stop()

    def statistics_accumulators_clear(self):
        """Clear the charge and energy accumulators."""
        self._statistics_offsets.clear()

    @property
    def calibration(self):
        return None

    def view_factory(self):
        """Construct a new View into the device's data.

        :return: A View-compatible instance.
        """
        view = View(self.stream_buffer, self.calibration)
        view.on_close = lambda: self.stream_process_unregister(view)
        self.stream_process_register(view)
        return view

    def parameters(self, name=None):
        if name is not None:
            for p in PARAMETERS:
                if p.name == name:
                    return copy.deepcopy(p)
            return None
        return copy.deepcopy(PARAMETERS)

    def parameter_set(self, name, value):
        p = PARAMETERS_DICT[name]
        if name == 'current_ranging':
            self._current_ranging_split(value)
            return
        if 'read_only' in p.flags:
            _log.warning('Attempting to set read_only parameter %s', name)
            return
        try:
            value = name_to_value(name, value)
        except KeyError:
            if p.validator is None:
                raise KeyError(f'value {value} not allowed for parameter {name}')
            else:
                value = p.validator(value)
        self._parameters[name] = value
        k = self._param_map.get(name)
        if k is not None:
            if not self.is_open:
                self._parameter_set_queue.append((name, value))
                return
            k(value)

    def _on_i_range(self, value):
        if value == 0x80:
            self.publish('s/i/range/mode', 'auto')
        elif value == 0:
            self.publish('s/i/range/mode', 'off')
        else:
            value = _I_RANGE_LOOKUP[value]
            self.publish('s/i/range/select', value)
            self.publish('s/i/range/mode', 'manual')

    def _on_v_range(self, value):
        if value == 'auto':
            self.publish('s/v/range/mode', 'auto')
        else:
            # JS110 current ranges are 5 and 15 V, cannot map to 2 V
            self.publish('s/v/range/select', '15 V')
            self.publish('s/v/range/mode', 'manual')

    def _on_buffer_duration(self, value):
        self._buffer_duration = value

    def _on_reduction_frequency(self, value):
        pass  # todo

    def _on_sampling_frequency(self, value):
        value = min(value, 1000000)
        if value not in _SAMPLING_FREQUENCIES:
            raise ValueError(f'invalid sampling frequency {value}')
        self._output_sampling_frequency = value

    def _stream_process_call(self, method, *args, **kwargs):
        rv = False
        b, self._stream_cbk_objs, self._stream_cbk_objs_add = self._stream_cbk_objs + self._stream_cbk_objs_add, [], []
        for obj in b:
            fn = getattr(obj, method, None)
            if not callable(fn):
                continue
            if obj.driver_active:
                try:
                    rv |= bool(fn(*args, **kwargs))
                    self._stream_cbk_objs.append(obj)
                except Exception:
                    _log.exception('%s %s() exception', obj, method)
                    obj.driver_active = False
            if not obj.driver_active:
                try:
                    if hasattr(obj, 'close'):
                        obj.close()
                except Exception:
                    _log.exception('%s close() exception', obj)
        return rv

    def _on_stream(self, topic, value):
        b = self.stream_buffer
        _, e1 = b.sample_id_range
        b.insert(topic, value)
        _, e2 = b.sample_id_range

        if e1 == e2:
            return False
        rv = self._stream_process_call('stream_notify', self.stream_buffer)
        if rv:
            self.stop()

    def start(self, stop_fn=None, duration=None, contiguous_duration=None):
        self.stop()
        self.stream_buffer.reset()
        self._stop_fn = stop_fn
        for topic in _STREAM_TOPICS:
            self.subscribe(topic + '!data', 'pub', self._on_stream_cbk)
            self.publish(topic + 'ctrl', 1)
        self._is_streaming = True
        self._stream_process_call('start', self.stream_buffer)

    def stop(self):
        if self._is_streaming:
            self._is_streaming = False
            for topic in _STREAM_TOPICS:
                self.unsubscribe(topic + '!data', self._on_stream_cbk)
                self.publish(topic + 'ctrl', 0)
            fn, self._stop_fn = self._stop_fn, None
            if callable(fn):
                fn(0, '')  # status, message
            self._stream_process_call('stop')

    def read(self, duration=None, contiguous_duration=None, out_format=None, fields=None):
        pass  # todo

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

    def info(self):
        info = {
            'type': 'info',
            'ver': 2,
            'model': self.model,
            'serial_number': self.serial_number,
            'ctl': {
                'hw': {
                    'rev': _version_u32_to_str(self.query('c/hw/version')),
                    'sn_mcu': self.serial_number,
                    'sn_mfg': self.serial_number,
                    'ver': _version_u32_to_str(self.query('c/hw/version')),
                },
                'fw': {
                    'ver': _version_u32_to_str(self.query('c/fw/version')),
                }
            },
            'sensor': {
                'fw': {
                    'ver': _version_u32_to_str(self.query('s/fpga/version')),
                },
                'fpga': {
                    'ver': _version_u32_to_str(self.query('s/fpga/version')),
                },
            },
        }
        return info

    def status(self):
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

    def extio_status(self):
        """Read the EXTIO GPI value.

        :return: A dict containing the extio status.  Each key is the status
            item name.  The value is itself a dict with the following keys:

            * name: The status name, which is the same as the top-level key.
            * value: The actual value
            * units: The units, if applicable.
            * format: The recommended formatting string (optional).
        """
        return {}  # todo

    def open(self, event_callback_fn=None, mode=None, timeout=None):
        super().open(mode, timeout)
        while len(self._parameter_set_queue):
            name, value = self._parameter_set_queue.pop(0)
            self.parameter_set(name, value)
        if len(self._statistics_callbacks):
            self._statistics_start()
        self.stream_buffer = StreamBuffer(self._buffer_duration)

    def close(self, timeout=None):
        if len(self._statistics_callbacks):
            self._statistics_stop()
        self.stop()
        super().close(timeout)
