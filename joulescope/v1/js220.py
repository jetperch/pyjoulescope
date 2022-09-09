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


PARAM_IGNORE = ['sensor_power', 'source', 'io_voltage',
                'current_ranging_type', 'current_ranging_samples_pre',
                'current_ranging_samples_window', 'current_ranging_samples_post',
                'reduction_frequency',
                ]


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
            'i_range': self._i_range,
            'v_range': self._v_range,
            # 'gpo0':
            # 'gpi1':
            # current_lsb,enables gpi0,✔
            # voltage_lsb,enables gpi1,✔
            'buffer_duration': self._buffer_duration,
            # reduction_frequency,✔,✔
            # sampling_frequency,✔,✔'
        }
        self._parameter_set_queue = []
        self._statistics_callbacks = []
        self._on_stats_cbk = self._on_stats  # hold reference for unsub
        self._statistics_offsets = []

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

    def view_factory(self):
        """Construct a new View into the device's data.

        :return: A View-compatible instance.
        """
        pass  # todo

    def parameter_set(self, name, value):
        if name in PARAM_IGNORE:
            return
        k = self._param_map[name]
        if not self.is_open:
            self._parameter_set_queue.append((name, value))
            return
        if callable(k):
            k(value)
        else:
            self.publish(k, value)

    def _i_range(self, value):
        if value == 'auto':
            self.publish('s/i/range/mode', value)
        else:
            self.publish('s/i/range/select', value)
            self.publish('s/i/range/mode', 'manual')

    def _v_range(self, value):
        if value == 'auto':
            self.publish('s/v/range/mode', 'auto')
        else:
            self.publish('s/v/range/select', '15 V')
            self.publish('s/v/range/mode', 'manual')

    def _buffer_duration(self, value):
        pass  # todo

    def start(self, stop_fn=None, duration=None, contiguous_duration=None):
        pass  # todo

    def stop(self):
        pass  # todo

    def read(self, duration=None, contiguous_duration=None, out_format=None, fields=None):
        pass  # todo

    @property
    def is_streaming(self):
        """Check if the device is streaming.

        :return: True if streaming.  False if not streaming.
        """
        return self._streaming

    def stream_process_register(self, obj):
        """Register a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi`.
            The instance must remain valid until its :meth:`close` is
            called.

        Call :meth:`stream_process_unregister` to disconnect the instance.
        """
        pass

    def stream_process_unregister(self, obj):
        """Unregister a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi` that was
            previously registered using :meth:`stream_process_register`.
        """
        pass

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
        return {}

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

    def open(self, mode=None, timeout=None):
        super().open(mode, timeout)
        while len(self._parameter_set_queue):
            name, value = self._parameter_set_queue.pop(0)
            self.parameter_set(name, value)
        if len(self._statistics_callbacks):
            self._statistics_start()

    def close(self, timeout=None):
        if len(self._statistics_callbacks):
            self._statistics_stop()
        super().close(timeout)
