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
from joulescope.parameters_v1 import PARAMETERS_DICT, name_to_value


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


def _signal_bool(value):
    if isinstance(value, str):
        value = value.lower()
    if value in [0, False, '0', 'off', 'disable']:
        return False
    elif value in [1, True, '1', 'on', 'enable']:
        return True
    else:
        raise ValueError('invalid signal level.')


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
            'gpo0': self._on_gpo0,
            'gpo1': self._on_gpo1,
        }
        self._input_sampling_frequency = 1000000
        self._output_sampling_frequency = 1000000
        self._parameters['sampling_frequency'] = self._output_sampling_frequency
        self._stream_topics = ['s/i/', 's/v/', 's/p/', 's/i/range/', 's/gpi/0/', 's/gpi/1/']

    def parameter_set(self, name, value):
        p = PARAMETERS_DICT[name]
        if 'read_only' in p.flags:
            self._log.warning('Attempting to set read_only parameter %s', name)
            return
        try:
            if name == 'v_range' and value in ['2V', '2 V', '2', 2]:
                value = '2 V'
            else:
                value = name_to_value(name, value)
        except KeyError:
            if p.validator is None:
                raise KeyError(f'value {value} not allowed for parameter {name}')
            else:
                value = p.validator(value)
        self._parameters[name] = value
        if not self.is_open:
            self._parameter_set_queue.append((name, value))
            return
        k = self._param_map.get(name)
        if k is not None:
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
            if value in ['2V', '2 V']:
                value = '2 V'
            elif value in ['15V', '15 V', 0]:
                value = '15 V'
            self.publish('s/v/range/select', value)
            self.publish('s/v/range/mode', 'manual')

    def _on_buffer_duration(self, value):
        self.buffer_duration = value

    def _on_reduction_frequency(self, value):
        scnt = int(1_000_000 / value)
        self.publish('s/stats/scnt', scnt)

    def _on_sampling_frequency(self, value):
        value = min(value, 1000000)
        if value not in _SAMPLING_FREQUENCIES:
            raise ValueError(f'invalid sampling frequency {value}')
        self._output_sampling_frequency = value

    def _on_gpo(self, index, value):
        index = int(index)
        if _signal_bool(value):
            topic = 's/gpo/+/!set'
        else:
            topic = 's/gpo/+/!clr'
        value = 1 << index
        self.publish(topic, value)

    def _on_gpo0(self, value):
        self._on_gpo(0, value)

    def _on_gpo1(self, value):
        self._on_gpo(1, value)

    def _config_apply(self, config=None):
        if config is None or config.lower() == 'auto':
            self.publish('s/i/range/mode', 'auto')
            self.publish('s/v/range/select', '15 V')
            self.publish('s/v/range/mode', 'manual')
        elif config == 'ignore':
            pass  # do nothing
        elif config == 'off':
            for topic in self._stream_topics:
                self.publish(topic + 'ctrl', 0)
        else:
            self._log.warning('Unsupported config %s', config)

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

