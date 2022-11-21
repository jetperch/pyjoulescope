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
from joulescope.parameters_v1 import PARAMETERS_DICT, name_to_value, PARAMETERS_DEFAULTS
import logging


_log = logging.getLogger(__name__)


class DeviceJs110(Device):

    def __init__(self, driver, device_path):
        super().__init__(driver, device_path)
        self._param_map = {
            'sensor_power': None,
            'source': None,
            'i_range': 's/i/range/select',
            'v_range': 's/v/range/select',
            'ovr_to_lsb': None,
            'trigger_source': None,
            'io_voltage': 's/extio/voltage',
            'gpo0': 's/gpo/0/value',
            'gpo1': 's/gpo/1/value',
            'current_lsb': 's/i/lsb_src',
            'voltage_lsb': 's/v/lsb_src',
            'current_ranging': None,  # parse
            'current_ranging_type': 's/i/range/mode',
            'current_ranging_samples_pre': 's/i/range/pre',
            'current_ranging_samples_window': self._on_current_ranging_samples_window,
            'current_ranging_samples_post': 's/i/range/post',
            'buffer_duration': self._on_buffer_duration,
            'reduction_frequency': self._on_reduction_frequency,
            'sampling_frequency': self._on_sampling_frequency,
        }
        self._input_sampling_frequency = 2000000
        self._output_sampling_frequency = 2000000
        self._stream_topics = ['s/i/', 's/v/', 's/p/', 's/i/range/', 's/gpi/0/', 's/gpi/1/']

    def _on_current_ranging_samples_window(self, value):
        if value in ['m', 'n']:
            self.publish('s/i/range/win', value)
        else:
            self.publish('s/i/range/win_sz', int(value))
            self.publish('s/i/range/win', 'manual')

    def _current_ranging_split(self, value):
        if value is None or value in [False, 'off']:
            self.parameter_set('current_ranging_type', 'off')
            return
        parts = value.split('_')
        if len(parts) != 4:
            raise ValueError(f'Invalid current_ranging value {value}')
        for p, v in zip(['type', 'samples_pre', 'samples_window', 'samples_post'], parts):
            self.parameter_set('current_ranging_' + p, v)

    def _on_buffer_duration(self, value):
        self.buffer_duration = value

    def _on_reduction_frequency(self, value):
        scnt = int(2_000_000 / value)
        self.publish('s/stats/scnt', scnt)

    def _on_sampling_frequency(self, value):
        value = int(value)
        self.publish('h/fs', int(value))
        self.output_sampling_frequency = value

    def _config_apply(self, config=None):
        for key, value in PARAMETERS_DEFAULTS.get(config, {}).items():
            self.parameter_set(key, value)

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
        if not self.is_open:
            self._parameter_set_queue.append((name, value))
            return
        k = self._param_map.get(name)
        if isinstance(k, str):
            _log.info(f'parameter_set({name}, {value}) -> {k}')
            self.publish(k, value)
        elif k is not None:
            _log.info(f'parameter_set({name}, {value})')
            k(value)

    def info(self):
        return {
            'type': 'info',
            'ver': 2,
            'model': self.model,
            'hardware_version': 'H',
            'serial_number': self.serial_number,
        }

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
        return {}   # todo
