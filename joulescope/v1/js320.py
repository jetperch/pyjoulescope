# Copyright 2022-2026 Jetperch LLC
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


from .js220 import DeviceJs220, sampling_frequency_parameter


_SAMPLING_FREQUENCIES = [
    # The JS320 streams i, v, p at 1 Msps with on-instrument downsampling
    # by N for N in {1, 4, ..., 1000} and host-side downsampling below
    # 1 kHz.  List the h/fs values that are also valid
    # 'sampling_frequency' parameter options.  500 kHz is not supported
    # (the gateware clamps downsampling factors 2 and 3 to 4).
    10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 1_000_000,
]


class DeviceJs320(DeviceJs220):

    def __init__(self, driver, device_path):
        super().__init__(driver, device_path)
        # The JS320 samples at 16 Msps and always downsamples on-instrument,
        # by 16 to 1 Msps by default.  Stream messages report
        # sample_rate=16000000 with decimate_factor >= 16.
        self._input_sampling_frequency = 16000000
        self._output_sampling_frequency = 1000000
        self._h_fs = 1000000  # h/fs is the i, v, p rate: 1 Msps max
        self._parameters['sampling_frequency'] = self._output_sampling_frequency
        self._parameters_override['sampling_frequency'] = \
            sampling_frequency_parameter(_SAMPLING_FREQUENCIES, 1_000_000)

    def _on_sampling_frequency(self, value):
        value = min(int(value), 1000000)
        if value not in _SAMPLING_FREQUENCIES:
            raise ValueError(f'invalid sampling frequency {value}')
        self._log.info('_on_sampling_frequency %s', value)
        self.publish('h/fs', value)
        self._output_sampling_frequency_set(value)
        self._parameters['sampling_frequency'] = value
