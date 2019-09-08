# Copyright 2018-2019 Jetperch LLC
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

from joulescope import span
import numpy as np
import logging


class View:

    def __init__(self, device):
        self._device = device  # joulescope.Device which must be opened
        self.x = None
        self.data = None  # NxMx4 np.float32 [length][current, voltage, power][mean, var, min, max]
        x_max = len(self._device.stream_buffer) / self.sampling_frequency
        self.x_range = [x_max - 1.0, x_max]  # the current range
        self.samples_per = 1
        self.data_idx = 0
        self._span = span.Span(limits=[0.0, x_max],
                               quant=1.0 / self.sampling_frequency,
                               length=100)
        self.changed = True
        self.log = logging.getLogger(__name__)

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    @property
    def sampling_frequency(self):
        return self._device.sampling_frequency

    @property
    def calibration(self):
        return self._device.calibration

    @property
    def limits(self):
        """Get the (x_min, x_max) limits for the view."""
        return self._span.limits

    def clear(self):
        self.changed = True
        self.data_idx = 0
        if self.data is not None:
            self.data[:, :, :] = np.nan

    def on_x_change(self, cmd, kwargs):
        x_range = list(self.x_range)
        if cmd == 'resize':  # {pixels: int}
            length = kwargs['pixels']
            if length is not None and length != len(self):
                self.log.info('resize %s', length)
                self._span.length = length
                self.data = np.full((length, 3, 4), np.nan, dtype=np.float32)
                self.changed = True  # invalidate
            x_range, self.samples_per, self.x = self._span.conform_discrete(x_range)
        elif cmd == 'span_absolute':  # {range: (start: float, stop: float)}]
            x_range, self.samples_per, self.x = self._span.conform_discrete(kwargs.get('range'))
        elif cmd == 'span_relative':  # {center: float, gain: float}]
            x_range, self.samples_per, self.x = self._span.conform_discrete(
                x_range, gain=kwargs.get('gain'), pivot=kwargs.get('pivot'))
        elif cmd == 'span_pan':
            delta = kwargs.get('delta', 0.0)
            x_range = [x_range[0] + delta, x_range[-1] + delta]
            x_range, self.samples_per, self.x = self._span.conform_discrete(x_range)
        elif cmd == 'refresh':
            self.log.warning('on_x_change(refresh)')
            self.changed = True
            return
        else:
            self.log.warning('on_x_change(%s) unsupported', cmd)
            return

        if self._device.is_streaming:
            x_max = self._span.limits[1]
            if x_range[1] < x_max:
                x_shift = x_max - x_range[1]
                x_range = [x_range[0] + x_shift, x_max]
            x_range, self.samples_per, self.x = self._span.conform_discrete(x_range)

        self.changed |= (self.x_range != x_range)
        self.clear()
        self.x_range = x_range
        self.log.info('changed=%s, length=%s, span=%s, range=%s, samples_per=%s',
                 self.changed, len(self), self.x_range,
                 self.x_range[1] - self.x_range[0], self.samples_per)

    def _view(self):
        buffer = self._device.stream_buffer
        _, sample_id_end = buffer.sample_id_range
        lag_time = self._span.limits[1] - self.x_range[1]
        lag_samples = int(lag_time * self.sampling_frequency) // self.samples_per
        data_idx_stream_end = sample_id_end // self.samples_per
        data_idx_view_end = data_idx_stream_end - lag_samples
        sample_id_end = data_idx_view_end * self.samples_per
        delta = data_idx_view_end - self.data_idx
        return data_idx_view_end, sample_id_end, delta

    def time_to_sample_id(self, t):
        idx_start, idx_end = self._device.stream_buffer.sample_id_range
        t_start, t_end = self._span.limits
        if not t_start <= t <= t_end:
            return None
        dx_end = t_end - t
        dx_idx_end = int(dx_end * self.sampling_frequency)
        s = idx_end - dx_idx_end
        return s

    def update(self):
        buffer = self._device.stream_buffer
        length = len(self)
        data_idx_view_end, sample_id_end, delta = self._view()

        if self.data is None:
            return False, (None, None)
        elif not self.changed and 0 == delta:
            return False, (self.x, self.data)
        elif self.changed or delta >= length:  # perform full recompute
            self.data[:, :, :] = np.nan
            if data_idx_view_end > 0:
                start_idx = (data_idx_view_end - length) * self.samples_per
                # self.log.debug('recompute(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
                buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data)
        elif data_idx_view_end > 0:
            start_idx = self.data_idx * self.samples_per
            # self.log.debug('update(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
            self.data = np.roll(self.data, -delta, axis=0)
            buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data[-delta:, :, :])
        else:
            self.data[:, :, :] = np.nan
        self.data_idx = data_idx_view_end
        self.changed = False
        return True, (self.x, self.data)

    def extract(self):
        buffer = self._device.stream_buffer
        length = len(self)
        data_idx_view_end, sample_id_end, delta = self._view()
        start_idx = (data_idx_view_end - length) * self.samples_per
        return buffer.data_get(start_idx, sample_id_end)

    def raw_get(self, start=None, stop=None):
        return self._device.stream_buffer.raw_get(start=start, stop=stop)

    def samples_get(self, start=None, stop=None):
        data = self._device.stream_buffer.data_get(start=start, stop=stop)
        return {
            'signals': {
                'current': {
                    'value': data[:, 0, 0],
                    'units': 'A',
                },
                'voltage': {
                    'value': data[:, 1, 0],
                    'units': 'V',
                },
                'raw': {
                    'value': self._device.stream_buffer.raw_get(start=start, stop=stop),
                    'units': 'LSBs',
                },
            },
        }
