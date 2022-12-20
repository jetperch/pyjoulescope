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
Utility functions for managing spans.
"""

import numpy as np
import math
import logging


log = logging.getLogger(__name__)


class Span:
    """Constrain a span.

    :param limits: The (min, max) limits for the span.
    """

    def __init__(self, limits, quant, length):
        self.limits = None
        self.quant = float(quant)
        self._length = 0
        self.limits = self._round_span(limits, 1)
        self.length = int(length)
        log.info(self)

    def __str__(self):
        return 'Span(%r, %r, %r)' % (self.limits, self.quant, self._length)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        value = int(value)
        if value <= 0:
            value = 0
        elif self.quant * (value - 1) > self.s_limit_max:
            length_max = int(self.s_limit_max / self.quant) + 1
            log.info('length %d too big.  Set to maximum length %d.',
                     value, length_max)
            value = length_max
        self._length = value

    @property
    def s_limit_max(self):
        return self.limits[1] - self.limits[0]

    @property
    def a_limit_min(self):
        return self.quant * (self.length - 1)

    def _round_span(self, span, quants_per):
        return [self.quantize_round(span[0], quants_per, 1),
                self.quantize_round(span[1], quants_per, -1)]

    def _slide_or_truncate(self, span, quants_per=None):
        quants_per = 1 if quants_per is None else int(quants_per)
        s = self._round_span(span, quants_per)
        limits = self._round_span(self.limits, quants_per)
        if s[0] < limits[0]:  # shift up if hits lower bound
            s[1] += limits[0] - s[0]
            s[0] = limits[0]
        if s[1] > limits[1]:  # shift down if hits upper bound
            s[0] -= s[1] - limits[1]
            s[1] = limits[1]
            if s[0] < limits[0]:  # but in no case bigger than upper
                s[0] = limits[0]
        return s

    def quantize_round(self, value, quants_per, direction=None):
        direction = 0 if direction is None else float(direction)
        step = float(self.quant * quants_per)
        value = float(value) / step
        if 0 == direction:
            value = round(value)
        elif direction > 0:
            value = math.ceil(value)
        else:
            value = math.floor(value)
        return value * step

    def _bound_steps_per(self, steps_per):
        steps_per = int(round(steps_per))
        if steps_per < 1:
            steps_per = 1
        if steps_per > 1 and ((self.length - 1) * self.quant * steps_per) > self.s_limit_max:
            steps_per = int(self.s_limit_max / ((self.length - 1) * self.quant))
            if steps_per < 1:
                steps_per = 1
        return steps_per

    def _quants_per(self, span):
        """Compute the number of quantization steps per sample in a given span"""
        t = span[1] - span[0]
        return int(round((t / (self.length - 1)) / self.quant))

    def conform(self, span):
        """Update a span range to conform to rules.

        :param span: The (min, max) span to conform.
        :return: span adjusted to be within limits and with a step
            size no smaller than the minimum quantization step.
        """
        s = list(span)
        t = s[1] - s[0]
        s_min = self.quant * (self.length - 1)
        if t < s_min:
            s[1] += s_min - t
        return self._slide_or_truncate(s)

    def quants_per(self, span):
        a = self._quants_per(span)
        return self._bound_steps_per(a)

    def scale(self, span, pivot=None, gain=None):
        if pivot is None:
            pivot = (span[1] + span[0]) / 2
        if gain is None:
            gain = 1.0
        z1 = pivot + (span[0] - pivot) * gain
        z2 = z1 + (span[1] - span[0]) * gain
        return [z1, z2]

    def conform_quant_per(self, span, incr=None, gain=None, pivot=None):
        """Conform quantized.

        :param span: The span to quantize.
        :param incr: Force steps_per to change by at least
            this amount (but never less than 1).  span_prev must not
            be None when incr is not None.
            None (default) behaves normally.
        :param gain: Gain fraction applied to increase/decrease the range.
            When not None, automatically compute incr.
            None (default) behaves normally.
        :param pivot: The pivot for the scale operation.
        :return: (span, steps_per) where span is the (min, max) range of the
            span and steps_per are the number of quantization steps between
            each sample.
        """
        if self.length <= 1:
            return span

        s_prev = [float(span[0]), float(span[1])]
        if gain is not None:
            if gain > 1:
                incr = 1
            elif gain < 1:
                incr = -1
            gain = float(gain)
        if pivot is None:
            pivot = (s_prev[1] + s_prev[0]) / 2  # center
        else:
            pivot = float(pivot)
        window_sz = s_prev[1] - s_prev[0]
        if window_sz > 0:
            pivot_fract = (pivot - s_prev[0]) / (s_prev[1] - s_prev[0])
        else:
            pivot_fract = 0.5
        s = self.scale(s_prev, pivot, gain)

        # adjust start base upon quantization
        steps_per = self.quants_per(s)
        if incr is not None:
            steps_per_prev = self._quants_per(s_prev)
            if incr < 0 and steps_per > (steps_per_prev - incr):
                steps_per = steps_per_prev - incr
            elif incr > 0 and steps_per < (steps_per_prev + incr):
                steps_per = steps_per_prev + incr
            steps_per = self._bound_steps_per(steps_per)
        s = self._slide_or_truncate(s, steps_per)

        # right justify
        # s[0] = s[1] - (steps_per * self.quant * (self.length - 1))
        # left justify
        # s[1] = s[0] + (steps_per * self.quant * (self.length - 1))
        # center
        pivot_quant = self.quantize_round(pivot, steps_per, direction=-1)
        start = pivot_quant - (steps_per * self.quant * round(pivot_fract * self.length - 1))
        s[0] = self.quantize_round(start, steps_per)
        s[1] = s[0] + (steps_per * self.quant * (self.length - 1))

        s = self._slide_or_truncate(s)

        return s, steps_per

    def conform_discrete(self, span, incr=None, gain=None, pivot=None):
        """Update a span to conform to rules.

        :param span: The (min, max) span to conform.
        :param incr: Force steps_per to change by at least
            this amount (but never less than 1).  span_prev must not
            be None when incr is not None.
            None (default) behaves normally.
        :param gain: Gain fraction applied to increase/decrease the range.
            When not None, automatically compute incr.
            None (default) behaves normally.
        :param pivot: The pivot point to maintain for the scale operation.
        :return: (span, steps_per, axis).  Span is the (min, max) adjusted to be
            within limits and no smaller than quant per step.  The steps_per
            are the number of quantized samples per resulting samples.  The axis
            is the np.ndarray of quantized values for use as a plotting axis.
        """
        log.debug('conform_discrete(span=%r, incr=%r, gain=%r, pivot=%r)',
                  span, incr, gain, pivot)
        steps_per = 1
        s = self._slide_or_truncate(span)

        # compute the axis value
        if self.length == 0:
            a = np.array([])
        elif self.length == 1:
            a = np.array([s[0]])
        else:
            s, steps_per = self.conform_quant_per(s, incr, gain, pivot)
            a = np.arange(self.length, dtype=float)
            a *= self.quant * steps_per
            a += s[0]
            log.debug('conform_discrete: span=%s, steps_per=%s', s, steps_per)
        return s, steps_per, a

