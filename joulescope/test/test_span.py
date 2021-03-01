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
Test spans
"""

import unittest
from joulescope.span import Span
import numpy as np


class TestSpan(unittest.TestCase):

    def test_trivial(self):
        s = Span([0.0, 20.0], 1.0, 11)
        self.assertEqual(20.0, s.s_limit_max)
        self.assertEqual(10.0, s.a_limit_min)
        self.assertEqual([1.0, 11.0], s.conform([1.0, 11.0]))
        self.assertEqual(1.0, s.quants_per([1.0, 11.0]))
        self.assertEqual(([2.0, 12.0], 1.0), s.conform_quant_per([1.0, 11.0]))
        span, steps_per, axis = s.conform_discrete([1.0, 11.0])
        self.assertEqual([2.0, 12.0], span)
        self.assertEqual(1.0, steps_per)
        self.assertEqual([1.0, 11.0], s.scale([1.0, 11.0]))

    def test_conform_discrete_range_trim(self):
        s = Span([10.0, 30.0], 1.0, 11)
        sc, spans_per, x = s.conform_discrete([0.0, 40.0])
        np.testing.assert_allclose([10.0, 30.0], sc)
        self.assertEqual(2, spans_per)
        np.testing.assert_allclose(np.arange(10, 31, 2, dtype=np.float64), x)

    def test_conform_discrete_min_gain(self):
        s = Span([10.0, 30.0], 1.0, 11)
        sc, spans_per, x = s.conform_discrete([19.5, 20.5])
        np.testing.assert_allclose([16, 26], sc)
        self.assertEqual(1, spans_per)
        np.testing.assert_allclose(np.arange(16, 27, dtype=np.float64), x)

    def test_conform_discrete_small(self):
        s = Span([10.0, 30.0], 1.0, 11)
        sc, spans_per, x = s.conform_discrete([19.0, 21.0])
        np.testing.assert_allclose([16.0, 26.0], sc)

    def test_conform_discrete_pivot(self):
        s = Span([10.0, 30.0], 0.001, 1000)
        span = [27.000, 27.999]
        pivot = 27.1
        sc, spans_per, x = s.conform_discrete([27.000, 27.999])
        np.testing.assert_allclose(span, sc)
        sc, spans_per, x = s.conform_discrete(span=span, gain=0.5, pivot=pivot)
        k0 = (pivot - span[0]) / (span[1] - span[0])
        k1 = (pivot - sc[0]) / (sc[1] - sc[0])
        np.testing.assert_allclose(k1, k0, rtol=0.015)

    def test_maximum_length(self):
        s = Span([10.0, 20.0], 0.01, 1000)
        self.assertEqual(1000, s.length)
        s.length = 101
        self.assertEqual(101, s.length)
        s.length = 10000
        self.assertEqual(1001, s.length)
