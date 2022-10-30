# Copyright 2019 Jetperch LLC
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

import unittest
from joulescope.v0.filter_fir import FilterFir
import numpy as np
import time


filter_ma3 = [1.0/3, 1.0/3, 1.0/3]
data_x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data_y1 = [0, 1.0/3, 1, 2, 3, 4, 5, 6, 7, 8, 9]
data_y2 = [1.0/3, 2, 4, 6, 8]


class TestFilterFir(unittest.TestCase):

    def setUp(self):
        self.y = []

    def cbk(self, y):
        self.y.append(y)

    def test_no_downsample(self):
        f = FilterFir([{'M': 1, 'taps': filter_ma3}])
        f.callback_set(self.cbk)
        for x in data_x1:
            f.process(x)
        np.testing.assert_allclose(data_y1, self.y)

    def test_downsample_by_2(self):
        f = FilterFir([{'M': 2, 'taps': filter_ma3}])
        f.callback_set(self.cbk)
        for x in data_x1:
            f.process(x)
        np.testing.assert_allclose(data_y2, self.y)

    def test_downsample_by_2_width_2(self):
        f = FilterFir([{'M': 2, 'taps': filter_ma3}], width=2)
        f.callback_set(self.cbk)
        x = np.zeros(2, dtype=np.float64)
        for idx in range(len(data_x1) - 1):
            x[0] = data_x1[idx]
            x[1] = data_x1[idx + 1]
            f.process(x)
        expect = np.array(data_y1[1:]).reshape((5, 2))
        np.testing.assert_allclose(expect, self.y)

    def test_performance(self):
        taps = np.ones(504)
        taps /= np.sum(taps)
        width = 1
        f = FilterFir([{'M': 10, 'taps': taps}], width=width)
        x = np.random.randn(2000000, width)
        time_start = time.time()
        f.process(x)
        f.process(x)
        f.process(x)
        time_stop = time.time()
        print(f'duration = {time_stop - time_start}')
