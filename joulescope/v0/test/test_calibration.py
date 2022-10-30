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
Test the circular plot buffer with data reduction
"""

import unittest
from joulescope.v0.calibration import Calibration
import monocypher
import numpy as np
import os

MYPATH = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(MYPATH, 'calibration_01.dat'), 'rb') as f:
    CAL1 = f.read()

CAL1_VK = [-1150.5926667302124, -2041.675208990041]
CAL1_VG = [0.0009468113102263853, 0.00029342128123136175]
CAL1_IK = [-2806.717811538373, -2822.6645675137756, -2820.88100055503, -2822.6200621623125,
           -2820.462616626318, -2820.5826040862453, -2818.676888822763, 0.0, float('nan')]
CAL1_1G = [0.0006828914450133605, 0.00016375303534373303, 1.4626174421742075e-05, 1.4937616926812478e-06,
           1.4970493483159016e-07, 1.498388811827842e-08, 1.4984542304875239e-09, 0.0, float('nan')]


PRIVATE_KEY = bytes(range(32))
PUBLIC_KEY = monocypher.compute_signing_public_key(PRIVATE_KEY)


class TestCalibrate(unittest.TestCase):

    def test_save_load_unsigned(self):
        c1 = Calibration()
        c1.current_offset = CAL1_IK[:7]
        c1.current_gain = CAL1_1G[:7]
        c1.voltage_offset = CAL1_VK
        c1.voltage_gain = CAL1_VG
        data = c1.save()
        c2 = Calibration().load(data, keys=[])
        self.assertFalse(c2.signed)
        np.testing.assert_allclose(CAL1_VK, c2.voltage_offset)
        np.testing.assert_allclose(CAL1_VG, c2.voltage_gain)
        np.testing.assert_allclose(CAL1_IK, c2.current_offset)
        np.testing.assert_allclose(CAL1_1G, c2.current_gain)

    def test_save_load_signed(self):
        c1 = Calibration()
        c1.current_offset = CAL1_IK[:7]
        c1.current_gain = CAL1_1G[:7]
        c1.voltage_offset = CAL1_VK
        c1.voltage_gain = CAL1_VG
        data = c1.save(PRIVATE_KEY)
        # with open(os.path.join(MYPATH, 'calibration_01.dat'), 'wb') as f:
        #     f.write(data)
        c2 = Calibration().load(data, keys=[PUBLIC_KEY])
        self.assertTrue(c2.signed)
        np.testing.assert_allclose(CAL1_VK, c2.voltage_offset)
        np.testing.assert_allclose(CAL1_VG, c2.voltage_gain)
        np.testing.assert_allclose(CAL1_IK, c2.current_offset)
        np.testing.assert_allclose(CAL1_1G, c2.current_gain)

    def test_cal1(self):
        c = Calibration().load(CAL1, keys=[PUBLIC_KEY])
        self.assertTrue(c.signed)
        np.testing.assert_allclose(CAL1_VK, c.voltage_offset)
        np.testing.assert_allclose(CAL1_VG, c.voltage_gain)
        np.testing.assert_allclose(CAL1_IK, c.current_offset)
        np.testing.assert_allclose(CAL1_1G, c.current_gain)

    def test_load_save_load(self):
        c1 = Calibration().load(CAL1, keys=[])
        self.assertFalse(c1.signed)
        data = c1.save(private_key=PRIVATE_KEY)
        c2 = Calibration().load(data, keys=[PUBLIC_KEY])
        np.testing.assert_allclose(CAL1_VK, c2.voltage_offset)
        np.testing.assert_allclose(CAL1_VG, c2.voltage_gain)
        np.testing.assert_allclose(CAL1_IK, c2.current_offset)
        np.testing.assert_allclose(CAL1_1G, c2.current_gain)
