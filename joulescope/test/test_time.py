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


import unittest
from joulescope import time


class TestTimestamp(unittest.TestCase):

    def test_epoch(self):
        self.assertEqual(1514764800.0, time.EPOCH)

    def test_seconds_to_timestamp(self):
        self.assertEqual(0, time.seconds_to_timestamp(time.EPOCH))

    def test_timestamp_to_seconds(self):
        self.assertEqual(time.EPOCH, time.timestamp_to_seconds(0))

    def test_conversion(self):
        now = time.timestamp_now()
        s = time.timestamp_to_seconds(now)
        t = time.seconds_to_timestamp(s)
        self.assertEqual(now, t)
