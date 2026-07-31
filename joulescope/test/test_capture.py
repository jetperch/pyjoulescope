# Copyright 2026 Jetperch LLC
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

"""Test the capture entry point format dispatch."""

import unittest
from joulescope.entry_points.capture import run


class FakeV0Device:
    """A device without the v1 publish API."""


class TestCaptureDispatch(unittest.TestCase):

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            run(FakeV0Device(), 'out.jls', duration=0.1,
                out_format='bogus')

    def test_jls2_requires_v1(self):
        with self.assertRaises(ValueError):
            run(FakeV0Device(), 'out.jls', duration=0.1,
                out_format='jls2')


if __name__ == '__main__':
    unittest.main()
