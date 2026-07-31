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

"""Test the joulescope public API exports."""

import unittest
import joulescope


class TestPublicApi(unittest.TestCase):

    def test_star_import(self):
        exec('from joulescope import *', {})

    def test_all_entries_exist(self):
        for name in joulescope.__all__:
            self.assertTrue(hasattr(joulescope, name), f'missing {name}')

    def test_version(self):
        self.assertEqual(joulescope.VERSION, joulescope.__version__)
