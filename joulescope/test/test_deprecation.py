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

"""Test the once-per-process deprecation warnings."""

import os
import unittest
import warnings
from unittest import mock
from joulescope import deprecation


def _library_function():
    deprecation.warn_once('key', 'deprecated')


class TestWarnOnce(unittest.TestCase):

    def setUp(self):
        # Treat only the deprecation module as library code, since this
        # test lives inside the joulescope package.
        patches = [
            mock.patch.object(deprecation, '_warned', set()),
            mock.patch.object(deprecation, '_PACKAGE_DIR', deprecation.__file__),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_once(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertTrue(deprecation.warn_once('key', 'deprecated'))
            self.assertFalse(deprecation.warn_once('key', 'deprecated'))
            self.assertTrue(deprecation.warn_once('other', 'deprecated'))
        self.assertEqual(2, len(w))
        self.assertIs(DeprecationWarning, w[0].category)
        self.assertEqual('deprecated', str(w[0].message))

    def test_caller_attribution(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _library_function()
        self.assertEqual(os.path.abspath(__file__), os.path.abspath(w[0].filename))
