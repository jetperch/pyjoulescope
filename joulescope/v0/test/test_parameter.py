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
Test parameter
"""

import unittest
from joulescope.parameter import Parameter


class TestParameter(unittest.TestCase):

    def test_initialize_keywords_simple(self):
        options = [
            ('there', 0),
            ('world', 1),
            ('you', 2)
        ]
        p = Parameter(name='hello', path='universe', default='world', options=options)
        self.assertEqual('hello', p.name)
        self.assertEqual('universe', p.path)
        self.assertEqual(0, p.str_to_value['there'])
        self.assertEqual(1, p.str_to_value['world'])
        self.assertEqual(2, p.str_to_value['you'])
        with self.assertRaises(KeyError):
            p.str_to_value['__invalid__']
        with self.assertRaises(KeyError):
            p.str_to_value[2]  # does not currently map the value
