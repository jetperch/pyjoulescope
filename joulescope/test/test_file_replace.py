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
import tempfile
import os
import sys
from joulescope.file_replace import FileReplace


class TestFileReplace(unittest.TestCase):

    def setUp(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self._path = f.name

    def tearDown(self) -> None:
        if os.path.isfile(self._path):
            os.remove(self._path)

    def confirm(self, contents):
        with open(self._path, 'rt', encoding='utf-8') as f:
            d = f.read()
        self.assertEqual(contents, d)

    def test_does_not_exist(self):
        os.remove(self._path)
        with FileReplace(self._path, mode='wt') as f:
            f.write('hello')
        self.confirm('hello')

    def test_file_exists(self):
        with FileReplace(self._path, mode='wt') as f:
            f.write('hello')
        self.confirm('hello')

    def test_file_delete_during_replace(self):
        with FileReplace(self._path, mode='wt') as f:
            f.write('hello')
            os.remove(self._path)
        self.confirm('hello')

    def test_exception_during_replace(self):
        with open(self._path, 'wt', encoding='utf-8') as f:
            f.write('original')
        with self.assertRaises(RuntimeError):
            with FileReplace(self._path, mode='wt') as f:
                f.write('hello')
                raise RuntimeError('doh')
        self.confirm('original')

    def test_file_open(self):
        if sys.platform.startswith('win'):
            # windows has file locking by default
            with self.assertRaises(PermissionError):
                with open(self._path, 'wt', encoding='utf-8') as f1:
                    f1.write('original')
                    with FileReplace(self._path, mode='wt') as f2:
                        f2.write('hello')
            self.confirm('original')

