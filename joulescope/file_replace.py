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


import os
import random
import shutil


class FileReplace:
    """Safely replace an existing file.

    :param filename: The full path and file name.
    :param mode: The mode for open().
    :param encoding: The encoding for open().
    """

    def __init__(self, filename, mode=None, encoding=None):
        self._filename = os.path.abspath(filename)
        unique = '_%d_%d' % (os.getpid(), random.randint(0, 9999))
        name, ext = os.path.splitext(self._filename)
        self._mode = mode
        self._encoding = encoding

        # use filenames in same directory.  NamedTemporaryFile may be on different volume.
        self._filename_new = '%s_new_%s%s' % (name, unique, ext)
        self._filename_bak = '%s_bak_%s%s' % (name, unique, ext)
        self._f = None

    def open(self):
        if self._f is not None:
            self.close()
        if os.path.isfile(self._filename):
            shutil.copy(self._filename, self._filename_bak)
        self._f = open(self._filename_new, mode=self._mode, encoding=self._encoding)
        return self._f

    def replace(self, source):
        if os.path.isfile(self._filename):
            os.replace(source, self._filename)
        else:
            os.rename(source, self._filename)

    def close(self):
        try:
            if self._f:
                self._f.close()
                self._f = None
                if os.path.isfile(self._filename):
                    os.replace(self._filename_new, self._filename)
                else:
                    os.rename(self._filename_new, self._filename)
        finally:
            self._cleanup()

    def _cleanup(self):
        if os.path.isfile(self._filename_new):
            os.unlink(self._filename_new)
        if os.path.isfile(self._filename_bak):
            os.unlink(self._filename_bak)

    def revert(self):
        if os.path.isfile(self._filename):
            os.replace(self._filename_bak, self._filename)
        else:
            os.rename(self._filename_bak, self._filename)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self._f:
                self._f.close()
                self._f = None
                # file should still be intact
        else:
            self.close()
