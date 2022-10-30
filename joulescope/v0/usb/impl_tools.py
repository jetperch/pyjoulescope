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

"""Common tools used by :class:`joulescope.usb.api.DeviceDriverApi` implementations."""


import time


class RunUntilDone:

    def __init__(self, timeout, name=''):
        self._timeout = timeout
        self._name = name
        self._value = None
        self._time_start = time.time()

    def __str__(self):
        return 'RunUntilDone(timeout=%r, name=%r)' % (self._timeout, self._name)

    @property
    def value(self):
        return self._value

    @property
    def value_args0(self):
        return self._value[0][0]

    def cbk_fn(self, *args, **kwargs):
        self._value = (args, kwargs)

    def is_done(self):
        if self._value is not None:
            return True
        time_delta = time.time() - self._time_start
        if time_delta > self._timeout:
            raise TimeoutError('RunUntilDone %s: timeout %s > %s' % (self._name, time_delta, self._timeout))
        return False
