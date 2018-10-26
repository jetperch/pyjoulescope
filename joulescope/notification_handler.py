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

import logging
import time

log = logging.getLogger(__name__)


class NotificationHandler:

    def __init__(self, period=None):
        self.period = period
        self._time_last = 0.0
        self._notify = []

    def emit(self, obj=None):
        if self.period is not None:
            time_now = time.time()
            time_delta = time_now - self._time_last
            if time_delta < self.period:
                return
            self._time_last = time_now
        self.emit_always(obj)

    def emit_always(self, obj=None):
        for fn in self._notify:
            try:
                fn(obj)
            except Exception:
                log.exception('while notify %s' % fn.__name__)

    def register(self, fn):
        """Attach a function to be notified of changes

        :param fn: The callback function which is called when the
            data buffer content changes.
        """
        if fn not in self._notify:
            self._notify.append(fn)
        else:
            log.info('NotificationHandler.register already has %s' % fn.__name__)

    def unregister(self, fn):
        if fn in self._notify:
            self._notify.remove(fn)
        else:
            log.info('NotificationHandler.unregister does not contain %s' % fn.__name__)
