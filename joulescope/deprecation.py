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

"""Issue each deprecation warning once per process."""

import os
import threading
import warnings


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_warned = set()
_lock = threading.Lock()


def warn_once(key, message):
    """Issue a DeprecationWarning on the first call for key.

    :param key: The unique deprecation identifier.
    :param message: The warning message.
    :return: True if issued, False if already issued.

    The warning names the first caller outside the joulescope package,
    so Python shows it by default when that caller is a script.
    """
    with _lock:
        if key in _warned:
            return False
        _warned.add(key)
    warnings.warn(message, DeprecationWarning, skip_file_prefixes=(_PACKAGE_DIR,))
    return True
