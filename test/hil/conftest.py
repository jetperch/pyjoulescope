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

"""Hardware-in-the-loop test fixtures for the joulescope package.

These tests require attached Joulescope hardware.  Tests parametrized by
model skip automatically for models that are not connected.  Run with:

    python -m pytest test/hil -v
"""

import pytest
import joulescope


MODELS = ['js110', 'js220', 'js320']

_devices = None


def _scan():
    global _devices
    if _devices is None:
        _devices = {}
        for d in joulescope.scan(config='auto'):
            _devices[d.model] = d
    return _devices


@pytest.fixture(scope='session', params=MODELS)
def model(request):
    if request.param not in _scan():
        pytest.skip(f'{request.param} not connected')
    return request.param


@pytest.fixture()
def device(model):
    """The opened device for the model; streaming stopped on teardown."""
    d = _scan()[model]
    d.open()
    try:
        yield d
    finally:
        try:
            d.stop()
        finally:
            d.close()


@pytest.fixture()
def device_closed(model):
    """The device instance, guaranteed closed."""
    d = _scan()[model]
    if d.is_open:
        d.close()
    yield d
    if d.is_open:
        d.close()
