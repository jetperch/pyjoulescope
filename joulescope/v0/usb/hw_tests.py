# Copyright 2017 Jetperch LLC
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
Perform functional validation testing on real hardware.
"""

import logging
log = logging.getLogger(__name__)


def control_loopback_wvalue(device, request_id, increment=None):
    """Test writing/reading from a single 16-bit location.

    :param device: The USB device instance.
    :param request_id: The request identifier.
    :param increment: The increment value.  The test runs from 0 to 2**16 - 1.

    The device firmware must implement the request.  On control OUT, the
    firmware must store the 16-bit value.  On a control IN, the firmware
    must return the 16-bit value.
    """
    request_id = int(request_id)
    increment = int(increment) if increment is not None else 1
    for value in range(0, 2 ** 16 - 1, increment):
        if value & 0x3ff == 0:
            log.info('test_control_short %.1f%%', value * 100 / 2 ** 16)
        rv = device.control_transfer_out('device', 'vendor',
                                         request=request_id, value=value, index=0)
        if rv.result != 0:
            raise RuntimeError('test_control_short out %d returned %d' % (value, rv.result))
        rv = device.control_transfer_in('device', 'vendor',
                                        request=request_id, value=value, index=0, length=2)
        if rv.result != 0:
            raise RuntimeError('test_control_short in %d returned %d' % (value, rv.result))
        actual = rv.data[0] + rv.data[1] * 256
        if value != actual:
            raise RuntimeError('test_control_short value mismatch: expected %d, received %d' %
                               (value, actual))


def control_loopback_buffer(device, request_id, max_length, iterations=None):
    """Test writing/reading from a memory buffer 16-bit location.

    :param device: The USB device instance.
    :param request_id: The request identifier.
    :param max_length: The maximum length for each data transfer.  The
        firmware must have a dedicated buffer for at least this many bytes.
    :param iterations: The total number of iterations.

    The device firmware must implement the request.  On control OUT, the
    firmware must store the data to the buffer.  On a control IN, the firmware
    must return the buffer in the data phase.
    """
    request_id = int(request_id)
    iterations = int(iterations) if iterations is not None else 256
    max_length = int(max_length)
    for i in range(iterations):
        if i & 0x3 == 0:
            log.info('test_control_long %.1f%%', i * 100 / iterations)
        for length in range(1, max_length):
            data = bytes(range(length))
            rv = device.control_transfer_out('device', 'vendor',
                                             request=request_id, value=0, index=0, data=data)
            if rv.result != 0:
                raise RuntimeError('test_control_data out %d.%d returned %d' % (i, length, rv.result))
            rv = device.control_transfer_in('device', 'vendor',
                                            request=request_id, value=0, index=0, length=length)
            if rv.result != 0:
                raise RuntimeError('test_control_data in %d.%d returned %d' % (i, length, rv.result))
            if rv.data != data:
                raise RuntimeError('ctest_control_data in %d.%d \n    expected = %s\n    received = %s' %
                                   (i, length, data, rv.data))
