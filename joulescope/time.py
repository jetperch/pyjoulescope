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
The Python time standard uses POSIX (UNIX) time which is defined relative
to Jan 1, 1970.  Python uses floating point seconds to express time.
This module defines a much simpler 64-bit fixed point integer for
representing time which is much friendlier for microcontrollers.
The value is 34Q30 with the upper 34 bits
to represent whole seconds and the lower 30 bits to represent fractional
seconds.  A value of 2**30 (1 << 30) represents 1 second.  This
representation gives a resolution of 2 ** -30 (approximately 1 nanosecond)
and a range of +/- 2 ** 33 (approximately 272 years).  The value is
signed to allow for simple arithmetic on the time either as a fixed value
or as deltas.

For more details on a compatible C implementation, see
https://github.com/mliberty1/embc/tree/master/include/embc
"""


import datetime
import time as pytime


EPOCH = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc).timestamp()


def timestamp_now():
    """Get the current timestamp.

    :return: The timestamp for the current time.
    """
    return seconds_to_timestamp(pytime.time())


def seconds_to_timestamp(time_s):
    """Convert python (POSIX) time in seconds to timestamp.

    :param time_s: The python time in float seconds.
    :return: The timestamp.
    """
    return int((time_s - EPOCH) * 2**30)


def timestamp_to_seconds(timestamp):
    """Convert timestamp to python (POSIX) time in seconds.

    :param timestamp: The timestamp.
    :return: The python time in float seconds.
    """
    return (timestamp / 2**30) + EPOCH
