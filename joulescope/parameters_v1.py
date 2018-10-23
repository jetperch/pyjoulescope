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
Define the JouleScope parameters available to the application.
"""

from .parameter import Parameter


# list of [param_name, permission, path, default, [(value_name, value), ...]]
# * permission is either 'r' (read-only) or 'rw' (read-write)
# * path is the USB (request, index) for the transaction
# * default is the name for the default value

PARAMETERS = [
    Parameter(
        'sensor_power',
        'rw',
        'setting',
        'on',
        [
            ('off', 0),
            ('on', 1),
        ]
    ),

    Parameter(
        'source',
        'rw',
        'setting',
        'off',
        [
            ('off',             0x00),
            ('raw',             0xC0),
            ('pattern_usb',     0x09),
            ('pattern_control', 0x0A),
            ('pattern_sensor',  0xAF),
        ],
    ),
    Parameter(
        'i_range',
        'rw',
        'setting',
        'off',
        [
            ('auto',   0x80),
            ('9 A',    0x01, ['0']),
            ('2 A',    0x02, ['1']),
            ('180 mA', 0x04, ['2']),
            ('18 mA',  0x08, ['3']),
            ('1.8 mA', 0x10, ['4']),
            ('180 µA', 0x20, ['5']),
            ('18 µA',  0x40, ['6']),
            ('off',    0x00),
        ],
    ),
    Parameter(
        'v_range',
        'rw',
        'setting',
        '13.2V',
        [
            ('13.2V', 0, ['low']),
            ('4V',    1, ['high']),
        ],
    ),
    Parameter(
        'ovr_to_lsb',
        'rw',
        'setting',
        'off',
        [
            ('off', 0),
            ('on', 1),
        ],
    ),
    Parameter(
        'control_test_mode',
        'rw',
        'setting',
        'normal',
        [
            ('normal', 0x03),
            ('usb',    0x81),
            ('fpga',   0x82),
            ('both',   0x83),  # also set 'source' to 'pattern_sensor'
        ],
    ),
    Parameter(
        'transfer_length',
        'rw',
        None,  # when stream is configured
        '256',
        [('%d' % (2**x), (2**x)) for x in range(0, 9)],
        'packets',
    ),
    Parameter(
        'transfer_outstanding',
        'rw',
        None,  # when stream is configured
        '8',
        [('%d' % (2**x), (2**x)) for x in range(0, 4)],
    ),
]


def _lookup_construct(x):
    fwd = {}
    rev = {}
    for p in x:
        d_fwd = {}
        d_rev = {}
        fwd[p.name] = d_fwd
        rev[p.name] = d_rev
        for value_name, value, aliases in p.values:
            d_fwd[value_name] = value
            d_rev[value] = value_name
            for alias in aliases:
                d_fwd[alias] = value
    return fwd, rev


PARAMETERS_DICT = dict((p.name, p) for p in PARAMETERS)

_TO_VALUE, _TO_NAME = _lookup_construct(PARAMETERS)


def name_to_value(param_name, value_name):
    return _TO_VALUE[param_name][value_name]


def value_to_name(param_name, value):
    return _TO_NAME[param_name][value]
