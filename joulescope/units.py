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
Manage units and unit display.

For full handling of units, see the excellent quantities module
http://pythonhosted.org/quantities/.
"""

import re


# https://www.regular-expressions.info/floatingpoint.html
#RE_IS_NUMBER = re.compile(r'^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+))')
RE_IS_NUMBER = re.compile(r'^\s*([-+]?[0-9]*\.?[0-9]+)\s*(.*)')


FIELD_UNITS = {
    'current': 'A',
    'charge': 'C',
    'voltage': 'V',
    'power': 'W',
    'energy': 'J',
    'current_range': '',
    'current_lsb': '',
    'voltage_lsb': '',
    'raw': '',
    'raw_current': 'LSBs',
    'raw_voltage': 'LSBs',
}


_UNIT_PREFIX = [
    (1e24, 'Y'),
    (1e21, 'Z'),
    (1e18, 'E'),
    (1e15, 'P'),
    (1e12, 'T'),
    (1e9, 'G'),
    (1e6, 'M'),
    (1e3, 'k'),
    (1e0,  ''),
    (1e-3, 'm'),
    (1e-6, 'µ'),
    (1e-9, 'n'),
    (1e-12, 'p'),
    (1e-15, 'f'),
    (1e-18, 'a'),
    (1e-21, 'z'),
    (1e-24, 'y'),
]

_PREFIX_MAP = dict([(p, v) for v, p in _UNIT_PREFIX])
_PREFIX_MAP['u'] = 1e-6  # common "misuse" of µ


def unit_prefix(value):
    """Get the unit prefix and adjust value.

    :param value: The value to convert to a power of 1000.
    :return: The tuple of (adjusted_value, prefix, scale)
        where value = scale * adjust_value
        Scale is always a power of 10 (10**k).
    """
    v = abs(value)
    for k, c in _UNIT_PREFIX:
        if v >= k:
            return value / k, c, k
    return 0.0, '', 1  # close enough to zero


def three_sig_figs(x, units=None, space1=None, space2=None):
    """Get the value x displayed to three significant figures.

    :param x: The value to convert to a string.
    :param units: The units for x.
    :param space1: The space to insert between the number and prefix.
        If None and units are provided, insert a single space, ' '.
        If None and units are not provide, insert nothing, ''.
    :param space2: The space to insert between the prefix and units.
        If None, insert nothing, ''.
    :return: The string formatted as 'value units'
    """
    units = '' if units is None else units
    if space1 is None:
        if len(units):
            space1 = ' '
        else:
            space1 = ''
    space2 = '' if space2 is None else space2
    x, prefix, _ = unit_prefix(x)
    z = abs(x)
    if z >= 100:
        s = '%.0f' % z
    elif z >= 10:
        s = '%.1f' % z
    elif z >= 1:
        s = '%.2f' % z
    else:
        s = '%.3f' % z
    if x < 0:
        s = '-' + s
    return '%s%s%s%s%s' % (s, space1, prefix, space2, units)


def str_to_number(s):
    if s is None:
        return s
    if not isinstance(s, str):
        float(s)
        return s
    match = RE_IS_NUMBER.match(s)
    if not match:
        raise ValueError(f'not a number: {s}')
    number = match.group(1)
    units = match.group(2)
    if '.' in number:
        number = float(number)
    else:
        number = int(number)
    if units.startswith('ppm'):
        pass
    elif len(units):
        v = _PREFIX_MAP.get(units[0])
        if v is not None:
            if v >= 1:
                v = int(v)
            number *= v
    return number


def duration_to_seconds(d):
    """Convert a duration to float seconds.

    :param d: The duration specification, which is one of:
        * A string formatted as fz where f is a valid floating-point value
          and z is either omitted, 's', 'm', 'h', 'd'.
        * An integer in seconds.
        * A floating-point value in seconds.
    """
    if d is None:
        raise ValueError('cannot convert None')
    if isinstance(d, str):
        if not len(d):
            raise ValueError('cannot convert empty string')
        if d[-1] == 's':
            return float(d[:-1])
        elif d[-1] == 'm':
            return 60 * float(d[:-1])
        elif d[-1] == 'h':
            return 60 * 60 * float(d[:-1])
        elif d[-1] == 'd':
            return 60 * 60 * 24 * float(d[:-1])
        else:
            return float(d)
    else:
        return float(d)
