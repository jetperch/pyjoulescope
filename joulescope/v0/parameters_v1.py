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


def _buffer_duration_validator(x):
    x = float(x)
    if x < 0:
        raise ValueError('too small')
    return int(x + 0.5001)

# list of [param_name, permission, path, default, [(value_name, value), ...]]
# * permission is either 'r' (read-only) or 'rw' (read-write)
# * path is the USB (request, index) for the transaction
# * default is the name for the default value

PARAMETERS = [
    Parameter(
        name='sensor_power',
        brief='Force the sensor power on or off.',
        detail='Altering this parameter during operation may cause errors.',
        path='setting',
        default='on',
        options=[('off', 0), ('on', 1)],
        flags=['developer'],
    ),

    Parameter(
        name='source',
        brief='Select the streaming data source.',
        path='setting',
        default='off',
        options=[
            ('off',             0x00),
            ('raw',             0xC0, ['on']),
            ('pattern_usb',     0x09),
            ('pattern_control', 0x0A),
            ('pattern_sensor',  0xAF),
        ],
        flags=['developer'],
    ),
    Parameter(
        name='i_range',
        brief='Select the current measurement range (shunt resistor)',
        path='setting',
        default='off',
        options=[
            ('auto',   0x80, ['on']),
            ('10 A',   0x01, ['0', 0]),
            ('2 A',    0x02, ['1', 1]),
            ('180 mA', 0x04, ['2', 2]),
            ('18 mA',  0x08, ['3', 3]),
            ('1.8 mA', 0x10, ['4', 4]),
            ('180 µA', 0x20, ['5', 5]),
            ('18 µA',  0x40, ['6', 6]),
            ('off',    0x00),
        ],
    ),
    Parameter(
        name='v_range',
        brief='Select the voltage measurement range (gain)',
        path='setting',
        default='15V',
        options=[
            ('15V', 0, ['low', 0]),
            ('5V',  1, ['high', 1]),
        ],
    ),
    Parameter(
        name='ovr_to_lsb',
        brief='Map overflow flags to the LSBs',
        path='setting',
        default='off',
        options=[('off', 0), ('on', 1)],
        flags=['developer'],
    ),
    Parameter(
        name='trigger_source',
        brief='Select the trigger source',
        path='extio',
        default='auto',
        options=[
            ('auto', 0),
            ('gpi0', 2),
            ('gpi1', 3),
        ],
        flags=['hidden'],  # feature not ready yet
    ),

    # --- EXTIO ---
    Parameter(
        name='io_voltage',
        brief='The GPI/O high-level voltage.',
        path='extio',
        default='3.3V',
        options=[
            ('1.8V', 1800),
            ('2.1V', 2100),
            ('2.5V', 2500),
            ('2.7V', 2700),
            ('3.0V', 3000),
            ('3.3V', 3300),
            ('3.6V', 3600),
            ('5.0V', 5000),
        ],
    ),
    Parameter(
        name='gpo0',
        brief='The GPO bit 0 output value.',
        path='extio',
        default='0',
        options=[
            ('0', 0, [0]),
            ('1', 1, [1]),
            # ('start_pulse', 2),   # reserved, but not yet implemented
            # ('sample_toggle', 3), # reserved, but not yet implemented
        ],
    ),
    Parameter(
        name='gpo1',
        brief='The GPO bit 1 output value.',
        path='extio',
        default='0',
        options=[
            ('0', 0, [0]),
            ('1', 1, [1]),
        ],
    ),
    Parameter(
        name='current_lsb',
        brief='The current signal least-significant bit mapping.',
        path='extio',
        default='normal',
        options=[
            ('normal', 0),
            ('gpi0', 2),
            ('gpi1', 3),
        ],
    ),
    Parameter(
        name='voltage_lsb',
        brief='The voltage signal least-significant bit mapping.',
        path='extio',
        default='normal',
        options=[
            ('normal', 0),
            ('gpi0', 2),
            ('gpi1', 3),
        ],
    ),

    Parameter(
        name='control_test_mode',
        brief='Set the test mode',
        path='setting',
        default='normal',
        options=[
            ('normal', 0x03),
            ('usb',    0x81),
            ('fpga',   0x82),
            ('both',   0x83),  # also set 'source' to 'pattern_sensor'
        ],
        flags=['developer'],
    ),
    Parameter(
        name='transfer_length',
        brief='Set the USB transfer length in packets',
        path='setting',
        default='256',
        options=[('%d' % (2**x), (2**x)) for x in range(0, 9)],
        units='packets',
        flags=['developer', 'skip_update'],
    ),
    Parameter(
        name='transfer_outstanding',
        brief='Set the maximum number of USB transfers issued simultaneously',
        path='setting',
        default='8',
        options=[('%d' % (2**x), (2**x)) for x in range(0, 4)],
        flags=['developer', 'skip_update']
    ),

    Parameter(
        name='current_ranging',  # virtual parameter!
        path='current_ranging',
        default=None,
        options=None,  # parse into current_ranging fields: type, samples_pre, samples_window, samples_post
        flags=['hidden'],
    ),
    Parameter(
        name='current_ranging_type',
        brief='The filter type.',
        path='current_ranging',
        default='interp',
        options=[
            ('off', 'off'),
            ('mean', 'mean'),
            ('interp', 'interp', ['interpolate']),
            ('NaN', 'nan', ['nan'])
        ],
    ),
    Parameter(
        name='current_ranging_samples_pre',
        brief='The number of samples before the range switch to include.',
        detail='Only valid for type "mean" - ignored for "off", "interp", and "NaN".',
        path='current_ranging',
        default='1',
        options=[(str(d), d, [d]) for d in range(9)]
    ),
    Parameter(
        name='current_ranging_samples_window',
        brief='The number of samples to adjust.',
        detail='Use "n" for automatic duration based upon known response time. ' +
               'Use "m" for shorter automatic duration that may result in min/max distortion.',
        path='current_ranging',
        default='n',
        options=[('m', 'm'), ('n', 'n')] + [(str(d), d, [d]) for d in range(13)]
    ),
    Parameter(
        name='current_ranging_samples_post',
        brief='The number of samples after the range switch to include.',
        detail='Only valid for type "mean" - ignored for "off", "interp", and "NaN".',
        path='current_ranging',
        default='1',
        options=[(str(d), d, [d]) for d in range(9)]
    ),

    Parameter(
        name='buffer_duration',
        brief='The amount of sample data to store in memory.',
        detail='Use care when setting this value. ' +
               'The software requires 1.5 GB of RAM for every 60 seconds at ' +
               'the full 2 MSPS rate.',
        path='setting',
        default='30 seconds',
        options=[
            ('15 seconds', 15, [15]),
            ('30 seconds', 30, [30]),
            ('1 minute', 60, [60]),
            ('2 minutes', 120, [120]),
            ('5 minutes', 300, [300]),
            ('10 minutes', 600, [600]),
            ('20 minutes', 60*20, [60*20]),
            ('1 hour', 3600, [3600]),
            ('2 hours', 2*3600, [2*3600]),
            ('5 hours', 5*3600, [5*3600]),
            ('10 hours', 10*3600, [10*3600]),
            ('1 day', 24 * 3600, [24*3600])],
        units='seconds',
        validator=_buffer_duration_validator,
        flags=['skip_update']
    ),
    Parameter(
        name='reduction_frequency',
        brief='The rate that the device produces statistics, including multimeter values.',
        path='setting',
        default='2 Hz',
        options=[
            ('100 Hz', 100, [100]),
            ('50 Hz', 50, [50]),
            ('20 Hz', 20, [20]),
            ('10 Hz', 10, [10]),
            ('5 Hz', 5, [5]),
            ('2 Hz', 2, [2]),
            ('1 Hz', 1, [1])],
        units='Hz',
        flags=['skip_update']
    ),
    Parameter(
        name='sampling_frequency',
        brief='The rate that the device produces samples.',
        path='setting',
        default='2 MHz',
        options=[
            ('2 MHz', 2000000, [2000000, 'auto', None, 'default']),
            ('1 MHz', 1000000, [1000000]),
            ('500 kHz', 500000, [500000]),
            ('200 kHz', 200000, [200000]),
            ('100 kHz', 100000, [100000]),
            ('50 kHz', 50000, [50000]),
            ('20 kHz', 20000, [20000]),
            ('10 kHz', 10000, [10000]),
            ('5 kHz', 5000, [5000]),
            ('2 kHz', 2000, [2000]),
            ('1 kHz', 1000, [1000]),
            ('500 Hz', 500, [500]),
            ('200 Hz', 200, [200]),
            ('100 Hz', 100, [100]),
            ('50 Hz', 50, [50]),
            ('20 Hz', 20, [20]),
            ('10 Hz', 10, [10])],
        units='Hz',
        flags=['skip_update']
    ),

    Parameter(name='model', path='info', flags=['read_only', 'hidden']),
    Parameter(name='device_serial_number', path='info', flags=['read_only', 'hidden']),
    Parameter(name='hardware_serial_number', path='info', flags=['read_only', 'hidden']),
]


def _lookup_construct(x):
    fwd = {}
    rev = {}
    for p in x:
        d_fwd = {}
        d_rev = {}
        fwd[p.name] = d_fwd
        rev[p.name] = d_rev
        for value_name, value, aliases in p.options:
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


PARAMETERS_DEFAULTS = {
    'auto': {
        'i_range': 'auto',
        'v_range': '15V',
        'source': 'on',
    },
    'off': {
        'source': 'off',
    },
    'ignore': {},
}
