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

import binascii
from joulescope.v0 import public_keys
from joulescope import datafile
from joulescope.time import seconds_to_timestamp
import io
import json
import dateutil.parser
import numpy as np
import logging


log = logging.getLogger(__name__)
PRODUCT = '1.1.1'


def _stuffed(x):
    while len(x) < 9:
        x.append(x[-1])
    x[7] = 0.0
    x[8] = float('nan')
    return np.array(x, dtype=np.float32)


def _version_str_to_u32(s: str):
    major, minor, patch = [int(x) for x in s.split('.')]
    return (((major & 0xff) << 24) |
            ((minor & 0xff) << 16) |
            (patch & 0xffff))


def _np_list(x):
    return np.array(x, dtype=np.float32).tolist()


def raw_split(data):
    """Split raw data into current, voltage, i_range and missing_sample_count.

    :param data: The (N, 2) or (N x 2, ) np.ndarray with dtype np.uint16.
        Either way, the current and voltage samples are interleaved as
        they arrive from Joulescope over USB.
        The least significant 2 bits as indicators.
    :return: (current, voltage, i_range, missing_sample_count) where missing
        samples are detected by the per-sample bit toggle.  Current and Voltage
        are N np.ndarray(dtype=np.uint16) with 14-bit right-shifted values.
        i_range is N np.ndarray(dtype=np.uint8) with 3-bit right shifted values.
        missing_sample_count is a scalar integer.
    """
    if len(data.shape) == 2:
        data = data.reshape((np.prod(data.shape), ))

    # get i_range
    data_i_range_low = np.bitwise_and(data[::2], 0x0003)
    data_i_range_high = np.bitwise_and(data[1::2], 0x0001)
    data_i_range_high = np.left_shift(data_i_range_high, 2, out=data_i_range_high)
    i_range = np.bitwise_or(data_i_range_low, data_i_range_high, out=data_i_range_low)
    i_range = i_range.astype(np.uint8)

    # every sample also contains a toggling sample indicator
    # to check proper packet construction.
    # This check will not detect samples lost at packet boundaries,
    # but this is good enough to detect system problems.  The hardware
    # ensures that individual samples should not be lost, just packets.
    toggle = np.bitwise_and(data[1::2], 0x0002)
    toggle = np.right_shift(toggle, 1, out=toggle)
    missing_sample_count = len(toggle) - 1 - \
        np.count_nonzero(np.bitwise_xor(toggle[:-1], toggle[1:]))

    # get 14-bit right justified current and voltage data
    data = np.right_shift(data, 2)
    i_raw = data[::2]
    v_raw = data[1::2]
    return i_raw, v_raw, i_range, missing_sample_count


class Calibration:
    """Manage Joulescope calibration data."""

    def __init__(self):
        self.version = '1.0.0'
        self.product_id = 1
        self.vendor_id = 1
        self.subtype_id = 1
        self.time = dateutil.parser.parse('2018-01-01T00:00:00.00000Z')
        self.serial_number = '00000000000000000000000000000000'
        self.current_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan], dtype=np.float32)
        self.current_gain = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, np.nan], dtype=np.float32)
        self.voltage_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.voltage_gain = np.array([1.0, 1.0], dtype=np.float32)
        self.signed = False
        self.data = None  # store load verbatim for future direct write

    def json(self):
        return {
            'version': self.version,
            'time': self.time.isoformat(),
            'timestamp': str(seconds_to_timestamp(self.time.timestamp())),
            'product_name': 'Joulescope JS110',
            'vendor_name': 'Jetperch LLC',
            'subtype_name': 'Calibration',
            'product': f'{self.product_id}.{self.vendor_id}.{self.subtype_id}',
            'serial_number': self.serial_number,
            'voltage': {
                'offset': _np_list(self.voltage_offset),
                'gain': _np_list(self.voltage_gain),
            },
            'current': {
                'offset': _np_list(self.current_offset),
                'gain': _np_list(self.current_gain),
            },
        }

    def save(self, private_key=None):
        """Save calibration to bytes.

        :param private_key: The private key used to sign the calibration.
            None (default) does not sign the calibration.
            The Joulescope software will display warnings if the calibration
            is not signed using a valid key.
        :return: The calibration as bytes.
        """
        if isinstance(self.time, str):
            self.time = dateutil.parser.parse(self.time)
        fh = io.BytesIO()
        dfw = datafile.DataFileWriter(fh)
        if private_key is not None:
            dfw.signature_start(private_key)
        dfw.append_header(
            timestamp=seconds_to_timestamp(self.time.timestamp()),
            version=_version_str_to_u32(self.version),
            product_id=self.product_id,
            vendor_id=self.vendor_id,
            subtype_id=self.subtype_id,
            hardware_compatibility=0,
            serial_number=binascii.unhexlify(self.serial_number),
        )
        self.current_offset = np.concatenate((self.current_offset[:7], [0.0, np.nan]))
        self.current_gain = np.concatenate((self.current_gain[:7], [0.0, np.nan]))
        if len(self.current_offset) != 9 or len(self.current_gain) != 9:
            raise ValueError('Invalid length for current')
        if len(self.voltage_offset) != 2 or len(self.voltage_gain) != 2:
            raise ValueError('Invalid length for voltage')
        cal = self.json()
        dfw.append(datafile.TAG_DATA_JSON, json.dumps(cal, allow_nan=True).encode('utf-8'))
        if private_key is not None:
            dfw.signature_end()
        dfw.finalize()
        return bytes(fh.getbuffer())

    def load(self, data, keys=None):
        """Load calibration from bytes.

        :param data: The bytes containing the calibration.
        :param keys: The list of allowed public keys.  None (default)
            uses the official signing key.
        :return: This instance.
        :raise ValueError: on invalid data.
        """
        self.data = data
        fh = io.BytesIO(data)
        r = datafile.DataFileReader(fh)
        self.signed = True
        tag, value = next(r)
        public_key = value[8:]
        if keys is None:
            keys = [public_keys.CALIBRATION_SIGNING, public_keys.RECALIBRATION_01_SIGNING]
        if datafile.TAG_SIGNATURE_START != tag:
            self.signed = False
            log.warning('Invalid format: missing signature start tag')
        else:
            if public_key not in keys:
                self.signed = False
                log.warning('Invalid signing key')
            tag, header = next(r)
        if datafile.TAG_HEADER != tag:
            raise ValueError('Invalid format: missing header tag')
        tag, cal = next(r)
        if tag not in [datafile.TAG_DATA_JSON, datafile.TAG_CALIBRATION_JSON]:
            raise ValueError('Invalid format: missing calibration tag')
        if self.signed:
            tag, _ = next(r)
            if datafile.TAG_SIGNATURE_END != tag:
                self.signed = False
                log.warning('Invalid format: missing signature end tag')
        cal = json.loads(cal.decode('utf-8'))
        if cal['product'] != PRODUCT:
            raise ValueError('Invalid calibration')
        self.version = cal['version']
        self.time = dateutil.parser.parse(cal['time'])
        self.serial_number = cal['serial_number']
        self.current_offset = _stuffed(cal['current']['offset'])
        self.current_gain = _stuffed(cal['current']['gain'])
        self.voltage_offset = np.array(cal['voltage']['offset'], dtype=np.float32)
        self.voltage_gain = np.array(cal['voltage']['gain'], dtype=np.float32)
        return self

    def transform(self, data, v_range=None):
        """Apply the calibration to transform raw data.

        :param data: The (N, 2) or (N x 2, ) np.ndarray with dtype np.uint16.
            Either way, the current and voltage samples are interleaved as
            they arrive from Joulescope over USB.
            The least significant 2 bits as indicators.
        :param v_range: The selected voltage ranged for the conversion.
        :return: (current, voltage, missing_sample_count) where missing samples
            is detected by the per-sample bit toggle.
        """
        v_range = 0 if v_range is None else int(v_range)
        if len(data.shape) == 2:
            data = data.reshape((np.prod(data.shape), ))

        i, v, i_range, missing_sample_count = raw_split(data)
        idx = np.logical_and(np.logical_and(i_range == 7, i == 0x3fff), v == 0x3fff)
        i_range[idx] += 1

        # get 14-bit right justified current and voltage data
        i = i.astype(np.float32)
        i += self.current_offset[i_range]
        i *= self.current_gain[i_range]
        v = v.astype(np.float32)
        v += self.voltage_offset[v_range]
        v *= self.voltage_gain[v_range]
        v[np.isnan(i)] = np.nan

        return i, v, missing_sample_count
