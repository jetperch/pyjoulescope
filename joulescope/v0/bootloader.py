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

from joulescope import datafile
import io
import struct
import binascii
import logging
log = logging.getLogger(__name__)

DeviceInterfaceGUID = '{76ab0534-eeda-48b5-b0c9-617fc6ce8296}'
VENDOR_ID = 0x1FC9
PRODUCT_ID = 0xFC94
HOST_API_VERSION = 1
INFO_REQUEST_LENGTH = 256
CHUNK_SIZE = 0x1000


def _ioerror_on_bad_result(rv):
    if 0 != rv.result:
        raise IOError('usb.Device %s' % (rv,))


class UsbdRequest:
    LOOPBACK_WVALUE = 1
    LOOPBACK_BUFFER = 2
    INFO = 3
    """Get the detailed device information"""

    WRITE_START = 4
    WRITE_CHUNK = 5
    WRITE_FINALIZE = 6
    READ_CHUNK = 7
    GO = 8
    ERASE = 9


class Segment:
    PERSONALITY = 1
    FIRMWARE = 2
    STORAGE1 = 3
    CALIBRATION_ACTIVE = 4
    CALIBRATION_FACTORY = 5


SEGMENTS = {
    'calibration_factory': Segment.CALIBRATION_FACTORY,
    'firmware': Segment.FIRMWARE,
    'storage1': Segment.STORAGE1,
    'personality': Segment.PERSONALITY,
    'calibration_active': Segment.CALIBRATION_ACTIVE,
}
for segments_value in list(SEGMENTS.values()):
    SEGMENTS[segments_value] = segments_value


class Bootloader:
    """The bootloader API and implementation for use by applications.

    :param usb_device: The backend USB :class:`usb.device` instance.
    """
    def __init__(self, usb_device):
        self._usb = usb_device
        self._is_open = False

    def __str__(self):
        return self._usb.serial_number

    @property
    def usb_device(self):
        """Get the USB backend device implementation.

        This method should only be used for unit and system tests.  Production
        code should *NEVER* access the underlying USB device directly.
        """
        return self._usb

    def open(self):
        """Open the device for use

        :raise IOError: on failure.
        """
        if self._is_open:
            self.close()
        self._is_open = True
        self._usb.open()

    def close(self):
        """Close the device and release resources"""
        self._is_open = False
        try:
            self._usb.close()
        except Exception:
            log.exception('USB close failed')

    def comm_test(self):
        rv = self._usb.control_transfer_out(
            None, 'device', 'vendor',
            request=UsbdRequest.LOOPBACK_WVALUE,
            value=0x1234, index=0)
        _ioerror_on_bad_result(rv)
        rv = self._usb.control_transfer_in(
            None, 'device', 'vendor',
            request=UsbdRequest.LOOPBACK_WVALUE,
            value=0, index=0, length=2)
        _ioerror_on_bad_result(rv)
        value = struct.unpack('<H', bytes(rv.data))[0]
        print(value)

    def info_get(self):
        rv = self._usb.control_transfer_in(
            None, 'device', 'vendor',
            request=UsbdRequest.INFO,
            value=0, index=0, length=INFO_REQUEST_LENGTH)
        _ioerror_on_bad_result(rv)
        values = struct.unpack('<IIIII', bytes(rv.data))
        names = ['bootloader_version', 'hardware_id', 'hardware_version', 'firmware_version', 'status']
        data_str = ['%s=0x%08x' % (name, value) for name, value in zip(names, values)]
        log.info('\n    '.join(['info_get:'] + data_str))
        return dict(zip(names, values))

    def chunk_read(self, segment, chunk):
        """Read a data chunk segment from the device.

        :param segment: The segment number.
        :param chunk: The chunk number within the segment.
        :return: The data read from the device.
        :raise IOError: on any error, including reading segments
            that do not have read permissions.
        """
        rv = self._usb.control_transfer_in(
            None, 'device', 'vendor',
            request=UsbdRequest.READ_CHUNK,
            value=segment, index=chunk, length=CHUNK_SIZE)
        _ioerror_on_bad_result(rv)
        return bytes(rv.data)

    def program(self, segment, data, metadata=None, progress_cbk=None):
        """Program a segment with data.

        :param segment: The segment to program in flash, either integer
            identifier or name from SEGMENTS.
        :param data: The raw data bytes for the flash.
        :param metadata: The metadata for firmware updates.
        :param progress_cbk:  The optional Callable[[float], None] which is called
            with the progress fraction from 0.0 to 1.0
        :return: 0 on success or error code.

        This function erases the existing application, even on failure.
        """
        log.info('%s: program', self)
        data = datafile.filename_or_bytes(data)
        segment = SEGMENTS[segment]
        if metadata is None:
            metadata = {
                'encryption': 0,
                'header': bytes([0] * 24),
                'mac': bytes([0] * 16),
                'signature': bytes([0] * 64),
            }
        if progress_cbk is None:
            progress_cbk = lambda x: None

        progress_cbk(0.000)
        metadata['size'] = len(data)
        msg = struct.pack('<II', metadata['size'], metadata['encryption'])
        msg = msg + metadata['header'] + metadata['mac'] + metadata['signature']
        log.info('write start: segment=%d, length=%d', segment, len(data))
        rv = self._usb.control_transfer_out(
            None, 'device', 'vendor',
            request=UsbdRequest.WRITE_START,
            value=segment, index=0, data=msg)
        _ioerror_on_bad_result(rv)

        total_size = len(data)
        chunk = 0
        while len(data):
            fraction_done = chunk * CHUNK_SIZE / total_size
            chunk_data = data[:CHUNK_SIZE]
            progress_cbk(fraction_done)
            log.info('write chunk %d, length=%d | %.1f%%', chunk, len(chunk_data), fraction_done * 100)
            rv = self._usb.control_transfer_out(
                None, 'device', 'vendor',
                request=UsbdRequest.WRITE_CHUNK,
                value=segment, index=chunk, data=chunk_data)
            _ioerror_on_bad_result(rv)
            data = data[CHUNK_SIZE:]
            chunk += 1
        log.info('write finalize')
        rv = self._usb.control_transfer_in(
            None, 'device', 'vendor',
            request=UsbdRequest.WRITE_FINALIZE,
            value=segment, index=0, length=1)
        _ioerror_on_bad_result(rv)
        log.info('write status=%d', rv.data[0])
        progress_cbk(1.0)
        return rv.data[0]

    def go(self):
        """Command the bootloader to run the application.

        :raise IOError: on failure.

        This method will close() the bootloader regardless of success.
        """
        log.info('%s: go', self)
        try:
            rv = self._usb.control_transfer_out(
                None, 'device', 'vendor',
                request=UsbdRequest.GO,
                value=0, index=0)
            _ioerror_on_bad_result(rv)
        finally:
            self.close()

    def firmware_program(self, filename, progress_cbk=None):
        """Program or update the application firmware.

        :param filename: The filename containing the firmware, which must be
            in Joulescope firmware format and correctly signed.
        :param progress_cbk:  The optional Callable[[float], None] which is called
            with the progress fraction from 0.0 to 1.0
        :return: 0 on success or error code.
        """
        log.info('%s: firmware_program', self)
        data = datafile.filename_or_bytes(filename)
        if len(data):
            fh = io.BytesIO(data)
            dr = datafile.DataFileReader(fh)
            # todo: check distribution signature
            tag, hdr_value = next(dr)
            if tag != datafile.TAG_HEADER:
                raise ValueError('incorrect format: expected header, received %r' % tag)
            tag, data = next(dr)
            if tag != datafile.TAG_DATA_BINARY:
                raise ValueError('incorrect format: expected data, received %r' % tag)
            tag, enc = next(dr)
            if tag != datafile.TAG_ENCRYPTION:
                raise ValueError('incorrect format: expected encryption, received %r' % tag)
            metadata = {
                'encryption': 1,
                'header': hdr_value[:24],
                'mac': enc[:16],
                'signature': enc[16:],
            }
            log.info('header    = %r', binascii.hexlify(metadata['header']))
            log.info('mac       = %r', binascii.hexlify(metadata['mac']))
            log.info('signature = %r', binascii.hexlify(metadata['signature']))
        else:
            metadata = None
        return self.program(Segment.FIRMWARE, data, metadata, progress_cbk)

    def calibration_program(self, filename, is_factory=False):
        """Program the calibration.

        :param filename: The filename containing the Joulescope calibration
            record for this device.
        :param is_factory: True to program factory calibration.  False to
            program the active calibration.  The device only allows
            factory calibration to be programmed once.  Attempting to
            overwrite factory calibration will result in an error.
        :return: 0 on success or error code.
        """
        log.info('%s: calibration_program', self)
        data = datafile.filename_or_bytes(filename)
        segment = Segment.CALIBRATION_FACTORY if bool(is_factory) else Segment.CALIBRATION_ACTIVE
        return self.program(segment, data)
        
    def _erase(self, magic, sector_start=None, sector_end=None):
        """Permanently erase flash on Joulescope.
        
        :param magic: The 32-byte erase key.
        :param sector_start: The starting sector to erase.
        :param sector_end: The ending sector to erase.
        
        WARNING: This operation renders Joulescope useless and should never be
        used other than during manufacturing.
        """
        if sector_end < sector_start:  # swap
            sector_start, sector_end = sector_end, sector_start
        sector_start = 0 if sector_start is None else int(sector_start)
        sector_end = 15 if sector_end is None else int(sector_end)
        if not 0 <= sector_start <= 15:
            raise ValueError('erase sector_start out of range: %s', sector_start)
        if not 0 <= sector_end <= 15:
            raise ValueError('erase sector_end out of range: %s', sector_end)
        log.info('%s ERASE (%d, %d)', self, sector_start, sector_end)
        # arm
        data = struct.pack('<III', 1, 0, 0)
        rv = self._usb.control_transfer_out(
            None, 'device', 'vendor',
            request=UsbdRequest.ERASE,
            value=0, index=0, data=magic + data)        
        _ioerror_on_bad_result(rv)
        
        # and do it... hope you meant it!
        data = struct.pack('<III', 2, sector_start, sector_end)
        rv = self._usb.control_transfer_out(
            None, 'device', 'vendor',
            request=UsbdRequest.ERASE,
            value=0, index=0, data=magic + data)        
        _ioerror_on_bad_result(rv)        
        return 0

    def erase(self, magic, operation=None):
        """Permanently erase flash on a Joulescope.  !!! WARNING: factory use only!!!

        :param magic: The 32-byte erase key.
        :param operation: The erase operation which is one of:
            * all: (default) Delete the everything: The application,
              bootloader, calibration, and personalization.
            * calibration: Delete all calibration information and storage.
        """
        sector_start, sector_end = {
            'all': (0, 15),
            'calibration': (13, 15),
        }.get(operation, (0, 15))
        return self._erase(magic, sector_start, sector_end)
