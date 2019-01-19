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

from joulescope.usb import core as usb_core
from joulescope.usb.core import SetupPacket, ControlTransferResponse
from joulescope.usb.scan_info import INFO
from typing import List
import time
import ctypes
import ctypes.util
from ctypes import Structure, c_uint8, c_uint16, c_uint, \
    c_int, c_char, c_ssize_t, c_void_p, POINTER, pointer
import logging

log = logging.getLogger(__name__)


STRING_LENGTH_MAX = 255
CONTROL_TRANSFER_TIMEOUT_MS = 1000  # default in milliseconds

_lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('usb-1.0'))


class DescriptorType:
    DEVICE = 0x01
    CONFIG = 0x02
    STRING = 0x03
    INTERFACE = 0x04
    ENDPOINT = 0x05
    BOS = 0x0f
    DEVICE_CAPABILITY = 0x10
    HID = 0x21
    REPORT = 0x22
    PHYSICAL = 0x23
    HUB = 0x29
    SUPERSPEED_HUB = 0x2a
    SS_ENDPOINT_COMPANION = 0x30


class TransferType:
    CONTROL = 0
    ISOCHRONOUS = 1
    BULK = 2
    INTERRUPT = 3
    BULK_STREAM = 4


class TransferStatus:
    COMPLETED = 0
    ERROR = 1
    TIMED_OUT = 2
    CANCELLED = 3
    STALL = 4
    NO_DEVICE = 5
    OVERFLOW = 6


class TransferFlags:
    SHORT_NOT_OK = 1 << 0
    FREE_BUFFER = 1 << 1
    FREE_TRANSFER = 1 << 2
    ADD_ZERO_PACKET = 1 << 3


class ReturnCodes:
    SUCCESS = 0
    ERROR_IO = -1
    ERROR_INVALID_PARAM = -2
    ERROR_ACCESS = -3
    ERROR_NO_DEVICE = -4
    ERROR_NOT_FOUND = -5
    ERROR_BUSY = -6
    ERROR_TIMEOUT = -7
    ERROR_OVERFLOW = -8
    ERROR_PIPE = -9
    ERROR_INTERRUPTED = -10
    ERROR_NO_MEM = -11
    ERROR_NOT_SUPPORTED = -12
    ERROR_OTHER = -99


class _libusb_device_descriptor(Structure):
    _fields_ = [
        ('bLength', c_uint8),
        ('bDescriptorType', c_uint8),
        ('bcdUSB', c_uint16),
        ('bDeviceClass', c_uint8),
        ('bDeviceSubClass', c_uint8),
        ('bDeviceProtocol', c_uint8),
        ('bMaxPacketSize0', c_uint8),
        ('idVendor', c_uint16),
        ('idProduct', c_uint16),
        ('bcdDevice', c_uint16),
        ('iManufacturer', c_uint8),
        ('iProduct', c_uint8),
        ('iSerialNumber', c_uint8),
        ('bNumConfigurations', c_uint8)]


# typedef void (LIBUSB_CALL *libusb_transfer_cb_fn)(struct libusb_transfer *transfer);
libusb_transfer_cb_fn = ctypes.CFUNCTYPE(None, c_void_p)

class _libusb_transfer(Structure):
    _fields_ = [
        ('dev_handle', c_void_p),
        ('flags', c_uint8),
        ('endpoint_id', c_uint8),
        ('endpoint_type', c_uint8),
        ('timeout_ms', c_uint),
        ('status', c_uint),
        ('length', c_int),
        ('actual_length', c_int),
        ('callback', libusb_transfer_cb_fn),
        ('user_data', c_void_p),
        ('buffer', POINTER(c_uint8)),
        ('num_iso_packets', c_int),
        # struct libusb_iso_packet_descriptor iso_packet_desc[ZERO_SIZED_ARRAY];
    ]

# typedef struct libusb_context libusb_context; - c_void_p
# typedef struct libusb_device libusb_device; - c_void_p
# typedef struct libusb_device_handle libusb_device_handle; c_void_p

# int LIBUSB_CALL libusb_init(libusb_context **ctx);
_lib.libusb_init.restype = c_int
_lib.libusb_init.argtypes = [POINTER(c_void_p)]

# void LIBUSB_CALL libusb_exit(libusb_context *ctx);
_lib.libusb_exit.restype = None
_lib.libusb_exit.argtypes = [c_void_p]

# ssize_t LIBUSB_CALL libusb_get_device_list(libusb_context *ctx,
#	libusb_device ***list);
_lib.libusb_get_device_list.restype = c_ssize_t
_lib.libusb_get_device_list.argtypes = [c_void_p, POINTER(POINTER(c_void_p))]


# void LIBUSB_CALL libusb_free_device_list(libusb_device **list,
#	int unref_devices);
_lib.libusb_free_device_list.restype = None
_lib.libusb_free_device_list.argtypes = [POINTER(c_void_p), c_int]


# int LIBUSB_CALL libusb_open(libusb_device *dev, libusb_device_handle **dev_handle);
_lib.libusb_open.restype = c_int
_lib.libusb_open.argtypes = [c_void_p, POINTER(c_void_p)]

# void LIBUSB_CALL libusb_close(libusb_device_handle *dev_handle);
_lib.libusb_close.restype = None
_lib.libusb_close.argtypes = [c_void_p]

# int LIBUSB_CALL libusb_set_configuration(libusb_device_handle *dev_handle,
#	int configuration);
_lib.libusb_set_configuration.restype = c_int
_lib.libusb_set_configuration.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_claim_interface(libusb_device_handle *dev_handle,
#	int interface_number);
_lib.libusb_claim_interface.restype = c_int
_lib.libusb_claim_interface.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_release_interface(libusb_device_handle *dev_handle,
#	int interface_number);
_lib.libusb_release_interface.restype = c_int
_lib.libusb_release_interface.argtypes = [c_void_p, c_int]

# int LIBUSB_CALL libusb_set_interface_alt_setting(libusb_device_handle *dev_handle,
#	int interface_number, int alternate_setting);
_lib.libusb_set_interface_alt_setting.restype = c_int
_lib.libusb_set_interface_alt_setting.argtypes = [c_void_p, c_int, c_int]


# int LIBUSB_CALL libusb_get_device_descriptor(libusb_device *dev,
#	struct libusb_device_descriptor *desc);
_lib.libusb_get_device_descriptor.restype = c_int
_lib.libusb_get_device_descriptor.argtypes = [c_void_p, POINTER(_libusb_device_descriptor)]

# int LIBUSB_CALL libusb_control_transfer(libusb_device_handle *dev_handle,
#	uint8_t request_type, uint8_t bRequest, uint16_t wValue, uint16_t wIndex,
#	unsigned char *data, uint16_t wLength, unsigned int timeout);
_lib.libusb_control_transfer.restype = c_int
_lib.libusb_control_transfer.argtypes = [c_void_p, c_uint8, c_uint8, c_uint16, c_uint16,
                                         POINTER(c_uint8), c_uint16, c_int]


_ctx = c_void_p()

rc = _lib.libusb_init(pointer(_ctx))
if rc:
    raise RuntimeError('Could not open libusb')


def _path_split(path):
    vid, pid, serial_number = path.split('/')
    return int(vid, 16), int(pid, 16), serial_number



def _get_string_descriptor(device, index):
    request_type = usb_core.RequestType(direction='in', type_='standard', recipient='device').u8
    byte_buffer = bytearray(STRING_LENGTH_MAX)
    buffer_type = c_uint8 * STRING_LENGTH_MAX
    buffer = buffer_type.from_buffer(byte_buffer)

    # determine default language
    rv = _lib.libusb_control_transfer(device, request_type, usb_core.Request.GET_DESCRIPTOR,
                                      (DescriptorType.STRING << 8), 0,
                                      buffer, STRING_LENGTH_MAX,
                                      1000)
    if rv < 0:
        raise RuntimeError('control_transfer could not get language: %d' % (rv, ))
    langid = int(byte_buffer[2]) | (int(byte_buffer[3]) << 8)

    rv = _lib.libusb_control_transfer(device, request_type, usb_core.Request.GET_DESCRIPTOR,
                                      (DescriptorType.STRING << 8) | (index & 0xff), langid,
                                      buffer, STRING_LENGTH_MAX,
                                      1000)

    if rv < 0:
        raise RuntimeError('control transfer could not get string descriptor: %d' % (rv, ))
    buffer_len = min(rv, byte_buffer[0])
    return byte_buffer[:buffer_len].decode('UTF-16-LE')


class LibUsbDevice:
    """The LibUSB :class:`usb.api.Device` implementation"""

    def __init__(self, path):
        self._path = path
        self._handle = c_void_p(None)

    def __str__(self):
        return f'Joulescope {self._path}'

    def _open(self):
        log.info('open: start %s', self._path)
        descriptor = _libusb_device_descriptor()
        devices = POINTER(c_void_p)()
        vid, pid, serial_number = _path_split(self._path)
        sz = _lib.libusb_get_device_list(_ctx, pointer(devices))
        try:
            for idx in range(sz):
                device = devices[idx]
                if _lib.libusb_get_device_descriptor(device, pointer(descriptor)):
                    continue
                if vid == descriptor.idVendor and pid == descriptor.idProduct:
                    dh = c_void_p()
                    rv = _lib.libusb_open(device, dh)
                    if rv < 0:
                        log.info('Could not open device: %04x/%04x', vid, pid)
                        continue
                    if serial_number == _get_string_descriptor(dh, descriptor.iSerialNumber):
                        self._handle = dh
                        log.info('open: success')
                        return
            log.warning('open:failed')
            raise IOError('open:failed')
        finally:
            _lib.libusb_free_device_list(devices, 0)


    def open(self):
        self.close()
        self._open()
        log.info('Configure device')
        if 0 != _lib.libusb_set_configuration(self._handle, 1):
            raise IOError('libusb_set_configuration 1 failed')
        if 0 != _lib.libusb_claim_interface(self._handle, 0):
            raise IOError('libusb_claim_interface 0 failed')
        if 0 != _lib.libusb_set_interface_alt_setting(self._handle, 0, 0):
            raise IOError('libusb_set_interface_alt_setting 0,0 failed')

    def close(self):
        if self._handle is not None:
            self._handle = None

    def control_transfer_out(self, recipient, type_, request, value=0, index=0, data=None) -> ControlTransferResponse:
        request_type = usb_core.RequestType(direction='out', type_=type_, recipient=recipient).u8
        length = 0 if data is None else len(data)
        setup_packet = usb_core.SetupPacket(request_type, request, value, index, length)
        if length:
            buffer = (c_uint8 * length)(*data)
        else:
            buffer = c_void_p()
        rv = _lib.libusb_control_transfer(self._handle, request_type, request, value, index,
                                          buffer, length, CONTROL_TRANSFER_TIMEOUT_MS)
        if rv < 0:
            log.warning('control_transfer_out rv=%d', rv)
            result = rv
        elif rv > length:
            log.warning('control_transfer_out: length too long (%d > %d)', rv, length)
            result = 1
        else:
            result = 0
        length = min(rv, length)
        return usb_core.ControlTransferResponse(setup_packet, result, bytes(buffer[:length]))

    def control_transfer_in(self, recipient, type_, request, value, index, length) -> ControlTransferResponse:
        request_type = usb_core.RequestType(direction='in', type_=type_, recipient=recipient).u8
        buffer = (c_uint8 * length)()
        setup_packet = usb_core.SetupPacket(request_type, request, value, index, length)
        rv = _lib.libusb_control_transfer(self._handle, request_type, request, value, index,
                                          buffer, length, CONTROL_TRANSFER_TIMEOUT_MS)
        if rv < 0:
            log.warning('control_transfer_in rv=%d', rv)
            result = rv
        elif rv > length:
            log.warning('control_transfer_in: length too long (%d > %d)', rv, length)
            result = 1
        else:
            result = 0
        length = min(length, rv)
        return usb_core.ControlTransferResponse(setup_packet, result, bytes(buffer[:length]))

    def read_stream_start(self, endpoint_id, transfers, block_size, data_fn, process_fn):
        pass  # todo

    def read_stream_stop(self, endpoint_id):
        pass  # todo

    def status(self):
        return {}

    def signal(self):
        pass  # todo

    def process(self, timeout=None):
        time.sleep(0.005)


class DeviceNotify:

    def __init__(self, cbk):
        self._cbk = cbk

    def close(self):
        """Close and stop the notifications."""
        pass


def scan(name: str) -> List[LibUsbDevice]:
    """Scan for attached devices.

    :param name: The case-insensitive name of the device to scan.
    :return: The list of discovered :class:`WinUsbDevice` instances.
    """
    paths = []
    infos = INFO[name.lower()]
    descriptor = _libusb_device_descriptor()
    devices = POINTER(c_void_p)()
    sz = _lib.libusb_get_device_list(_ctx, pointer(devices))
    try:
        for idx in range(sz):
            device = devices[idx]
            if _lib.libusb_get_device_descriptor(device, pointer(descriptor)):
                raise RuntimeError('descriptor')
            for info in infos:
                vid = info['vendor_id']
                pid = info['product_id']
                if vid == descriptor.idVendor and pid == descriptor.idProduct:
                    dh = c_void_p()
                    rv = _lib.libusb_open(device, pointer(dh))
                    if rv < 0:
                        log.info('Could not open device: %04x/%04x', vid, pid)
                        continue
                    serial_number = _get_string_descriptor(dh, descriptor.iSerialNumber)
                    _lib.libusb_close(dh)
                    path = '%04x/%04x/%s' % (vid, pid, serial_number)
                    paths.append(path)
    finally:
        _lib.libusb_free_device_list(devices, 0)

    if not len(paths):
        log.info('scan found no devices')
        return []
    log.info('scan found %s' % paths)
    devices = [LibUsbDevice(p) for p in paths]
    return devices


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    devices = scan('Joulescope')
    print('\n'.join([str(d) for d in devices]))
    if len(devices):
        d = devices[0]
        d.open()
        rv = d.control_transfer_in('device', 'vendor', request=4,
            value=0, index=0, length=128)
        print(rv)
        d.close()
