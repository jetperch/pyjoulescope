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

from .. import core as usb_core
from joulescope.v0.usb.api import DeviceEvent
from joulescope.v0.usb.impl_tools import RunUntilDone
from joulescope.v0.usb.scan_info import INFO
from .setupapi import device_interface_guid_to_paths
from ctypes import windll, Structure, POINTER, byref, sizeof, \
    c_ubyte, c_ushort, c_ulong, c_void_p, pointer
from ctypes.wintypes import DWORD, HANDLE, BOOL
import numpy as np
from typing import List
import time

from . import kernel32
import logging
log = logging.getLogger(__name__)

_winusb = windll.winusb

TICK_INTERVAL = 1.0  # seconds
BULK_IN_LENGTH = 512

# Set the control timeout in milliseconds
# WinUSB cannot cancel control transfers.
# In order to prevent heap corruption, the timeout must be less than the
# any forced closing time.
CONTROL_TIMEOUT = 1.0  # seconds


class PipeInfo(Structure):
    _fields_ = [
        ('pipe_type', c_ulong),
        ('pipe_id', c_ubyte),
        ('maximum_packet_size', c_ushort), 
        ('interval', c_ubyte)
    ]

    def __repr__(self):
        return usb_core.structure_to_repr(self)


# https://docs.microsoft.com/en-us/windows-hardware/drivers/usbcon/winusb-functions-for-pipe-policy-modification
class PipePolicy:
    SHORT_PACKET_TERMINATE = 0x01
    AUTO_CLEAR_STALL = 0x02
    PIPE_TRANSFER_TIMEOUT = 0x03
    IGNORE_SHORT_PACKETS = 0x04
    ALLOW_PARTIAL_READS = 0x05
    AUTO_FLUSH = 0x06
    RAW_IO = 0x07
    MAXIMUM_TRANSFER_SIZE = 0x08
    RESET_PIPE_ON_RESUME = 0x09


# BOOL __stdcall WinUsb_Initialize(
#   _In_  HANDLE                   DeviceHandle,
#   _Out_ PWINUSB_INTERFACE_HANDLE InterfaceHandle);
WinUsb_Initialize = _winusb.WinUsb_Initialize
WinUsb_Initialize.restype = BOOL
WinUsb_Initialize.argtypes = [HANDLE, POINTER(c_void_p)]

# BOOL __stdcall WinUsb_Free(
#   _In_ WINUSB_INTERFACE_HANDLE InterfaceHandle);
WinUsb_Free = _winusb.WinUsb_Free
WinUsb_Free.restype = BOOL
WinUsb_Free.argtypes = [c_void_p]

# BOOL __stdcall WinUsb_GetDescriptor(
#   _In_  WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_  UCHAR                   DescriptorType,
#   _In_  UCHAR                   Index,
#   _In_  USHORT                  LanguageID,
#   _Out_ PUCHAR                  Buffer,
#   _In_  ULONG                   BufferLength,
#   _Out_ PULONG                  LengthTransferred);
WinUsb_GetDescriptor = _winusb.WinUsb_GetDescriptor
WinUsb_GetDescriptor.restype = BOOL
WinUsb_GetDescriptor.argtypes = [c_void_p, c_ubyte, c_ubyte, c_ushort, POINTER(c_ubyte), c_ulong, POINTER(c_ulong)]


# BOOL __stdcall WinUsb_QueryDeviceInformation(
#   _In_    WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_    ULONG                   InformationType,
#   _Inout_ PULONG                  BufferLength,
#   _Out_   PVOID                   Buffer);
WinUsb_QueryDeviceInformation = _winusb.WinUsb_QueryDeviceInformation
WinUsb_QueryDeviceInformation.restype = BOOL
WinUsb_QueryDeviceInformation.argtypes = [c_void_p, c_ulong, POINTER(c_ulong), c_void_p]

# BOOL __stdcall WinUsb_QueryInterfaceSettings(
#   _In_  WINUSB_INTERFACE_HANDLE   InterfaceHandle,
#   _In_  UCHAR                     AlternateSettingNumber,
#   _Out_ PUSB_INTERFACE_DESCRIPTOR UsbAltInterfaceDescriptor);
WinUsb_QueryInterfaceSettings = _winusb.WinUsb_QueryInterfaceSettings
WinUsb_QueryInterfaceSettings.restype = BOOL
WinUsb_QueryInterfaceSettings.argtypes = [c_void_p, c_ubyte, POINTER(usb_core.InterfaceDescriptor)]

# BOOL __stdcall WinUsb_GetAssociatedInterface(
#   _In_  WINUSB_INTERFACE_HANDLE  InterfaceHandle,
#   _In_  UCHAR                    AssociatedInterfaceIndex,
#   _Out_ PWINUSB_INTERFACE_HANDLE AssociatedInterfaceHandle); -- must call WinUsb_Free
WinUsb_GetAssociatedInterface = _winusb.WinUsb_GetAssociatedInterface
WinUsb_GetAssociatedInterface.restype = BOOL
WinUsb_GetAssociatedInterface.argtypes = [c_void_p, c_ubyte, POINTER(c_void_p)]

# BOOL __stdcall WinUsb_ControlTransfer(
#   _In_      WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_      WINUSB_SETUP_PACKET     SetupPacket,
#   _Out_     PUCHAR                  Buffer,
#   _In_      ULONG                   BufferLength,
#   _Out_opt_ PULONG                  LengthTransferred,
#   _In_opt_  LPOVERLAPPED            Overlapped);
WinUsb_ControlTransfer = _winusb.WinUsb_ControlTransfer
WinUsb_ControlTransfer.restype = BOOL
WinUsb_ControlTransfer.argtypes = [c_void_p, usb_core.SetupPacket, POINTER(c_ubyte), c_ulong, POINTER(c_ulong), POINTER(
    kernel32.Overlapped)]

# BOOL __stdcall WinUsb_QueryPipe(
#   _In_  WINUSB_INTERFACE_HANDLE  InterfaceHandle,
#   _In_  UCHAR                    AlternateInterfaceNumber,
#   _In_  UCHAR                    PipeIndex,
#   _Out_ PWINUSB_PIPE_INFORMATION PipeInformation);
WinUsb_QueryPipe = _winusb.WinUsb_QueryPipe
WinUsb_QueryPipe.restype = BOOL
WinUsb_QueryPipe.argtypes = [c_void_p, c_ubyte, c_ubyte, POINTER(PipeInfo)]

# BOOL __stdcall WinUsb_ReadPipe(
#   _In_      WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_      UCHAR                   PipeID,
#   _Out_     PUCHAR                  Buffer,
#   _In_      ULONG                   BufferLength,
#   _Out_opt_ PULONG                  LengthTransferred,
#   _In_opt_  LPOVERLAPPED            Overlapped);
WinUsb_ReadPipe = _winusb.WinUsb_ReadPipe
WinUsb_ReadPipe.restype = BOOL
WinUsb_ReadPipe.argtypes = [c_void_p, c_ubyte, POINTER(c_ubyte), c_ulong, POINTER(c_ulong),
                            POINTER(kernel32.Overlapped)]

# BOOL WinUsb_SetPipePolicy(
#   WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   UCHAR                   PipeID,
#   ULONG                   PolicyType,
#   ULONG                   ValueLength,
#   PVOID                   Value);
WinUsb_SetPipePolicy = _winusb.WinUsb_SetPipePolicy
WinUsb_SetPipePolicy.restype = BOOL
WinUsb_SetPipePolicy.argtypes = [c_void_p, c_ubyte, c_ulong, c_ulong, c_void_p]

# BOOL __stdcall WinUsb_WritePipe(
#   _In_      WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_      UCHAR                   PipeID,
#   _In_      PUCHAR                  Buffer,
#   _In_      ULONG                   BufferLength,
#   _Out_opt_ PULONG                  LengthTransferred,
#   _In_opt_  LPOVERLAPPED            Overlapped);
WinUsb_WritePipe = _winusb.WinUsb_WritePipe
WinUsb_WritePipe.restype = BOOL
WinUsb_WritePipe.argtypes = [c_void_p, c_ubyte, POINTER(c_ubyte), c_ulong, POINTER(c_ulong),
                             POINTER(kernel32.Overlapped)]

# BOOL __stdcall WinUsb_AbortPipe(
#   _In_ WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_ UCHAR                   PipeID);
WinUsb_AbortPipe = _winusb.WinUsb_AbortPipe
WinUsb_AbortPipe.restype = BOOL
WinUsb_AbortPipe.argtypes = [c_void_p, c_ubyte]

# BOOL __stdcall WinUsb_FlushPipe(
#   _In_ WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_ UCHAR                   PipeID);
WinUsb_FlushPipe = _winusb.WinUsb_FlushPipe
WinUsb_FlushPipe.restype = BOOL
WinUsb_FlushPipe.argtypes = [c_void_p, c_ubyte]

# BOOL __stdcall WinUsb_GetOverlappedResult(
#   _In_  WINUSB_INTERFACE_HANDLE InterfaceHandle,
#   _In_  LPOVERLAPPED            lpOverlapped,
#   _Out_ LPDWORD                 lpNumberOfBytesTransferred,
#   _In_  BOOL                    bWait);
WinUsb_GetOverlappedResult = _winusb.WinUsb_GetOverlappedResult
WinUsb_GetOverlappedResult.restype = BOOL
WinUsb_GetOverlappedResult.argtypes = [c_void_p, POINTER(kernel32.Overlapped), POINTER(DWORD), BOOL]


def wrap_no_throw(f):
    if f is None:
        return lambda *args, **kwargs: None

    def fn(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            log.exception('While calling %s' % fn.__name__)
    return fn


def sanitize_boolean_return_code(result):
    if result:
        return 0
    else:
        return kernel32.GetLastError()


class TransferOverlapped:
    def __init__(self, event, size):
        self.ov = kernel32.Overlapped(event)
        self.ptr = pointer(self.ov)
        self.data = np.full(size, 0, dtype=np.uint8)
        self.b = self.data.ctypes.data_as(POINTER(c_ubyte))
        self.size = size

    def reset(self):
        self.ov.reset()


class EndpointIn:

    def __init__(self, winusb, pipe_id, transfers, block_size, data_fn, process_fn, stop_fn):
        """Manage an in endpoint.

        :param winusb: The underlying winusb handle.
        :param pipe_id: The endpoint IN pipe identifier.
        :param transfers: The number of outstanding transfers to pend.
        :param block_size: The size of each transfer in bytes.
        :param data_fn: The function to call with the received endpoint IN data.
            The data_fn should normally return False, but can return True to
            stop the endpoint streaming.
        :param process_fn: The function() called after data_fn was called.
            This function can have more latency than data_fn.
            The process_fn should normally return False, but can return True to
            stop the endpoint streaming.
        :param stop_fn: The function(event, message) called when this endpoint
            stops streaming data.
        """
        self._winusb = winusb
        self.pipe_id = pipe_id  # (endpoint_id & 0x7f) | 0x80
        self._overlapped_free = []
        self._overlapped_pending = []
        self._transfers = transfers
        self._transfer_size = ((block_size + BULK_IN_LENGTH - 1) // BULK_IN_LENGTH) * BULK_IN_LENGTH
        self._data_fn = data_fn
        self._process_fn = process_fn
        self._stop_fn = stop_fn
        self._process_transfers = 0
        self._time_last = 0.0

        self._state = self.ST_IDLE
        self.stop_code = None
        self.stop_message = ''
        self.byte_count_this = 0
        self.byte_count_total = 0
        self.transfer_count = 0
        self.transfer_expire_max = 0

        self.event = None
        self._overlapped_free = []
        self._overlapped_pending = []

    ST_IDLE = 0
    ST_RUNNING = 1
    ST_STOPPING = 2

    def _open(self):
        self.stop_code = None
        self.event = kernel32.CreateEvent(
            None,   # default security
            True,   # manual reset
            False,  # not initially signalled
            None    # no name
        )
        if self.event is None:
            raise ValueError('could not create event')

        for i in range(self._transfers):
            ov = TransferOverlapped(self.event, self._transfer_size)
            self._overlapped_free.append(ov)

    def _close(self):
        if self.event is not None:
            kernel32.CloseHandle(self.event)
        self.event = None

    def _issue(self, ov):
        ov.reset()
        result = WinUsb_ReadPipe(self._winusb, self.pipe_id, ov.b, ov.size, None, ov.ptr)
        ec = kernel32.GetLastError()
        if not result and (ec != kernel32.ERROR_IO_PENDING):
            msg = 'EndpointIn %d issue failed: %s' % (self.pipe_id, kernel32.get_error_str(ec))
            self._overlapped_free.append(ov)
            self._halt(DeviceEvent.COMMUNICATION_ERROR, msg)
            return True
        self._overlapped_pending.append(ov)
        return False

    def _pend(self):
        while len(self._overlapped_free):
            ov = self._overlapped_free.pop(0)
            if self._issue(ov):
                return True
        return False

    def _expire(self):
        rv = False
        length_transferred = c_ulong(0)
        count = 0
        while not rv and len(self._overlapped_pending):
            ov = self._overlapped_pending[0]
            if WinUsb_GetOverlappedResult(self._winusb, ov.ptr, byref(length_transferred), False):
                ov = self._overlapped_pending.pop(0)
                self.transfer_count += 1
                length = length_transferred.value
                self.byte_count_this += length
                count += 1
                try:
                    rv = self._data_fn(ov.data[:length])
                    rv = bool(rv)
                except Exception:
                    log.exception('data_fn exception: stop streaming')
                    rv = True
                if rv:
                    msg = f'EndpointIn {self.pipe_id} terminated by data_fn'
                    self._halt(DeviceEvent.ENDPOINT_CALLBACK_STOP, msg)
                    self._overlapped_free.append(ov)
                else:
                    rv = self._issue(ov)
            else:
                ec = kernel32.GetLastError()
                if ec in [kernel32.ERROR_IO_INCOMPLETE, kernel32.ERROR_IO_PENDING]:
                    break
                ov = self._overlapped_pending.pop(0)
                self._overlapped_free.append(ov)
                msg = 'EndpointIn WinUsb_GetOverlappedResult fatal: %s' % (kernel32.get_error_str(ec),)
                rv = True
                self._halt(DeviceEvent.COMMUNICATION_ERROR, msg)
        if count > self.transfer_expire_max:
            self.transfer_expire_max = count
        self._process_transfers += count
        return rv

    def _cancel(self):
        length_transferred = c_ulong(0)
        if not WinUsb_AbortPipe(self._winusb, self.pipe_id):
            log.warning('WinUsb_AbortPipe pipe_id %d: %s', self.pipe_id, kernel32.get_last_error())
        while len(self._overlapped_pending):
            ov = self._overlapped_pending.pop(0)
            if not WinUsb_GetOverlappedResult(self._winusb, ov.ptr, byref(length_transferred), True):
                ec = kernel32.GetLastError()
                if ec == kernel32.ERROR_IO_OPERATION_ABORTED:
                    pass  # aborted as requested, no issue
                elif ec in [kernel32.ERROR_FILE_NOT_FOUND, kernel32.ERROR_GEN_FAILURE, kernel32.ERROR_DEVICE_NOT_CONNECTED]:
                    log.debug('cancel overlapped: %s', kernel32.get_last_error())
                else:
                    log.warning('cancel overlapped: %s', kernel32.get_last_error())
            self._overlapped_free.append(ov)

    def _halt(self, stop_code, msg=None):
        if self._state != self.ST_STOPPING:
            self._state = self.ST_STOPPING
            self._cancel()
        if stop_code:
            lvl = logging.INFO if stop_code <= 0 else logging.ERROR
            if self.stop_code is None:
                self.stop_code = stop_code
                self.stop_message = '' if msg is None else str(msg)
                log.log(lvl, 'endpoint halt %d: %s', stop_code, msg)
            else:
                log.log(lvl, 'endpoint halt %d duplicate: %s', stop_code, msg)

    def process(self):
        """Process pending data.

        :return: True to stop process, False to continue processing.
        """
        if self._state != self.ST_RUNNING:
            return self.stop_code > 0
        rv = self._expire()
        if not rv:
            rv = self._pend()
        return rv

    def process_signal(self):  #: NoRaise
        if self._process_transfers:
            self._process_transfers = 0
            try:
                if callable(self._process_fn):
                    return self._process_fn()
            except Exception:
                log.exception('_process_fn exception: stop streaming')
                return True  # force stop
        return False

    def start(self):
        log.info("endpoint start 0x%02x transfer size = %d bytes" % (self.pipe_id, self._transfer_size))
        self._open()
        self._state = self.ST_RUNNING
        self._process_transfers = 0
        self._time_last = time.time()
        self._pend()

    def stop(self):
        if self._state != self.ST_IDLE:
            log.info("endpoint stop")
            if self._state != self.ST_STOPPING:
                self._cancel()  # was already called in halt
            if self.stop_code is None:
                self.stop_code = 0
                self.process_signal()  # ensure that all received data is processed
            self._close()
            try:
                self._stop_fn(self.stop_code, self.stop_message)
            except Exception:
                log.exception('_stop_fn exception')
            self._state = self.ST_IDLE

    def status(self):
        """Get the endpoint status.

        :return: A dict mapping status name to a value..  The value is a dict
            containing 'value' and 'units'.
        """
        time_now = time.time()
        duration = time_now - self._time_last
        if duration < 0.01:
            throughput = 0.0
        else:
            throughput = self.byte_count_this / duration
        self.byte_count_total += self.byte_count_this
        status = {
            'bytes': {'value': self.byte_count_total, 'units': 'bytes'},
            'transfers': {'value': self.transfer_count, 'units': 'transfers'},
            'duration': {'value': duration, 'units': 'seconds'},
            'throughput': {'value': throughput, 'units': 'Bps'},
            'transfer_expire': {'value': self.transfer_expire_max, 'units': 'transfers'},

        }
        self.byte_count_this = 0
        self.transfer_expire_max = 0
        self._time_last = time_now
        return status


class ControlTransferAsync:

    def __init__(self, winusb):
        self._winusb = winusb
        self._overlapped = None
        self._event = None
        self._commands = []  # list of pending control transfer commands
        self._time_start = None
        self.stop_code = None

    @property
    def event(self):
        return self._event

    def open(self):
        self.stop_code = None
        self._event = kernel32.CreateEvent(
            None,   # default security
            True,   # manual reset
            False,  # not initially signalled
            None    # no name
        )
        if self._event is None:
            raise ValueError('could not create control event')
        self._overlapped = TransferOverlapped(self._event, 4096)

    def close(self):
        self._commands, commands = [], self._commands
        # cannot abort control pipe using WinUSB!
        commands_len = len(commands)
        if commands:
            command = commands.pop(0)
            rc = kernel32.WaitForSingleObject(self._event, int(CONTROL_TIMEOUT * 1100))
            if rc != kernel32.WAIT_OBJECT_0:  # transfer done
                log.warning('ControlTransferAsync.close but transaction hung')
            self._finish(command)
        while commands:
            cbk_fn, setup_packet, _ = commands.pop(0)
            cbk_fn(usb_core.ControlTransferResponse(setup_packet, False, None))
        if commands_len == 0:
            if self._event:
                kernel32.CloseHandle(self._event)
                self._event = None
            if self._overlapped:
                self._overlapped = None
        else:
            pass  # defer to _close_event

    def _close_event(self):
        if self._event:
            kernel32.CloseHandle(self._event)
            self._event = None
            if self._overlapped:
                self._overlapped = None

    def pend(self, cbk_fn, setup_packet: usb_core.SetupPacket, buffer=None):
        """Pend an asynchronous Control Transfer.

        :param cbk_fn: The function to call when the control transfer completes.
            A :class:`usb_core.ControlTransferResponse` is the sole argument.
        :param setup_packet:
        :param buffer: The buffer (if length > 0) for write transactions.
        :return: True on pending, False on error.
        """
        if self.stop_code is not None:
            response = usb_core.ControlTransferResponse(setup_packet, self.stop_code, None)
            cbk_fn(response)
            return False
        command = [cbk_fn, setup_packet, buffer]
        was_empty = not bool(self._commands)
        self._commands.append(command)
        if was_empty:
            return self._issue()
        return True

    def _issue(self):
        if not self._commands:
            return True
        cbk_fn, setup_packet, buffer = self._commands[0]
        pkt = usb_core.RequestType(value=setup_packet.request_type)
        self._overlapped.reset()
        if pkt.direction == 'out':
            if setup_packet.length > 0:
                log.debug('ControlTransferAsync._issue buffer type: %s', type(buffer))
                self._overlapped.data[:setup_packet.length] = np.frombuffer(buffer, dtype=np.uint8)
        result = WinUsb_ControlTransfer(
            self._winusb,
            setup_packet,
            self._overlapped.b,
            setup_packet.length,
            None,
            self._overlapped.ptr)
        result = sanitize_boolean_return_code(result)
        self._time_start = time.time()
        if result != kernel32.ERROR_IO_PENDING:
            if self.stop_code is None:
                self.stop_code = DeviceEvent.COMMUNICATION_ERROR
            log.warning('ControlTransferAsync._issue %s', kernel32.get_error_str(result))
            response = usb_core.ControlTransferResponse(setup_packet, result, None)
            cbk_fn(response)
            return False
        return True

    def _finish(self, command):
        buffer = None
        cbk_fn, setup_packet, _ = command
        length_transferred = c_ulong()
        rc = WinUsb_GetOverlappedResult(self._winusb, self._overlapped.ptr, byref(length_transferred), True)
        if rc == 0:  # failed (any value other than 0 is success)
            rc = kernel32.GetLastError()
            if rc not in [kernel32.ERROR_IO_INCOMPLETE, kernel32.ERROR_IO_PENDING]:
                kernel32.ResetEvent(self._event)
                log.warning('ControlTransferAsync._finish result: %s', kernel32.get_error_str(rc))
        else:
            kernel32.ResetEvent(self._event)
            rc = 0
            pkt = usb_core.RequestType(value=setup_packet.request_type)
            duration = time.time() - self._time_start
            if pkt.direction == 'in' and setup_packet.length:
                log.debug('ControlTransferAsync._finish duration=%.6f s, length: %s, %s',
                          duration, setup_packet.length, length_transferred.value)
                buffer = bytes(self._overlapped.data[:length_transferred.value])
            else:
                log.debug('ControlTransferAsync._finish duration=%.6f s', duration)
        response = usb_core.ControlTransferResponse(setup_packet, rc, buffer)
        cbk_fn(response)

    def process(self):
        """Check if control transfer completed"""
        if not self._commands:
            return
        rc = kernel32.WaitForSingleObject(self._event, 0)
        if rc == kernel32.WAIT_OBJECT_0:  # transfer done
            command = self._commands.pop(0)
            self._finish(command)
            if self.stop_code is None:
                self._issue()
            else:
                self._close_event()
        elif rc == kernel32.WAIT_TIMEOUT:
            pass  # not ready yet
        else:
            log.warning('ControlTransferAsync.process wait %d', rc)


class WinUsbDevice:
    """The WinUSB :class:`usb.api.Device` implementation"""

    def __init__(self, path):
        self._path = path
        self._file = None  # The file handle for this device
        self._winusb = HANDLE()  # The WinUSB handle for the device's default interface
        self._interface = 0
        self._endpoints = {}  # type: Dict[int, EndpointIn]
        self._event = None
        self._event_list = (HANDLE * kernel32.MAXIMUM_WAIT_OBJECTS)()
        self._event_callback_fn = None
        self._control_transfer = None
        self._update_event_list()

    def __str__(self):
        # path like: \\?\usb#vid_1fc9&pid_fc93#0001#{99a06894-3518-41a5-a207-8519746da89f}
        # not recommended by Microsoft, but good enough for now (20181003)
        try:
            parts = self._path.split('#')
            # vid, pid = [x.split('_')[1] for x in parts[1].split('&')]
            serial_number = parts[2]
            return 'Joulescope:' + serial_number
        except Exception:
            return 'Joulescope'

    def _update_event_list(self):
        self.event_list_count = 0

        def _event_append(event):
            if event is not None:
                self._event_list[self.event_list_count] = event
                self.event_list_count += 1

        _event_append(self._event)
        if self._control_transfer:
            _event_append(self._control_transfer.event)
        for endpoint in self._endpoints.values():
            _event_append(endpoint.event)

    @property
    def path(self):
        return self._path

    @property
    def serial_number(self):
        parts = self._path.split('#')
        if len(parts) != 4:
            return '0000-0000'
        return parts[2]

    def open(self, event_callback_fn=None):
        self.close()
        log.info('WinUsbDevice.open')
        self._event_callback_fn = event_callback_fn

        # create event for self.signal()
        self._event = kernel32.CreateEvent(
            None,   # default security
            True,   # manual reset
            False,  # not initially signalled
            None    # no name
        )

        try:
            if self._event is None:
                raise RuntimeError('could not create event')

            self._file = kernel32.CreateFile(
                self._path,
                kernel32.GENERIC_WRITE | kernel32.GENERIC_READ,
                kernel32.FILE_SHARE_WRITE | kernel32.FILE_SHARE_READ,
                None,
                kernel32.OPEN_EXISTING,
                kernel32.FILE_ATTRIBUTE_NORMAL | kernel32.FILE_FLAG_OVERLAPPED,
                None)

            if self._file == kernel32.INVALID_HANDLE_VALUE:
                raise IOError('%s : open failed on invalid handle' % self)
            result = WinUsb_Initialize(self._file, byref(self._winusb))
            if result == 0:
                raise IOError('%s : open failed %s' % (self, kernel32.get_last_error()))
            log.info('is_high_speed = %s', self._is_high_speed)
            log.info('interface_settings = %s', self._query_interface_settings(0))
            self._control_transfer = ControlTransferAsync(self._winusb)
            self._control_transfer.open()
            timeout = DWORD(int(CONTROL_TIMEOUT * 1000))
            result = WinUsb_SetPipePolicy(self._winusb, 0, PipePolicy.PIPE_TRANSFER_TIMEOUT,
                                          sizeof(timeout), byref(timeout))
            if not result:
                log.warning('WinUsb_SetPipePolicy: %s', kernel32.get_error_str())
            self._update_event_list()
        except Exception:
            self.close()
            raise

    def close(self):
        log.info('WinUsbDevice.close')
        for endpoint in self._endpoints.values():
            endpoint.stop()
        self._endpoints.clear()
        if self._control_transfer is not None:
            self._control_transfer.close()
            self._control_transfer = None
        if self._file is not None:
            WinUsb_Free(self._winusb)
            self._winusb = HANDLE()
            kernel32.CloseHandle(self._file)
            self._file = None
        self._interface = 0
        if self._event:
            kernel32.CloseHandle(self._event)
            self._event = None
        self._event_callback_fn = None

    @property
    def _is_high_speed(self):
        if self._file is None:
            raise IOError('WinUSB device not open')
        buff = (c_void_p * 1)()
        buff_length = c_ulong(sizeof(buff))
        result = WinUsb_QueryDeviceInformation(self._winusb, 1, byref(buff_length), buff)
        if result != 0:
            return buff[0] >= 3
        else:
            raise IOError('WinUsb_QueryDeviceInformation failed')
    
    def _query_interface_settings(self, interface, alternate_setting=0):
        if interface != 0:
            handle = HANDLE()
            result = WinUsb_GetAssociatedInterface(self._winusb, interface, byref(handle))
            if result == 0:
                raise IOError('WinUsb_GetAssociatedInterface failed: ' + kernel32.get_last_error())
        else:
            handle = self._winusb
        interface_descriptor = usb_core.InterfaceDescriptor()
        result = WinUsb_QueryInterfaceSettings(handle, alternate_setting, byref(interface_descriptor))
        if interface != 0:
            WinUsb_Free(handle)
        if result == 0:
            raise IOError('WinUsb_QueryInterfaceSettings failed: ' + kernel32.get_last_error())
        return interface_descriptor

    def _query_pipe(self, pipe_index):
        pipe_info = PipeInfo()
        result = WinUsb_QueryPipe(self._winusb, 0, pipe_index, byref(pipe_info))
        if result == 0:
            raise IOError('WinUsb_QueryPipe failed: ' + kernel32.get_last_error())
        return pipe_info

    def control_transfer_out(self, cbk_fn, recipient, type_, request, value=0, index=0, data=None):
        if cbk_fn is None:
            run_until_done = RunUntilDone(1.0, 'control_transfer_out')
            self.control_transfer_out(run_until_done.cbk_fn, recipient, type_, request, value, index, data)
            while not run_until_done.is_done():
                self.process(0.01)
            return run_until_done.value_args0
        log.debug('control_transfer_out')
        request_type = usb_core.RequestType(direction='out', type_=type_, recipient=recipient).u8
        length = 0 if data is None else len(data)
        pkt = usb_core.SetupPacket(request_type, request, value, index, length)
        return self._control_transfer.pend(cbk_fn, pkt, data)

    def control_transfer_in(self, cbk_fn, recipient, type_, request, value, index, length):
        if cbk_fn is None:
            run_until_done = RunUntilDone(1.0, 'control_transfer_in')
            self.control_transfer_in(run_until_done.cbk_fn, recipient, type_, request, value, index, length)
            while not run_until_done.is_done():
                self.process(0.01)
            return run_until_done.value_args0
        log.debug('control_transfer_in')
        request_type = usb_core.RequestType(direction='in', type_=type_, recipient=recipient).u8
        pkt = usb_core.SetupPacket(request_type, request, value, index, length)
        return self._control_transfer.pend(cbk_fn, pkt)

    def read_stream_start(self, endpoint_id, transfers, block_size, data_fn, process_fn, stop_fn):
        log.info('read_stream_start %d', endpoint_id)
        pipe_id = (endpoint_id & 0x7f) | 0x80
        endpoint = self._endpoints.pop(pipe_id, None)
        if endpoint is not None:
            log.warning('repeated start')
            endpoint.stop()
        endpoint = EndpointIn(self._winusb, pipe_id, transfers,
                              block_size, data_fn, process_fn, stop_fn)
        self._endpoints[endpoint.pipe_id] = endpoint
        endpoint.start()
        self._update_event_list()

    def read_stream_stop(self, endpoint_id):
        log.info('read_stream_stop %d', endpoint_id)
        pipe_id = (endpoint_id & 0x7f) | 0x80
        endpoint = self._endpoints.pop(pipe_id, None)
        if endpoint is not None:
            endpoint.stop()
            self._update_event_list()

    def status(self):
        e = {}
        s = {'endpoints': e}
        for endpoint in self._endpoints.values():
            e[endpoint.pipe_id] = endpoint.status()
        return s

    def signal(self):
        kernel32.SetEvent(self._event)

    def _abort(self, stop_code, msg):
        for endpoint in self._endpoints.values():
            endpoint.stop()
        self._endpoints.clear()
        if self._control_transfer.stop_code is None:
            self._control_transfer.stop_code = DeviceEvent.ENDPOINT_CALLBACK_STOP
        self._update_event_list()
        event_callback_fn, self._event_callback_fn = self._event_callback_fn, None
        if callable(event_callback_fn):
            try:
                event_callback_fn(stop_code, msg)
            except Exception:
                log.exception('while in _event_callback_fn')

    def process(self, timeout):
        timeout_ms = int(timeout * 1000)
        rv = kernel32.WaitForMultipleObjects(self.event_list_count, self._event_list, False, timeout_ms)
        if rv < kernel32.MAXIMUM_WAIT_OBJECTS:
            kernel32.ResetEvent(self._event)
            stop_endpoint_ids = []
            for endpoint in self._endpoints.values():
                if endpoint.process():
                    stop_endpoint_ids.append(endpoint.pipe_id)
            for endpoint in self._endpoints.values():
                if endpoint.process_signal() or endpoint.stop_code is not None:
                    stop_endpoint_ids.append(endpoint.pipe_id)
            pipe_id = None
            for pipe_id in stop_endpoint_ids:
                endpoint = self._endpoints.pop(pipe_id, None)
                if endpoint is None:
                    continue  # duplicate
                endpoint.stop()
                self._update_event_list()
                msg = 'Endpoint pipe_id %r stopped: %s' % (pipe_id, endpoint.stop_code)
                log.info(msg)
                if endpoint.stop_code > 0:
                    self._abort(endpoint.stop_code, msg)
            if self._control_transfer is not None:
                self._control_transfer.process()
                if self._control_transfer.stop_code is not None and self._control_transfer.stop_code > 0:
                    msg = 'Control pipe %r stopped: %s' % (pipe_id, self._control_transfer.stop_code)
                    log.info(msg)
                    self._abort(self._control_transfer.stop_code, msg)


def scan(name: str) -> List[WinUsbDevice]:
    """Scan for attached devices.

    :param name: The case-insensitive name of the device to scan.
    :return: The list of discovered :class:`WinUsbDevice` instances.
    """
    paths = []
    infos = INFO[name.lower()]
    for info in infos:
        guid = info['DeviceInterfaceGUID']
        p = device_interface_guid_to_paths(guid)
        paths.extend(p)
    if not len(paths):
        log.debug('scan found no devices')
        return []
    log.debug('scan found %s' % paths)
    devices = [WinUsbDevice(p) for p in paths]
    return devices
