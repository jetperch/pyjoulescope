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

from joulescope import usb
from joulescope import JOULESCOPE_DIR
from joulescope import span
from joulescope.usb.device_thread import DeviceThread
from .parameters_v1 import PARAMETERS, PARAMETERS_DICT, name_to_value, value_to_name
from . import datafile
from . import bootloader
from . data_recorder import DataRecorder, construct_record_filename, DataRecorderConfiguration
from joulescope.stream_buffer import StreamBuffer
from joulescope.calibration import Calibration
import struct
import copy
import time
import os
import queue
import numpy as np
import binascii
from typing import List
import logging
log = logging.getLogger(__name__)

DeviceInterfaceGUID = '{99a06894-3518-41a5-a207-8519746da89f}'
VENDOR_ID = 0x1FC9
PRODUCT_ID = 0xFC93
STATUS_REQUEST_LENGTH = 128
SERIAL_NUMBER_LENGTH = 16
HOST_API_VERSION = 1
CALIBRATION_SIZE_MAX = 0x8000


def _ioerror_on_bad_result(rv):
    if 0 != rv.result:
        raise IOError('usb.Device %s' % (rv,))


class UsbdRequest:

    LOOPBACK_WVALUE = 1
    """USB testing for 16-bit value, use wValue"""

    LOOPBACK_BUFFER = 2
    """USB testing for messages up to USBD_APP_LOOPBACK_BUFFER_SIZE bytes"""

    SETTINGS = 3
    """"Configure operation, including starting streaming"""

    STATUS = 4
    """Get current status (GET only)"""

    SENSOR_BOOTLOADER = 5
    """Sensor bootloader operations"""

    CONTROLLER_BOOTLOADER = 6
    """Request reboot into the controller bootloader"""

    SERIAL_NUMBER = 7
    """Request the 16-bit unique serial number."""

    CALIBRATION = 8
    """Request the calibration. wIndex 0=factory, 1=active."""


class SensorBootloader:

    START = 1
    RESUME = 2
    ERASE = 3
    WRITE = 4


class PacketType:
    SETTINGS = 1
    STATUS = 2


LOOPBACK_BUFFER_SIZE = 132
"""The maximum size of the hardware control loopback buffer for testing."""

SAMPLING_FREQUENCY = 2000000  # samples per second (Hz)
REDUCTIONS = [200, 100, 50]   # in samples in sample units of the previous reduction
STREAM_BUFFER_DURATION = 30   # seconds


class Device:
    """The device implementation for use by applications.

    :param usb_device: The backend USB :class:`usb.device` instance.
    """
    def __init__(self, usb_device):
        os.makedirs(JOULESCOPE_DIR, exist_ok=True)
        self._usb = DeviceThread(usb_device)
        self._parameters = {}
        self._reductions = REDUCTIONS
        self._sampling_frequency = SAMPLING_FREQUENCY
        sb_len = self._sampling_frequency * STREAM_BUFFER_DURATION
        self.stream_buffer = StreamBuffer(sb_len, self._reductions)
        self.view = None  #
        self._streaming = False
        self._stop_fn = None
        self._data_recorder = None
        self.calibration = None
        self._statistics_callback = None
        for p in PARAMETERS:
            if p.permission == 'rw':
                self._parameters[p.name] = name_to_value(p.name, p.default)

    def __str__(self):
        return str(self._usb)

    @property
    def usb_device(self):
        """Get the USB backend device implementation.

        This method should only be used for unit and system tests.  Production
        code should *NEVER* access the underlying USB device directly.
        """
        return self._usb

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def statistics_callback(self):
        return self.stream_buffer.callback

    @statistics_callback.setter
    def statistics_callback(self, cbk):
        """Set the statistics callback.

        :param cbk: The callable(stats, energy) where stats is
            a np.array((3, 4)) of [current, voltage, power]
            [mean, variance, min, max].
        """
        idx = len(self._reductions)
        if idx:
            self.stream_buffer.callback = cbk

    def parameters(self, name=None):
        """Get the list of :class:`joulescope.parameter.Parameter` instances"""
        if name is not None:
            for p in PARAMETERS:
                if p.name == name:
                    return copy.deepcopy(p)
            return None
        return copy.deepcopy(PARAMETERS)

    def parameter_set(self, name, value):
        """Set a parameter value.

        :param name: The parameter name
        :param value: The new parameter value
        :raise KeyError: if name not found.
        :raise ValueError: if value is not allowed
        """
        value = name_to_value(name, value)
        self._parameters[name] = value
        p = PARAMETERS_DICT[name]
        if p.path == 'setting':
            self._stream_settings_send()

    def parameter_get(self, name):
        """Get a parameter value.

        :param name: The parameter name.
        :raise KeyError: if name not found.
        """
        value = self._parameters[name]
        return value_to_name(name, value)

    @property
    def serial_number(self):
        """Get the unique 16-byte LPC54608 serial number."""
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.SERIAL_NUMBER,
            value=0, index=0, length=SERIAL_NUMBER_LENGTH)
        if 0 != rv.result:
            log.warning('usb control transfer failed %d', rv.result)
            return {}
        sn = bytes(rv.data)
        serial_number = binascii.hexlify(sn).decode('utf-8')
        log.info('serial number = %s', serial_number)
        return serial_number

    def open(self):
        """Open the device for use

        :raise IOError: on failure.
        """
        self.stream_buffer.reset()
        self._usb.open()
        self.calibration = self._calibration_read()
        self.view = View(self)

    def _calibration_read_raw(self, factory=None):
        value = 0 if bool(factory) else 1
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.CALIBRATION,
            value=value, index=0, length=datafile.HEADER_SIZE)
        if 0 != rv.result:
            log.warning('calibration_read transfer failed %d', rv.result)
            return None
        try:
            length, _ = datafile.validate_file_header(bytes(rv.data))
        except Exception:
            log.warning('invalid calibration file')
            return None

        calibration = b''
        offset = 0
        while offset < length:
            # note: can only transfer 4096 (0x1000) bytes in one transfer
            # https://docs.microsoft.com/en-us/windows/desktop/api/winusb/nf-winusb-winusb_controltransfer
            k = 4096
            if k > length:
                k = length
            rv = self._usb.control_transfer_in(
                'device', 'vendor',
                request=UsbdRequest.CALIBRATION,
                value=value, index=offset, length=k)
            if 0 != rv.result:
                log.warning('calibration_read transfer failed %d', rv.result)
                return None
            chunk = bytes(rv.data)
            offset += len(chunk)
            calibration += chunk
        return calibration

    def _calibration_read(self) -> Calibration:
        cal = Calibration()
        serial_number = self.serial_number
        cal.serial_number = serial_number
        try:
            cal_data = self._calibration_read_raw()
            if cal_data is None:
                log.info('no calibration present')
            else:
                cal.load(cal_data)
        except (ValueError, IOError):
            log.info('failed reading calibration')
        if cal.serial_number != serial_number:
            log.info('calibration serial number mismatch')
            return None
        self.calibration = cal
        self.stream_buffer.calibration_set(cal.current_offset, cal.current_gain, cal.voltage_offset, cal.voltage_gain)
        return cal

    def close(self):
        """Close the device and release resources"""
        try:
            self.stop()
        except:
            log.exception('USB stop failed')
        try:
            self._usb.close()
        except:
            log.exception('USB close failed')
        self.view = None

    def _wait_for_sensor_command(self, timeout=None):
        timeout = 2.0 if timeout is None else float(timeout)
        time_start = time.time()
        while True:
            s = self._status()
            rv = s.get('settings_result', {}).get('value', -1)
            if rv in [-1, 19]:
                dt = time.time() - time_start
                if dt >= timeout:
                    raise RuntimeError('timed out')
                time.sleep(0.005)
                continue
            return rv

    def _stream_settings_send(self):
        version = 1
        length = 16
        if self._streaming:
            streaming = self._parameters['control_test_mode']
        else:
            streaming = 0
        options = (self._parameters['v_range'] << 1) | self._parameters['ovr_to_lsb']
        msg = struct.pack('<BBBBIBBBBBBBB',
                          version,
                          length,
                          PacketType.SETTINGS,
                          0,  # reserved
                          0,  # reserved
                          self._parameters['sensor_power'],
                          self._parameters['i_range'],
                          self._parameters['source'],
                          options,
                          streaming,
                          0, 0, 0)
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SETTINGS,
            value=0, index=0, data=msg)
        _ioerror_on_bad_result(rv)
        if streaming == 0:
            self._wait_for_sensor_command()

    def _on_data(self, data):
        # invoked from USB thread
        is_done = True
        if data is not None:
            is_done = self.stream_buffer.insert(data)
        if is_done:
            stop_fn, self._stop_fn = self._stop_fn, None
            if callable(stop_fn):
                stop_fn()
        return is_done

    def _on_process(self):
        """Perform data processing.

        WARNING: called from USB thread!
        Return True to stop streaming.
        """
        self.stream_buffer.process()
        if self._data_recorder is not None:
            self._data_recorder.process(self.stream_buffer)
        return False

    def start(self, stop_fn=None, duration=None, contiguous_duration=None):
        """Start data streaming

        :param stop_fn: The function() called when the device stops.  The
            device can stop "automatically" on errors.
            Call :meth:`read_stream_stop` to stop from
            the caller.  This function will be called from the USB
            processing thread.  Any calls back into self MUST BE
            resynchronized.

        :param duration: The duration in seconds for the capture.
        :param contiguous_duration: The contiguous duration in seconds for
            the capture.  As opposed to duration, this ensures that the
            duration has no missing samples.  Missing samples usually
            occur when the device first starts.

        If streaming was already in progress, it will be restarted.
        """
        if self._streaming:
            self.stop()
        self.stream_buffer.reset()
        self._stop_fn = stop_fn
        if duration is not None:
            self.stream_buffer.sample_id_max = int(duration * self.sampling_frequency)
        if contiguous_duration is not None:
            c = int(contiguous_duration * self.sampling_frequency)
            c += self._reductions[0]
            self.stream_buffer._contiguous_max = c
            log.info('contiguous_samples=%s', c)
        self._streaming = True
        self._stream_settings_send()
        self._usb.read_stream_start(
            endpoint_id=2,
            transfers=self._parameters['transfer_outstanding'],
            block_size=self._parameters['transfer_length'] * usb.BULK_IN_LENGTH,
            data_fn=self._on_data,
            process_fn=self._on_process)
        return True

    def stop(self):
        """Stop data streaming.

        :return: True if stopped.  False if was already stopped.

        This method is always safe to call, even after the device has been
        stopped or removed.
        """
        log.info('stop')
        if self._streaming:
            self._usb.read_stream_stop(2)
            self._streaming = False
            self._stream_settings_send()
            self.recording_stop()
            return True
        return False

    def read(self, duration=None, contiguous_duration=None, out_format=None):
        """Read data from the device

        :param duration: The duration in seconds for the capture.
        :param contiguous_duration: The contiguous duration in seconds for
            the capture.  As opposed to duration, this ensures that the
            duration has no missing samples.  Missing samples usually
            occur when the device first starts.
        :param out_format: The output format which is one of
            ['raw', 'calibrated', None].
            None (default) is the same as 'calibrated'.

        If streaming was already in progress, it will be restarted.
        """
        log.info('read(duration=%s, contiguous_duration=%s, out_format=%s)',
                 duration, contiguous_duration, out_format)
        q = queue.Queue()

        def on_stop():
            log.info('received stop callback: pending stop')
            q.put(None)

        self.start(on_stop, duration=duration, contiguous_duration=contiguous_duration)
        q.get()
        self.stop()
        start_id, end_id = self.stream_buffer.sample_id_range
        log.info('%s, %s', start_id, end_id)
        if contiguous_duration is not None:
            start_id = end_id - int(contiguous_duration * self.sampling_frequency)
            if start_id < 0:
                start_id = 0
            log.info('%s, %s', start_id, end_id)
        if out_format == 'raw':
            return self.stream_buffer.raw(start_id, end_id).reshape((-1, 2))
        else:
            r = self.stream_buffer.get(start_id, end_id, increment=1)
            i = r[:, 0, 0].reshape((-1, 1))
            v = r[:, 1, 0].reshape((-1, 1))
            return np.hstack((i, v))

    def recording_start(self, filename=None):
        self.recording_stop()
        if not self._streaming:
            return False
        log.info('recording_start(%s)', filename)
        if filename is None:
            filename = construct_record_filename()
        c = DataRecorderConfiguration()
        c.sampling_frequency = self._sampling_frequency
        c.reductions = self._reductions
        c.sample_id_offset = self.stream_buffer.sample_id_range[1]
        self._data_recorder = DataRecorder(
            filename,
            configuration=c,
            calibration=self.calibration.data)
        # todo save voltage gain to _data_recorder
        return True

    def recording_stop(self):
        if self._data_recorder is not None:
            log.info('recording_stop')
            self._data_recorder.close()
            self._data_recorder = None

    @property
    def is_recording(self):
        """Check if the device is recording"""
        return self._streaming and self._data_recorder is not None

    @property
    def is_streaming(self):
        """Check if the device is streaming.

        :return: True if streaming.  False if not streaming.
        """
        return self._streaming

    def _status(self):
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.STATUS,
            value=0, index=0, length=STATUS_REQUEST_LENGTH)
        if 0 != rv.result:
            log.warning('usb control transfer failed %d', rv.result)
            return {}
        pdu = bytes(rv.data)
        expected_length = 8 + 16
        if len(pdu) < expected_length:
            log.warning('status msg pdu too small: %d < %d',
                        len(pdu), expected_length)
            return {}
        version, hdr_length, pdu_type = struct.unpack('<BBB', pdu[:3])
        if version != HOST_API_VERSION:
            log.warning('status msg API version mismatch: %d != %d',
                        version, HOST_API_VERSION)
            return {}
        if pdu_type != PacketType.STATUS:
            return {}
        if hdr_length != expected_length:
            log.warning('status msg length mismatch: %d != %d',
                        hdr_length, expected_length)
            return {}
        values = struct.unpack('<iIIBBBx', pdu[8:])
        status = {
            'settings_result': {
                'value': values[0],
                'units': ''},
            'fpga_frame_counter': {
                'value': values[1],
                'units': 'frames'},
            'fpga_discard_counter': {
                'value': values[2],
                'units': 'frames'},
            'sensor_flags': {
                'value': values[3],
                'format': '0x{:02x}',
                'units': ''},
            'sensor_i_range': {
                'value': values[4],
                'format': '0x{:02x}',
                'units': ''},
            'sensor_source': {
                'value': values[5],
                'format': '0x{:02x}',
                'units': ''},
        }
        for key, value in status.items():
            value['name'] = key
        return status

    def status(self):
        status = self._usb.status()
        status['driver'] = self._status()
        status['buffer'] = self.stream_buffer.status()
        return status

    def sensor_firmware_program(self, data):
        log.info('sensor_firmware_program')
        self.stop()
        log.info('sensor bootloader: start')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.START)
        _ioerror_on_bad_result(rv)
        log.info('sensor bootloader: erase all flash')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.ERASE)
        _ioerror_on_bad_result(rv)
        chunk_size = 2 ** 10  # 16 kB
        assert(0 == (chunk_size % 256))
        index = 0
        while len(data):
            sz = chunk_size if len(data) > chunk_size else len(data)
            log.info('sensor bootloader: program chunk index=%d, sz=%d', index, sz)
            rv = self._usb.control_transfer_out(
                'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
                value=SensorBootloader.WRITE, index=index, data=data[:sz])
            _ioerror_on_bad_result(rv)
            data = data[sz:]
            index += chunk_size // 256
        log.info('sensor bootloader: resume')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.RESUME)
        _ioerror_on_bad_result(rv)

    def run_from_bootloader(self, fn):
        """Run commands from the bootloader and then return to the app.

        :param fn: The function(bootloader) to execute the commands.
        """
        _, existing_devices, _ = scan_for_changes([self])
        existing_bootloaders = bootloader.scan()
        log.info('controller_firmware_upgrade')
        log.info('my_device = %s', str(self))
        log.info('existing_devices = %s', existing_devices)
        log.info('existing_bootloaders = %s', existing_bootloaders)
        self.stop()
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.CONTROLLER_BOOTLOADER,
            value=SensorBootloader.START)
        _ioerror_on_bad_result(rv)
        self.close()
        b = []
        while not len(b):
            time.sleep(0.1)
            _, b, _ = bootloader.scan_for_changes(existing_bootloaders)
        time.sleep(0.1)
        _, b, _ = bootloader.scan_for_changes(existing_bootloaders)
        b = b[0]
        b.open()
        fn(b)
        b.go()  # closes automatically
        d = []
        while not len(d):
            time.sleep(0.1)
            _, d, _ = scan_for_changes(existing_devices)
        time.sleep(0.5)
        self._usb = d[0]._usb
        self.open()

    def controller_firmware_program(self, data):
        return self.run_from_bootloader(lambda b: b.firmware_program(data))

    def calibration_program(self, data, is_factory=False):
        return self.run_from_bootloader(lambda b: b.calibration_program(data, is_factory))


class View:

    def __init__(self, device: Device):
        self._device = device  # which must be opened
        self.x = None
        self.data = None  # NxMx4 np.float32 [length][current, voltage, power][mean, var, min, max]
        x_max = len(self._device.stream_buffer) / self.sampling_frequency
        self.x_range = [x_max - 1.0, x_max]  # the current range
        self.samples_per = 1
        self.data_idx = 0
        self.span = span.Span(limits=[0.0, x_max],
                              quant=1.0/self.sampling_frequency,
                              length=100)
        self.changed = True

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    @property
    def sampling_frequency(self):
        return self._device.sampling_frequency

    def on_x_change(self, cmd, kwargs):
        x_range = list(self.x_range)
        if cmd == 'resize':  # {pixels: int}
            length = kwargs['pixels']
            if length is not None and length != len(self):
                log.info('resize %s', length)
                self.span.length = length
                self.data = np.full((length, 3, 4), np.nan, dtype=np.float32)
                self.changed = True  # invalidate
            x_range, self.samples_per, self.x = self.span.conform_discrete(x_range)
        elif cmd == 'span_absolute':  # {range: (start: float, stop: float)}]
            x_range, self.samples_per, self.x = self.span.conform_discrete(kwargs.get('range'))
        elif cmd == 'span_relative':  # {center: float, gain: float}]
            x_range, self.samples_per, self.x = self.span.conform_discrete(
                x_range, gain=kwargs.get('gain'), center=kwargs.get('center'))
        else:
            log.warning('on_x_change(%s) unsupported', cmd)
            return

        if True:  # only when tracking head
            self.x -= self.x[0]
        self.changed |= self.x_range != x_range
        self.x_range = x_range
        log.info('changed=%s, length=%s, span=%s, range=%s, samples_per=%s',
                 self.changed, len(self), self.x_range,
                 self.x_range[1] - self.x_range[0], self.samples_per)

    def update(self):
        # todo allow "lag" to shift view relative to head
        length = len(self)
        buffer = self._device.stream_buffer
        sample_id_start, sample_id_end = buffer.sample_id_range
        data_idx_start = (sample_id_start + self.samples_per - 1) // self.samples_per
        if self.data_idx < data_idx_start or self.changed:
            self.data[:, :, :] = np.nan
            self.data_idx = 0
        data_idx_end = sample_id_end // self.samples_per
        delta = data_idx_end - self.data_idx

        if not self.changed and 0 == delta:
            return False, (self.x, self.data)
        if delta >= length:
            start_idx = (data_idx_end - length) * self.samples_per
            if start_idx < 0:
                start_idx = 0
            # log.debug('recompute(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
            buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data)
        else:
            start_idx = self.data_idx * self.samples_per
            # log.debug('update(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
            self.data = np.roll(self.data, -delta, axis=0)
            buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data[-delta:, :, :])
        self.data_idx = data_idx_end
        self.changed = False
        return True, (self.x, self.data)


def scan() -> List[Device]:
    """Scan for connected devices.

    :return: The list of :class:`Device` instances.  A new instance is created
        for each detected device.  Use :func:`scan_for_changes` to preserved
        existing instances.
    """
    devices = usb.scan(DeviceInterfaceGUID)
    devices = [Device(d) for d in devices]
    return devices


def scan_require_one() -> Device:
    """Scan for one and only one device.

    :return: The :class:`Device` found.
    :raise RuntimeError: If no devices or more than one device was found.
    """
    devices = scan()
    if not len(devices):
        raise RuntimeError("no devices found")
    if len(devices) > 1:
        raise RuntimeError("multiple devices found")
    return devices[0]


def scan_for_changes(devices=None):
    """Scan for device changes.

    :param devices: The list of existing :class:`Device` instances returned
        by a previous scan.  Pass None or [] if no scan has yet been performed.
    :return: The tuple of lists (devices_now, devices_added, devices_removed).
        "devices_now" is the list of all currently connected devices.  If the
        device was in "devices", then return the :class:`Device` instance from
        "devices".
        "devices_added" is the list of connected devices not in "devices".
        "devices_removed" is the list of devices in "devices" but not "devices_now".
    """
    devices_prev = [] if devices is None else devices
    devices_next = scan()
    devices_added = []
    devices_removed = []
    devices_now = []

    for d in devices_next:
        matches = [x for x in devices_prev if str(x) == str(d)]
        if len(matches):
            devices_now.append(matches[0])
        else:
            devices_added.append(d)
            devices_now.append(d)

    for d in devices_prev:
        matches = [x for x in devices_next if str(x) == str(d)]
        if not len(matches):
            devices_removed.append(d)

    return devices_now, devices_added, devices_removed
