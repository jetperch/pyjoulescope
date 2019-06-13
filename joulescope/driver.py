# Copyright 2018-2019 Jetperch LLC
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
from . data_recorder import DataRecorder, construct_record_filename
from joulescope.stream_buffer import StreamBuffer, stats_to_api
from joulescope.calibration import Calibration
import struct
import copy
import time
import json
import os
import queue
import numpy as np
import binascii
from typing import List
import logging
log = logging.getLogger(__name__)

STATUS_REQUEST_LENGTH = 128
EXTIO_REQUEST_LENGTH = 128
SERIAL_NUMBER_LENGTH = 16
HOST_API_VERSION = 1
CALIBRATION_SIZE_MAX = 0x8000
PACKET_VERION = 1


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

    EXTIO = 9
    """Get/set the external GPI/O settings."""

    INFO = 10
    """Get the current device information metadata JSON string."""

    TEST_MODE = 11
    """Enter a test mode."""


class SensorBootloader:

    START = 1
    RESUME = 2
    ERASE = 3
    WRITE = 4


class PacketType:
    SETTINGS = 1
    STATUS = 2
    EXTIO = 3
    INFO = 4


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
        self.stream_buffer = None
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
        """Get the list of :class:`joulescope.parameter.Parameter` instances.

        :param name: The optional name of the parameter to retrieve.
            None (default) returns a list of all parameters.
        :return: The list of all parameters.  If name is provided, then just
            return that single parameters.
        """
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
        elif p.path == 'extio':
            self._extio_set()

    def parameter_get(self, name):
        """Get a parameter value.

        :param name: The parameter name.
        :raise KeyError: if name not found.
        """
        value = self._parameters[name]
        return value_to_name(name, value)

    @property
    def serial_number(self):
        """Get the unique 16-byte LPC54608 microcontroller serial number.

        :return: The microcontroller serial number.

        The serial number assigned during manufacturing is available using
        self.info()['ctl']['hw']['sn_mfg'].
        """
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

    def open(self, event_callback_fn=None):
        """Open the device for use

        :param event_callback_fn: The function(event, message) to call on
            asynchronous events, mostly to allow robust handling of device
            errors.  "event" is one of the :class:`DeviceEvent` values,
            and the message is a more detailed description of the event.

        :raise IOError: on failure.

        The event_callback_fn may be called asynchronous and from other
        threads.  The event_callback_fn must implement any thread safety.
        """
        self._usb.open(event_callback_fn)
        sb_len = self._sampling_frequency * STREAM_BUFFER_DURATION
        self.stream_buffer = StreamBuffer(sb_len, self._reductions)
        try:
            info = self.info()
            if info is not None:
                log.info('info:\n%s', json.dumps(info, indent=2))
        except Exception:
            log.warning('could not fetch info record')
        self.calibration = self._calibration_read()
        self.view = View(self)

    def info(self):
        """Get the device information structure.

        :return: The device information structure.

        First implemented in 0.3.  Older firmware returns None.
        """
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.INFO,
            value=0, index=0, length=1024)
        if 0 != rv.result:  # firmware prior to 0.3
            return None
        if len(rv.data) < 8:
            log.warning('info record too short')
            return None
        version, hdr_length, pdu_type = struct.unpack('<BBB', rv.data[:3])
        if version != HOST_API_VERSION:
            log.warning('info msg API version mismatch: %d != %d' % (version, HOST_API_VERSION))
            return None
        if pdu_type != PacketType.INFO:
            log.warning('info msg pdu_type mismatch: %d != %d' % (pdu_type, PacketType.INFO))
            return None
        if hdr_length != len(rv.data):
            log.warning('info msg length mismatch: %d != %d' % (hdr_length, len(rv.data)))
            return None
        try:
            return json.loads(rv.data[8:].decode('utf-8'))
        except UnicodeDecodeError:
            log.exception('INFO has invalid unicode: %s', binascii.hexlify(rv.data[8:]))
        except json.decoder.JSONDecodeError:
            log.exception('Could not decode INFO: %s', rv.data[8:])
        return None

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
            log.info('calibration = %s', binascii.hexlify(rv.data))
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
        self.stream_buffer = None

    def _wait_for_sensor_command(self, timeout=None):
        timeout = 1.0 if timeout is None else float(timeout)
        time_start = time.time()
        while True:
            dt = time.time() - time_start
            if dt > timeout:
                raise RuntimeError('timed out')
            s = self._status()
            if 0 != s['return_code']['value']:
                log.warning('Error while getting status: %s', s['return_code']['str'])
                time.sleep(0.4)
                continue
            rv = s.get('settings_result', {}).get('value', -1)
            if rv in [-1, 19]:
                time.sleep(0.010)
                continue
            return rv

    def _stream_settings_send(self):
        length = 16
        if self._streaming:
            streaming = self._parameters['control_test_mode']
        else:
            streaming = 0
        options = (self._parameters['v_range'] << 1) | self._parameters['ovr_to_lsb']
        msg = struct.pack('<BBBBIBBBBBBBB',
                          PACKET_VERION,
                          length,
                          PacketType.SETTINGS,
                          0,  # rsvl (1 byte)
                          0,  # rsv4 (4 byte)
                          self._parameters['sensor_power'],
                          self._parameters['i_range'],  # select
                          self._parameters['source'],
                          options,
                          streaming,
                          0,  # rsv1_u8
                          0,  # rsv2_u8
                          0   # rsv3_u8
                          )
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SETTINGS,
            value=0, index=0, data=msg)
        _ioerror_on_bad_result(rv)
        if streaming == 0:
            self._wait_for_sensor_command()

    def _extio_set(self):
        msg = struct.pack('<BBBBIBBBBBBBBII',
                          PACKET_VERION,
                          24,
                          PacketType.EXTIO,
                          0,  # hdr_rsv1
                          0,  # hdr_rsv4
                          0,  # flags
                          self._parameters['trigger_source'],
                          self._parameters['current_gpi'],
                          self._parameters['voltage_gpi'],
                          self._parameters['gpo0'],
                          self._parameters['gpo1'],
                          0,  # uart_tx mapping reserved
                          0,  # rsv1_u8
                          0,  # rsv3_u32, baudrate reserved
                          self._parameters['io_voltage'],
        )
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.EXTIO,
            value=0, index=0, data=msg)
        _ioerror_on_bad_result(rv)

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
        self.view.clear()
        self._stop_fn = stop_fn
        if duration is not None:
            self.stream_buffer.sample_id_max = int(duration * self.sampling_frequency)
        if contiguous_duration is not None:
            c = int(contiguous_duration * self.sampling_frequency)
            c += self._reductions[0]
            self.stream_buffer.contiguous_max = c
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
            try:
                self._stream_settings_send()
            except:
                log.warning('Device.stop() while attempting _stream_settings_send')
            try:
                self.recording_stop()
            except:
                log.warning('Device.stop() while attempting recording_stop')
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
            return self.stream_buffer.raw_get(start_id, end_id).reshape((-1, 2))
        else:
            r = self.stream_buffer.data_get(start_id, end_id, increment=1)
            i = r[:, 0, 0].reshape((-1, 1))
            v = r[:, 1, 0].reshape((-1, 1))
            return np.hstack((i, v))

    def recording_start(self, filename=None):
        """Begin recording to a file.

        :param filename: The target filename or file-like object for the
            recording.  None (default) constructs a filename in the
            default path.
        """
        self.recording_stop()
        log.info('recording_start(%s)', filename)
        if filename is None:
            filename = construct_record_filename()
        self._data_recorder = DataRecorder(
            filename,
            calibration=self.calibration.data,
            sampling_frequency=self._sampling_frequency)
        # todo save voltage gain to _data_recorder
        return True

    def recording_stop(self):
        """Stop recording to a file."""
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
        def _status_error(ec, msg_str):
            log.warning('status failed %d: %s', ec, msg_str)
            return {'return_code': {'value': ec, 'str': msg_str, 'units': ''}}

        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.STATUS,
            value=0, index=0, length=STATUS_REQUEST_LENGTH)
        if 0 != rv.result:
            s = 'usb control transfer failed: {}'.format(usb.get_error_str(rv.result))
            return _status_error(rv.result, s)
        pdu = bytes(rv.data)
        expected_length = 8 + 16
        if len(pdu) < expected_length:
            msg = 'status msg pdu too small: %d < %d' % (len(pdu), expected_length)
            return _status_error(1, msg)
        version, hdr_length, pdu_type = struct.unpack('<BBB', pdu[:3])
        if version != HOST_API_VERSION:
            msg = 'status msg API version mismatch: %d != %d' % (version, HOST_API_VERSION)
            return _status_error(1, msg)
        if pdu_type != PacketType.STATUS:
            msg = 'status msg pdu_type mismatch: %d != %d' % (pdu_type, PacketType.STATUS)
            return _status_error(1, msg)
        if hdr_length != expected_length:
            msg = 'status msg length mismatch: %d != %d' % (hdr_length, expected_length)
            return _status_error(1, msg)
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
            'return_code': {
                'value': 0,
                'format': '{}',
                'units': '',
            },
        }
        for key, value in status.items():
            value['name'] = key
        return status

    def status(self):
        """Get the current device status.

        :return: A dict containing status information.
        """
        status = self._usb.status()
        status['driver'] = self._status()
        status['buffer'] = self.stream_buffer.status()
        return status

    def _sensor_status_check(self):
        rv = self._status()
        ec = rv.get('settings_result', {}).get('value', 1)
        if 0 != ec:
            raise RuntimeError('sensor_firmware_program failed %d' % (ec,))

    def extio_status(self):
        """Read the EXTIO GPI value.

        :return: A dict containing the extio status.  Each key is the status
            item name.  The value is itself a dict with the following keys:

            * name: The status name, which is the same as the top-level key.
            * value: The actual value
            * units: The units, if applicable.
            * format: The recommended formatting string (optional).
        """
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.EXTIO,
            value=0, index=0, length=EXTIO_REQUEST_LENGTH)
        if 0 != rv.result:
            s = usb.get_error_str(rv.result)
            log.warning('usb control transfer failed: %s', s)
            return {'return_code': {'value': rv.result, 'str': s, 'units': ''}}
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
        if pdu_type != PacketType.EXTIO:
            return {}
        if hdr_length != expected_length:
            log.warning('status msg length mismatch: %d != %d',
                        hdr_length, expected_length)
            return {}
        values = struct.unpack('<BBBBBBBBII', pdu[8:24])
        status = {
            'flags': {
                'value': values[0],
                'units': ''},
            'trigger_source': {
                'value': values[1],
                'units': ''},
            'current_gpi': {
                'value': values[2],
                'units': ''},
            'voltage_gpi': {
                'value': values[3],
                'units': ''},
            'gpo0': {
                'value': values[4],
                'units': ''},
            'gpo1': {
                'value': values[5],
                'units': ''},
            'gpi_value': {
                'value': values[7],
                'units': '',
            },
            'io_voltage': {
                'value': values[9],
                'units': 'mV',
            },
        }
        for key, value in status.items():
            value['name'] = key
        return status

    def statistics_get(self, t1, t2):
        """Get the statistics for the collected sample data over a time range.

        :param t1: The starting time in seconds relative to the streaming start time.
        :param t2: The ending time in seconds.
        :return: The statistics data structure.  Here is an example:

            {
              "time": {
                "range": [4.2224105, 4.7224105],
                "delta": 0.5,
                "units": "s"
              },
              "signals": {
                "current": {
                  "statistics": {
                    "μ": 1.1410409683776379e-07,
                    "σ": 3.153094851882088e-08,
                    "min": 2.4002097531727884e-10,
                    "max": 2.77493541034346e-07,
                    "p2p": 2.772535200590287e-07
                  },
                  "units": "A",
                  "integral_units": "C"
                },
                "voltage": {
                  "statistics": {
                    "μ": 3.2984893321990967,
                    "σ": 0.0010323672322556376,
                    "min": 3.293551445007324,
                    "max": 3.3026282787323,
                    "p2p": 0.009076833724975586
                  },
                  "units": "V",
                  "integral_units": null
                },
                "power": {
                  "statistics": {
                    "μ": 3.763720144434046e-07,
                    "σ": 1.0400773930996365e-07,
                    "min": 7.916107769290193e-10,
                    "max": 9.155134534921672e-07,
                    "p2p": 9.147218427152382e-07
                  },
                  "units": "W",
                  "integral_units": "J"
                }
              }
            }
        """
        v = self.view
        s1 = v.view_time_to_sample_id(t1)
        s2 = v.view_time_to_sample_id(t2)
        log.info('buffer %s, %s => %s, %s : %s', t1, t2, s1, s2, v.span)
        d = self.stream_buffer.stats_get(start=s1, stop=s2)
        t_start = s1 / self.sampling_frequency
        t_stop = s2 / self.sampling_frequency
        return stats_to_api(d, t_start, t_stop)

    def sensor_firmware_program(self, data, progress_cbk=None):
        """Program the sensor microcontroller firmware

        :param data: The firmware to program as a raw binary file.
        :param progress_cbk:  The optional Callable[[float], None] which is called
            with the progress fraction from 0.0 to 1.0
        :raise: on error.
        """
        log.info('sensor_firmware_program')
        if progress_cbk is None:
            progress_cbk = lambda x: None
        self.stop()

        log.info('sensor bootloader: start')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.START)
        _ioerror_on_bad_result(rv)
        self._sensor_status_check()

        log.info('sensor bootloader: erase all flash')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.ERASE)
        _ioerror_on_bad_result(rv)
        self._sensor_status_check()

        log.info('sensor bootloader: program')
        total_size = len(data)
        chunk_size = 2 ** 10  # 16 kB
        assert(0 == (chunk_size % 256))
        index = 0
        while len(data):
            sz = chunk_size if len(data) > chunk_size else len(data)
            fraction_done = (index * 256) / total_size
            progress_cbk(fraction_done)
            log.info('sensor bootloader: program chunk index=%d, sz=%d | %.1f%%', index, sz, fraction_done * 100)
            rv = self._usb.control_transfer_out(
                'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
                value=SensorBootloader.WRITE, index=index, data=data[:sz])
            _ioerror_on_bad_result(rv)
            self._sensor_status_check()
            data = data[sz:]
            index += chunk_size // 256
        log.info('sensor bootloader: resume')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.RESUME)
        progress_cbk(1.0)
        _ioerror_on_bad_result(rv)

    def bootloader(self):
        """Start the bootloader for this device.
        
        :return: (bootloader, existing_devices)  Use the bootloader instance
            to perform operations.  Use existing_devices to assist in 
            determining when this device returns from bootloader mode.
        """
        _, existing_devices, _ = scan_for_changes(name='Joulescope', devices=[self])
        existing_bootloaders = scan(name='bootloader')
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
            _, b, _ = scan_for_changes(name='bootloader', devices=existing_bootloaders)
        time.sleep(0.1)
        _, b, _ = scan_for_changes(name='bootloader', devices=existing_bootloaders)
        b = b[0]
        b.open()
        return b, existing_devices
        
    def run_from_bootloader(self, fn):
        """Run commands from the bootloader and then return to the app.

        :param fn: The function(bootloader) to execute the commands.
        """
        b, existing_devices = self.bootloader()
        try:
            rc = fn(b)
        finally:
            b.go()  # go closes bootloader automatically
            d = []
            while not len(d):
                time.sleep(0.1)
                _, d, _ = scan_for_changes(name='Joulescope', devices=existing_devices)
            time.sleep(0.5)
            self._usb = d[0]._usb
            self.open()
        return rc

    def controller_firmware_program(self, data, progress_cbk=None):
        return self.run_from_bootloader(lambda b: b.firmware_program(data, progress_cbk))

    def calibration_program(self, data, is_factory=False):
        return self.run_from_bootloader(lambda b: b.calibration_program(data, is_factory))

    def enter_test_mode(self, index=None, value=None):
        """Enter a custom test mode.

        :param index: The test mode index.
        :param value: The test mode value.

        You probably should not be using this method.  You will not destroy
        anything, but you will likely stop your Joulescope from working
        normally until you unplug it.
        """
        index = 0 if index is None else int(index)
        value = 0 if value is None else int(value)
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.TEST_MODE,
            index=index,
            value=value)
        _ioerror_on_bad_result(rv)


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

    def clear(self):
        self.changed = True
        self.data_idx = 0
        if self.data is not None:
            self.data[:, :, :] = np.nan

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
                x_range, gain=kwargs.get('gain'), pivot=kwargs.get('pivot'))
        elif cmd == 'span_pan':
            delta = kwargs.get('delta', 0.0)
            x_range = [x_range[0] + delta, x_range[-1] + delta]
            x_range, self.samples_per, self.x = self.span.conform_discrete(x_range)
        elif cmd == 'refresh':
            log.warning('on_x_change(refresh)')
            self.changed = True
            return
        else:
            log.warning('on_x_change(%s) unsupported', cmd)
            return

        if self._device.is_streaming:
            x_max = self.span.limits[1]
            if x_range[1] < x_max:
                x_shift = x_max - x_range[1]
                x_range = [x_range[0] + x_shift, x_max]
            x_range, self.samples_per, self.x = self.span.conform_discrete(x_range)

        self.changed |= (self.x_range != x_range)
        self.clear()
        self.x_range = x_range
        log.info('changed=%s, length=%s, span=%s, range=%s, samples_per=%s',
                 self.changed, len(self), self.x_range,
                 self.x_range[1] - self.x_range[0], self.samples_per)

    def _view(self):
        buffer = self._device.stream_buffer
        _, sample_id_end = buffer.sample_id_range
        lag_time = self.span.limits[1] - self.x_range[1]
        lag_samples = int(lag_time * self.sampling_frequency) // self.samples_per
        data_idx_stream_end = sample_id_end // self.samples_per
        data_idx_view_end = data_idx_stream_end - lag_samples
        sample_id_end = data_idx_view_end * self.samples_per
        delta = data_idx_view_end - self.data_idx
        return data_idx_view_end, sample_id_end, delta

    def view_time_to_sample_id(self, t):
        idx_start, idx_end = self._device.stream_buffer.sample_id_range
        t_start, t_end = self.span.limits
        if not t_start <= t <= t_end:
            return None
        dx_end = t_end - t
        dx_idx_end = int(dx_end * self.sampling_frequency)
        s = idx_end - dx_idx_end
        return s

    def update(self):
        buffer = self._device.stream_buffer
        length = len(self)
        data_idx_view_end, sample_id_end, delta = self._view()

        if not self.changed and 0 == delta:
            return False, (self.x, self.data)
        if self.changed or delta >= length:
            self.data[:, :, :] = np.nan
            if data_idx_view_end > 0:
                start_idx = (data_idx_view_end - length) * self.samples_per
                # log.info('recompute(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
                buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data)
        elif data_idx_view_end > 0:
            start_idx = self.data_idx * self.samples_per
            # log.debug('update(start=%s, stop=%s, increment=%s)', start_idx, sample_id_end, self.samples_per)
            self.data = np.roll(self.data, -delta, axis=0)
            buffer.data_get(start_idx, sample_id_end, self.samples_per, self.data[-delta:, :, :])
        self.data_idx = data_idx_view_end
        self.changed = False
        return True, (self.x, self.data)

    def extract(self):
        buffer = self._device.stream_buffer
        length = len(self)
        data_idx_view_end, sample_id_end, delta = self._view()
        start_idx = (data_idx_view_end - length) * self.samples_per
        return buffer.data_get(start_idx, sample_id_end)


def scan(name: str=None) -> List[Device]:
    """Scan for connected devices.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :return: The list of :class:`Device` instances.  A new instance is created
        for each detected device.  Use :func:`scan_for_changes` to preserved
        existing instances.
    """
    if name is None:
        name = 'joulescope'
    devices = usb.scan(name)
    if name == 'bootloader':
        devices = [bootloader.Bootloader(d) for d in devices]
    else:
        devices = [Device(d) for d in devices]
    return devices


def scan_require_one(name: str=None) -> Device:
    """Scan for one and only one device.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :return: The :class:`Device` found.
    :raise RuntimeError: If no devices or more than one device was found.
    """
    devices = scan(name)
    if not len(devices):
        raise RuntimeError("no devices found")
    if len(devices) > 1:
        raise RuntimeError("multiple devices found")
    return devices[0]


def scan_for_changes(name: str=None, devices=None):
    """Scan for device changes.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
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
    devices_next = scan(name)
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

    log.info('scan_for_changes %d devices: %d added, %d removed',
             len(devices_now), len(devices_added), len(devices_removed))
    return devices_now, devices_added, devices_removed
