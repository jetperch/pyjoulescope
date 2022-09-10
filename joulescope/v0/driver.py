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

from joulescope.v0.usb.device_thread import DeviceThread
from joulescope.parameters_v1 import PARAMETERS, PARAMETERS_DICT, PARAMETERS_DEFAULTS, name_to_value, value_to_name
from . import bootloader, usb
from joulescope import datafile
from joulescope.v0.stream_buffer import StreamBuffer, DownsamplingStreamBuffer
from joulescope.v0.calibration import Calibration
from joulescope.view import View
import struct
import copy
import time
import json
import io
import queue
import binascii
from typing import List
import logging

log = logging.getLogger(__name__)

STATUS_REQUEST_LENGTH = 128
EXTIO_REQUEST_LENGTH = 128
SERIAL_NUMBER_LENGTH = 16
HOST_API_VERSION = 1
CALIBRATION_SIZE_MAX = 0x8000
PACKET_VERSION = 1
USB_RECONNECT_TIMEOUT_SECONDS = 15.0
SENSOR_COMMAND_TIMEOUT_SECONDS = 3.0


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


CALIBRATION_INDEX_TO_NAME = {
    0: 'factory',
    1: 'active',
}


CALIBRATION_INDEX = {
    None: 1,
    1: 1,
    'active': 1,
    0: 0,
    'factory': 0,
}


LOOPBACK_BUFFER_SIZE = 132
"""The maximum size of the hardware control loopback buffer for testing."""

SAMPLING_FREQUENCY = 2000000  # samples per second (Hz)


class StreamProcessApi:
    """This API is used to chain multiple processing callbacks that are called
    when new data is received from the Joulescope.

    The Joulescope driver will call notify() when new data is available.  It
    will call close() when streaming is stopped.
    """

    def start(self, stream_buffer):
        """Start a new streaming session. [optional]

        :param stream_buffer: The :class:`StreamBuffer` instance which contains
            the new data from the Joulescope.  This same instance will be passed
            to future :meth:`stream_notify` calls, too.

        This method will be called from the USB thread.  The processing
        duration must be very short to prevent dropping samples.  You
        should post any significant processing to a separate thread.
        """
        raise NotImplementedError()

    def stop(self):
        """Stop the existing streaming session. [optional]

        This method will be called from the USB thread.
        """
        raise NotImplementedError()

    def stream_notify(self, stream_buffer):
        """Notify that new data is available from the Joulescope.

        :param stream_buffer: The :class:`StreamBuffer` instance which contains
            the new data from the Joulescope.
        :return: False to continue streaming.  True to stop streaming.

        This method will be called from the USB thread.  The processing
        duration must be very short to prevent dropping samples.  You
        should post any significant processing to a separate thread.
        """
        raise NotImplementedError()

    def close(self):
        """Close the processing instance. [optional]

        This method may be called from the USB thread.
        """
        raise NotImplementedError()


def _reduction_frequency_to_reductions(frequency):
    if frequency < 1 or frequency > 100:
        raise ValueError(f'reduction_frequency {frequency} not between 1 Hz and 100 Hz')
    elif frequency < 50:
        return [200, 100, int(100 // frequency)]  # three reduction levels
    else:
        return [200, int((200 * 50) // frequency)]  # two reduction levels


def _statistics_unpack(pdu_section):
    values = struct.unpack('<qqqqiiiiiiiiiiii', pdu_section)
    samples_total, power_mean, charge, energy, samples_this, \
        samples_per_update, samples_per_second, \
        current_mean, current_min, current_max, \
        voltage_mean, voltage_min, voltage_max, \
        power_min, power_max, _ = values

    if not samples_this:
        return None

    s_start = (samples_total - samples_this)
    s_end = samples_total
    t_start = s_start / samples_per_second
    t_end = s_end / samples_per_second
    t_delta = t_end - t_start
    nan = float('nan')

    current_mean /= (1 << 27)
    current_min /= (1 << 27)
    current_max /= (1 << 27)
    voltage_mean /= (1 << 17)
    voltage_min /= (1 << 17)
    voltage_max /= (1 << 17)
    power_mean /= (1 << 34)
    power_min /= (1 << 21)
    power_max /= (1 << 21)
    charge /= (1 << 27)
    energy /= (1 << 27)

    s = {
        'time': {
            'range': {'value': [t_start, t_end], 'units': 's'},
            'delta': {'value': t_delta, 'units': 's'},
            'samples': {'value': samples_this, 'units': 'samples'},
        },
        'signals': {
            'current': {
                'µ': {'value': current_mean, 'units': 'A'},
                'σ2': {'value': nan, 'units': 'A'},
                'min': {'value': current_min, 'units': 'A'},
                'max': {'value': current_max, 'units': 'A'},
                'p2p': {'value': current_max - current_min, 'units': 'A'},
                '∫': {'value': current_mean * t_delta, 'units': 'C'},
            },
            'voltage': {
                'µ': {'value': voltage_mean, 'units': 'V'},
                'σ2': {'value': nan, 'units': 'V'},
                'min': {'value': voltage_min, 'units': 'V'},
                'max': {'value': voltage_max, 'units': 'V'},
                'p2p': {'value': voltage_max - voltage_min, 'units': 'V'},
            },
            'power': {
                'µ': {'value': power_mean, 'units': 'W'},
                'σ2': {'value': nan, 'units': 'W'},
                'min': {'value': power_min, 'units': 'W'},
                'max': {'value': power_max, 'units': 'W'},
                'p2p': {'value': power_max - power_min, 'units': 'W'},
                '∫': {'value': power_mean * t_delta, 'units': 'J'},
            },
        },
        'accumulators': {
            'charge': {
                'value': charge,
                'units': 'C',
            },
            'energy': {
                'value': energy,
                'units': 'J',
            },
        },
        'source': 'sensor',
    }
    return s


class Device:
    """The device implementation for use by applications.

    :param usb_device: The backend USB :class:`usb.device` instance.
    :param config: The initial default configuration following device open.
        Choices are ['auto', 'off', 'ignore', None].
        * 'auto': enable the sensor and start collecting data with
        current sensor autoranging.
        * 'ignore' or None: Leave the device in its existing state.
        * 'off': Turn the sensor off and disable data collection.
    """

    def __init__(self, usb_device, config=None):
        self._usb = DeviceThread(usb_device)
        self._config = config
        self._parameters = {}
        self._input_sampling_frequency = SAMPLING_FREQUENCY
        self._output_sampling_frequency = SAMPLING_FREQUENCY  # cached on open
        self._statistics_callbacks = {'stream_buffer': [], 'sensor': []}
        self._statistics_offsets = {'stream_buffer': [], 'sensor': []}
        self.stream_buffer = None
        self._streaming = False
        self._stop_fn = None
        self._process_objs = []  #: list of :class:`StreamProcessApi` compatible instances
        self._process_objs_add = []  #: list of :class:`StreamProcessApi` compatible instances
        self.calibration = None
        self._parameters_defaults = PARAMETERS_DEFAULTS
        for p in PARAMETERS:
            if 'read_only' not in p.flags and p.default is not None:
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
    def input_sampling_frequency(self):
        """The original input sampling frequency."""
        return self._input_sampling_frequency

    @property
    def output_sampling_frequency(self):
        """The output sampling frequency."""
        return self.parameter_get('sampling_frequency', dtype='actual')

    @property
    def sampling_frequency(self):
        """The output sampling frequency."""
        return self.parameter_get('sampling_frequency', dtype='actual')

    @property
    def statistics_callback(self):
        """Get the registered statistics callback."""
        cbks = self._statistics_callbacks['stream_buffer']
        if len(cbks):
            return cbks[0]
        else:
            return None

    @statistics_callback.setter
    def statistics_callback(self, cbk):
        """Set the statistics callback.

        :param cbk: The callable(data) where data is a statistics data
            structure.  See the `statistics documentation <statistics.html>`_
            for details on the data format.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
        """
        self._statistics_callbacks['stream_buffer'] = [cbk]

    def statistics_callback_register(self, cbk, source=None):
        """Register a statistics callback.

        :param cbk: The callable(data) where data is a statistics data
            structure.  See the `statistics documentation <statistics.html>`_
            for details on the data format.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
        :param source: The statistics source where the computation is performed.
            which is one of:
            * stream_buffer: The host-side stream buffer (default).
            * sensor: The device-side FPGA.

        WARNING: calling :meth:`statistics_callback` after calling this method
        may result in unusual behavior.  Do not mix these API calls.
        """
        source = 'stream_buffer' if source is None else source
        if source not in self._statistics_callbacks:
            raise ValueError(f'Invalid source: {source}')
        self._statistics_callbacks[source].append(cbk)

    def statistics_callback_unregister(self, cbk, source=None):
        """Unregister a statistics callback.

        :param cbk: The callback previously provided to
            :meth:`statistics_callback_register`.
        :param source: The callback source.
        """
        source = 'stream_buffer' if source is None else source
        if source not in self._statistics_callbacks:
            raise ValueError(f'Invalid source: {source}')
        self._statistics_callbacks[source].remove(cbk)

    def _statistics_callback_handler(self, s):
        if s is None:
            return
        source = s['source']
        offsets = self._statistics_offsets.get(source, [])
        if not len(offsets):
            duration = s['time']['range']['value'][0]
            charge = s['accumulators']['charge']['value']
            energy = s['accumulators']['energy']['value']
            offsets = [duration, charge, energy]
            self._statistics_offsets[source] = offsets
            return
        duration_offset, charge_offset, energy_offset = offsets
        s['time']['range']['value'] = [x - duration_offset for x in s['time']['range']['value']]
        s['accumulators']['charge']['value'] -= charge_offset
        s['accumulators']['energy']['value'] -= energy_offset
        cbks = self._statistics_callbacks.get(source, [])
        for cbk in cbks:
            cbk(s)

    def statistics_accumulators_clear(self):
        """Clear the charge and energy accumulators."""
        for offsets in self._statistics_offsets.values():
            offsets.clear()

    def view_factory(self):
        """Construct a new View into the device's data.

        :return: A View-compatible instance.
        """
        view = View(self.stream_buffer, self.calibration)
        view.on_close = lambda: self.stream_process_unregister(view)
        self.stream_process_register(view)
        return view

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
        p = PARAMETERS_DICT[name]
        if name == 'current_ranging':
            self._current_ranging_split(value)
            return
        if 'read_only' in p.flags:
            log.warning('Attempting to set read_only parameter %s', name)
            return
        try:
            value = name_to_value(name, value)
        except KeyError:
            if p.validator is None:
                raise KeyError(f'value {value} not allowed for parameter {name}')
            else:
                value = p.validator(value)
        self._parameters[name] = value
        if p.path == 'setting':
            if 'skip_update' not in p.flags:
                self._stream_settings_send()
        elif p.path == 'extio':
            self._extio_set()
        elif p.path == 'current_ranging':
            self._current_ranging_set()

    def parameter_get(self, name, dtype=None):
        """Get a parameter value.

        :param name: The parameter name.
        :param dtype: The data type for the parameter.  None (default)
            attempts to convert the value to the enum string.
            'actual' returns the value in its actual type used by the driver.
        :raise KeyError: if name not found.
        """
        if name == 'current_ranging':
            pnames = ['type', 'samples_pre', 'samples_window', 'samples_post']
            values = [str(self.parameter_get('current_ranging_' + p)) for p in pnames]
            return '_'.join(values)
        p = PARAMETERS_DICT[name]
        if p.path == 'info':
            return self._parameter_get_info(name)
        value = self._parameters[name]
        if dtype == 'actual':
            return value
        try:
            return value_to_name(name, value)
        except KeyError:
            return value

    def _parameter_get_info(self, name):
        if name == 'model':
            return 'JS110'
        elif name == 'device_serial_number':
            if self.stream_buffer is None or self.calibration is None:
                return self.device_serial_number
            return self.calibration.serial_number
        elif name == 'hardware_serial_number':
            if self.stream_buffer is None:
                return None
            return self.serial_number

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

    @property
    def device_serial_number(self):
        """Get the Joulescope device serial number assigned during manufacturing.

        :return: The serial number string.
        """
        if self._usb is None:
            return None
        return self._usb.serial_number

    def _stream_buffer_open(self):
        if self.stream_buffer:
            buffer, self.stream_buffer = self.stream_buffer, None
            del buffer
        stream_buffer_duration = self._parameters['buffer_duration']
        reduction_frequency = self._parameters['reduction_frequency']
        self._output_sampling_frequency = self.parameter_get('sampling_frequency', dtype='actual')
        reductions = _reduction_frequency_to_reductions(reduction_frequency)
        if self._input_sampling_frequency == self._output_sampling_frequency:
            log.info('Create StreamBuffer')
            self.stream_buffer = StreamBuffer(stream_buffer_duration, reductions,
                                              self._input_sampling_frequency)
        else:
            log.info('Create DownsamplingStreamBuffer')
            self.stream_buffer = DownsamplingStreamBuffer(stream_buffer_duration, reductions,
                                                          self._input_sampling_frequency,
                                                          self._output_sampling_frequency)
        self.stream_buffer.callback = self._statistics_callback_handler

    def open(self, event_callback_fn=None):
        """Open the device for use.

        :param event_callback_fn: The function(event, message) to call on
            asynchronous events, mostly to allow robust handling of device
            errors.  "event" is one of the :class:`DeviceEvent` values,
            and the message is a more detailed description of the event.
        :raise IOError: on failure.

        The event_callback_fn may be called asynchronous and from other
        threads.  The event_callback_fn must implement any thread safety.
        """
        if self.stream_buffer:
            self.close()
        self.statistics_accumulators_clear()
        self._usb.open(event_callback_fn)
        self._stream_buffer_open()
        self._current_ranging_set()
        try:
            info = self.info()
            if info is not None:
                log.info('info:\n%s', json.dumps(info, indent=2))
        except Exception:
            log.warning('could not fetch info record')
        try:
            self.calibration = self._calibration_read()
        except Exception:
            log.warning('could not fetch active calibration')
        try:
            cfg = self._parameters_defaults.get(self._config, {})
            for key, value in cfg.items():
                self.parameter_set(key, value)
        except Exception:
            log.warning('could not set defaults')
        return self

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
        if rv.data[0] != b'{'[0]:  # has header (firmware prior to 1.1.0)
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
            json_bytes = rv.data[8:]
        else:  # just JSON string
            json_bytes = rv.data
        try:
            return json.loads(json_bytes.decode('utf-8'))
        except UnicodeDecodeError:
            log.exception('INFO has invalid unicode: %s', binascii.hexlify(rv.data[8:]))
        except json.decoder.JSONDecodeError:
            log.exception('Could not decode INFO: %s', rv.data[8:])
        return None

    def _calibration_read_raw(self, cal_idx):
        rv = self._usb.control_transfer_in(
            'device', 'vendor',
            request=UsbdRequest.CALIBRATION,
            value=cal_idx, index=0, length=datafile.HEADER_SIZE)
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
                value=cal_idx, index=offset, length=k)
            if 0 != rv.result:
                log.warning('calibration_read transfer failed %d', rv.result)
                return None
            chunk = bytes(rv.data)
            offset += len(chunk)
            calibration += chunk
        return calibration

    def _calibration_read(self, calibration=None) -> Calibration:
        cal_idx = CALIBRATION_INDEX[calibration]
        cal = Calibration()
        serial_number = self.serial_number
        cal.serial_number = serial_number
        try:
            cal_data = self._calibration_read_raw(cal_idx)
            if cal_data is None:
                log.info('no calibration present')
            else:
                cal.load(cal_data)
        except (ValueError, IOError):
            log.info('failed reading %s calibration', CALIBRATION_INDEX_TO_NAME[cal_idx])
        if cal.serial_number != serial_number:
            log.info('calibration serial number mismatch')
            return None
        self.calibration = cal
        self.stream_buffer.calibration_set(cal.current_offset, cal.current_gain, cal.voltage_offset, cal.voltage_gain)
        return cal

    def close(self):
        """Close the device and release resources"""
        if self.stream_buffer is not None:
            try:
                self.stop()
            except Exception:
                log.exception('USB stop failed')
            try:
                self._usb.close()
            except Exception:
                log.exception('USB close failed')
            self._stream_process_call('close')
            self.stream_buffer.callback = None
            self.stream_buffer = None

    def _wait_for_sensor_command(self, timeout=None):
        timeout = SENSOR_COMMAND_TIMEOUT_SECONDS if timeout is None else float(timeout)
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
                          PACKET_VERSION,
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
                          0  # rsv3_u8
                          )
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SETTINGS,
            value=0, index=0, data=msg)
        _ioerror_on_bad_result(rv)
        if streaming == 0:
            self._wait_for_sensor_command()

    def _extio_set(self):
        msg = struct.pack('<BBBBIBBBBBBBBII',
                          PACKET_VERSION,
                          24,
                          PacketType.EXTIO,
                          0,  # hdr_rsv1
                          0,  # hdr_rsv4
                          0,  # flags
                          self._parameters['trigger_source'],
                          self._parameters['current_lsb'],
                          self._parameters['voltage_lsb'],
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

    def _current_ranging_split(self, value):
        if value is None or value in [False, 'off']:
            self.parameter_set('current_ranging_type', 'off')
            return
        parts = value.split('_')
        if len(parts) != 4:
            raise ValueError(f'Invalid current_ranging value {value}')
        for p, v in zip(['type', 'samples_pre', 'samples_window', 'samples_post'], parts):
            self.parameter_set('current_ranging_' + p, v)

    def _current_ranging_set(self):
        if self.stream_buffer is None:
            return
        names = ['current_ranging_type', 'current_ranging_samples_pre',
                 'current_ranging_samples_window', 'current_ranging_samples_post']
        s = '_'.join([str(self.parameter_get(n)) for n in names])
        self.stream_buffer.suppress_mode = s

    def _on_data(self, data):
        # DeviceDriverApi.read_stream_start data_fn callback
        # invoked from USB thread when new sample data is available
        # VERY time critical - keep as short as possible
        # return False to continue streaming, True to stop streaming
        return self.stream_buffer.insert(data)

    def _on_process(self):
        # DeviceDriverApi.read_stream_start process_fn callback
        # invoked from USB thread when new sample data is available after _on_data
        # Time critical, but less so than _on_data
        # return False to continue streaming, True to stop streaming
        rv = False
        try:
            self.stream_buffer.process()
        except Exception:
            log.exception('stream_buffer.process exception: stop streaming')
            return True

        objs = self._process_objs + self._process_objs_add
        self._process_objs = []
        self._process_objs_add = []
        for obj in objs:
            if obj.driver_active:
                try:
                    rv |= bool(obj.stream_notify(self.stream_buffer))
                except Exception:
                    log.exception('%s stream_notify() exception - stop streaming', obj)
                    obj.driver_active = False
                    rv = True
                self._process_objs.append(obj)
            else:
                obj.driver_active = False
                try:
                    if hasattr(obj, 'close'):
                        obj.close()
                except Exception:
                    log.exception('%s close() exception', obj)
        return rv

    def _on_stop(self, status, message):
        # DeviceDriverApi.read_stream_start stop_fn callback
        # invoked from USB thread
        log.info('streaming done(%d, %s)', status, message)
        stop_fn, self._stop_fn = self._stop_fn, None
        if callable(stop_fn):
            stop_fn(status, message)
        for obj in self._process_objs:
            if obj.driver_active and hasattr(obj, 'stop'):
                try:
                    obj.stop()
                except Exception:
                    log.exception('%s stop() exception', obj)

    def _stream_process_call(self, method, *args, **kwargs):
        for obj in self._process_objs:
            fn = getattr(obj, method, None)
            if obj.driver_active and callable(fn):
                try:
                    fn(*args, **kwargs)
                except Exception:
                    log.exception('%s %s() exception', obj, method)
                    obj.driver_active = False

    def start(self, stop_fn=None, duration=None, contiguous_duration=None):
        """Start data streaming.

        :param stop_fn: The function(event, message) called when the device stops.
            The device can stop "automatically" on errors.
            Call :meth:`stop` to stop from the caller.
            This function will be called from the USB processing thread.
            Any calls back into self MUST BE resynchronized.
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
            self.stream_buffer.sample_id_max = int(duration * self._input_sampling_frequency)
        if contiguous_duration is not None:
            c = int(contiguous_duration * self._input_sampling_frequency)
            self.stream_buffer.contiguous_max = c
            log.info('contiguous_samples=%s', c)

        self._process_objs = self._process_objs + self._process_objs_add
        self._stream_process_call('start', stream_buffer=self.stream_buffer)
        self._streaming = True
        self._stream_settings_send()
        self._usb.read_stream_start(
            endpoint_id=2,
            transfers=self._parameters['transfer_outstanding'],
            block_size=self._parameters['transfer_length'] * usb.BULK_IN_LENGTH,
            data_fn=self._on_data,
            process_fn=self._on_process,
            stop_fn=self._on_stop)
        return True

    def stop(self):
        """Stop data streaming.

        :return: True if stopped.  False if was already stopped.

        This method is always safe to call, even after the device has been
        stopped or removed.
        """
        log.info('stop : streaming=%s', self._streaming)
        if self._streaming:
            self._usb.read_stream_stop(2)
            self._streaming = False
            try:
                self._stream_settings_send()
            except Exception:
                log.warning('Device.stop() while attempting _stream_settings_send')
            self._stream_process_call('stop')
            return True
        return False

    def read(self, duration=None, contiguous_duration=None, out_format=None, fields=None):
        """Read data from the device.

        :param duration: The duration in seconds for the capture.
            The duration must fit within the stream_buffer.
        :param contiguous_duration: The contiguous duration in seconds for
            the capture.  As opposed to duration, this ensures that the
            duration has no missing samples.  Missing samples usually
            occur when the device first starts.
            The duration must fit within the stream_buffer.
        :param out_format: The output format which is one of:

            * calibrated: The Nx2 np.ndarray(float32) with columns current and voltage.
            * raw: The raw Nx2 np.ndarray(uint16) Joulescope data.
            * samples_get: The StreamBuffer samples get format.  Use the fields
              parameter to optionally specify the signals to include.
            * None: equivalent to 'calibrated'.

        :param fields: The fields for samples_get when out_format=samples_get.

        If streaming was already in progress, it will be restarted.
        If neither duration or contiguous duration is specified, the capture
        will only be stopped by callbacks registered through
        :meth:`stream_process_register`.
        """
        log.info('read(duration=%s, contiguous_duration=%s, out_format=%s)',
                 duration, contiguous_duration, out_format)
        if out_format not in ['raw', 'calibrated', 'samples_get', None]:
            raise ValueError(f'Invalid out_format {out_format}')
        if duration is None and contiguous_duration is None:
            raise ValueError('Must specify duration or contiguous_duration')
        duration_max = len(self.stream_buffer) / self._output_sampling_frequency
        if contiguous_duration is not None and contiguous_duration > duration_max:
            raise ValueError(f'contiguous_duration {contiguous_duration} > {duration_max} max seconds')
        if duration is not None and duration > duration_max:
            raise ValueError(f'duration {duration} > {duration_max} max seconds')
        q = queue.Queue()

        def on_stop(*args, **kwargs):
            log.info('received stop callback: pending stop')
            q.put(None)

        self.start(on_stop, duration=duration, contiguous_duration=contiguous_duration)
        q.get()
        self.stop()
        start_id, end_id = self.stream_buffer.sample_id_range
        log.info('read available range %s, %s', start_id, end_id)
        if contiguous_duration is not None:
            start_id = end_id - int(contiguous_duration * self._output_sampling_frequency)
        elif duration is not None:
            start_id = end_id - int(duration * self._output_sampling_frequency)
        if start_id < 0:
            start_id = 0
        log.info('read actual %s, %s', start_id, end_id)

        if out_format == 'raw':
            return self.stream_buffer.samples_get(start_id, end_id, fields='raw').reshape((-1, 2))
        elif out_format in ['calibrated', None]:
            return self.stream_buffer.samples_get(start_id, end_id, fields='current_voltage')
        else:
            return self.stream_buffer.samples_get(start_id, end_id, fields=fields)

    @property
    def is_streaming(self):
        """Check if the device is streaming.

        :return: True if streaming.  False if not streaming.
        """
        return self._streaming

    def stream_process_register(self, obj):
        """Register a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi`.
            The instance must remain valid until its :meth:`close` is
            called.

        Call :meth:`stream_process_unregister` to disconnect the instance.
        """
        obj.driver_active = True
        self._process_objs_add.append(obj)

    def stream_process_unregister(self, obj):
        """Unregister a stream process object.

        :param obj: The instance compatible with :class:`StreamProcessApi` that was
            previously registered using :meth:`stream_process_register`.
        """
        obj.driver_active = False

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
        v1_expected_length = 8 + 16
        v2_expected_length = 8 + 16 + 4 * 8 + 12 * 4
        expected_lengths = [v1_expected_length, v2_expected_length]
        if len(pdu) not in expected_lengths:
            msg = 'status msg size unexpected: %d not in %s' % (len(pdu), expected_lengths)
            return _status_error(1, msg)
        version, hdr_length, pdu_type = struct.unpack('<BBB', pdu[:3])
        if version != HOST_API_VERSION:
            msg = 'status msg API version mismatch: %d != %d' % (version, HOST_API_VERSION)
            return _status_error(1, msg)
        if pdu_type != PacketType.STATUS:
            msg = 'status msg pdu_type mismatch: %d != %d' % (pdu_type, PacketType.STATUS)
            return _status_error(1, msg)
        if hdr_length != len(pdu):
            msg = 'status msg length mismatch: %d != %d' % (hdr_length, len(pdu))
            return _status_error(1, msg)
        values = struct.unpack('<iIIBBBx', pdu[8:v1_expected_length])
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
        if hdr_length == v2_expected_length:
            s = _statistics_unpack(pdu[v1_expected_length:])
            self._statistics_callback_handler(s)
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
            'current_lsb': {
                'value': values[2],
                'units': ''},
            'voltage_lsb': {
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

    def sensor_firmware_program(self, data, progress_cbk=None):
        """Program the sensor microcontroller firmware

        :param data: The firmware to program as a raw binary file.
        :param progress_cbk:  The optional Callable[float] which is called
            with the progress fraction from 0.0 to 1.0
        :raise: on error.
        """
        log.info('sensor_firmware_program')
        if progress_cbk is None:
            progress_cbk = lambda x: None
        self.stop()

        log.info('sensor bootloader: start')
        data = datafile.filename_or_bytes(data)
        if not len(data):
            # erase without programming
            metadata = {
                'size': 0,
                'encryption': 1,
                'header': bytes([0] * 24),
                'mac': bytes([0] * 16),
                'signature': bytes([0] * 64),
            }
        else:
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
                'size': len(data),
                'encryption': 1,
                'header': hdr_value[:24],
                'mac': enc[:16],
                'signature': enc[16:],
            }
        log.info('header    = %r', binascii.hexlify(metadata['header']))
        log.info('mac       = %r', binascii.hexlify(metadata['mac']))
        log.info('signature = %r', binascii.hexlify(metadata['signature']))
        msg = struct.pack('<II', metadata['size'], metadata['encryption'])
        msg = msg + metadata['header'] + metadata['mac'] + metadata['signature']

        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.START, data=msg)
        # firmware holds of control transaction complete until done
        _ioerror_on_bad_result(rv)
        self._sensor_status_check()
        time.sleep(1.0)  # give sensor extra time to power up

        log.info('sensor bootloader: erase all flash')
        rv = self._usb.control_transfer_out(
            'device', 'vendor', request=UsbdRequest.SENSOR_BOOTLOADER,
            value=SensorBootloader.ERASE)
        # firmware holds of control transaction complete until done
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
            # firmware holds of control transaction complete until done
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

    def bootloader(self, progress_cbk=None):
        """Start the bootloader for this device.

        :param progress_cbk:  The optional Callable[float] which is called
            with the progress fraction from 0.0 to 1.0
        :return: (bootloader, existing_devices)  Use the bootloader instance
            to perform operations.  Use existing_devices to assist in
            determining when this device returns from bootloader mode.
        """
        if progress_cbk is None:
            progress_cbk = lambda x: None
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
        time_start = time.time()
        while not len(b):
            time_elapsed = time.time() - time_start
            if time_elapsed > USB_RECONNECT_TIMEOUT_SECONDS:
                raise IOError('Timed out waiting for bootloader to connect')
            progress_cbk(time_elapsed / USB_RECONNECT_TIMEOUT_SECONDS)
            time.sleep(0.25)
            _, b, _ = scan_for_changes(name='bootloader', devices=existing_bootloaders)
        if len(b) != 1:
            raise IOError('More than one new bootloader found')
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
            time_start = time.time()
            while not len(d):
                if (time.time() - time_start) > USB_RECONNECT_TIMEOUT_SECONDS:
                    raise IOError('Timed out waiting for application to connect')
                time.sleep(0.25)
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

    def __enter__(self):
        """Device context manager, automatically open."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Device context manager, automatically close."""
        self.close()


def scan(name: str = None, config=None) -> List[Device]:
    """Scan for connected devices.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param config: The configuration for the :class:`Device`.
    :return: The list of :class:`Device` instances.  A new instance is created
        for each detected device.  Use :func:`scan_for_changes` to preserved
        existing instances.
    :raises: None - guaranteed not to raise an exception
    """
    if name is None:
        name = 'joulescope'
    try:
        devices = usb.scan(name)
        if name == 'bootloader':
            devices = [bootloader.Bootloader(d) for d in devices]
        else:
            devices = [Device(d, config=config) for d in devices]
        return devices
    except Exception:
        log.exception('while scanning for devices')
        return []


def scan_require_one(name: str = None, config=None) -> Device:
    """Scan for one and only one device.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param config: The configuration for the :class:`Device`.
    :return: The :class:`Device` found.
    :raise RuntimeError: If no devices or more than one device was found.
    """
    devices = scan(name, config=config)
    if not len(devices):
        raise RuntimeError("no devices found")
    if len(devices) > 1:
        raise RuntimeError("multiple devices found")
    return devices[0]


def scan_for_changes(name: str = None, devices=None, config=None):
    """Scan for device changes.

    :param name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param devices: The list of existing :class:`Device` instances returned
        by a previous scan.  Pass None or [] if no scan has yet been performed.
    :param config: The configuration for the :class:`Device`.
    :return: The tuple of lists (devices_now, devices_added, devices_removed).
        "devices_now" is the list of all currently connected devices.  If the
        device was in "devices", then return the :class:`Device` instance from
        "devices".
        "devices_added" is the list of connected devices not in "devices".
        "devices_removed" is the list of devices in "devices" but not "devices_now".
    """
    devices_prev = [] if devices is None else devices
    devices_next = scan(name, config=config)
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

    _log.info('scan_for_changes %d devices: %d added, %d removed',
              len(devices_now), len(devices_added), len(devices_removed))
    return devices_now, devices_added, devices_removed


def bootloaders_run_application():
    """Command all connected bootloaders to run the application.

    :raises: None - guaranteed not to raise an exception.
    """
    log.info('Find all Joulescope bootloaders and run the application')
    for d in scan(name='bootloader'):
        try:
            d.open()
        except Exception:
            log.exception('while attempting to open bootloader')
            continue
        try:
            d.go()
        except Exception:
            log.exception('while attempting to run the application')


def bootloader_go(bootloader, device_name=None, timeout=None, config=None, progress_cbk=None):
    """Command the bootloader to run the application and return the matching device.

    :param bootloader: The target bootloader, which is already open.
    :param device_name: The case-insensitive device name to scan.
        None (default) is equivalent to 'Joulescope'.
    :param timeout: The timeout in seconds while waiting for the application.
    :param config: The configuration for the device.
    :param progress_cbk:  The optional Callable[float] which is called
        with the progress fraction from 0.0 to 1.0
    :return: The matching device, not yet opened.
    :raise IOError: on failure.
    """
    timeout = USB_RECONNECT_TIMEOUT_SECONDS if timeout is None else float(timeout)
    if progress_cbk is None:
        progress_cbk = lambda x: None
    existing_devices = scan(device_name)
    bootloader.go()
    time_start = time.time()
    while True:
        time_elapsed = time.time() - time_start
        if time_elapsed > timeout:
            raise IOError('could not find device')
        progress_cbk(time_elapsed / timeout)
        _, devices, _ = scan_for_changes(device_name, existing_devices, config)
        if len(devices):
            return devices[0]
        time.sleep(0.25)
