# Copyright 2018 Jetperch LLC.  All rights reserved.

"""
Python ctypes binding to native Joulescope code.
"""

import os
import numpy as np
import ctypes
from ctypes import c_void_p, c_uint8, c_uint16, c_uint32, c_uint64, \
    c_int32, c_size_t, c_float, POINTER
import logging


STREAM_BUFFER_REDUCTION_MAX = 20

log = logging.getLogger(__name__)
MY_PATH = os.path.dirname(os.path.abspath(__file__))


NAME_TO_COLUMN = {
    'current': 0,
    'i': 0,
    'i_raw': 0,
    'voltage': 1,
    'v': 1,
    'v_raw': 1,
    'power': 1,
    'p': 1,
}


def load_lib():
    LIB_PATHS = [
        os.path.join(MY_PATH),
        os.path.join(MY_PATH, '..', '..', 'host', 'cmake-build-debug', 'source')
    ]
    if hasattr(ctypes, 'windll'):
        loader = ctypes.windll.LoadLibrary
        filenames = ['libjoulescope.dll']
    else:
        loader = ctypes.cdll.LoadLibrary
        filenames = ['libjoulescope.so', 'libjoulescope.dylib']
    for lib_path in LIB_PATHS:
        for f in filenames:
            p = os.path.abspath(os.path.join(lib_path, f))
            if os.path.exists(p):
                return loader(p)
    raise RuntimeError('Could not load native Joulescope library')


lib = load_lib()


CBK = ctypes.CFUNCTYPE(None, c_void_p, POINTER(c_float))


class Calibration(ctypes.Structure):
    _fields_ = [
        ('current_offset', c_float * 8),
        ('current_gain', c_float * 8),
        ('voltage_offset', c_float * 2),
        ('voltage_gain', c_float * 2),
    ]


class BufferReduction(ctypes.Structure):
    _fields_ = [
        ('enabled', c_int32),
        ('samples_per_step', c_uint32),
        ('sample_counter', c_uint32),
        ('length', c_uint32),
        ('cbk_fn', CBK),
        ('cbk_user_data', c_void_p),
        ('data', POINTER(c_float)),
    ]


class StreamBufferNative(ctypes.Structure):
    _fields_ = [
        ('packet_index', c_uint64),
        ('packet_index_offset', c_uint64),
        ('device_sample_id', c_uint64),
        ('processed_sample_id', c_uint64),
        ('sample_missing_count', c_uint64),
        ('skip_count', c_uint64),
        ('sample_sync_count', c_uint64),
        ('contiguous_count', c_uint64),
        ('length', c_uint32),
        ('raw', POINTER(c_uint16)),
        ('data', POINTER(c_float)),
        ('cal', Calibration),
        ('reductions', BufferReduction * STREAM_BUFFER_REDUCTION_MAX),
        ('stats_counter', c_uint64),
        ('stats_remaining', c_uint64),
        ('stats', c_float * 12),
        ('sample_toggle_last', c_uint16),
        ('sample_toggle_mask', c_uint16),
        ('voltage_range', c_uint8),
    ]


# struct js_stream_buffer_s * js_stream_buffer_new(
#     uint32_t length,
#     struct js_stream_buffer_calibration_s const * calibration,
#     uint32_t const * reductions, size_t reductions_length);
lib.js_stream_buffer_new.argtypes = [c_uint32, POINTER(c_uint32), c_size_t]
lib.js_stream_buffer_new.restype = POINTER(StreamBufferNative)

# void js_stream_buffer_free(struct js_stream_buffer_s * self);
lib.js_stream_buffer_free.argtypes = [POINTER(StreamBufferNative)]
lib.js_stream_buffer_free.restype = None

# void js_stream_buffer_callback_set(struct js_stream_buffer_s * self, uint32_t reduction,
#                                    js_stream_buffer_cbk cbk_fn, void * cbk_user_data);
lib.js_stream_buffer_callback_set.argtypes = [POINTER(StreamBufferNative), c_uint32, CBK, c_void_p]
lib.js_stream_buffer_callback_set.restype = None

# void js_stream_buffer_calibration_set(struct js_stream_buffer_s * self,
#                                       struct js_stream_buffer_calibration_s const * calibration);
lib.js_stream_buffer_calibration_set.argtypes = [POINTER(StreamBufferNative), POINTER(Calibration)]
lib.js_stream_buffer_calibration_set.restype = None

# void js_stream_buffer_reset(struct js_stream_buffer_s * self);
lib.js_stream_buffer_reset.argtypes = [POINTER(StreamBufferNative)]
lib.js_stream_buffer_reset.restype = None

# void js_stream_buffer_insert_usb_bulk(
#     struct js_stream_buffer_s * self,
#     uint8_t const * data,
#     size_t length);
lib.js_stream_buffer_insert_usb_bulk.argtypes = [POINTER(StreamBufferNative), POINTER(c_uint8), c_size_t]
lib.js_stream_buffer_insert_usb_bulk.restype = None

# void js_stream_buffer_process(struct js_stream_buffer_s * self)
lib.js_stream_buffer_process.argtypes = [POINTER(StreamBufferNative)]
lib.js_stream_buffer_process.restype = None


# uint32_t js_stream_buffer_get(struct js_stream_buffer_s * self,
#        float * buffer, uint64_t start, uint64_t stop, uint64_t increment);
lib.js_stream_buffer_get.argtypes = [POINTER(StreamBufferNative), POINTER(c_float), c_uint64, c_uint64, c_uint64]
lib.js_stream_buffer_get.restype = c_uint32

# uint32_t js_stream_buffer_raw(struct js_stream_buffer_s * self,
#                               uint8_t * buffer, uint64_t start, uint64_t stop);
lib.js_stream_buffer_raw.argtypes = [POINTER(StreamBufferNative), POINTER(c_uint8), c_uint64, c_uint64]
lib.js_stream_buffer_raw.restype = c_uint32

buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
buf_from_mem.restype = ctypes.py_object
buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)


class StreamBuffer:

    def __init__(self, length, reductions):
        r = (c_uint32 * len(reductions))(*reductions)
        self.length = int(length)
        self._impl = lib.js_stream_buffer_new(self.length, r, len(r))
        if not self._impl:
            raise RuntimeError('Could not allocate StreamBuffer')
        self.sample_id_max = 0  # used to automatically stop streaming
        self.contiguous_max = 0  # used to automatically stop streaming
        self.callback = None  # fn(np.array [3][4] of statistics, energy)
        self._energy_picojoules = 0  # integer for infinite precision
        self.reset()

        # get numpy array into raw data
        p = ctypes.cast(self._impl[0].raw, c_void_p)
        b = buf_from_mem(p.value, self.length * 4, 0x100)
        self.raw_buffer = np.ndarray((self.length * 2), dtype=np.uint16, buffer=b)

        p = ctypes.cast(self._impl[0].data, c_void_p)
        b = buf_from_mem(p.value, self.length * 8, 0x100)
        self.data_buffer = np.ndarray((self.length * 2), dtype=np.float32, buffer=b)
        self._ctypes_cbk = CBK(self._on_cbk)
        lib.js_stream_buffer_callback_set(self._impl, len(reductions) - 1, self._ctypes_cbk, None)

        self.reduction_data = []
        for idx in range(len(reductions)):
            d = self._impl[0].reductions[idx].data
            k = self._impl[0].reductions[idx].length
            b = buf_from_mem(ctypes.cast(d, c_void_p).value, k * 3 * 4 * 4, 0x100)
            self.reduction_data.append(np.ndarray((k, 3, 4), dtype=np.float32, buffer=b))
        self.reduction_step = int(np.prod(reductions))

    def __del__(self):
        if self._impl:
            lib.js_stream_buffer_free(self._impl)

    def __len__(self):
        return self.length

    @property
    def sample_id_range(self):
        s_end = int(self._impl[0].processed_sample_id)
        s_start = s_end - self.length + self.reduction_step
        if s_start < 0:
            s_start = 0
        return [s_start, s_end]

    def status(self):
        return {
            'device_sample_id': {'value': self._impl[0].device_sample_id, 'units': 'samples'},
            'sample_id': {'value': self._impl[0].processed_sample_id, 'units': 'samples'},
            'sample_missing_count': {'value': self._impl[0].sample_missing_count, 'units': 'samples'},
            'skip_count': {'value': self._impl[0].skip_count, 'units': ''},
            'sample_sync_count': {'value': self._impl[0].sample_sync_count, 'units': 'samples'},
            'contiguous_count': {'value': self._impl[0].contiguous_count, 'units': 'samples'},
        }

    def calibration_set(self, current_offset, current_gain, voltage_offset, voltage_gain):
        c = Calibration()
        for i in range(7):
            c.current_offset[i] = current_offset[i]
            c.current_gain[i] = current_gain[i]
        c.current_offset[7] = float('nan')
        c.current_gain[7] = float('nan')
        for i in range(2):
            c.voltage_offset[i] = voltage_offset[i]
            c.voltage_gain[i] = voltage_gain[i]
        lib.js_stream_buffer_calibration_set(self._impl, c)

    def reset(self):
        self.sample_id_max = 1 << 64
        self.contiguous_max = 1 << 64
        self._energy_picojoules = 0
        lib.js_stream_buffer_reset(self._impl)

    def insert(self, data):
        """Insert new data into the buffer.

        :param data: The new data to insert.
        :return: False to continue streaming, True to end.
        """
        buffer = data.ctypes.data_as(POINTER(c_uint8))
        lib.js_stream_buffer_insert_usb_bulk(self._impl, buffer, len(data))
        duration_stop = self._impl[0].device_sample_id >= self.sample_id_max
        contiguous_stop = self._impl[0].contiguous_count >= self.contiguous_max
        rv = duration_stop or contiguous_stop
        if rv:
            if duration_stop:
                log.info('insert causing duration stop %d >= %d',
                         self._impl[0].device_sample_id, self.sample_id_max)
            elif duration_stop:
                log.info('insert causing contiguous stop %d >= %d',
                         self._impl[0].contiguous_count, self.contiguous_max)
        return rv

    def process(self):
        lib.js_stream_buffer_process(self._impl)

    def data_get(self, start, stop, increment=None, out=None):
        """Get the samples with statistics"""
        if stop <= start:
            return np.empty((0, 3, 4), dtype=np.float32)
        expected_length = (stop - start) // increment
        if out is None:
            out = np.empty((expected_length, 3, 4), dtype=np.float32)
        data = out.ctypes.data_as(ctypes.POINTER(c_float))
        length = lib.js_stream_buffer_get(self._impl, data, start, stop, increment)
        if length != expected_length:
            log.warning('length mismatch: expected=%s, returned=%s', expected_length, length)
        return out[:length, :, :]

    def raw_get(self, start, stop):
        if stop <= start:
            log.warning('raw %d <= %d', start, stop)
            return np.empty((0, 2), dtype=np.uint16)
        total_length = self._impl[0].length
        start_idx = start % total_length
        stop_idx = stop % total_length
        if 0 == stop_idx:
            stop_idx = total_length
        if stop_idx > start_idx:
            return self.raw_buffer[(start_idx * 2):(stop_idx * 2)]
        # on wrap, call native function to be sure
        expected_length = stop - start
        out = np.empty(expected_length * 4, dtype=np.uint8)
        buffer = out.ctypes.data_as(ctypes.POINTER(c_uint8))
        length = lib.js_stream_buffer_raw(self._impl, buffer, start, stop)
        if length != expected_length:
            raise RuntimeError('invalid length: %d != %d for (%d, %d)' %
                               (length, expected_length, start, stop))
        return out

    def get_reduction(self, idx, start, stop):
        """Get reduction data directly.

        :param idx: The reduction index.
        :param start: The starting sample_id (inclusive).
        :param stop: The ending sample_id (exclusive).
        :return: The reduction data which normally is memory mapped to the
            underlying data, but will be copied on rollover.
        """
        log.info('get_reduction(%s, %s, %s)', idx, start, stop)
        total_length = self._impl[0].length
        if stop - start > total_length:
            raise ValueError('requested size is too large')
        reduction = 1
        for i in range(idx + 1):
            reduction *= self._impl[0].reductions[i].samples_per_step
        r_len = self._impl[0].reductions[idx].length
        start = (start % total_length) // reduction
        stop = (stop % total_length) // reduction
        k = stop - start
        r = self.reduction_data[idx]
        if k == 0:
            return np.empty((0, 3, 4), dtype=np.float32)
        elif k < 0:
            k += r_len
            d = np.empty((k, 3, 4), dtype=np.float32)
            d[:(r_len - start), :, :] = r[start:, :, :]
            d[r_len - start:, :, :] = r[:stop, :, :]
            return d
        else:
            return r[start:stop, :, :]

    def _on_cbk(self, user_data, stats):
        if callable(self.callback):
            b = np.empty(12, dtype=np.float32)
            for i in range(12):
                b[i] = stats[i]
            b = b.reshape((3, 4))
            # todo handle variable sampling frequencies and reductions
            time_interval = 0.5  # seconds
            power_picowatts = b[2][0] * 1e12
            energy_picojoules = power_picowatts * time_interval
            if np.isfinite(energy_picojoules):
                self._energy_picojoules += int(energy_picojoules)
            energy = self._energy_picojoules * 1e-12
            self.callback(b, energy)


def usb_packet_factory(packet_index, count=None):
    """Construct USB Bulk packets for testing.

    :param packet_index: The USB packet index for the first packet.
    :param count: The number of consecutive packets to construct.
    :return: The bytes containing the packet data
    """
    count = 1 if count is None else int(count)
    if count < 1:
        count = 1
    frame = (ctypes.c_uint8 * ((packet_index + 1) * 512 * count))()
    for i in range(count):
        idx = packet_index + i
        k = i * 512
        frame[k + 0] = 1     # packet type raw
        frame[k + 1] = 0     # status = 0
        frame[k + 2] = 0x00  # length LSB
        frame[k + 3] = 0x02  # length MSB
        frame[k + 4] = idx & 0xff
        frame[k + 5] = (idx >> 8) & 0xff
        frame[k + 6] = 0
        frame[k + 7] = 0
        k += 8
        for j in range(126 * 2):
            v = (idx * 126 * 2 + j) << 2
            if j & 1:
                v |= j & 0x0002
            frame[k + j * 2] = v & 0xff
            frame[k + j * 2 + 1] = (v >> 8) & 0xff
    return frame