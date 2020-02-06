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

from . import datafile
from joulescope.calibration import Calibration
from joulescope.stream_buffer import reduction_downsample, Statistics, stats_to_api, \
    stats_factory, stats_array_factory, stats_array_clear, stats_array_invalidate, \
    STATS_DTYPE, STATS_FIELD_NAMES, STATS_FIELD_COUNT, NP_STATS_NAMES, \
    I_RANGE_MISSING, SUPPRESS_SAMPLES_MAX, RawProcessor
import json
import numpy as np
import datetime
import logging

log = logging.getLogger(__name__)

DATA_RECORDER_FORMAT_VERSION = '2'
SAMPLES_PER_BLOCK = 100000
_REDUCTION_SAMPLES_MINIMUM = 200
_STATS_VALUES_V1 = 4

_DTYPE_SIZE_MASK = 0x03
_DTYPE_SIZE_1B = 0x0
_DTYPE_SIZE_2B = 0x1
_DTYPE_SIZE_4B = 0x2
_DTYPE_SIZE_8B = 0x3

_DTYPE_SUBTYPE_MASK = 0x6
_DTYPE_SUBTYPE_FLOAT = 0x0
_DTYPE_SUBTYPE_UINT = 0x8
_DTYPE_SUBTYPE_INT = 0xC

_DTYPE_FLOAT32 = _DTYPE_SUBTYPE_FLOAT + _DTYPE_SIZE_4B
_DTYPE_FLOAT64 = _DTYPE_SUBTYPE_FLOAT + _DTYPE_SIZE_8B
_DTYPE_UINT8 = _DTYPE_SUBTYPE_UINT + _DTYPE_SIZE_1B
_DTYPE_UINT16 = _DTYPE_SUBTYPE_UINT + _DTYPE_SIZE_2B
_DTYPE_UINT32 = _DTYPE_SUBTYPE_UINT + _DTYPE_SIZE_4B
_DTYPE_UINT64 = _DTYPE_SUBTYPE_UINT + _DTYPE_SIZE_8B
_DTYPE_INT8 = _DTYPE_SUBTYPE_INT + _DTYPE_SIZE_1B
_DTYPE_INT16 = _DTYPE_SUBTYPE_INT + _DTYPE_SIZE_2B
_DTYPE_INT32 = _DTYPE_SUBTYPE_INT + _DTYPE_SIZE_4B
_DTYPE_INT64 = _DTYPE_SUBTYPE_INT + _DTYPE_SIZE_8B


_DTYPE_MAP = {
    _DTYPE_FLOAT32: np.float32,
    _DTYPE_FLOAT64: np.float64,
    _DTYPE_UINT8: np.uint8,
    _DTYPE_UINT16: np.uint16,
    _DTYPE_UINT32: np.uint32,
    _DTYPE_UINT64: np.uint64,
    _DTYPE_INT8: np.int8,
    _DTYPE_INT16: np.int16,
    _DTYPE_INT32: np.int32,
    _DTYPE_INT64: np.int64,
}


_SIGNALS_UNITS = {
    'current': 'A',
    'voltage': 'V',
    'power': 'W',
    'raw_current': 'LSBs',
    'raw_voltage': 'LSBs',
    'raw': '',
    'bits': '',
    'current_range': '',
    'current_lsb': '',
    'voltage_lsb': '',
}


def construct_record_filename():
    time_start = datetime.datetime.utcnow()
    timestamp_str = time_start.strftime('%Y%m%d_%H%M%S')
    return f'{timestamp_str}.jls'


class DataRecorder:
    """Record Joulescope data to a file."""

    def __init__(self, filehandle, calibration=None, user_data=None):
        """Create a new instance.

        :param filehandle: The file-like object or file name.
        :param calibration: The calibration bytes in datafile format.
            None (default) uses the unit gain calibration.
        :param user_data: Arbitrary JSON-serializable user data that is
            added to the file.
        """
        log.info('init')
        if isinstance(filehandle, str):
            self._fh = open(filehandle, 'wb')
            filehandle = self._fh
        else:
            self._fh = None

        self._sampling_frequency = 0
        self._samples_per_tlv = 0
        self._samples_per_block = 0
        self._samples_per_reduction = 0
        self._reductions_per_tlv = 0
        self._reduction = None

        self._sample_id_tlv = 0  # sample id for start of next TLV
        self._sample_id_block = None  # sample id for start of current block, None if not started yet

        self._stream_buffer = None  # to ensure same
        self._sb_sample_id_last = None
        self._voltage_range = None

        self._writer = datafile.DataFileWriter(filehandle)
        self._closed = False
        self._total_size = 0

        if user_data is not None:
            b = json.dumps(user_data).encode('utf-8')
            self._writer.append(datafile.TAG_USER_JSON, b)
        if calibration is not None:
            if isinstance(calibration, Calibration):
                calibration = calibration.data
            self._writer.append_subfile('calibration', calibration)

    def _initialize(self):
        self._sampling_frequency = self._stream_buffer.sampling_frequency
        if self._sampling_frequency > 100 * _REDUCTION_SAMPLES_MINIMUM:
            self._samples_per_reduction = int(self._sampling_frequency) // 100  # 100 Hz @ 2 MSPS
        else:
            self._samples_per_reduction = _REDUCTION_SAMPLES_MINIMUM
        self._samples_per_tlv = self._samples_per_reduction * 20  # ~ 5 Hz @ 2 MSPS
        self._samples_per_block = self._samples_per_tlv * 5  # ~1 Hz @ 2 MSPS

        # dependent vars
        self._reductions_per_tlv = self._samples_per_tlv // self._samples_per_reduction
        reduction_block_size = self._samples_per_block // self._samples_per_reduction

        self._reduction = stats_array_factory(reduction_block_size)
        self._append_configuration()
        self._writer.collection_start(0, 0)

    def _append_configuration(self):
        if self._stream_buffer is None:
            data_format = 'none'
        elif self._stream_buffer.has_raw:
            data_format = 'raw'
        else:
            data_format = 'float32_v1'
        config = {
            'type': 'config',
            'data_recorder_format_version': DATA_RECORDER_FORMAT_VERSION,
            'sampling_frequency': self._sampling_frequency,
            'samples_per_reduction': self._samples_per_reduction,
            'samples_per_tlv': self._samples_per_tlv,
            'samples_per_block': self._samples_per_block,
            'reduction_fields': ['current', 'voltage', 'power',
                                 'current_range', 'current_lsb', 'voltage_lsb'],
            'data_format': data_format,
            'reduction_format': 'v2',
        }
        cfg_data = json.dumps(config).encode('utf-8')
        self._writer.append(datafile.TAG_META_JSON, cfg_data)

    def _collection_start(self, data=None):
        log.debug('_collection_start()')
        c = self._writer.collection_start(1, 0, data=data)
        c.metadata = {'start_sample_id': self._sample_id_tlv}
        c.on_end = self._collection_end
        self._sample_id_block = self._sample_id_tlv

    def _collection_end(self, collection):
        sample_count = (self._sample_id_tlv - self._sample_id_block)
        r_stop = sample_count // self._samples_per_reduction
        log.debug('_collection_end(%s, %s)', r_stop, len(self._reduction))
        data = self._reduction[:r_stop, :]
        arrays = [(0, 0, _DTYPE_UINT64, data[:, 0]['length'])]
        for field_idx, field in enumerate(STATS_FIELD_NAMES):
            for stat_idx, stat in enumerate(NP_STATS_NAMES[1:]):
                x = data[:, field_idx][stat]
                arrays.append((field_idx, stat_idx + 1, _DTYPE_FLOAT32, x))
        value = self._arrays_format(r_stop, arrays)
        value = value.tobytes()
        self._writer.collection_end(collection, value)
        self._sample_id_block = None
        stats_array_clear(self._reduction)

    def stream_notify(self, stream_buffer):
        """Process data from a stream buffer.

        :param stream_buffer: The stream_buffer instance which has:
            * "sample_id_range" member
            * "voltage_range" member
            * samples_get(start_sample_id, stop_sample_id, dtype)
            * data_get(start_sample_id, stop_sample_id, increment, out)
        """
        finalize = False
        if stream_buffer is None:
            if self._stream_buffer is None:
                return
            finalize = True
            stream_buffer = self._stream_buffer
        sb_start, sb_stop = stream_buffer.sample_id_range
        if self._stream_buffer is None:
            self._stream_buffer = stream_buffer
            self._initialize()
            self._sample_id_tlv = sb_stop
            self._sample_id_block = None
            if self._samples_per_tlv > len(stream_buffer):
                raise ValueError('stream_buffer length too small.  %s > %s' %
                                 (self._samples_per_tlv, len(stream_buffer)))
        elif self._stream_buffer != stream_buffer:
            raise ValueError('Supports only a single stream_buffer instance')
        if sb_start > self._sample_id_tlv:
            raise ValueError('stream_buffer does not contain sample_id')

        while True:
            finalize_now = False
            sample_id_next = self._sample_id_tlv + self._samples_per_tlv
            if sb_stop < sample_id_next:
                if finalize and self._sample_id_tlv < sb_stop:
                    finalize_now = True
                    sample_id_next = sb_stop
                else:
                    break

            self._voltage_range = stream_buffer.voltage_range
            if self._sample_id_block is None:
                collection_data = {
                    'v_range': stream_buffer.voltage_range,
                    'sample_id': self._sample_id_tlv,
                }
                collection_data = json.dumps(collection_data).encode('utf-8')
                self._collection_start(data=collection_data)

            log.debug('_process() add tlv %d', self._sample_id_tlv)
            self._append_raw_data(self._sample_id_tlv, sample_id_next)
            self._total_size += (sample_id_next - self._sample_id_tlv)
            tlv_offset = (self._sample_id_tlv - self._sample_id_block) // self._samples_per_tlv
            r_start = tlv_offset * self._reductions_per_tlv
            r_stop = r_start + self._reductions_per_tlv
            stream_buffer.data_get(self._sample_id_tlv, sample_id_next,
                                   self._samples_per_reduction, out=self._reduction[r_start:r_stop, :])
            self._sample_id_tlv = sample_id_next
            if finalize_now or self._sample_id_tlv - self._sample_id_block >= self._samples_per_block:
                self._collection_end(self._writer.collections[-1])

    def _append_raw_data(self, start, stop):
        if self._closed:
            return
        b = self._stream_buffer.samples_get(start, stop, fields='raw')
        data = b.tobytes()
        self._writer.append(datafile.TAG_DATA_BINARY, data, compress=False)

    def _append_float_data(self, start, stop):
        if self._closed:
            return
        data = self._stream_buffer.data_get(start, stop)
        arrays = []
        for field_idx, field in enumerate(STATS_FIELD_NAMES):
            x = data[:, field]['mean']
            if field_idx < 3:
                x = x.astype(dtype=np.float32)
                dtype = _DTYPE_FLOAT32
            elif field_idx == 3:
                x = (x * 16).astype(dtype=np.uint8)
                dtype = _DTYPE_UINT8
            else:
                x = (x * 15).astype(dtype=np.uint8)
                dtype = _DTYPE_UINT8
            arrays.append((field_idx, 1, dtype, x))
        return self._arrays_format(stop - start, arrays)

    def _arrays_format(self, sample_count, arrays):
        """Format arrays.

        :param sample_count: The number of samples in all arrays.
        :param arrays: The list of (field_id, stats_id, dtype, array) for
            each item to format.
        """
        # tag-length index followed by values.
        # In tag-length, value format: tag[31:0], length[23:0]
        #  tag: compress[31], field_idx[14:11], stats_idx[10:8], dtype[7:0]
        # All values start on multiple of 8 bytes (values padded).
        # Length is the length of the value in bytes (may be zero).
        #  data: compressed[7], rsv[6], field_idx[5:3], stats_idx[2:0]
        #  0: offset to first value (multiple of 8 bytes)
        #  1: array_count[31:24], sample_count[23:0]
        #  hdr_N: tag-length data for each value for fast indexing
        if sample_count >= 2**24:
            raise ValueError(f'sample_count {sample_count} too big')
        arrays_count = len(arrays)
        header_size = (((2 + 2 * arrays_count) * 4 + 7) // 8) * 8

        sz = 0
        arrays_fmt = []
        for (field_idx, stat_idx, dtype, x) in arrays:
            sz_mask = dtype & _DTYPE_SIZE_MASK
            if sz_mask != _DTYPE_SIZE_1B:
                np_dtype = _DTYPE_MAP[dtype]
                if not x.dtype == np_dtype:
                    x = x.astype(np_dtype)
                if not x.flags['C_CONTIGUOUS']:
                    x = np.ascontiguousarray(x, dtype=np_dtype)
                x = x.view(dtype=np.uint8)
            hdr = ((field_idx & 0x7) << 11) + ((stat_idx & 0x7) << 8) + (dtype & 0xFf)
            arrays_fmt.append((hdr, x))
            sz += ((len(x) + 7) // 8) * 8
        v_u32 = np.zeros(header_size + sz, dtype=np.uint32)
        v_u8 = v_u32.view(dtype=np.uint8)
        v_u32[0] = header_size
        v_u32[1] = (arrays_count << 24) + sample_count
        offset = header_size
        for idx, (hdr, x) in enumerate(arrays_fmt):
            x_len = len(x)
            value_len = ((x_len + 7) // 8) * 8
            v_u32[2 + idx * 2] = hdr
            v_u32[2 + idx * 2 + 1] = x_len
            v_u8[offset:offset + x_len] = x
            offset += value_len
        return v_u8[:offset]

    def _append_meta(self):
        index = {
            'type': 'footer',
            'size': self._total_size,  # in samples
        }
        data = json.dumps(index).encode('utf-8')
        self._writer.append(datafile.TAG_META_JSON, data)

    def close(self):
        if self._closed:
            return
        if self._stream_buffer is None:
            self._append_configuration()
        self.stream_notify(None)
        self._closed = True
        while len(self._writer.collections):
            collection = self._writer.collections[-1]
            if len(collection.metadata):
                self._collection_end(collection)
            else:
                self._writer.collection_end()
        self._append_meta()
        self._writer.finalize()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class DataReader:

    def __init__(self):
        self.calibration = None
        self.config = None
        self.footer = None
        self._fh_close = False
        self._fh = None
        self._f = None  # type: datafile.DataFileReader
        self._data_start_position = 0
        self._voltage_range = 0
        self._sample_cache = None
        self._data_get_handler = None
        self._reduction_handler = None
        self._samples_get_handler = None
        self._statistics_get_handler = None
        self.raw_processor = RawProcessor()
        self.user_data = None

    def __str__(self):
        if self._f is not None:
            return 'DataReader %.2f seconds (%d samples)' % (self.duration, self.footer['size'])

    def close(self):
        if self._fh_close:
            self._fh.close()
        self._fh_close = False
        self._fh = None
        self._f = None
        self._sample_cache = None
        self._reduction_cache = None

    def open(self, filehandle):
        self.close()
        self.calibration = Calibration()  # default calibration
        self.config = None
        self.footer = None
        self._data_start_position = 0

        if isinstance(filehandle, str):
            log.info('DataReader(%s)', filehandle)
            self._fh = open(filehandle, 'rb')
            self._fh_close = True
        else:
            self._fh = filehandle
            self._fh_close = False
        self._f = datafile.DataFileReader(self._fh)
        while True:
            tag, value = self._f.peek()
            if tag is None:
                raise ValueError('could not read file')
            elif tag == datafile.TAG_SUBFILE:
                name, data = datafile.subfile_split(value)
                if name == 'calibration':
                    self.calibration = Calibration().load(data)
            elif tag == datafile.TAG_COLLECTION_START:
                self._data_start_position = self._f.tell()
            elif tag == datafile.TAG_META_JSON:
                meta = json.loads(value.decode('utf-8'))
                type_ = meta.get('type')
                if type_ == 'config':
                    self._configure(meta)
                elif type_ == 'footer':
                    self.footer = meta
                    break
                else:
                    log.warning('Unknown JSON section type=%s', type_)
            elif tag == datafile.TAG_USER_JSON:
                self.user_data = json.loads(value.decode('utf-8'))
            self._f.skip()
        if self.config is None or self.footer is None:
            raise ValueError('could not read file')
        log.info('DataReader with %d samples:\n%s', self.footer['size'], json.dumps(self.config, indent=2))
        if self._data_start_position == 0 and self.footer['size']:
            raise ValueError(f"data not found, but expect {self.footer['size']} samples")
        if int(self.config['data_recorder_format_version']) > int(DATA_RECORDER_FORMAT_VERSION):
            raise ValueError('Invalid file format')
        self.config.setdefault('reduction_fields', ['current', 'voltage', 'power'])
        cal = self.calibration
        self.raw_processor.calibration_set(cal.current_offset, cal.current_gain, cal.voltage_offset, cal.voltage_gain)
        return self

    def _configure(self, config):
        config.setdefault('data_format', 'raw')
        config.setdefault('reduction_format', 'v1')
        self.config = config
        self._samples_get_handler = getattr(self, '_samples_get_handler_' + config['data_format'])
        self._data_get_handler = getattr(self, '_data_get_handler_' + config['data_format'])
        self._statistics_get_handler = getattr(self, '_statistics_get_handler_' + config['data_format'])
        self._reduction_handler = getattr(self, '_reduction_handler_' + config['reduction_format'])

    @property
    def sample_id_range(self):
        if self._f is not None:
            s_start = 0
            s_end = int(s_start + self.footer['size'])
            return [s_start, s_end]
        return 0

    @property
    def sampling_frequency(self):
        if self._f is not None:
            return float(self.config['sampling_frequency'])
        return 0.0

    @property
    def reduction_frequency(self):
        if self._f is not None:
            return self.config['sampling_frequency'] / self.config['samples_per_reduction']
        return 0.0

    @property
    def duration(self):
        f = self.sampling_frequency
        if f > 0:
            r = self.sample_id_range
            return (r[1] - r[0]) / f
        return 0.0

    @property
    def voltage_range(self):
        return self._voltage_range

    def _validate_range(self, start=None, stop=None, increment=None):
        idx_start, idx_end = self.sample_id_range
        if increment is not None:
            idx_end = ((idx_end + increment - 1) // increment) * increment
        # log.debug('[%d, %d] : [%d, %d]', start, stop, idx_start, idx_end)
        if not idx_start <= start < idx_end:
            raise ValueError('start out of range: %d <= %d < %d' % (idx_start, start, idx_end))
        if not idx_start <= stop <= idx_end:
            raise ValueError('stop out of range: %d <= %d <= %d: %s' %
                             (idx_start, stop, idx_end, increment))

    def _sample_tlv(self, sample_idx):
        """Get the sample data entry for the TLV containing sample_idx.

        :param sample_idx: The sample index.
        :return: The dict containing the data.
        """

        if self._sample_cache and self._sample_cache['start'] <= sample_idx < self._sample_cache['stop']:
            # cache hit
            return self._sample_cache

        idx_start, idx_end = self.sample_id_range
        if not idx_start <= sample_idx < idx_end:
            raise ValueError('sample index out of range: %d <= %d < %d' % (idx_start, sample_idx, idx_end))

        if self._sample_cache is not None:
            log.debug('_sample_cache cache miss: %s : %s %s',
                      sample_idx, self._sample_cache['start'], self._sample_cache['stop'])

        # seek
        samples_per_tlv = self.config['samples_per_tlv']
        samples_per_block = self.config['samples_per_block']
        tgt_block = sample_idx // samples_per_block

        if self._sample_cache is not None and sample_idx > self._sample_cache['start']:
            # continue forward
            self._fh.seek(self._sample_cache['tlv_pos'])
            voltage_range = self._sample_cache['voltage_range']
            block_fh_pos = self._sample_cache['block_pos']
            current_sample_idx = self._sample_cache['start']
            block_counter = current_sample_idx // samples_per_block
        else:  # add case for rewind?
            log.debug('_sample_tlv resync to beginning')
            self._fh.seek(self._data_start_position)
            voltage_range = 0
            block_fh_pos = 0
            block_counter = 0
            current_sample_idx = 0
            if self._f.advance() != datafile.TAG_COLLECTION_START:
                raise ValueError('data section must be single collection')

        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None:
                log.error('sample_tlv not found before end of file: %s > %s', sample_idx, current_sample_idx)
                break
            if tag == datafile.TAG_COLLECTION_START:
                if block_counter < tgt_block:
                    self._f.skip()
                    block_counter += 1
                else:
                    block_fh_pos = self._f.tell()
                    tag, collection_bytes = next(self._f)
                    c = datafile.Collection.decode(collection_bytes)
                    if c.data:
                        collection_start_meta = json.loads(c.data)
                        voltage_range = collection_start_meta.get('v_range', 0)
                        self._voltage_range = voltage_range
                    current_sample_idx = block_counter * samples_per_block
            elif tag == datafile.TAG_COLLECTION_END:
                block_counter += 1
                self._f.advance()
            elif tag == datafile.TAG_DATA_BINARY:
                tlv_stop = current_sample_idx + samples_per_tlv
                if current_sample_idx <= sample_idx < tlv_stop:
                    # found it!
                    tlv_pos = self._f.tell()
                    tag, value = next(self._f)
                    self._sample_cache = {
                        'voltage_range': voltage_range,
                        'start': current_sample_idx,
                        'stop': tlv_stop,
                        'value': value,
                        'tlv_pos': tlv_pos,
                        'block_pos': block_fh_pos,
                    }
                    return self._sample_cache
                else:
                    self._f.advance()
                current_sample_idx = tlv_stop
            else:
                self._f.advance()

    def _raw(self, start=None, stop=None):
        """Get the raw data.

        :param start: The starting sample (must already be validated).
        :param stop: The ending sample (must already be validated).
        :return: The output which is (out_raw, bits, out_cal).
        """
        x_start, x_stop = self.sample_id_range
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)
        length = stop - start
        if length <= 0:
            return np.empty((0, 2), dtype=np.uint16)
        # process extra before & after to handle filtering
        if start > SUPPRESS_SAMPLES_MAX:
            sample_idx = start - SUPPRESS_SAMPLES_MAX
            prefix_count = SUPPRESS_SAMPLES_MAX
        else:
            sample_idx = 0
            prefix_count = start
        if stop + SUPPRESS_SAMPLES_MAX <= x_stop:
            end_idx = stop + SUPPRESS_SAMPLES_MAX
        else:
            end_idx = x_stop
        out_idx = 0
        d_raw = np.empty((end_idx - sample_idx, 2), dtype=np.uint16)
        if self._f.advance() != datafile.TAG_COLLECTION_START:
            raise ValueError('data section must be single collection')
        while sample_idx < end_idx:
            sample_cache = self._sample_tlv(sample_idx)
            if sample_cache is None:
                break
            value = sample_cache['value']
            data = np.frombuffer(value, dtype=np.uint16).reshape((-1, 2))
            b_start = sample_idx - sample_cache['start']
            length = sample_cache['stop'] - sample_cache['start'] - b_start
            out_remaining = end_idx - sample_idx
            length = min(length, out_remaining)
            if length <= 0:
                break
            b_stop = b_start + length
            d = data[b_start:b_stop, :]
            d_raw[out_idx:(out_idx + length), :] = d
            out_idx += length
            sample_idx += length

        d_raw = d_raw[:out_idx, :]
        self.raw_processor.reset()
        self.raw_processor.voltage_range = self._voltage_range
        d_cal, d_bits = self.raw_processor.process_bulk(d_raw.reshape((-1, )))
        j = prefix_count
        k = min(prefix_count + stop - start, out_idx)
        return d_raw[j:k, :], d_bits[j:k], d_cal[j:k, :]

    def _reduction_tlv(self, reduction_idx):
        sz = self.config['samples_per_reduction']
        incr = self.config['samples_per_block'] // sz
        tgt_r_idx = reduction_idx

        if self._reduction_cache and self._reduction_cache['r_start'] <= tgt_r_idx < self._reduction_cache['r_stop']:
            return self._reduction_cache

        if self._reduction_cache is not None:
            log.debug('_reduction_tlv cache miss: %s : %s %s',
                      tgt_r_idx, self._reduction_cache['r_start'], self._reduction_cache['r_stop'])

        idx_start, idx_end = self.sample_id_range
        r_start = idx_start // sz
        r_stop = idx_end // sz
        if not r_start <= tgt_r_idx < r_stop:
            raise ValueError('reduction index out of range: %d <= %d < %d', r_start, tgt_r_idx, r_stop)

        if self._reduction_cache is not None and tgt_r_idx > self._reduction_cache['r_start']:
            # continue forward
            self._fh.seek(self._reduction_cache['next_collection_pos'])
            r_idx = self._reduction_cache['r_stop']
        else:  # add case for rewind?
            log.debug('_reduction_tlv resync to beginning')
            self._fh.seek(self._data_start_position)
            r_idx = 0
            if self._f.advance() != datafile.TAG_COLLECTION_START:
                raise ValueError('data section must be single collection')
            self._fh.seek(self._data_start_position)
            if self._f.advance() != datafile.TAG_COLLECTION_START:
                raise ValueError('data section must be single collection')

        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None or tag == datafile.TAG_COLLECTION_END:
                log.error('reduction_tlv not found before end of file: %s > %s', r_stop, r_idx)
                break
            elif tag != datafile.TAG_COLLECTION_START:
                raise ValueError('invalid file format: not collection start')
            r_idx_next = r_idx + incr
            if tgt_r_idx >= r_idx_next:
                self._f.skip()
                r_idx = r_idx_next
                continue
            self._f.collection_goto_end()
            tag, value = next(self._f)
            if tag != datafile.TAG_COLLECTION_END:
                raise ValueError('invalid file format: not collection end')

            data = self._reduction_handler(value)
            self._reduction_cache = {
                'r_start': r_idx,
                'r_stop': r_idx_next,
                'buffer': data,
                'next_collection_pos': self._f.tell()
            }
            return self._reduction_cache

    def _reduction_handler_v1(self, value):
        field_count = len(self.config['reduction_fields'])
        b = np.frombuffer(value, dtype=np.float32).reshape((-1, field_count, _STATS_VALUES_V1))
        data = stats_array_factory(len(b))
        stats_array_invalidate(data)
        samples_per_reduction = self.config['samples_per_reduction']
        data[:, :]['length'] = samples_per_reduction
        for idx in range(field_count):
            data[:, idx]['mean'] = b[:, 0, 0]
            data[:, idx]['variance'] = (samples_per_reduction - 1) * b[:, 0, 1] * b[:, 0, 1]
            data[:, idx]['min'] = b[:, 0, 2]
            data[:, idx]['min'] = b[:, 0, 3]
        return data

    def _reduction_handler_v2(self, value):
        value = np.frombuffer(value, dtype=np.uint8)
        sample_count, data = self._arrays_parse(value)
        stats_array = stats_array_factory(sample_count)
        stats_array_invalidate(stats_array)
        for (field_idx, stats_idx), x in data.items():
            stats_name = NP_STATS_NAMES[stats_idx]
            # todo optional decompresssion
            stats_array[:, field_idx][stats_name] = x  # .astype(dtype=np.float64)
        return stats_array

    def _arrays_parse(self, value):
        # See DataRecorder._arrays_format for format
        array_header = value[:8].view(dtype=np.uint32)
        offset = array_header[0]
        sample_count = array_header[1] & 0xffffff
        field_count = (array_header[1] >> 24) & 0xff
        offset_expect = 8 + field_count * 8
        if offset_expect > offset:
            raise ValueError('Invalid format')
        tag_length = value[8:offset_expect].view(dtype=np.uint32)
        result = {}
        for idx in range(field_count):
            hdr = tag_length[idx * 2]
            x_len = tag_length[idx * 2 + 1] & 0xffffff
            field_idx = (hdr >> 11) & 0x7
            stat_idx = (hdr >> 8) & 0x7
            dtype = hdr & 0xff
            np_dtype = _DTYPE_MAP[dtype]
            x = value[offset:offset + x_len]
            x = x.view(dtype=np_dtype)
            result[(field_idx, stat_idx)] = x
            offset += ((x_len + 7) // 8) * 8
        return sample_count, result

    def get_reduction(self, start=None, stop=None, units=None, out=None):
        """Get the fixed reduction with statistics.

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indices.
        :return: The The np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE)
            reduction data which normally is memory mapped to the underlying
            data, but will be copied on rollover.
        """
        start, stop = self.normalize_time_arguments(start, stop, units)
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)

        sz = self.config['samples_per_reduction']
        r_start = start // sz
        total_length = (stop - start) // sz
        r_stop = r_start + total_length
        log.info('DataReader.get_reduction(r_start=%r,r_stop=%r)', r_start, r_stop)

        if total_length <= 0:
            return stats_array_factory(0)
        if out is None:
            out = stats_array_factory(total_length)
        elif len(out) < total_length:
            raise ValueError('out too small')

        r_idx = r_start
        out_idx = 0

        while r_idx < r_stop:
            reduction_cache = self._reduction_tlv(r_idx)
            if reduction_cache is None:
                break
            data = reduction_cache['buffer']
            b_start = r_idx - reduction_cache['r_start']
            length = reduction_cache['r_stop'] - reduction_cache['r_start'] - b_start
            out_remaining = r_stop - r_idx
            length = min(length, out_remaining)
            if length <= 0:
                break
            out[out_idx:(out_idx + length), :] = data[b_start:(b_start + length), :]
            out_idx += length
            r_idx += length
        if out_idx != total_length:
            log.warning('DataReader length mismatch: out_idx=%s, length=%s', out_idx, total_length)
            total_length = min(out_idx, total_length)
        return out[:total_length, :]

    def _get_reduction_stats(self, start, stop):
        """Get statistics over the reduction

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :return: The tuple of ((sample_start, sample_stop), :class:`Statistics`).
        """
        # log.debug('_get_reduction_stats(%s, %s)', start, stop)
        s = Statistics()
        sz = self.config['samples_per_reduction']
        incr = self.config['samples_per_block'] // sz
        r_start = start // sz
        if (r_start * sz) < start:
            r_start += 1
        r_stop = stop // sz
        if r_start >= r_stop:  # cannot use the reductions
            s_start = r_start * sz
            return (s_start, s_start), s
        r_idx = r_start

        while r_idx < r_stop:
            reduction_cache = self._reduction_tlv(r_idx)
            if reduction_cache is None:
                break
            data = reduction_cache['buffer']
            b_start = r_idx - reduction_cache['r_start']
            length = reduction_cache['r_stop'] - reduction_cache['r_start'] - b_start
            out_remaining = r_stop - r_idx
            length = min(length, out_remaining)
            if length <= 0:
                break
            r = reduction_downsample(data, b_start, b_start + length, length)
            s.combine(Statistics(stats=r[0, :]))
            r_idx += length
        return (r_start * sz, r_stop * sz), s

    def _data_get_handler_none(self, start, stop, increment, out):
        return out

    def _data_get_handler_raw(self, start, stop, increment, out):
        out_len = len(out)
        if increment == 1:
            _, d_bits, d_cal = self._raw(start, stop)
            i, v = d_cal[:, 0], d_cal[:, 1]
            out[:, 0]['mean'] = i
            out[:, 1]['mean'] = v
            out[:, 2]['mean'] = i * v
            out[:, 3]['mean'] = np.bitwise_and(d_bits, 0x0f)
            out[:, 4]['mean'] = np.bitwise_and(np.right_shift(d_bits, 4), 0x01)
            out[:, 5]['mean'] = np.bitwise_and(np.right_shift(d_bits, 5), 0x01)
            out[:, :]['variance'] = 0.0  # zero variance, only one sample!
            out[:, :]['min'] = np.nan  # min
            out[:, :]['max'] = np.nan  # max
        elif increment == self.config['samples_per_reduction']:
            out = self.get_reduction(start, stop, out=out)
        elif increment > self.config['samples_per_reduction']:
            r_out = self.get_reduction(start, stop)
            increment = int(increment / self.config['samples_per_reduction'])
            out = reduction_downsample(r_out, 0, len(r_out), increment)
        else:
            k_start = start
            for idx in range(out_len):
                k_stop = k_start + increment
                out[idx, :] = self._statistics_get_handler(k_start, k_stop).value
                k_start = k_stop
        return out

    def data_get(self, start, stop, increment=None, units=None, out=None):
        """Get the samples with statistics.

        :param start: The starting sample id (inclusive).
        :param stop: The ending sample id (exclusive).
        :param increment: The number of raw samples.
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indices.
        :param out: The optional output np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE) to populate with
            the result.  None (default) will construct and return a new array.
        :return: The np.ndarray((N, STATS_FIELD_COUNT), dtype=DTYPE) data.
        """
        log.debug('DataReader.get(start=%r,stop=%r,increment=%r)', start, stop, increment)
        start, stop = self.normalize_time_arguments(start, stop, units)
        if increment is None:
            increment = 1
        if self._fh is None:
            raise IOError('file not open')
        increment = max(1, int(np.round(increment)))
        out_len = (stop - start) // increment
        if out_len <= 0:
            return stats_array_factory(0)
        if out is not None:
            if len(out) < len(out_len):
                raise ValueError('out too small')
        else:
            out = stats_array_factory(out_len)
        return self._data_get_handler(start, stop, increment, out)

    def summary_string(self):
        s = [str(self)]
        config_fields = ['sampling_frequency', 'samples_per_reduction', 'samples_per_tlv', 'samples_per_block']
        for field in config_fields:
            s.append('    %s = %r' % (field, self.config[field]))
        return '\n'.join(s)

    def time_to_sample_id(self, t):
        if t is None:
            return None
        s_min, s_max = self.sample_id_range
        s = int(t * self.sampling_frequency)
        if s < s_min or s > s_max:
            return None
        return s

    def sample_id_to_time(self, s):
        if s is None:
            return None
        return s / self.sampling_frequency

    def normalize_time_arguments(self, start, stop, units):
        s_min, s_max = self.sample_id_range
        if units == 'seconds':
            start = self.time_to_sample_id(start)
            stop = self.time_to_sample_id(stop)
        elif units is None or units == 'samples':
            if start is not None and start < 0:
                start = s_max + start
            if stop is not None and stop < 0:
                stop = s_max + start
        else:
            raise ValueError(f'invalid time units: {units}')
        s1 = s_min if start is None else start
        s2 = s_max if stop is None else stop
        if not s_min <= s1 < s_max:
            raise ValueError(f'start sample out of range: {s1}')
        if not s_min <= s2 <= s_max:
            raise ValueError(f'start sample out of range: {s2}')
        return s1, s2

    def _statistics_get_handler_none(self, start, stop):
        return None

    def _statistics_get_handler_raw(self, start, stop):
        s1, s2 = start, stop
        (k1, k2), s = self._get_reduction_stats(s1, s2)
        if k1 >= k2:
            # compute directly over samples
            stats = stats_factory()
            _, d_bits, z = self._raw(s1, s2)
            i, v = z[:, 0], z[:, 1]
            p = i * v
            zi = np.isfinite(i)
            i_view = i[zi]
            length = len(i_view)
            stats[:]['length'] = length
            if length:
                i_range = np.bitwise_and(d_bits, 0x0f)
                i_lsb = np.right_shift(np.bitwise_and(d_bits, 0x10), 4)
                v_lsb = np.right_shift(np.bitwise_and(d_bits, 0x20), 5)
                for idx, field in enumerate([i_view, v[zi], p[zi], i_range, i_lsb[zi], v_lsb[zi]]):
                    stats[idx]['mean'] = np.mean(field, axis=0)
                    stats[idx]['variance'] = np.var(field, axis=0)
                    stats[idx]['min'] = np.amin(field, axis=0)
                    stats[idx]['max'] = np.amax(field, axis=0)
            else:
                stats[3]['mean'] = I_RANGE_MISSING
                stats[3]['variance'] = 0
                stats[3]['min'] = I_RANGE_MISSING
                stats[3]['max'] = I_RANGE_MISSING
            s = Statistics(stats=stats)
        else:
            if s1 < k1:
                s_start = self._statistics_get_handler(s1, k1)
                s.combine(s_start)
            if s2 > k2:
                s_stop = self._statistics_get_handler(k2, s2)
                s.combine(s_stop)
        return s

    def _samples_get_handler_none(self, start, stop, fields, rv):
        return rv

    def _samples_get_handler_raw(self, start, stop, fields, rv):
        raw, bits, cal = self._raw(start, stop)
        signals = rv['signals']
        for field in fields:
            units = _SIGNALS_UNITS.get(field, '')
            if field == 'current':
                v = cal[:, 0]
            elif field == 'voltage':
                v = cal[:, 1]
            elif field == 'power':
                v = cal[:, 0] * cal[:, 1]
            elif field == 'raw':
                v = raw
            elif field == 'raw_current':
                v = np.right_shift(raw[0::2], 2)
            elif field == 'raw_voltage':
                v = np.right_shift(raw[1::2], 2)
            elif field == 'bits':
                v = bits
            elif field == 'current_range':
                v = np.bitwise_and(bits, 0x0f)
            elif field == 'current_lsb':
                v = np.bitwise_and(np.right_shift(bits, 4), 1)
            elif field == 'voltage_lsb':
                v = np.bitwise_and(np.right_shift(bits, 5), 1)
            else:
                log.warning('Unsupported field %s', field)
                v = np.array([])
            signals[field] = {'value': v, 'units': units}
        return rv

    def samples_get(self, start=None, stop=None, units=None, fields=None):
        """Get exact samples over a range.

        :param start: The starting time.
        :param stop: The ending time.
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indicies.
        :param fields: The fields to get.
        """
        log.debug('samples_get(%s, %s, %s)', start, stop, units)
        s1, s2 = self.normalize_time_arguments(start, stop, units)
        t1, t2 = start / self.sampling_frequency, stop / self.sampling_frequency
        rv = {
            'time': {
                'range': {'value': [t1, t2], 'units': 's'},
                'delta':  {'value': t2 - t1, 'units': 's'},
                'sample_id_range':  {'value': [start, stop], 'units': 'samples'},
                'sample_id_limits': {'value': self.sample_id_range, 'units': 'samples'},
                'sampling_frequency': {'value': self.sampling_frequency, 'units': 'Hz'},
            },
            'signals': {},
        }
        return self._samples_get_handler(s1, s2, fields, rv)

    def statistics_get(self, start=None, stop=None, units=None):
        """Get the statistics for the collected sample data over a time range.

        :param start: The starting time relative to the first sample.
        :param stop: The ending time.
        :param units: The units for start and stop.
            'seconds' is in floating point seconds relative to the view.
            'samples' or None is in stream buffer sample indices.
        :return: The statistics data structure.
            See :func:`joulescope.stream_buffer.stats_to_api` for details.
        """
        log.debug('statistics_get(%s, %s, %s)', start, stop, units)
        s1, s2 = self.normalize_time_arguments(start, stop, units)
        if s1 == s2:
            s2 = s1 + 1  # always try to produce valid statistics
        s = self._statistics_get_handler(s1, s2)
        t_start = s1 / self.sampling_frequency
        t_stop = s2 / self.sampling_frequency
        return stats_to_api(s.value, t_start, t_stop)
