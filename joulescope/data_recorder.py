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

from joulescope.v0.calibration import Calibration
from joulescope.v0.stream_buffer import reduction_downsample, Statistics, stats_to_api, \
    stats_invalidate, \
    stats_factory, stats_array_factory, stats_array_clear, stats_array_invalidate, \
    STATS_FIELD_NAMES, NP_STATS_NAMES, \
    I_RANGE_MISSING, SUPPRESS_SAMPLES_MAX, RawProcessor, stats_compute
from joulescope.v0 import array_storage
from joulescope import datafile
import json
import numpy as np
import datetime
import logging

log = logging.getLogger(__name__)

DATA_RECORDER_FORMAT_VERSION = '2'
SAMPLES_PER_REDUCTION = 20000   # 100 Hz @ 2 MSPS
REDUCTIONS_PER_TLV = 20         # 5 Hz @ 2 MSPS
TLVS_PER_BLOCK = 5              # 1 Hz @ 2 MSPS

_STATS_VALUES_V1 = 4


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


_DOWNSAMPLE_FORMATTERS = {
    0: lambda x: x.astype(dtype=np.float32),
    1: lambda x: x.astype(dtype=np.float32),
    2: lambda x: x.astype(dtype=np.float32),
    3: lambda x: (x * 16).astype(dtype=np.uint8),
    4: lambda x: (x * 15).astype(dtype=np.uint8),
    5: lambda x: (x * 15).astype(dtype=np.uint8),
}


_DOWNSAMPLE_UNFORMATTERS = {
    0: lambda x: x,
    1: lambda x: x,
    2: lambda x: x,
    3: lambda x: x.astype(dtype=np.float32) * (1.0 / 16),
    4: lambda x: x.astype(dtype=np.float32) * (1.0 / 15),
    5: lambda x: x.astype(dtype=np.float32) * (1.0 / 15),
}


def construct_record_filename():
    time_start = datetime.datetime.utcnow()
    timestamp_str = time_start.strftime('%Y%m%d_%H%M%S')
    return f'{timestamp_str}.jls'


class DataRecorder:
    """Record Joulescope data to a file.

    :param filehandle: The file-like object or file name.
    :param calibration: The calibration bytes in datafile format.
        None (default) uses the unit gain calibration.
    :param user_data: Arbitrary JSON-serializable user data that is
        added to the file.
    """
    def __init__(self, filehandle, calibration=None, user_data=None):

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

        self._sample_id_tlv = None  # sample id for start of next TLV
        self._sample_id_block = None  # sample id for start of current block, None if not started yet

        self._stream_buffer = None  # to ensure same
        self._sb_sample_id_last = None
        self._voltage_range = None
        self._data_buffer = []
        self._sample_buffers = {}

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

    def _initialize(self, sampling_frequency, data_format):
        self._sampling_frequency = sampling_frequency
        self._samples_per_reduction = SAMPLES_PER_REDUCTION
        self._samples_per_tlv = self._samples_per_reduction * REDUCTIONS_PER_TLV
        self._samples_per_block = self._samples_per_tlv * TLVS_PER_BLOCK

        # dependent vars
        self._reductions_per_tlv = self._samples_per_tlv // self._samples_per_reduction
        reduction_block_size = self._samples_per_block // self._samples_per_reduction

        self._reduction = stats_array_factory(reduction_block_size)
        self._append_configuration(data_format)
        self._writer.collection_start(0, 0)

    def _append_configuration(self, data_format=None):
        data_format = 'none' if data_format is None else str(data_format)
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
        arrays = {0x00: data[:, 0]['length'].astype(np.uint64)}
        for field_idx, field in enumerate(STATS_FIELD_NAMES):
            for stat_idx, stat in enumerate(NP_STATS_NAMES[1:]):
                x_id = ((field_idx & 0x0f) << 4) | ((stat_idx + 1) & 0x0f)
                x = data[:, field_idx][stat]
                arrays[x_id] = x.astype(np.float32)
        value = array_storage.pack(arrays, r_stop)
        value = value.tobytes()
        self._writer.collection_end(collection, value)
        self._sample_id_block = None
        stats_array_clear(self._reduction)

    def stream_notify(self, stream_buffer):
        """Process data from a stream buffer.

        :param stream_buffer: The stream_buffer instance which has:
            * "output_sampling_frequency" member -> float
            * "has_raw" member -> in [True, False]
            * "sample_id_range" member => (start, stop)
            * "voltage_range" member -> in [0, 1]
            * samples_get(start_sample_id, stop_sample_id, dtype)
            * data_get(start_sample_id, stop_sample_id, increment, out)
            * __len__ : maximum number of samples that stream_buffer holds
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
            data_format = 'raw' if stream_buffer.has_raw else 'float32_v2'
            self._initialize(stream_buffer.output_sampling_frequency, data_format)
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
                    'v_range': self._voltage_range,
                    'sample_id': self._sample_id_tlv,
                }
                collection_data = json.dumps(collection_data).encode('utf-8')
                self._collection_start(data=collection_data)

            log.debug('_process() add tlv %d', self._sample_id_tlv)
            if stream_buffer.has_raw:
                self._append_raw_data(self._sample_id_tlv, sample_id_next)
            else:
                self._append_float_data(self._sample_id_tlv, sample_id_next)
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
        data = self._stream_buffer.samples_get(start, stop, fields=STATS_FIELD_NAMES)
        arrays = {}
        for field_idx, field in enumerate(STATS_FIELD_NAMES):
            x_id = (field_idx << 4) | 0x01  # stat_idx=mean
            x = data['signals'][field]['value']
            x = _DOWNSAMPLE_FORMATTERS[field_idx](x)
            arrays[x_id] = x
        data = array_storage.pack(arrays, stop - start)
        data = data.tobytes()
        self._writer.append(datafile.TAG_DATA_BINARY, data, compress=False)

    def _insert_remove_processed_data(self):
        while len(self._data_buffer) and self._data_buffer[0]['time']['sample_id_range']['value'][1] < self._sample_id_tlv:
            self._data_buffer.pop()

    def _insert_size_pending(self):
        self._insert_remove_processed_data()
        if not len(self._data_buffer):
            return 0
        sz = np.sum([x['time']['samples']['value'] for x in self._data_buffer])
        start_idx = self._data_buffer[0]['time']['sample_id_range']['value'][0]
        sz -= self._sample_id_tlv - start_idx
        return sz

    def _insert_next(self):
        # Will write any pending data up to a full data TLV
        # WARNING: does not check for sufficient data to fill a data TLV
        # since it needs to also handle the final partial.
        if self._sample_id_block is None:
            if not len(self._data_buffer):
                return
            has_raw = 'raw' in self._data_buffer[0]['signals']
            collection_data = {'sample_id': self._sample_id_tlv}
            if has_raw:
                collection_data['v_range'] = self._data_buffer[0]['signals']['raw']['voltage_range']
            collection_data = json.dumps(collection_data).encode('utf-8')
            self._collection_start(data=collection_data)
        offset = 0
        finalize_now = True
        while len(self._data_buffer):
            do_pop = True
            data = self._data_buffer[0]
            s0, s1 = data['time']['sample_id_range']['value']  # samples, not indices
            assert(self._sample_id_tlv + offset >= s0)
            idx0 = self._sample_id_tlv + offset - s0  # input buffer starting index
            idx1 = s1 - s0  # input buffer ending index
            if s1 > self._sample_id_tlv + self._samples_per_tlv:
                idx1 = self._sample_id_tlv + self._samples_per_tlv - s0
                do_pop = False
            idx_end = offset + idx1 - idx0
            for field, d in self._sample_buffers.items():
                if field == 'raw':
                    self._sample_buffers[field][offset:idx_end, :] = data['signals'][field]['value'][idx0:idx1, :]
                else:
                    self._sample_buffers[field][offset:idx_end] = data['signals'][field]['value'][idx0:idx1]
            offset += idx1 - idx0
            if do_pop:
                self._data_buffer.pop(0)
            if offset >= self._samples_per_tlv:
                finalize_now = False
                break

        # write the data TLV
        if 'raw' in self._sample_buffers:
            data = self._sample_buffers['raw'][:offset, :].tobytes()
            self._writer.append(datafile.TAG_DATA_BINARY, data, compress=False)
        else:
            arrays = {}
            for field_idx, field in enumerate(STATS_FIELD_NAMES):
                x_id = (field_idx << 4) | 0x01  # stat_idx=mean
                x = self._sample_buffers[field][:offset]
                x = _DOWNSAMPLE_FORMATTERS[field_idx](x)
                arrays[x_id] = x
            data = array_storage.pack(arrays, self._samples_per_tlv)
            data = data.tobytes()
            self._writer.append(datafile.TAG_DATA_BINARY, data, compress=False)

        # update reductions
        tlv_offset = (self._sample_id_tlv - self._sample_id_block) // self._samples_per_tlv
        r_start = tlv_offset * self._reductions_per_tlv
        r_stop = r_start + (offset * self._reductions_per_tlv) // self._samples_per_tlv
        for field_idx, field in enumerate(STATS_FIELD_NAMES):
            d = self._sample_buffers[field]
            for idx in range(0, r_stop - r_start):
                r = r_start + idx
                k0 = idx * self._samples_per_reduction
                k1 = k0 + self._samples_per_reduction
                stats_compute(d[k0:k1], self._reduction[r, field_idx:field_idx + 1])

        self._sample_id_tlv += self._samples_per_tlv
        self._total_size += offset
        if finalize_now or self._sample_id_tlv - self._sample_id_block >= self._samples_per_block:
            self._collection_end(self._writer.collections[-1])
        return

    def insert(self, data):
        """Insert sample data.

        :param data: A dict in the :meth:`StreamBuffer.samples_get` format.
        """
        if self._sample_id_tlv is None:  # first call
            data_format = 'raw' if 'raw' in data['signals'] else 'float32_v2'
            self._initialize(data['time']['sampling_frequency']['value'], data_format)
            self._sample_id_tlv = data['time']['sample_id_range']['value'][0]
            self._sample_buffers = {}
            for field in data['signals'].keys():
                if field == 'raw':
                    d = np.empty((self._samples_per_tlv, 2), dtype=np.uint16)
                else:
                    d = np.empty(self._samples_per_tlv, dtype=np.float32)
                self._sample_buffers[field] = d
        if data is not None:
            self._data_buffer.append(data)
        while True:
            if self._insert_size_pending() < self._samples_per_tlv:
                break
            self._insert_next()
        if data is None:
            self._insert_next()  # short data finalize

    def _append_meta(self, footer_user_data=None):
        index = {
            'type': 'footer',
            'size': self._total_size,  # in samples
        }
        data = json.dumps(index).encode('utf-8')
        self._writer.append(datafile.TAG_META_JSON, data)
        if footer_user_data is not None:
            data = json.dumps(footer_user_data).encode('utf-8')
            self._writer.append(datafile.TAG_USER_JSON, data)

    def close(self, footer_user_data=None):
        """Finalize and close the recording."""
        if self._closed:
            return
        if self._sample_id_tlv is None:
            self._append_configuration()
        if self._stream_buffer:
            self.stream_notify(None)
        if self._data_buffer:
            self.insert(None)
        self._closed = True
        while len(self._writer.collections):
            collection = self._writer.collections[-1]
            if len(collection.metadata):
                self._collection_end(collection)
            else:
                self._writer.collection_end()
        self._append_meta(footer_user_data)
        self._writer.finalize()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class DataReader:
    """Read Joulescope data from a file."""

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
        self.footer_user_data = None

    def __str__(self):
        if self._f is not None:
            return 'DataReader %.2f seconds (%d samples)' % (self.duration, self.footer['size'])

    def close(self):
        """Close the recording file."""
        if self._fh_close:
            self._fh.close()
        self._fh_close = False
        self._fh = None
        self._f = None
        self._sample_cache = None
        self._reduction_cache = None

    def open(self, filehandle):
        """Open a recording file.

        :param filehandle: The seekable filehandle or filename string.
        """
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
                if self._data_start_position and self.config is not None:
                    log.warning('Unexpected file truncation, attempting recovery')
                    sample_count = self._sample_count()
                    log.warning('Recovery found %d samples', sample_count)
                    self.footer = {
                        'type': 'footer',
                        'size': sample_count,  # in samples
                    }
                else:
                    raise ValueError('could not read file')
                break
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
                else:
                    log.warning('Unknown JSON section type=%s', type_)
            elif tag == datafile.TAG_USER_JSON:
                if self.footer is None:
                    self.user_data = json.loads(value.decode('utf-8'))
                else:
                    self.footer_user_data = json.loads(value.decode('utf-8'))
            elif tag == datafile.TAG_END:
                break
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
        """The sample ID range.

        :return: [start, stop] sample identifiers.
        """
        if self._f is not None:
            s_start = 0
            s_end = int(s_start + self.footer['size'])
            return [s_start, s_end]
        return [0, 0]

    @property
    def sampling_frequency(self):
        """The output sampling frequency."""
        f = 0.0
        if self._f is not None:
            f = float(self.config['sampling_frequency'])
        if f <= 0.0:
            log.warning('Invalid sampling frequency %r, assume 1.0 Hz.', f)
            f = 1.0
        return f

    @property
    def input_sampling_frequency(self):
        """The original input sampling frequency."""
        f = 0.0
        if self._f is None:
            pass
        elif 'input_sampling_frequency' in self.config:
            f = float(self.config['sampling_frequency'])
        else:
            f = self.sampling_frequency
        if f <= 0.0:
            log.warning('Invalid input sampling frequency %r, assume 1.0 Hz.', f)
            f = 1.0
        return f

    @property
    def output_sampling_frequency(self):
        """The output sampling frequency."""
        return self.sampling_frequency

    @property
    def reduction_frequency(self):
        """The reduction frequency or 1 if no reduction."""
        f = 0.0
        try:
            if self._f is not None:
                f = self.config['sampling_frequency'] / self.config['samples_per_reduction']
        except Exception:
            log.warning('Could not get reduction frequency.')
        if f <= 0.0:
            log.warning('Invalid input sampling frequency %r, assume 1.0 Hz.', f)
            f = 1.0
        return f

    @property
    def duration(self):
        """The data file duration, in seconds."""
        f = self.sampling_frequency
        if f > 0:
            r = self.sample_id_range
            return (r[1] - r[0]) / f
        return 0.0

    @property
    def voltage_range(self):
        """The data file voltage range."""
        return self._voltage_range

    def _sample_count(self):
        """Count the actual samples in the file.

        WARNING: this operation may be slow.  Use footer when possible.
        """
        samples_per_block = self.config['samples_per_block']
        sample_count = 0
        self._fh.seek(self._data_start_position)

        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None:
                break
            if tag == datafile.TAG_COLLECTION_START:
                self._f.skip()
                sample_count += samples_per_block
            else:
                self._f.advance()
        return sample_count

    def _validate_range(self, start=None, stop=None, increment=None):
        idx_start, idx_end = self.sample_id_range
        if increment is not None:
            idx_end = ((idx_end + increment - 1) // increment) * increment
        # log.debug('[%d, %d] : [%d, %d]', start, stop, idx_start, idx_end)
        if start == idx_start and start == stop:
            pass  # empty is allowed
        elif not idx_start <= start < idx_end:
            raise ValueError('start out of range: %d <= %d < %d' % (idx_start, start, idx_end))
        elif not idx_start <= stop <= idx_end:
            raise ValueError('stop out of range: %d <= %d <= %d: %s' %
                             (idx_start, stop, idx_end, increment))
        if stop < start:
            stop = start
        return start, stop

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

    def _downsampled(self, start, stop, fields):
        """Get the _downsampled data.

        :param start: The starting sample (must already be validated).
        :param stop: The ending sample (must already be validated).
        :return: The output which is dict of field name to values.
        """
        self._fh.seek(self._data_start_position)
        start, stop = self._validate_range(start, stop)
        length = stop - start
        field_idxs = []
        if fields is None:
            fields = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb']
        for field in fields:
            if field not in STATS_FIELD_NAMES:
                raise ValueError(f'Field {field} not available')
            idx = (STATS_FIELD_NAMES.index(field) << 4) | 1
            field_idxs.append((field, idx))
        rv = {}
        for _, field_idx in field_idxs:
            rv[field_idx] = np.empty(length, dtype=np.float32)
        out_idx = 0
        while start < stop:
            sample_cache = self._sample_tlv(start)
            if sample_cache is None:
                break
            v = np.frombuffer(sample_cache['value'], dtype=np.uint8)
            value, sample_count = array_storage.unpack(v)
            b_start = start - sample_cache['start']
            length = sample_cache['stop'] - sample_cache['start'] - b_start
            out_remaining = stop - start
            length = min(length, out_remaining)
            if length <= 0:
                break
            b_stop = b_start + length
            for _, field_idx in field_idxs:
                rv[field_idx][out_idx:(out_idx + length)] = value[field_idx][b_start:b_stop]
            out_idx += length
            start += length

        result = {}
        for field, field_idx in field_idxs:
            x = rv[field_idx][:out_idx]
            idx = STATS_FIELD_NAMES.index(field)
            fn = _DOWNSAMPLE_UNFORMATTERS[idx]
            result[field] = fn(x)
        return result

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
        data, sample_count = array_storage.unpack(value)
        stats_array = stats_array_factory(sample_count)
        stats_array_invalidate(stats_array)
        for key, x in data.items():
            field_idx = (key >> 4) & 0x0f
            stats_idx = key & 0x0f
            stats_name = NP_STATS_NAMES[stats_idx]
            stats_array[:, field_idx][stats_name] = x  # .astype(dtype=np.float64)
        for idx in range(1, len(STATS_FIELD_NAMES)):
            stats_array[:, idx]['length'] = stats_array[:, 0]['length']
        return stats_array

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
        if sz < 1:
            sz = 1
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

    def _data_get_handler_none(self, start, stop, out):
        return None

    def _data_get_handler_raw(self, start, stop, out):
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

    def _data_get_handler_float32_v2(self, start, stop, out):
        k = self._downsampled(start, stop, STATS_FIELD_NAMES)
        out[:, 0]['mean'] = k['current']
        out[:, 1]['mean'] = k['voltage']
        out[:, 2]['mean'] = k['power']
        out[:, 3]['mean'] = k['current_range']
        out[:, 4]['mean'] = k['current_lsb']
        out[:, 5]['mean'] = k['voltage_lsb']
        out[:, :]['variance'] = 0.0  # zero variance, only one sample!
        out[:, :]['min'] = np.nan  # min
        out[:, :]['max'] = np.nan  # max

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

        out_len = len(out)
        if increment == 1:
            self._data_get_handler(start, stop, out)
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
                self._statistics_get_handler(k_start, k_stop, out[idx, :])
                k_start = k_stop
        return out

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
            log.warning(f'time %.6f out of range', t)
            return None
        return s

    def sample_id_to_time(self, s):
        if s is None:
            return None
        return s / self.sampling_frequency

    def normalize_time_arguments(self, start, stop, units=None):
        """Normalize time arguments to range.

        :param start: The start time or samples.
            None gets the first sample, equivalent to self.sample_id_range[0].
        :param stop: The stop time or samples.
            None gets the last sample, equivalent to self.sample_id_range[1].
        :param units: The time units which is one of ['seconds', 'samples']
            None (default) id equivalent to 'samples'.
        :return: (start_sample, stop_sample).

        When units=='samples', negative values are interpreted in standard
        Python notation relative to the last sample.  None

        """
        s_min, s_max = self.sample_id_range
        if units == 'seconds':
            start = self.time_to_sample_id(start)
            stop = self.time_to_sample_id(stop)
        elif units is None or units == 'samples':
            if start is not None and start < 0:
                start = s_max + start
            if stop is not None and stop < 0:
                stop = s_max + stop
        else:
            raise ValueError(f'invalid time units: {units}')
        s1 = s_min if start is None else start
        s2 = s_max if stop is None else stop
        if s1 == s_min and s1 == s2:
            pass  # ok, zero length capture
        elif not s_min <= s1 < s_max:
            raise ValueError(f'start sample out of range: {s1}')
        elif not s_min <= s2 <= s_max:
            raise ValueError(f'start sample out of range: {s2}')
        return s1, s2

    def _stats_update(self, stats, x, length):
        stats['mean'] = np.mean(x, axis=0)
        stats['variance'] = np.var(x, axis=0) * length
        stats['min'] = np.amin(x, axis=0)
        stats['max'] = np.amax(x, axis=0)

    def _statistics_get_handler_none(self, start, stop, stats):
        return stop - start

    def _statistics_get_handler_raw(self, s1, s2, stats):
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
                self._stats_update(stats[idx], field, length)
        else:
            stats_invalidate(stats)
        return length

    def _statistics_get_handler_float32_v2(self, s1, s2, stats):
        rv = {'signals': {}}
        self._samples_get_handler_float32_v2(s1, s2, STATS_FIELD_NAMES, rv)
        zi = np.isfinite(rv['signals']['current']['value'])
        length = np.count_nonzero(zi)
        stats[:]['length'] = length
        if length:
            for idx, field in enumerate(STATS_FIELD_NAMES):
                self._stats_update(stats[idx], rv['signals'][field]['value'][zi], length)
        else:
            stats_invalidate(stats)
        return length

    def _samples_get_handler_none(self, start, stop, fields, rv):
        return None

    def _samples_get_handler_raw(self, start, stop, fields, rv):
        raw, bits, cal = self._raw(start, stop)
        signals = rv['signals']
        if fields is None:
            fields = ['current', 'voltage', 'power', 'current_range', 'current_lsb', 'voltage_lsb', 'raw']
        for field in fields:
            d = {'units': _SIGNALS_UNITS.get(field, '')}
            if field == 'current':
                v = cal[:, 0]
            elif field == 'voltage':
                v = cal[:, 1]
            elif field == 'power':
                v = cal[:, 0] * cal[:, 1]
            elif field == 'raw':
                v = raw
                d['voltage_range'] = self._voltage_range
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
            d['value'] = v
            signals[field] = d
        return rv

    def _samples_get_handler_float32_v2(self, start, stop, fields, rv):
        k = self._downsampled(start, stop, fields)
        for field, value in k.items():
            units = _SIGNALS_UNITS.get(field, '')
            rv['signals'][field] = {'value': value, 'units': units}
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
        t1, t2 = s1 / self.sampling_frequency, s2 / self.sampling_frequency
        rv = {
            'time': {
                'range': {'value': [t1, t2], 'units': 's'},
                'delta':  {'value': t2 - t1, 'units': 's'},
                'sample_id_range':  {'value': [s1, s2], 'units': 'samples'},
                'sample_id_limits': {'value': self.sample_id_range, 'units': 'samples'},
                'samples': {'value': s2 - s1, 'units': 'samples'},
                'input_sampling_frequency': {'value': self.input_sampling_frequency, 'units': 'Hz'},
                'output_sampling_frequency': {'value': self.output_sampling_frequency, 'units': 'Hz'},
                'sampling_frequency': {'value': self.sampling_frequency, 'units': 'Hz'},
            },
            'signals': {},
        }
        return self._samples_get_handler(s1, s2, fields, rv)

    def _statistics_get(self, start, stop):
        s1, s2 = start, stop
        (k1, k2), s = self._get_reduction_stats(s1, s2)
        if k1 >= k2:
            # compute directly over samples
            stats = stats_factory()
            if not self._statistics_get_handler(s1, s2, stats):
                stats[3]['mean'] = I_RANGE_MISSING
                stats[3]['variance'] = 0
                stats[3]['min'] = I_RANGE_MISSING
                stats[3]['max'] = I_RANGE_MISSING
            s = Statistics(stats=stats)
        else:
            if s1 < k1:
                s_start = stats_factory()
                self._statistics_get_handler(s1, k1, s_start)
                s.combine(Statistics(stats=s_start))
            if s2 > k2:
                s_stop = stats_factory()
                self._statistics_get_handler(k2, s2, s_stop)
                s.combine(Statistics(stats=s_stop))
        return s

    def statistics_get(self, start=None, stop=None, units=None):
        """Get the statistics for the collected sample data over a time range.

        :param start: The starting time relative to the first sample.
        :param stop: The ending time.
        :param units: The units for start and stop.
            'seconds' is in floating point seconds relative to the view.
            'samples' or None is in stream buffer sample indices.
        :return: The statistics data structure.
            See the `statistics documentation <statistics.html>`_
            for details.
        """
        log.debug('statistics_get(%s, %s, %s)', start, stop, units)
        s1, s2 = self.normalize_time_arguments(start, stop, units)
        if s1 == s2:
            s2 = s1 + 1  # always try to produce valid statistics
        s = self._statistics_get(s1, s2)
        t_start = s1 / self.sampling_frequency
        t_stop = s2 / self.sampling_frequency
        return stats_to_api(s.value, t_start, t_stop)
