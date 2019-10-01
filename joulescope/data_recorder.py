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
from joulescope import JOULESCOPE_DIR
from joulescope.calibration import Calibration
from joulescope.stream_buffer import reduction_downsample, Statistics, stats_to_api, \
    STATS_FIELDS, STATS_VALUES, I_RANGE_MISSING
import json
import numpy as np
import datetime
import os
import logging

log = logging.getLogger(__name__)

DATA_RECORDER_FORMAT_VERSION = '1'
SAMPLING_FREQUENCY = 2000000
SAMPLES_PER_BLOCK = 100000


def construct_record_filename():
    time_start = datetime.datetime.utcnow()
    timestamp_str = time_start.strftime('%Y%m%d_%H%M%S')
    name = '%s.jls' % (timestamp_str, )
    return os.path.join(JOULESCOPE_DIR, name)


class DataRecorder:
    """Record Joulescope data to a file."""

    def __init__(self, filehandle, sampling_frequency, calibration=None):
        """Create a new instance.

        :param filehandle: The file-like object or file name.
        :param sampling_frequency: The sampling frequency in Hertz.
        :param calibration: The calibration bytes in datafile format.
            None (default) uses the unit gain calibration.
        """
        log.info('init')
        if isinstance(filehandle, str):
            self._fh = open(filehandle, 'wb')
            filehandle = self._fh
        else:
            self._fh = None

        self._sampling_frequency = sampling_frequency
        # constraints:
        #    int1 * _samples_per_reduction = _samples_per_block
        #    int2 * _samples_per_reduction = _samples_per_tlv
        #    int3 * _samples_per_tlv = _samples_per_block
        self._samples_per_reduction = int(sampling_frequency) // 100  # ~100 Hz
        self._samples_per_tlv = self._samples_per_reduction * 20  # ~ 5 Hz
        self._samples_per_block = self._samples_per_tlv * 5  # ~1 Hz

        # dependent vars
        self._reductions_per_tlv = self._samples_per_tlv // self._samples_per_reduction
        reduction_block_size = self._samples_per_block // self._samples_per_reduction

        self._reduction = np.empty((reduction_block_size, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        self._reduction[:] = np.nan

        self._sample_id_tlv = 0  # sample id for start of next TLV
        self._sample_id_block = None  # sample id for start of current block, None if not started yet

        self._stream_buffer = None  # to ensure same
        self._sb_sample_id_last = None
        self._voltage_range = None

        self._writer = datafile.DataFileWriter(filehandle)
        self._closed = False
        self._total_size = 0
        self._append_configuration()
        if calibration is not None:
            if isinstance(calibration, Calibration):
                calibration = calibration.data
            self._writer.append_subfile('calibration', calibration)
        self._writer.collection_start(0, 0)

    def _append_configuration(self):
        config = {
            'type': 'config',
            'data_recorder_format_version': DATA_RECORDER_FORMAT_VERSION,
            'sampling_frequency': self._sampling_frequency,
            'samples_per_reduction': self._samples_per_reduction,
            'samples_per_tlv': self._samples_per_tlv,
            'samples_per_block': self._samples_per_block,
            'reduction_fields': ['current', 'voltage', 'power',
                                 'current_range', 'current_lsb', 'voltage_lsb']
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
        tlv_offset = (self._sample_id_tlv - self._sample_id_block) // self._samples_per_tlv
        r_stop = tlv_offset * self._reductions_per_tlv
        log.debug('_collection_end(%s, %s)', r_stop, len(self._reduction))
        self._writer.collection_end(collection, self._reduction[:r_stop, :, :].tobytes())
        self._sample_id_block = None
        self._reduction[:] = np.nan

    def stream_notify(self, stream_buffer):
        """Process data from a stream buffer.

        :param stream_buffer: The stream_buffer instance which has a
            "sample_id_range" member, "voltage_range" member,
            raw(start_sample_id, stop_sample_id) and
            get_reduction(reduction_idx, start_sample_id, stop_sample_id).
        """
        sb_start, sb_stop = stream_buffer.sample_id_range
        if self._stream_buffer is None:
            self._stream_buffer = stream_buffer
            self._sample_id_tlv = sb_stop
            self._sample_id_block = None
        elif self._stream_buffer != stream_buffer:
            raise ValueError('Supports only a single stream_buffer instance')

        sample_id_next = self._sample_id_tlv + self._samples_per_tlv
        while sb_stop >= sample_id_next:  # have at least one block
            if self._samples_per_tlv > len(stream_buffer):
                raise ValueError('stream_buffer length too small.  %s > %s' %
                                 (self._samples_per_tlv, len(stream_buffer)))

            self._voltage_range = stream_buffer.voltage_range
            if self._sample_id_block is None:
                collection_data = {
                    'v_range': stream_buffer.voltage_range,
                    'sample_id': sample_id_next,
                }
                collection_data = json.dumps(collection_data).encode('utf-8')
                self._collection_start(data=collection_data)

            log.debug('_process() add tlv %d', self._sample_id_tlv)
            b = stream_buffer.raw_get(self._sample_id_tlv, sample_id_next)
            self._append_data(b.tobytes())
            tlv_offset = (self._sample_id_tlv - self._sample_id_block) // self._samples_per_tlv
            r_start = tlv_offset * self._reductions_per_tlv
            r_stop = r_start + self._reductions_per_tlv
            stream_buffer.data_get(self._sample_id_tlv, sample_id_next,
                                   self._samples_per_reduction, out=self._reduction[r_start:r_stop, :, :])
            self._sample_id_tlv = sample_id_next

            if self._sample_id_tlv - self._sample_id_block >= self._samples_per_block:
                self._collection_end(self._writer.collections[-1])

            sample_id_next += self._samples_per_tlv

    def _append_data(self, data):
        if self._closed:
            return
        expected_len = self._samples_per_tlv * 2 * 2  # two uint16's per sample
        if expected_len != len(data):
            raise ValueError('invalid data length: %d != %d', expected_len, len(data))
        self._writer.append(datafile.TAG_DATA_BINARY, data, compress=False)
        self._total_size += len(data) // 4

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
                    self.config = meta
                elif type_ == 'footer':
                    self.footer = meta
                    break
                else:
                    log.warning('Unknown JSON section type=%s', type_)
            self._f.skip()
        if self._data_start_position == 0 or self.config is None or self.footer is None:
            raise ValueError('could not read file')
        log.info('DataReader with %d samples:\n%s', self.footer['size'], json.dumps(self.config, indent=2))
        if self.config['data_recorder_format_version'] != DATA_RECORDER_FORMAT_VERSION:
            raise ValueError('Invalid file format')
        self.config.setdefault('reduction_fields', ['current', 'voltage', 'power'])
        return self

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
        if self._sample_cache and self._sample_cache['start'] <= sample_idx < self._sample_cache['stop']:
            # cache hit
            return self._sample_cache

        idx_start, idx_end = self.sample_id_range
        if not idx_start <= sample_idx < idx_end:
            raise ValueError('sample index out of range: %d <= %d < %d', idx_start, sample_idx, idx_end)

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
                    data = np.frombuffer(value, dtype=np.uint16).reshape((-1, 2))
                    self._sample_cache = {
                        'voltage_range': voltage_range,
                        'start': current_sample_idx,
                        'stop': tlv_stop,
                        'buffer': data,
                        'tlv_pos': tlv_pos,
                        'block_pos': block_fh_pos,
                    }
                    return self._sample_cache
                else:
                    self._f.advance()
                current_sample_idx = tlv_stop
            else:
                self._f.advance()

    def raw(self, start=None, stop=None, units=None):
        """Get the raw data.

        :param start: The starting time relative to the first sample.
        :param stop: The ending time.
        :param units: The units for start and stop: ['seconds', 'samples'].  None (default) is 'samples'.
        :return: The output which is (out_raw, bits, out_cal).
        """
        start, stop = self.normalize_time_arguments(start, stop, units)
        x_start, x_stop = self.sample_id_range
        if start is None:
            start = x_start
        if stop is None:
            stop = x_stop
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)
        length = stop - start
        if length <= 0:
            return np.empty((0, 2), dtype=np.uint16)
        d_raw = np.empty((length, 2), dtype=np.uint16)
        d_bits = np.empty(length, dtype=np.uint8)
        d_cal = np.empty((length, 2), dtype=np.float32)

        sample_idx = start
        out_idx = 0
        if self._f.advance() != datafile.TAG_COLLECTION_START:
            raise ValueError('data section must be single collection')
        while sample_idx < stop:
            sample_cache = self._sample_tlv(start)
            if sample_cache is None:
                break
            data = sample_cache['buffer']
            b_start = sample_idx - sample_cache['start']
            length = sample_cache['stop'] - sample_cache['start'] - b_start
            out_remaining = stop - sample_idx
            length = min(length, out_remaining)
            if length <= 0:
                break
            b_stop = b_start + length
            d = data[b_start:b_stop, :]
            d_raw[out_idx:(out_idx + length), :] = d
            i_range = np.bitwise_or(np.bitwise_and(d[:, 0], 0x03), np.left_shift(np.bitwise_and(d[:, 1], 0x01), 2))
            i_range[d[:, 0] == 0xffff] = I_RANGE_MISSING
            np.bitwise_or(i_range, np.left_shift(np.bitwise_and(d[:, 0], 0x0004), 2), out=i_range)
            np.bitwise_or(i_range, np.left_shift(np.bitwise_and(d[:, 1], 0x0004), 3), out=i_range)
            d_bits[out_idx:(out_idx + length)] = i_range
            v, i, _ = self.calibration.transform(d, v_range=sample_cache['voltage_range'])
            d_cal[out_idx:(out_idx + length), 0] = v
            d_cal[out_idx:(out_idx + length), 1] = i
            out_idx += length
            sample_idx += length
        return d_raw[:out_idx, :], d_bits[:out_idx], d_cal[:out_idx, :]

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
            field_count = len(self.config['reduction_fields'])
            data = np.frombuffer(value, dtype=np.float32).reshape((-1, field_count, STATS_VALUES))
            if field_count != STATS_FIELDS:
                d_nan = np.full((len(data), STATS_FIELDS - field_count, STATS_VALUES), np.nan, dtype=np.float32)
                data = np.concatenate((data, d_nan), axis=1)
            self._reduction_cache = {
                'r_start': r_idx,
                'r_stop': r_idx_next,
                'buffer': data,
                'next_collection_pos': self._f.tell()
            }
            return self._reduction_cache

    def get_reduction(self, start=None, stop=None, units=None, out=None):
        """Get the fixed reduction with statistics.

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indices.
        :return: The Nx3x4 sample data.
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
            return np.empty((0, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        if out is None:
            out = np.empty((total_length, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
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
            out[out_idx:(out_idx + length), :, :] = data[b_start:(b_start + length), :, :]
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
            s.combine(Statistics(length=length * sz, stats=r[0, :, :]))
            r_idx += length
        return (r_start * sz, r_stop * sz), s

    def get_calibrated(self, start=None, stop=None, units=None):
        """Get the calibrated data (no statistics).

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indices.
        :return: The tuple of (current, voltage), each as np.ndarray
            with dtype=np.float32.
        """
        log.debug('get_calibrated(%s, %s, %s)', start, stop, units)
        _, _, d = self.raw(start, stop)
        i, v = d[:, 0], d[:, 1]
        return i, v

    def get(self, start=None, stop=None, increment=None, units=None):
        """Get the calibrated data with statistics.

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param increment: The number of raw samples per output sample.
        :param units: The units for start and stop.
            'seconds' or None is in floating point seconds relative to the view.
            'samples' is in stream buffer sample indices.
        :return: The Nx3x4 sample data.
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
            return np.empty((0, STATS_FIELDS, STATS_VALUES), dtype=np.float32)
        out = np.empty((out_len, STATS_FIELDS, STATS_VALUES), dtype=np.float32)

        if increment == 1:
            _, d_bits, d_cal = self.raw(start, stop)
            i, v = d_cal[:, 0], d_cal[:, 1]
            out[:, 0, 0] = i
            out[:, 1, 0] = v
            out[:, 2, 0] = i * v
            out[:, 3, 0] = np.bitwise_and(d_bits, 0x0f)
            out[:, 4, 0] = np.bitwise_and(np.right_shift(d_bits, 4), 0x01)
            out[:, 5, 0] = np.bitwise_and(np.right_shift(d_bits, 5), 0x01)
            out[:, :, 1] = 0.0  # zero variance, only one sample!
            out[:, :, 2] = np.nan  # min
            out[:, :, 3] = np.nan  # max
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
                out[idx, :, :] = self._stats_get(k_start, k_stop).value
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

    def _stats_get(self, start, stop):
        s1, s2 = start, stop
        (k1, k2), s = self._get_reduction_stats(s1, s2)
        if k1 >= k2:
            # compute directly over samples
            stats = np.empty((STATS_FIELDS, STATS_VALUES), dtype=np.float32)
            _, d_bits, z = self.raw(s1, s2)
            i, v = z[:, 0], z[:, 1]
            p = i * v
            zi = np.isfinite(i)
            i_view = i[zi]
            if len(i_view):
                i_range = np.bitwise_and(d_bits, 0x0f)
                i_lsb = np.right_shift(np.bitwise_and(d_bits, 0x10), 4)
                v_lsb = np.right_shift(np.bitwise_and(d_bits, 0x20), 5)
                for idx, field in enumerate([i_view, v[zi], p[zi], i_range, i_lsb[zi], v_lsb[zi]]):
                    stats[idx, 0] = np.mean(field, axis=0)
                    stats[idx, 1] = np.var(field, axis=0)
                    stats[idx, 2] = np.amin(field, axis=0)
                    stats[idx, 3] = np.amax(field, axis=0)
            else:
                stats[:, :] = np.full((1, STATS_FIELDS, STATS_VALUES), np.nan, dtype=np.float32)
                stats[3, 0] = I_RANGE_MISSING
                stats[3, 1] = 0
                stats[3, 2] = I_RANGE_MISSING
                stats[3, 3] = I_RANGE_MISSING
            s = Statistics(length=len(i_view), stats=stats)
        else:
            if s1 < k1:
                s_start = self._stats_get(s1, k1)
                s.combine(s_start)
            if s2 > k2:
                s_stop = self._stats_get(k2, s2)
                s.combine(s_stop)
        return s

    def statistics_get(self, start=None, stop=None, units=None):
        """Get the statistics for the collected sample data over a time range.

        :param start: The starting time relative to the first sample.
        :param stop: The ending time.
        :param units: The units for start and stop.
            'seconds' is in floating point seconds relative to the view.
            'samples' or None is in stream buffer sample indices.
        :return: The statistics data structure.  See :meth:`joulescope.driver.Driver.statistics_get`
            for details.
        """
        log.debug('statistics_get(%s, %s, %s)', start, stop, units)
        s1, s2 = self.normalize_time_arguments(start, stop, units)
        if s1 == s2:
            s2 = s1 + 1  # always try to produce valid statistics
        s = self._stats_get(s1, s2)
        t_start = s1 / self.sampling_frequency
        t_stop = s2 / self.sampling_frequency
        return stats_to_api(s.value, t_start, t_stop)
