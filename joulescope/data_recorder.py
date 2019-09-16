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
from joulescope.stream_buffer import reduction_downsample, Statistics, stats_to_api
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

        self._reduction = np.empty((reduction_block_size, 3, 4), dtype=np.float32)
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
        while sb_stop > sample_id_next:  # have at least one block
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

    def __str__(self):
        if self._f is not None:
            return 'DataReader %.2f seconds (%d samples)' % (self.duration, self.footer['size'])

    def close(self):
        if self._fh_close:
            self._fh.close()
        self._fh_close = False
        self._fh = None
        self._f = None

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
        idx_start = 0
        idx_end = idx_start + self.footer['size']
        if increment is not None:
            idx_end = ((idx_end + increment - 1) // increment) * increment
        log.debug('[%d, %d] : [%d, %d]', start, stop, idx_start, idx_end)
        if not idx_start <= start < idx_end:
            raise ValueError('start out of range: %d <= %d < %d' % (idx_start, start, idx_end))
        if not idx_start <= stop <= idx_end:
            raise ValueError('stop out of range: %d <= %d <= %d: %s' %
                             (idx_start, stop, idx_end, increment))

    def raw(self, start=None, stop=None, units=None, calibrated=None, out=None):
        """Get the raw data.

        :param start: The starting time relative to the first sample.
        :param stop: The ending time.
        :param units: The units for start and stop: ['seconds', 'samples'].  None (default) is 'samples'.
        :param calibrated: When true, return calibrated np.float32 data.
            When false, return raw np.uint16 data.
        :param out: The optional output Nx2 output array.
            N must be >= (stop - start).
        :return: The output which is either a new array or (when provided) out.
        """
        start, stop = self.normalize_time_arguments(start, stop, units)
        r_start, r_stop = self.sample_id_range
        if start is None:
            start = r_start
        if stop is None:
            stop = r_stop        
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)
        length = stop - start
        if length <= 0:
            return np.empty((0, 2), dtype=np.uint16)
        if out is None:
            if calibrated:
                out = np.empty((length, 2), dtype=np.float32)
            else:
                out = np.empty((length, 2), dtype=np.uint16)

        sample_idx = 0
        samples_per_tlv = self.config['samples_per_tlv']
        samples_per_block = self.config['samples_per_block']
        block_start = start // samples_per_block
        block_counter = 0
        out_idx = 0
        if self._f.advance() != datafile.TAG_COLLECTION_START:
            raise ValueError('data section must be single collection')
        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None:
                break
            if tag == datafile.TAG_COLLECTION_START:
                if block_counter < block_start:
                    self._f.skip()
                    block_counter += 1
                else:
                    tag, collection_bytes = next(self._f)
                    c = datafile.Collection.decode(collection_bytes)
                    if c.data is None:
                        self._voltage_range = 0
                    else:
                        collection_start_meta = json.loads(c.data)
                        self._voltage_range = collection_start_meta.get('v_range', 0)
                    sample_idx = block_counter * samples_per_block
            elif tag == datafile.TAG_COLLECTION_END:
                block_counter += 1
                self._f.advance()
            elif tag == datafile.TAG_DATA_BINARY:
                tlv_stop = sample_idx + samples_per_tlv
                if start < tlv_stop:
                    tag, value = next(self._f)
                    data = np.frombuffer(value, dtype=np.uint16).reshape((-1, 2))
                    idx_start = 0
                    idx_stop = samples_per_tlv
                    if start > sample_idx:
                        idx_start = start - sample_idx
                    if stop < tlv_stop:
                        idx_stop = stop - sample_idx
                    length = idx_stop - idx_start
                    if calibrated:
                        v, i, _ = self.calibration.transform(data[idx_start:idx_stop, :],
                                                             v_range=self._voltage_range)
                        out[out_idx:(out_idx + length), 0] = v
                        out[out_idx:(out_idx + length), 1] = i
                    else:
                        out[out_idx:(out_idx + length), :] = data[idx_start:idx_stop, :]
                    out_idx += length
                else:
                    self._f.advance()
                sample_idx = tlv_stop
                if sample_idx > stop:
                    break
            else:
                self._f.advance()
        return out[:out_idx, :]

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
        sz = self.config['samples_per_reduction']
        incr = self.config['samples_per_block'] // sz
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)
        r_start = start // sz
        length = (stop - start) // sz
        r_stop = r_start + length
        log.info('DataReader.get_reduction(r_start=%r,r_stop=%r)', r_start, r_stop)
        if length <= 0:
            return np.empty((0, 3, 4), dtype=np.float32)
        if out is None:
            out = np.empty((length, 3, 4), dtype=np.float32)
        elif len(out) < length:
            raise ValueError('out too small')

        out_idx = 0
        r_idx = 0

        if self._f.advance() != datafile.TAG_COLLECTION_START:
            raise ValueError('data section must be single collection')
        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None or tag == datafile.TAG_COLLECTION_END:
                break
            elif tag != datafile.TAG_COLLECTION_START:
                raise ValueError('invalid file format: not collection start')
            r_idx_next = r_idx + incr
            if r_start >= r_idx_next:
                self._f.skip()
                r_idx = r_idx_next
                continue
            self._f.collection_goto_end()
            tag, value = next(self._f)
            if tag != datafile.TAG_COLLECTION_END:
                raise ValueError('invalid file format: not collection end')
            data = np.frombuffer(value, dtype=np.float32).reshape((-1, 3, 4))
            r_idx_start = 0
            r_idx_stop = incr
            if r_idx < r_start:
                r_idx_start = r_start - r_idx
            if r_idx_next > r_stop:
                r_idx_stop = r_stop - r_idx
            if r_idx_stop > len(data):
                r_idx_stop = len(data)
            copy_len = r_idx_stop - r_idx_start
            out[out_idx:(out_idx + copy_len), :, :] = data[r_idx_start:r_idx_stop, :, :]
            out_idx += copy_len
            r_idx = r_idx_next
            if r_idx_next >= r_stop:
                break
        if out_idx != length:
            log.warning('DataReader length mismatch: out_idx=%s, length=%s', out_idx, length)
            length = min(out_idx, length)
        return out[:length, :]

    def _get_reduction_stats(self, start, stop):
        """Get statistics over the reduction

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :return: The tuple of ((sample_start, sample_stop), :class:`Statistics`).
        """
        log.debug('_get_reduction_stats(%s, %s)', start, stop)
        s = Statistics()
        sz = self.config['samples_per_reduction']
        incr = self.config['samples_per_block'] // sz
        r_start = start // sz
        if (r_start * sz) < start:
            r_start += 1
        r_stop = stop // sz
        if r_start >= r_stop:  # use the reductions
            s_start = r_start * sz
            return (s_start, s_start), s
        r_idx = 0

        self._fh.seek(self._data_start_position)
        if self._f.advance() != datafile.TAG_COLLECTION_START:
            raise ValueError('data section must be single collection')
        while True:
            tag, _ = self._f.peek_tag_length()
            if tag is None or tag == datafile.TAG_COLLECTION_END:
                break
            elif tag != datafile.TAG_COLLECTION_START:
                raise ValueError('invalid file format: not collection start')
            r_idx_next = r_idx + incr
            if r_start >= r_idx_next:
                self._f.skip()
                r_idx = r_idx_next
                continue
            self._f.collection_goto_end()
            tag, value = next(self._f)
            if tag != datafile.TAG_COLLECTION_END:
                raise ValueError('invalid file format: not collection end')
            data = np.frombuffer(value, dtype=np.float32).reshape((-1, 3, 4))
            r_idx_start = 0
            r_idx_stop = incr
            if r_idx < r_start:
                r_idx_start = r_start - r_idx
            if r_idx_next > r_stop:
                r_idx_stop = r_stop - r_idx
            if r_idx_stop > len(data):
                r_idx_stop = len(data)
            length = r_idx_stop - r_idx_start
            r = reduction_downsample(data, r_idx_start, r_idx_stop, length)
            s.combine(Statistics(length=length * sz, stats=r[0, :, :]))
            r_idx = r_idx_next
            if r_idx_next >= r_stop:
                break
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
        d = self.raw(start, stop, calibrated=True)
        i, v = d[:, 0], d[:, 1]
        return i, v

    def get(self, start=None, stop=None, increment=None, units=None):
        """Get the data with statistics.

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
            return np.empty((0, 3, 4), dtype=np.float32)
        out = np.empty((out_len, 3, 4), dtype=np.float32)

        if increment == 1:
            d = self.raw(start, stop, calibrated=True)
            i, v = d[:, 0], d[:, 1]
            out[:, 0, 0] = i
            out[:, 1, 0] = v
            out[:, 2, 0] = i * v
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
            z = self.raw(start, stop, calibrated=True)
            i, v = z[:, 0], z[:, 1]
            p = i * v
            for idx in range(out_len):
                idx_start = idx * increment
                idx_stop = (idx + 1) * increment
                i_view = i[idx_start:idx_stop]
                zi = np.isfinite(i_view)
                i_view = i_view[zi]
                if len(i_view):
                    v_view = v[idx_start:idx_stop][zi]
                    p_view = p[idx_start:idx_stop][zi]
                    out[idx, 0, :] = np.vstack((np.mean(i_view, axis=0), np.var(i_view, axis=0),
                                                np.amin(i_view, axis=0), np.amax(i_view, axis=0))).T
                    out[idx, 1, :] = np.vstack((np.mean(v_view, axis=0), np.var(v_view, axis=0),
                                                np.amin(v_view, axis=0), np.amax(v_view, axis=0))).T
                    out[idx, 2, :] = np.vstack((np.mean(p_view, axis=0), np.var(p_view, axis=0),
                                                np.amin(p_view, axis=0), np.amax(p_view, axis=0))).T
                else:
                    out[idx, :, :] = np.full((1, 3, 4), np.nan, dtype=np.float32)
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

        (k1, k2), s = self._get_reduction_stats(s1, s2)
        if s1 < k1:
            length = k1 - s1
            s_start = self.get(s1, k1, increment=length)
            s.combine(Statistics(length=length, stats=s_start[0, :, :]))
        if s2 > k2:
            length = s2 - k2
            s_stop = self.get(k2, s2, increment=length)
            s.combine(Statistics(length=length, stats=s_stop[0, :, :]))

        t_start = s1 / self.sampling_frequency
        t_stop = s2 / self.sampling_frequency
        return stats_to_api(s.value, t_start, t_stop)
