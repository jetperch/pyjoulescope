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
import json
import numpy as np
import datetime
import copy
import os
import logging

log = logging.getLogger(__name__)

SAMPLING_FREQUENCY = 2000000
SAMPLES_PER_BLOCK = 100000


def construct_record_filename():
    time_start = datetime.datetime.utcnow()
    timestamp_str = time_start.strftime('%Y%m%d_%H%M%S')
    name = '%s.jls' % (timestamp_str, )
    return os.path.join(JOULESCOPE_DIR, name)


class DataRecorderConfiguration:

    def __init__(self):
        self.sampling_frequency = SAMPLING_FREQUENCY  # type: float
        """The sampling frequency in samples per second"""

        self.samples_per_block = SAMPLES_PER_BLOCK    # type: int
        """The number of samples stored in each data block."""

        self.reductions = [200, 100, 50]  # type: List[int]
        """The reductions as a list of samples per reduction in units of
        samples in the previous reduction.  The first entry is based upon
        the raw samples."""

        self.blocks_per_reduction = [5, 10, 10]  # type: List[int]
        """Each reduction is represented as a collection, and the reduction 
        data is stored in the collection end payload.  This field determines
        the number of blocks in each collection level with respect to the
        previous level.  Each block must fall on a reduction boundary!
        The total length must not exceed the stream_buffer total length."""

        self.sample_id_offset = 0   # type: int
        """The starting sample ID."""

    def validate(self):
        if len(self.reductions) != len(self.blocks_per_reduction):
            raise ValueError('reduction length mismatch')
        reductions = 1
        blocks = 1
        for idx, (r, b) in enumerate(zip(self.reductions, self.blocks_per_reduction)):
            reductions *= r
            blocks *= b
            samples = blocks * self.samples_per_block
            if 0 != samples % reductions:
                raise ValueError('reduction %d mismatch' % reductions)
            if 0 != samples % blocks:
                raise ValueError('blocks %d mismatch' % blocks)
        self.sample_id_offset = (self.sample_id_offset // reductions) * reductions

    def as_dict(self):
        return {
            'type': 'config',
            'sampling_frequency': self.sampling_frequency,
            'samples_per_block': self.samples_per_block,
            'reductions': self.reductions,
            'blocks_per_reduction': self.blocks_per_reduction,
            'sample_id_offset': self.sample_id_offset
        }


class DataRecorder:
    """Record Joulescope data to a file."""

    def __init__(self, filehandle, calibration=None, configuration=None):
        """Create a new instance.

        :param filehandle: The file-like object or file name.
        :param calibration: The calibration bytes in datafile format.
            None (default) uses the unit gain calibration.
        :param configuration: The :class:`DataRecorderConfiguration`.
        """
        log.info('init')
        if isinstance(filehandle, str):
            self._fh = open(filehandle, 'wb')
            filehandle = self._fh
        else:
            self._fh = None
        if configuration is None:
            self._config = DataRecorderConfiguration()
        else:
            self._config = copy.deepcopy(configuration)
        self.stream_buffer = None
        self._config.validate()
        self._sample_id = 0
        self._blocks_remaining = copy.copy(self._config.blocks_per_reduction)  # type: List[int]
        self._writer = datafile.DataFileWriter(filehandle)
        self._closed = False
        self._total_size = 0
        self._append_configuration()
        if calibration is None:
            calibration = Calibration().save()  # default calibration
        self._writer.append_subfile('calibration', calibration)
        self._writer.collection_start(0, 0)

    def _append_configuration(self):
        cfg_data = json.dumps(self._config.as_dict()).encode('utf-8')
        self._writer.append(datafile.TAG_META_JSON, cfg_data)

    def _collection_start(self):
        idx = len(self._writer.collections)
        c = self._writer.collection_start(idx, 0)
        reduction_idx = len(self._config.reductions) - idx
        c.metadata = {'reduction_idx': reduction_idx, 'start_sample_id': self._sample_id}
        c.on_end = self._collection_end

    def _collection_end(self, collection):
        samples_per_block = self._config.samples_per_block
        idx = collection.metadata['reduction_idx']
        start_sample_id = collection.metadata['start_sample_id']
        stop_sample_id = (self._sample_id // samples_per_block) * samples_per_block
        data = self.stream_buffer.get_reduction(idx, start_sample_id, stop_sample_id)
        self._writer.collection_end(collection, data.tobytes())

    def process(self, stream_buffer):
        """Process data from a stream buffer.

        :param stream_buffer: The stream_buffer instance which has a
            "sample_id_range" member, raw(start_sample_id, stop_sample_id) and
            get_reduction(reduction_idx, start_sample_id, stop_sample_id).
        """
        if self.stream_buffer is not None and self.stream_buffer != stream_buffer:
            raise ValueError('stream buffer mismatch')
        self.stream_buffer = stream_buffer

        sample_id_next = self._sample_id + self._config.samples_per_block
        while stream_buffer.sample_id_range[1] > sample_id_next:  # have at least one block
            # start collections as needed
            while len(self._writer.collections) <= len(self._blocks_remaining):
                self._collection_start()

            b = stream_buffer.raw_get(self._sample_id, sample_id_next)
            self.append_data(b.tobytes())
            self._sample_id = sample_id_next

            # end collections as needed
            for idx in range(len(self._config.reductions)):
                self._blocks_remaining[idx] -= 1
                if self._blocks_remaining[idx] > 0:
                    break
                self._collection_end(self._writer.collections[-1])
                self._blocks_remaining[idx] = self._config.blocks_per_reduction[idx]

            sample_id_next += self._config.samples_per_block

    def append_data(self, data):
        if self._closed:
            return
        expected_len = self._config.samples_per_block * 2 * 2  # two uint16's per sample
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

    def close(self):
        if self._fh_close:
            self._fh.close()
        self._fh_close = False
        self._fh = None
        self._f = None

    def open(self, filehandle):
        self.close()
        self.calibration = None
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
        log.info('DataReader with %d samples', self.footer['size'])
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
    def duration(self):
        f = self.sampling_frequency
        if f > 0:
            r = self.sample_id_range
            return (r[1] - r[0]) / f
        return 0.0

    def _validate_range(self, start, stop, increment=None):
        idx_start = 0
        idx_end = idx_start + self.footer['size']
        if increment is not None:
            idx_end = ((idx_end + increment - 1) // increment) * increment
        log.info('[%d, %d] : [%d, %d]', start, stop, idx_start, idx_end)
        if not idx_start <= start < idx_end:
            raise ValueError('start out of range: %d <= %d < %d' % (idx_start, start, idx_end))
        if not idx_start <= stop <= idx_end:
            raise ValueError('stop out of range: %d <= %d <= %d: %s' %
                             (idx_start, stop, idx_end, increment))

    def raw(self, start, stop, out=None):
        """Get the raw data.

        :param start: The starting sample identifier.
        :param stop: The ending sample identifier.
        :param out: The optional output Nx2 np.uint16 output array.
            N must be >= (stop - start).
        :return: The output which is either a new array or (when provided) out.
        """
        self._fh.seek(self._data_start_position)
        self._validate_range(start, stop)
        length = stop - start
        if length <= 0:
            return np.empty((0, 2), dtype=np.uint16)
        if out is None:
            out = np.empty((length, 2), dtype=np.uint16)

        start -= 0
        block_start = start // self.config['samples_per_block']
        block_counter = 0
        while True:
            tag, _ = self._f.peek_tag_length()
            if tag == datafile.TAG_DATA_BINARY:
                if block_counter >= block_start:
                    break
                else:
                    block_counter += 1
            self._f.advance()
        block_start_idx = start - block_counter * self.config['samples_per_block']
        out_idx = 0
        while out_idx < length:
            tag, value = next(self._f)
            if tag != datafile.TAG_DATA_BINARY:
                continue
            data = np.frombuffer(value, dtype=np.uint16).reshape((-1, 2))
            block_end_idx = len(data)
            if (block_end_idx - block_start_idx) >= (length - out_idx):
                block_end_idx = block_start_idx + length - out_idx
            k = block_end_idx - block_start_idx
            out[out_idx:(out_idx + k), :] = data[block_start_idx:block_end_idx, :]
            out_idx += k
            block_start_idx = 0
        return out[:length, :]

    def reduction(self, reduction_idx, start, stop, out=None):
        """Get the raw data.

        :param reduction_idx: The reduction index.  0 represents the fewest
            raw samples per sample.
        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param out: The optional output Nx2 np.uint16 output array.
            N must be >= (stop - start) / reduction.
        :return: The output which is either a new array or (when provided) out.
        """
        self._fh.seek(self._data_start_position)
        reductions = self.config['reductions']
        if len(reductions) <= reduction_idx:
            raise ValueError('Invalid reduction index')
        collection_idx = len(reductions) - reduction_idx
        raw_samples_per_sample = np.prod(reductions[:(reduction_idx + 1)])
        blocks_per_collection = np.prod(self.config['blocks_per_reduction'][:(reduction_idx + 1)])
        samples_per_collection = (self.config['samples_per_block'] * blocks_per_collection) // raw_samples_per_sample
        self._validate_range(start, stop, increment=raw_samples_per_sample)

        start = (start // raw_samples_per_sample) * raw_samples_per_sample
        length = (stop - start) // raw_samples_per_sample
        reduction_start = (start - 0) // raw_samples_per_sample

        if length <= 0:
            return np.empty((0, 3, 4), dtype=np.float32)
        if out is None:
            out = np.empty((length, 3, 4), dtype=np.float32)

        sample_idx = 0  # in reduction samples
        collection_depth = 0
        out_idx = 0
        while True:
            tag, _ = self._f.peek_tag_length()
            if tag != datafile.TAG_COLLECTION_START:
                if tag is None:
                    out[out_idx:, :, :] = np.nan
                    break
                self._f.advance()
            elif collection_depth < collection_idx:
                collection_depth += 1
                self._f.advance()
            elif sample_idx + samples_per_collection < reduction_start:
                sample_idx += samples_per_collection
                self._f.skip()
            else:
                self._f.collection_goto_end()
                _, value = next(self._f)
                d = np.frombuffer(value, dtype=np.float32)
                d = d.reshape((len(d) // 12, 3, 4))
                if sample_idx < reduction_start:
                    block_idx_start = reduction_start - sample_idx
                else:
                    block_idx_start = 0
                block_idx_end = len(d)
                if block_idx_end - block_idx_start > length - out_idx:
                    block_idx_end = block_idx_start + (length - out_idx)
                k = block_idx_end - block_idx_start
                r = d[block_idx_start:block_idx_end, :, :]
                out[out_idx:(out_idx + len(r)), :, :] = r
                out_idx += k
                sample_idx += samples_per_collection
                if sample_idx >= reduction_start + length:
                    break
        return out[:length, :, :]

    def _reduction_find(self, increment):
        reductions = self.config.get('reductions')
        reductions = [] if reductions is None else reductions
        idx = -1
        r_scale = 1
        for idx, reduction in enumerate(reductions):
            r_scale_next = r_scale * reduction
            if r_scale_next > increment:
                return idx - 1, r_scale
            r_scale = r_scale_next
        return idx, r_scale

    def get(self, start, stop, increment):
        """Get the data with statistics.

        :param start: The starting sample identifier (inclusive).
        :param stop: The ending sample identifier (exclusive).
        :param increment: The number of raw samples per output sample.
        :return: The tuple (x, data).  X is the sample_id values,The output which is either a new array or (when provided) out.
        """
        if self._fh is None:
            raise IOError('file not open')
        increment = max(1, int(np.round(increment)))

        # Adjust start and stop bounds
        idx_start = 0
        idx_start_q = (idx_start + increment - 1) // increment  # round up
        idx_end = idx_start + self.footer['size']
        idx_end_q = (idx_end + increment - 1) // increment  # round up
        tgt_start_q = (start + increment - 1) // increment  # round up
        tgt_stop_q = (stop + increment - 1) // increment  # round up (exclusive)
        out_len = tgt_stop_q - tgt_start_q
        if out_len <= 0:
            return (np.zeros([], dtype=float),
                    np.empty((0, 3, 4), dtype=np.float32))

        start_q = max(tgt_start_q, idx_start_q)
        tgt_start_offset_q = max(0, start_q - tgt_start_q)
        tgt_stop_q = min(tgt_stop_q, idx_end_q)
        tgt_stop_offset_q = tgt_stop_q - tgt_start_q
        idx_len = tgt_stop_q - start_q
        start = start_q * increment
        stop = tgt_stop_q * increment

        x = np.arange(out_len, dtype=np.float)
        x *= increment
        x += tgt_start_q * increment

        out = np.empty((out_len, 3, 4), dtype=np.float32)
        out[:tgt_start_offset_q, :, :] = np.nan
        out[tgt_stop_offset_q:, :, :] = np.nan
        outv = out[tgt_start_offset_q:tgt_stop_offset_q, :, :]

        r_idx, r_scale = self._reduction_find(increment)

        if increment == 1:
            d = self.raw(start, stop)
            i, v, _ = self.calibration.transform(d)
            outv[:, 0, 0] = i
            outv[:, 1, 0] = v
            outv[:, 2, 0] = i * v
            outv[:, :, 1] = 0.0  # zero variance, only one sample!
            outv[:, :, 2] = np.nan  # min
            outv[:, :, 3] = np.nan  # max
        elif r_idx < 0:
            z = self.raw(start, stop)
            i, v, _ = self.calibration.transform(z)
            p = i * v
            for idx in range(idx_len):
                idx_start = idx * increment
                idx_stop = (idx + 1) * increment
                i_view = i[idx_start:idx_stop]
                zi = np.isfinite(i_view)
                if len(i_view):
                    v_view = v[idx_start:idx_stop][zi]
                    p_view = p[idx_start:idx_stop][zi]
                    outv[idx, 0, :] = np.vstack((np.mean(i_view, axis=0), np.var(i_view, axis=0),
                                                 np.amin(i_view, axis=0), np.amax(i_view, axis=0))).T
                    outv[idx, 1, :] = np.vstack((np.mean(v_view, axis=0), np.var(v_view, axis=0),
                                                 np.amin(v_view, axis=0), np.amax(v_view, axis=0))).T
                    outv[idx, 2, :] = np.vstack((np.mean(p_view, axis=0), np.var(p_view, axis=0),
                                                 np.amin(p_view, axis=0), np.amax(p_view, axis=0))).T
                else:
                    out[idx, :, :] = np.full((1, 3, 4), np.nan, dtype=np.float32)
        else:
            z = self.reduction(r_idx, start, stop)
            increment_fract = increment / r_scale
            for idx in range(idx_len):
                idx_start = int(idx * increment_fract)
                idx_stop = int((idx + 1) * increment_fract)
                i_mean = z[idx_start:idx_stop, 0, 0]
                zi = np.isfinite(i_mean)
                v = z[idx_start:idx_stop, :, :][zi, :, :]
                if len(v):
                    v_mean = np.mean(v[:, :, 0], axis=0)
                    v_min = np.amin(v[:, :, 2], axis=0)
                    v_max = np.amax(v[:, :, 3], axis=0)
                    v_delta = z[:, :, 0] - v_mean
                    v_var = np.mean(v_delta * v_delta + z[:, :, 1], axis=0)
                    outv[idx, :, 0] = v_mean
                    outv[idx, :, 1] = v_var
                    outv[idx, :, 2] = v_min
                    outv[idx, :, 3] = v_max
                else:
                    outv[idx, :, :] = np.full((1, 3, 4), np.nan, dtype=np.float32)

        return x, out
