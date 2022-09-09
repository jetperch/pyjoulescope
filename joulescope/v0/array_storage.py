# Copyright 2018-2020 Jetperch LLC
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


"""
Pack and unpack arrays to a space-efficient format.

The format consists of a constant header followed by tag-length fields
followed by value.

The constant header consists of 4 x uint32:
0: 'ARRY'
1: total size in bytes, including the entire header
2: offset to first value (multiple of 8 bytes)
3: array_count[31:24], sample_count[23:0]

The tag-length fields consist of:
* 4-byte tag: [31:0]: compress_type[31:24], rsv[23:16], array_id[15:8], dtype[7:0]
* 4-byte length: rsv[31:24], length[23:0] of value in bytes (may be zero)

The tag-length fields are place up front (rather than traditional tag-length-value)
to allow for rapid indexing of the values.

The remainder of the payload consists of the values.
All values are padded so that each value starts on on multiple of 8 bytes.


Future work:
* Added SZ 1.x style compression: https://github.com/disheng222/SZ, https://www.mcs.anl.gov/papers/P5437-1115.pdf

"""

import numpy as np


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
_NPDTYPE_TO_DTYPE_MAP = {value: key for key, value in _DTYPE_MAP.items()}


_HEADER = ord('A') + (ord('R') << 8) + (ord('R') << 16) + (ord('Y') << 24)


def pack(arrays, sample_count=None):
    """Pack arrays.

    :param arrays: The dict of 1D np.ndarrays for each item to pack.
        Keys MUST be uint8 values that the application can use to identify
        each array.  Must be contain less that 256 keys.
    :param sample_count: The number of samples in all arrays which must
        be a uint between 0 and 2**24 - 1.  None (default) uses 0.
    :return: The packed np.ndarray(dtype=np.uint8).
    :raise ValueError: On invalid input.
    """
    arrays_count = len(arrays)
    sample_count = 0 if sample_count is None else int(sample_count)
    if not (0 <= sample_count < 2**24):
        raise ValueError(f'sample_count {sample_count} out of range')
    if len(arrays) > 256:
        raise ValueError(f'arrays length {len(arrays)} too big')
    header_size = (((4 + 2 * arrays_count) * 4 + 7) // 8) * 8

    sz = 0
    arrays_fmt = []
    for x_id, x in arrays.items():
        if not (0 <= x_id <= 255):
            raise ValueError(f'invalid array_id {x_id}')
        x_id = int(x_id) & 0xff
        dtype = _NPDTYPE_TO_DTYPE_MAP[x.dtype.type]
        sz_mask = dtype & _DTYPE_SIZE_MASK
        if sz_mask != _DTYPE_SIZE_1B:
            np_dtype = _DTYPE_MAP[dtype]
            if not x.dtype == np_dtype:
                x = x.astype(np_dtype)
            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x, dtype=np_dtype)
            x = x.view(dtype=np.uint8)
        hdr = (x_id << 8) + (dtype & 0xFf)
        arrays_fmt.append((hdr, x))
        sz += ((len(x) + 7) // 8) * 8
    v_total_length = header_size + sz
    v_u32 = np.zeros(v_total_length, dtype=np.uint32)
    v_u8 = v_u32.view(dtype=np.uint8)
    v_u32[0] = _HEADER
    v_u32[1] = v_total_length
    v_u32[2] = header_size
    v_u32[3] = (arrays_count << 24) + sample_count
    offset = header_size
    for idx, (hdr, x) in enumerate(arrays_fmt):
        x_len = len(x)
        value_len = ((x_len + 7) // 8) * 8
        v_u32[4 + idx * 2] = hdr
        v_u32[4 + idx * 2 + 1] = x_len
        v_u8[offset:offset + x_len] = x
        offset += value_len
    return v_u8[:offset]


def unpack(value):
    """Unpack arrays.

    :param value: The np.ndarray(dtype=np.uint8) array returned by :func:`pack`.
    :return: The (arrays, sample_count) provided to pack().
    :raise ValueError: On invalid input.
    """
    # See DataRecorder._arrays_format for format
    if len(value) < 16:
        raise ValueError('Invalid value: too short')
    array_header = value[:16].view(dtype=np.uint32)
    if array_header[0] != _HEADER:
        raise ValueError('Invalid value: bad prefix')
    k = array_header[1]
    if len(value) < k:
        raise ValueError(f'value too short: {len(value)} < {k}')
    offset = array_header[2]
    sample_count = array_header[3] & 0xffffff
    field_count = (array_header[3] >> 24) & 0xff
    offset_expect = 16 + field_count * 8
    if offset_expect > offset:
        raise ValueError('Invalid format')
    tag_length = value[16:offset_expect].view(dtype=np.uint32)
    result = {}
    for idx in range(field_count):
        hdr = tag_length[idx * 2]
        x_len = tag_length[idx * 2 + 1] & 0xffffff
        x_id = (hdr >> 8) & 0xff
        dtype = hdr & 0xff
        np_dtype = _DTYPE_MAP[dtype]
        x = value[offset:offset + x_len]
        x = x.view(dtype=np_dtype)
        result[x_id] = x
        offset += ((x_len + 7) // 8) * 8
    return result, sample_count
