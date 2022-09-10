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

"""
This module implements readers and writers for tag-length-value files
used by Joulescope.

This file format is used for multiple purposes, including:

* Raw data capture, playback & browsing
* Processed data capture, playback & browsing
* Calibration storage
* Firmware update storage

The file format must meet multiple objectives:

* Support a streaming interface, such as over TCP.
* Support fast access loading from disk.
* Support incremental processing, such as by a microcontroller
  for firmware updates.


The streaming requirement means that seeking back to the start of the
file is not allowed.  Any collections or sections must be indicated with
tags.  However, reading and writing files can seek, so tags may be
rewritten with offset information for improved performance.

The file format starts with a 32 byte header:

* 16 bytes: [0xd3, 0x74, 0x61, 0x67, 0x66, 0x6d, 0x74, 0x20, 0x0d, 0x0a, 0x20, 0x0a, 0x20, 0x20, 0x1a, 0x1c]
* 8 bytes: total length in bytes (0="not provided" or "streaming")
* 3 bytes: reserved (0)
* 1 byte: file version (1)
* 4 bytes: crc32 over this header

The prefix is specially selected to ensure:

* Identification: Help the application determine that
  this file is in the correct format with minimal uncertainty.
* Correct endianness:  Little endian has won, so this entire format is
  stored in little endian format.
* Proper binary processing:  The different line ending combinations
  ensure that the reader is not "fixing" the line endings, since this
  is a binary file format.
* Display: Include "substitute" and "file separator" so that text
  printers to not show the rest of the file.

The remaining file contents are in tag-length-value (TLV) format with CRC32:

* 3 bytes: tag
* 1 byte: TLV flags (compression, encryption)
* 4 bytes: length of data in bytes (may be zero)
* length bytes: The data value
* pad bytes: zero padding to 8 byte + 4 boundary so that crc ends on 8 byte boundary
* 4 bytes: crc32

Tags are selected such that the upper byte is 0.  Since the file format is
little endian, this means that the tag has three usable characters.  The
upper tag bits have the following definitions:

* bit 31: 1=compressed, 0=uncompressed
* bit 30: 1=encrypted, 0=unencrypted, ChaCha20 + Poly1305 with EdDSA signature
* bits [28:24]: reserved
* bits [23:0]: Unique tag

The supported tags include:

* b'HDR': common header information.  Must be the first tag, but
  SGS is optionally allowed before.  Files with encrypted tags will
  typically use the first 24 bytes of this field as the nonce, and then
  increment the last uint32 with each new encrypted block.

  * 8 byte timestamp for data creation.  See time.py for timestamp
    format information.
  * 4 byte version of the file data contents: major8, minor8, patch16.
    If this field is not used, set to 0.
  * 4 byte vendor_id: For USB products the MSB is 0 and LSB is the
    USB VID.
  * 2 byte product_id: unique within vendor_id
  * 2 byte subtype_id: application-defined, unique within product_id.
    A single product may include multiple subtypes, such as
    firmware, FPGA bitstreams and calibration data.  Each product may
    assign values for this field or not use it.
  * 4 byte hardware_compatibility: application-defined.
    Each bit represents a potentially incompatible hardware revision.
    This field should set the bit for each hardware version supported.
    If this field is not used, set to 0.
  * 16 byte serial number identifying device associated with this data.
    If this field is not used, set to 0.

* b'END': Indicate data file end.  Must be the last tag.
* b'CLS': collection start.  The payload is:

  * 8 byte position to the collection end tag.
    This allows fast seeking to skip the collection data.
    In streaming datafile mode, the offset is 0.
  * 2 byte file-specific collection identifier
  * 1 byte collection type: 0=unstructured, 1=list, 2=map
  * 1 byte reserved (0)
  * N bytes: optional application specific data.

* b'CLE': collection end.  May contain application-specific data
  such as indices to increase access performance.
* b'SUB': A subfile, which is often used for storing the calibration
  record inside the data capture.  The payload starts with 128 bytes
  127 bytes of UTF-8 encoded characters) that contains the null-terminated
  file name.  Unused bytes MUST be set to 0.  The remaining payload is
  the file in this datafile format.
* b'IDX': application-specific index information.
* b'MJS': application-specific metadata, JSON formatted.
* b'AJS': application-specific data, JSON formatted.
* b'ABN': application-specific data, binary formatted.
* b'UJS': arbitrary end-user data, JSON formatted.
* b'UBN': arbitrary end-user data, binary formatted.
* b'ENC': encryption authenticity and integrity information.
  This tag must follow every block with the encryption bit set.

  * 16 bytes: ChaCha20 + Poly1305 MAC
  * 64 bytes: EdDSA curve25519 using Blake2b hash (monocypher)
    The signature is computed on the UNENCRYPTED data (sign-then-encrypt)
    For firmware updates, we care more that the firmware is valid than
    who created the cryptotext.  If you want to prevent cryptotext
    forgeries, use encrypt-then-sign with use SGS/SGE and the payload only
    flag.

* b'SGS': signature start.  This field (inclusive) and all others
  up to SGE (exclusive) are included in the signature.  Note that this file
  format makes no provisions for managing keys or ensuring key validity.

  * 1 byte: signature type

    * 1 = EdDSA curve25519 using Blake2b hash (monocypher).

  * 1 byte: flags

    * 1 = include this field (default is exclude)
    * 2 = payload only (exclude tag, length & crc32)

  * 6 bytes: reserved zero
  * 32 bytes: public key

* b'SGE': signature end.  This field is exclude from the signature.
  Payload is the signature.

"""

# Use monocypher crypto library rather than libsodium/NaCl
# Design choice allows for same library on microcontrollers
# http://loup-vaillant.fr/tutorials/chacha20-design
# https://github.com/LoupVaillant/Monocypher/blob/master/src/monocypher.c

import struct
import binascii
from joulescope import time
import zlib
import monocypher
import logging

log = logging.getLogger(__name__)

MAGIC = b'\xd3tagfmt \r\n \n  \x1a\x1c'
assert(len(MAGIC) == 16)
HEADER_SIZE = 32
VERSION = 1

FLAG_COMPRESS = 0x80
FLAG_ENCRYPT = 0x40

TAG_HEADER = b'HDR'  # JSON-formatted header
TAG_END = b'END'
TAG_USB = b'USB'
TAG_COLLECTION_START = b'CLS'
TAG_COLLECTION_END = b'CLE'
TAG_SUBFILE = b'SUB'    # subfile: filename + datafile
TAG_INDEX = b'IDX'
TAG_META_JSON = b'MJS'  # JSON formatted file metadata
TAG_DATA_BINARY = b'ABN'  # binary formatted file data
TAG_DATA_JSON = b'AJS'  # JSON formatted file data
TAG_USER_BINARY = b'UBN'  # binary formatted user data
TAG_USER_JSON = b'UJS'  # JSON formatted user data
TAG_CALIBRATION_JSON = b'CJS'  # deprecated, use TAG_DATA_JSON
TAG_SIGNATURE_START = b'SGS'
TAG_SIGNATURE_END = b'SGE'
TAG_ENCRYPTION = b'ENC'


SIGNATURE_FLAG_KEY_INCLUDE = 1
SIGNATURE_FLAG_PAYLOAD_ONLY = 2


def filename_or_bytes(x):
    if x is None:
        return b''
    elif isinstance(x, str):
        with open(x, 'rb') as f:
            return f.read()
    else:
        return x


def default(value, d):
    if value is None:
        return d
    return value


class Header:
    FORMAT = '<qIIHHI'

    def __init__(self, timestamp=None, version=None,
                 product_id=None, vendor_id=None,
                 subtype_id=None, hardware_compatibility=None,
                 serial_number=None):
        self.timestamp = default(timestamp, time.timestamp_now())
        self.version = default(version, 0)
        self.product_id = default(product_id, 0)
        self.vendor_id = default(vendor_id, 0)
        self.subtype_id = default(subtype_id, 0)
        self.hardware_compatibility = default(hardware_compatibility, 0)
        self.serial_number = default(serial_number, bytes([0] * 16))

    def encode(self):
        v = struct.pack(self.FORMAT,
                        self.timestamp, self.version,
                        self.product_id, self.vendor_id,
                        self.subtype_id, self.hardware_compatibility)
        v += self.serial_number
        return v

    @staticmethod
    def decode(b):
        parts = struct.unpack(Header.FORMAT, b[24:])
        serial_number = b[24:]
        return Header(*parts, serial_number)


class Collection:
    FORMAT = '<QHBB'

    def __init__(self, id_=None, type_=None, end_position=None, start_position=None, data=None):
        self.id_ = 0 if id_ is None else id_
        self.type_ = 0 if type_ is None else type_
        self.end_position = 0 if end_position is None else end_position
        self.start_position = 0 if start_position is None else start_position
        self.data = data
        self.metadata = {}  # application-defined metadata

    def encode(self):
        contents = struct.pack(self.FORMAT, self.end_position, self.id_, self.type_, 0)
        if self.data is not None:
            contents = contents + self.data
        return contents

    @staticmethod
    def decode(data):
        end_position, id_, type_, _ = struct.unpack(Collection.FORMAT, data[:12])
        if len(data) > 12:
            collection_data = data[12:]
        else:
            collection_data = None
        return Collection(id_=id_, type_=type_, end_position=end_position, data=collection_data)

    def __repr__(self):
        parts = ['%s=%r' % (x, getattr(self, x)) for x in ['id_', 'type_', 'end_position', 'start_position']]
        return 'Collection(%s)' % (', '.join(parts), )


def _length_to_pad(length):
    remainder = (length + 4) & 0x7
    if remainder == 0:
        return 0
    else:
        return 8 - remainder


def subfile_split(value):
    """Split a subfile into the name and payload.

    :param value: The value in the SUBFILE tag.
    :return: (name, data).
    """
    name = value.split(b'\x00', 1)[0].decode('utf-8')
    data = value[128:]
    return name, data


def _maybe_compress(data):
    length = len(data)
    flags = 0
    if length:
        dataz = zlib.compress(data, level=1)
        lengthz = len(dataz)
        if lengthz < length:
            data = dataz
            flags = FLAG_COMPRESS
    return flags, data


class DataFileWriter:
    """Create a new instance.

    :param filehandle: The file open for write which must support
        the write, seek and tell methods.
    """

    def __init__(self, filehandle):
        self._fh = filehandle
        self.collections = []  # type: List[Collection]
        self._write_file_header()
        self._signature = None

    def _write_file_header(self):
        length = self._fh.tell()
        self._fh.seek(0)
        header = MAGIC + struct.pack('<QBBBB', length, 0, 0, 0, VERSION)
        crc = binascii.crc32(header)
        header = header + struct.pack('<I', crc)
        assert(HEADER_SIZE == len(header))
        self._fh.write(header)
        if length:
            self._fh.seek(length)

    def _append(self, tag, flags, data):
        """Append a new tag-length-value field to the file.

        :param tag: The 3-byte tag as either an integer or bytes.
        :param flags: The flag bytes value
        :param data: The associated data for the tag (optional).
        :return: The starting position for the tag.
        """
        position = self._fh.tell()
        flags = int(flags)
        if flags < 0 or flags > 255:
            raise ValueError('Invalid flags')
        if isinstance(tag, bytes):
            if len(tag) != 3:
                raise ValueError('Invalid tag length %d' % len(tag))
            tag, = struct.unpack('<I', tag + b'\x00')
        tag &= 0xFFFFFF
        tag |= flags << 24
        length = len(data)
        value = bytearray(8 + length + _length_to_pad(length) + 4)
        value[:8] = struct.pack('<II', tag, length)
        value[8:8 + length] = data
        crc = binascii.crc32(value[:-4])
        value[-4:] = struct.pack('<I', crc)
        self._fh.write(value)
        if self._signature is not None:
            if 0 != (self._signature['flags'] & SIGNATURE_FLAG_PAYLOAD_ONLY):
                self._signature['data'] += data
            else:
                self._signature['data'] += value
        return position

    def append(self, tag, data=None, compress=None):
        """Append a new tag-length-value field to the file.

        :param tag: The 3-byte tag as either an integer or bytes.
        :param data: The associated data for the tag (optional).
        :param compress: When False or None, do not attempt to
            compress the data.  When True, attempt to compress
            the data.
        :return: The starting position for the tag.
        """
        data = b'' if data is None else data
        flags = 0
        if bool(compress):
            flags, data = _maybe_compress(data)
        return self._append(tag, flags, data)

    def append_encrypted(self, tag, data, signing_key, encryption_key, nonce, associated_data=None, compress=None):
        flags = 0
        if bool(compress):
            flags, data = _maybe_compress(data)
        flags |= FLAG_ENCRYPT
        signature = monocypher.signature_sign(signing_key, data)
        mac, data = monocypher.lock(encryption_key, nonce, data, associated_data)
        log.info('signature = %r', binascii.hexlify(signature))
        log.info('mac       = %r', binascii.hexlify(mac))
        pos = self._append(tag, flags, data)
        self._append(TAG_ENCRYPTION, 0, mac + signature)
        return pos

    def append_subfile(self, name: str, data, compress=None):
        """Append a subfile.

        :param name: The name of the subfile, which must fit into 127
            bytes encoded as utf-8.
        :param data: The data in this datafile format.
        :param compress: When False or None, do not attempt to
            compress the data.  When True, attempt to compress
            the data.
        """
        n = name.encode('utf-8')
        if len(n) > 127:
            raise ValueError('name too long: %s' % name)
        payload = n + bytes([0] * (128 - len(n))) + data
        self.append(TAG_SUBFILE, payload, compress)

    def signature_start(self, private_key, flags=None):
        self._signature = {
            'flags': 0 if flags is None else int(flags),
            'key': private_key,
            'data': b'',
        }
        payload = bytearray(8)
        payload[0] = 1  # Ed25519 using Blake2b
        payload[1] = self._signature['flags']
        payload += monocypher.compute_signing_public_key(private_key)
        self.append(TAG_SIGNATURE_START, payload)
        if 0 == (self._signature['flags'] & SIGNATURE_FLAG_KEY_INCLUDE):
            self._signature['data'] = b''

    def signature_end(self):
        s = monocypher.signature_sign(self._signature['key'], self._signature['data'])
        self.append(TAG_SIGNATURE_END, s)

    def append_header(self,
                      timestamp=None, version=None,
                      product_id=None, vendor_id=None,
                      subtype_id=None, hardware_compatibility=None,
                      serial_number=None):
        h = Header(timestamp=timestamp, version=version,
                   product_id=product_id,
                   vendor_id=vendor_id, subtype_id=subtype_id,
                   hardware_compatibility=hardware_compatibility,
                   serial_number=serial_number)
        return self.append(TAG_HEADER, h.encode())

    def collection_start(self, id_=None, type_=None, data=None):
        c = Collection(id_, type_, data=data)
        # note: append writes start_position=0 and end_position=0.
        # Both are later updated by collection_end() in normal operation.
        c.start_position = self.append(TAG_COLLECTION_START, c.encode())
        self.collections.append(c)
        return c

    def collection_end(self, collection=None, data=None):
        c = self.collections.pop()
        if collection is not None and collection != c:
            raise RuntimeError('collection mismatch')
        c.end_position = self.append(TAG_COLLECTION_END, data=data)
        current_position = self._fh.tell()
        self._fh.seek(c.start_position)
        self.append(TAG_COLLECTION_START, c.encode())
        self._fh.seek(current_position)

    def finalize(self):
        for c in self.collections[-1::-1]:
            log.info('collection unclosed: id=%d, type=%d' % (c.id_, c.type_))
            self.collection_end(c)
        self.append(TAG_END)
        self._write_file_header()


def validate_file_header(data):
    if len(data) < HEADER_SIZE:
        raise ValueError('data too short')
    header = data[:HEADER_SIZE]
    if MAGIC != header[:16]:
        raise IOError('invalid file')
    length, version, crc_read = struct.unpack('<QxxxBI', header[16:])
    crc_compute = binascii.crc32(header[:(HEADER_SIZE - 4)])
    if crc_read != crc_compute:
        raise ValueError('invalid header crc: 0x%08x != 0x%08x' % (crc_read, crc_compute))
    if VERSION != version:
        raise ValueError('unsupported file version %d' % version)
    return length, version


def _parse_tag_start(data):
    flags = data[3]
    tag = data[:3]
    length, = struct.unpack('<I', data[4:])
    return tag, flags, length


def _maybe_decompress(value, flags):
    if flags & FLAG_COMPRESS:
        if flags & FLAG_ENCRYPT:
            raise ValueError('Connect decompress encrypted data')
        value = zlib.decompress(value)
    return value


class DataFileReader:
    """Create a new instance.

    :param filehandle: The file-like object open for read.  The file must
        support read, seek and tell.
    """

    def __init__(self, filehandle):
        self._signature_key = None
        self._signature_data = None
        self._fh = filehandle
        header = self._fh.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise IOError('file too small')
        self.length, self.version = validate_file_header(header)
        pos1 = self._fh.tell()
        self._fh.seek(0, 2)
        file_length = self._fh.tell()
        self._fh.seek(pos1)
        if self.length == 0:
            log.warning('length was not written, use file length %d', file_length)
            self.length = file_length

    @property
    def remaining_length(self):
        pos = self._fh.tell()
        remaining = self.length - pos
        if remaining < 0:
            remaining = 0
        return remaining

    def __iter__(self):
        return self

    def _read_tag(self):
        remaining_length = self.remaining_length
        if 0 <= remaining_length < 16:
            log.warning('tag read truncated at %d of %d', remaining_length, self._fh.tell())
            return None, 0, b'', b''
        tag_length = self._fh.read(8)
        tag, flags, length = _parse_tag_start(tag_length)
        log.debug('tag read %s length=%d at %d of %d', tag, length, self._fh.tell(), self.length)
        entry = bytearray(8 + length + _length_to_pad(length) + 4)
        entry[:8] = tag_length
        pad = _length_to_pad(length)
        tlv_size = 8 + length + pad + 4
        if remaining_length < tlv_size:
            raise ValueError('File too short')
        entry[8:] = self._fh.read(tlv_size - 8)
        crc = binascii.crc32(entry[:-4])
        crc_read, = struct.unpack('<I', entry[-4:])
        if crc != crc_read:
            log.warning('invalid tag crc: 0x%08x != 0x%08x' % (crc_read, crc))
            # raise ValueError('invalid tag crc: 0x%08x != 0x%08x' % (crc_read, crc))
        value = entry[8:8+length]
        return tag, flags, value, entry

    def tell(self):
        """Give the location of the current entry.

        :return: The position suitable for :meth:`seek`.
        """
        return self._fh.tell()

    def seek(self, position):
        """Change to the location of another entry.

        :param position: The position returned by a previous call
            to :meth:`tell`.
        """
        self._fh.seek(position)
        return position

    def peek_tag_length(self):
        """Peek at the next available entry.

        :return: tuple (tag, value_length)

        This method gets the tag and length quickly.  It does not
        load the data or validate the checksum.
        """
        if 0 <= self.remaining_length < 8:
            return None, 0
        position = self._fh.tell()
        tag_length = self._fh.read(8)
        tag, _, length = _parse_tag_start(tag_length)
        self._fh.seek(position)
        return tag, length

    def peek(self):
        """Peek at the next available entry.

        :return: tuple (tag, value)
        """
        position = self._fh.tell()
        tag, flags, value, _ = self._read_tag()
        self._fh.seek(position)
        value = _maybe_decompress(value, flags)
        return tag, value

    def advance(self):
        """Advance to the next TLV, ignoring data.

        :return: The tag that was skipped.
        """
        if 0 <= self.remaining_length < 16:
            return None
        position = self._fh.tell()
        tag_length = self._fh.read(8)
        tag, _, length = _parse_tag_start(tag_length)
        pad = _length_to_pad(length)
        self._fh.seek(position + 8 + length + pad + 4)
        return tag

    def skip(self):
        """Skip the next available entry, skipping entire collections.

        :return: The tag that was skipped.
        """
        if 0 <= self.remaining_length < 16:
            return None
        position = self._fh.tell()
        tag_length = self._fh.read(8)
        tag, _, length = _parse_tag_start(tag_length)
        if tag == TAG_COLLECTION_START:
            self._fh.seek(position)
            self.collection_goto_end()
            self.skip()
        else:
            pad = _length_to_pad(length)
            self._fh.seek(position + 8 + length + pad + 4)
        return tag

    def collection_goto_end(self):
        """Skip to the collection end.

        :raise RuntimeError: If the current tag is not a COLLECTION_START.
        """
        if 0 <= self.remaining_length < 16:
            return None
        tag, value = next(self)
        if tag != TAG_COLLECTION_START:
            raise RuntimeError('Not a collection start')
        c = Collection.decode(value)
        return self._fh.seek(c.end_position)

    def decrypt(self, signing_key, encryption_key, nonce, associated_data=None):
        """Decrypt the next tag, if needed"""
        tag, flags, value, _ = self._read_tag()
        if flags & FLAG_ENCRYPT:
            tag2, _, value2, _ = self._read_tag()
            if tag2 != TAG_ENCRYPTION:
                raise ValueError('Encrypted data must be followed by ENC tag')
            mac = value2[:16]
            signature = value2[16:]
            value = monocypher.unlock(encryption_key, nonce, mac, value, associated_data)
            if value is None:
                raise ValueError('Decryption failed')
            if not monocypher.signature_check(signature, signing_key, value):
                raise ValueError('Signature check failed')
        if flags & FLAG_COMPRESS:
            value = zlib.decompress(value)
        return tag, value

    def __next__(self):
        """Get the next available entry.

        :return: tuple (tag, value).
        :raise StopIteration: when no more entries remain.
        :raise ValueError: on signature check failure.
        """
        tag, flags, value, entry = self._read_tag()
        if tag is None or tag == TAG_END:
            raise StopIteration()
        if tag == TAG_SIGNATURE_START:
            self._signature_key = bytes(value[8:])
            if value[1] & 1:
                self._signature_data = bytes(entry)
            else:
                self._signature_data = b''
        elif tag == TAG_SIGNATURE_END:
            signature = bytes(value)
            value = monocypher.signature_check(signature, self._signature_key, self._signature_data)
            if not value:
                raise ValueError('signature check failed')
            self._signature_key = None
            self._signature_data = None
        elif self._signature_data is not None:
            self._signature_data += entry
        value = _maybe_decompress(value, flags)
        return tag, value

    def pretty_print(self):
        """Pretty print the datafile structure."""
        indent_str = '  '
        indent = 0
        for tag, value in self:
            if tag == TAG_COLLECTION_END:
                indent -= 1
            tag_str = tag[:3].decode('utf-8')
            if tag == TAG_COLLECTION_START:
                c = Collection.decode(value)
                print("%s%s id=%d, type=%d, start=%d, end=%d" %
                      (indent_str * indent, tag_str, c.id_, c.type_, c.start_position, c.end_position))
                indent += 1
            elif tag == TAG_SIGNATURE_END:
                print("%s%s %s" % (indent_str * indent, tag_str, value))
            else:
                print("%s%s %d" % (indent_str * indent, tag_str, len(value)))
