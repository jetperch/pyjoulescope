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
Test the datafile

"""

import unittest
from joulescope import datafile
import io
import binascii
import struct
import numpy as np
import monocypher


F1 = b'\xd3tagfmt \r\n \n  \x1a\x1cH\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x2f\x3c\x0b\x52TAG\x00\n\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\x00\x00\xb7\xe9d%END\x00\x00\x00\x00\x00\x00\x00\x00\x00\xbc\x93z\xc0'


class TestDataFile(unittest.TestCase):

    def test_crc32(self):
        data = bytes(range(256))
        crc1 = binascii.crc32(data)
        crc2 = binascii.crc32(data[:8])
        crc2 = binascii.crc32(data[8:-4], crc2)
        crc2 = binascii.crc32(data[-4:], crc2)
        self.assertEqual(crc1, crc2)

    def test_write_known_value(self):
        fh = io.BytesIO()
        f = datafile.DataFileWriter(fh)
        f.append(b'TAG', bytes(range(10)))
        f.finalize()
        value = fh.getvalue()
        self.assertEqual(F1, value)

    def test_read_known_value(self):
        fh = io.BytesIO(F1)
        f = datafile.DataFileReader(fh)
        tag, value = next(f)
        self.assertEqual(b'TAG', tag)
        self.assertEqual(bytes(range(10)), value)
        with self.assertRaises(StopIteration):
            next(f)

    def test_write_read(self):
        tags = [[42, bytes(range(1, 17))], (43, b'1234')]
        fh = io.BytesIO(F1)
        fw = datafile.DataFileWriter(fh)
        for tag, value in tags:
            fw.append(tag, value)
        fw.finalize()
        fh.seek(0)

        fr = datafile.DataFileReader(fh)
        for tag, value in tags:
            tag_rd, value_rd = yield fr
            tag_rd, = struct.unpack('<I', tag_rd)
            self.assertEqual(tag, tag_rd)
            self.assertEqual(value, value_rd)
        with self.assertRaises(StopIteration):
            next(fr)

    def test_compressed_write_read(self):
        data = np.arange(2**16, dtype=np.uint16).tobytes()
        fh = io.BytesIO(F1)
        fw = datafile.DataFileWriter(fh)
        fw.append(b'RAW', data, compress=True)
        fw.finalize()
        fh.seek(0)

        fr = datafile.DataFileReader(fh)
        tag, value = next(fr)
        self.assertEqual(b'RAW', tag)
        self.assertEqual(data, value)
        with self.assertRaises(StopIteration):
            next(fr)

    def make_encrypted(self, compress=False):
        data = np.arange(2**16, dtype=np.uint16).tobytes()
        fh = io.BytesIO(F1)
        fw = datafile.DataFileWriter(fh)
        signing_key_prv = bytes(range(32))
        encryption_key = bytes(range(1, 33))
        nonce = bytes(range(16, 16 + 24 * 2, 2))
        associated_data = bytes(range(2, 50, 2))
        fw.append_encrypted(b'RAW', data, signing_key_prv, encryption_key, nonce, associated_data, compress=compress)
        fw.finalize()
        fh.seek(0)
        fr = datafile.DataFileReader(fh)
        return fr, {
            'data': data,
            'signing_key_prv': signing_key_prv,
            'signing_key_pub': monocypher.compute_signing_public_key(signing_key_prv),
            'encryption_key': encryption_key,
            'nonce': nonce,
            'associated_data': associated_data,
            'fh': fh,
        }

    def test_encrypted(self):
        fr, d = self.make_encrypted()
        tag, value = fr.decrypt(d['signing_key_pub'], d['encryption_key'], d['nonce'], d['associated_data'])
        self.assertEqual(b'RAW', tag)
        self.assertEqual(d['data'], value)
        with self.assertRaises(StopIteration):
            next(fr)

    def test_encrypted_compressed(self):
        fr, d = self.make_encrypted(compress=True)
        tag, value = fr.decrypt(d['signing_key_pub'], d['encryption_key'], d['nonce'], d['associated_data'])
        self.assertEqual(b'RAW', tag)
        self.assertEqual(d['data'], value)
        with self.assertRaises(StopIteration):
            next(fr)

    def test_encrypted_bad_signing_key(self):
        fr, d = self.make_encrypted(compress=True)
        sk = d['signing_key_pub'][:-1] + b'\x00'
        with self.assertRaises(ValueError):
            fr.decrypt(sk, d['encryption_key'], d['nonce'], d['associated_data'])

    def test_encrypted_bad_encryption_key(self):
        fr, d = self.make_encrypted(compress=True)
        ek = d['encryption_key'][:-1] + b'\x00'
        with self.assertRaises(ValueError):
            fr.decrypt(d['signing_key_pub'], ek, d['nonce'], d['associated_data'])

    def test_encrypted_bad_nonce(self):
        fr, d = self.make_encrypted(compress=True)
        with self.assertRaises(ValueError):
            fr.decrypt(d['signing_key_pub'], d['encryption_key'], bytes([0]*24), d['associated_data'])

    def test_missing_associated_data(self):
        fr, d = self.make_encrypted(compress=True)
        with self.assertRaises(ValueError):
            fr.decrypt(d['signing_key_pub'], d['encryption_key'], d['nonce'])

    def _construct_collection(self, collection_data=None):
        data = [bytes(range(00, 10)), bytes(range(10, 20)), bytes(range(20, 30))]
        fh = io.BytesIO()
        fw = datafile.DataFileWriter(fh)
        c = fw.collection_start(id_=0, type_=1, data=collection_data)
        for d in data:
            fw.append(b'TAG', d)
        fw.collection_end(c)
        fw.finalize()
        fh.seek(0)
        return data, fh

    def test_collection(self):
        data, fh = self._construct_collection()

        fr = datafile.DataFileReader(fh)
        tag, collection_bytes = next(fr)
        c = datafile.Collection.decode(collection_bytes)
        self.assertEqual(datafile.TAG_COLLECTION_START, tag)
        self.assertIsNone(c.data)
        for d in data:
            tag, value = next(fr)
            self.assertEqual(b'TAG', tag)
            self.assertEqual(d, value)
        self.assertEqual(c.end_position, fh.tell())
        tag, value = next(fr)
        self.assertEqual(datafile.TAG_COLLECTION_END, tag)
        with self.assertRaises(StopIteration):
            next(fr)

    def test_collection_with_data(self):
        data, fh = self._construct_collection(collection_data=b'hello world')

        fr = datafile.DataFileReader(fh)
        tag, collection_bytes = next(fr)
        c = datafile.Collection.decode(collection_bytes)
        self.assertEqual(datafile.TAG_COLLECTION_START, tag)
        self.assertEqual(c.data, b'hello world')
        for d in data:
            tag, value = next(fr)
            self.assertEqual(b'TAG', tag)
            self.assertEqual(d, value)
        self.assertEqual(c.end_position, fh.tell())
        tag, value = next(fr)
        self.assertEqual(datafile.TAG_COLLECTION_END, tag)
        with self.assertRaises(StopIteration):
            next(fr)

    def test_collection_unclosed(self):
        fh1 = io.BytesIO()
        fw1 = datafile.DataFileWriter(fh1)
        fw1.collection_start(id_=0, type_=1)
        fw1.collection_end()
        fw1.finalize()

        fh2 = io.BytesIO()
        fw2 = datafile.DataFileWriter(fh2)
        fw2.collection_start(id_=0, type_=1)
        fw2.finalize()

        self.assertEqual(fh1.getbuffer(), fh2.getbuffer())

    def test_peek_tag_length(self):
        fh = io.BytesIO(F1)
        f = datafile.DataFileReader(fh)
        tag, length = f.peek_tag_length()
        self.assertEqual(b'TAG', tag)
        self.assertEqual(10, length)
        tag, value = next(f)
        self.assertEqual(b'TAG', tag)
        self.assertEqual(bytes(range(10)), value)
        with self.assertRaises(StopIteration):
            next(f)

    def test_peek(self):
        fh = io.BytesIO(F1)
        f = datafile.DataFileReader(fh)
        tag, value = f.peek()
        self.assertEqual(b'TAG', tag)
        self.assertEqual(bytes(range(10)), value)
        tag, value = next(f)
        self.assertEqual(b'TAG', tag)
        self.assertEqual(bytes(range(10)), value)
        with self.assertRaises(StopIteration):
            next(f)

    def test_skip(self):
        fh = io.BytesIO(F1)
        f = datafile.DataFileReader(fh)
        self.assertEqual(b'TAG', f.skip())
        with self.assertRaises(StopIteration):
            next(f)

    def test_skip_collection(self):
        data, fh = self._construct_collection()
        fr = datafile.DataFileReader(fh)
        fr.skip()
        with self.assertRaises(StopIteration):
            next(fr)

    def test_seek(self):
        fh = io.BytesIO(F1)
        f = datafile.DataFileReader(fh)
        pos = f.tell()
        self.assertEqual(b'TAG', f.skip())
        f.seek(pos)
        self.assertEqual(b'TAG', f.skip())
        with self.assertRaises(StopIteration):
            next(f)

    def test_valid_signature(self):
        key = b';\xfa\xe7\xa7(\xa5\xc8M\xcb\xb8\xe1H\x84\x95rB\x99\xafW\x91T\x10\nE\x80\xb2]AT\xfd\xf3\xcb'
        fh = io.BytesIO()
        fw = datafile.DataFileWriter(fh)
        fw.signature_start(key)
        fw.append(datafile.TAG_DATA_BINARY, bytes(range(256)))
        fw.signature_end()
        fw.finalize()

        fh.seek(0)
        fr = datafile.DataFileReader(fh)
        tag, value = next(fr)
        self.assertEqual(datafile.TAG_SIGNATURE_START, tag)
        next(fr)
        tag, value = next(fr)
        self.assertEqual(datafile.TAG_SIGNATURE_END, tag)
        self.assertTrue(value)

    def test_invalid_signature(self):
        key = b';\xfa\xe7\xa7(\xa5\xc8M\xcb\xb8\xe1H\x84\x95rB\x99\xafW\x91T\x10\nE\x80\xb2]AT\xfd\xf3\xcb'
        fh = io.BytesIO()
        fw = datafile.DataFileWriter(fh)
        fw.signature_start(key)
        fw.append(datafile.TAG_DATA_BINARY, bytes(range(256)))

        # add data to signature computation, but mess with underlying file
        pos = fh.tell()
        fw.append(datafile.TAG_DATA_BINARY, bytes(range(256)))
        fh.seek(pos)

        fw.signature_end()
        fw.finalize()
        fh.seek(0)
        fr = datafile.DataFileReader(fh)
        tag, value = next(fr)
        self.assertEqual(datafile.TAG_SIGNATURE_START, tag)
        next(fr)
        with self.assertRaises(ValueError):
            next(fr)
