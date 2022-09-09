# Copyright 2019 Jetperch LLC
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

import unittest
from joulescope.v0.pattern_buffer import PatternBuffer
import numpy as np


def construct(count, counter_first=None, sample_drop=None):
    packet_buffer = np.zeros(count * 512, dtype=np.uint8)
    pkt = packet_buffer.view(dtype=np.uint32)
    pkt_count = 0
    sample_id = 0
    bitshift = 0
    counter = 0
    while len(pkt):
        pkt[0] = 0x02000080
        pkt[1] = pkt_count
        pkt_count += 1
        for idx in range(2, 128):
            while True:
                if bool(counter_first) != ((sample_id & 1) == 0):
                    pkt[idx] = ((~counter & 0xffff) << 16) | counter
                    counter = (counter + 1) & 0xffff
                else:
                    pkt[idx] = bitshift
                    if bitshift == 0:
                        bitshift = 1
                    else:
                        bitshift = (bitshift << 1) & 0xffffffff
                sample_id += 1
                if sample_drop is not None and len(sample_drop) and sample_drop[0] == sample_id:
                    sample_drop.pop(0)
                else:
                    break
        pkt = pkt[128:]
    return packet_buffer


class TestPatternBuffer(unittest.TestCase):

    def test_counter_first(self):
        p = PatternBuffer()
        data = construct(1, counter_first=True)
        self.assertEqual(512, len(data))
        p.insert(data)
        s = p.status()
        self.assertEqual(126, s['sample_id'])
        self.assertEqual(0, s['header_error'])
        self.assertEqual(0, s['pkt_index_error'])
        self.assertEqual(0, s['pattern_error'])

    def test_bitshift_first(self):
        p = PatternBuffer()
        data = construct(2, counter_first=False)
        self.assertEqual(1024, len(data))
        p.insert(data)
        s = p.status()
        self.assertEqual(252, s['sample_id'])
        self.assertEqual(0, s['header_error'])
        self.assertEqual(0, s['pkt_index_error'])
        self.assertEqual(0, s['pattern_error'])

    def test_sample_drop(self):
        p = PatternBuffer()
        data = construct(2, sample_drop=[10, 19])
        p.insert(data)
        s = p.status()
        self.assertEqual(252, s['sample_id'])
        self.assertEqual(0, s['header_error'])
        self.assertEqual(0, s['pkt_index_error'])
        self.assertEqual(2, s['pattern_error'])

    def test_packet_index_error(self):
        p = PatternBuffer()
        data = construct(2)
        data[516] = 3
        p.insert(data)
        s = p.status()
        print(s)
        self.assertEqual(252, s['sample_id'])
        self.assertEqual(0, s['header_error'])
        self.assertEqual(1, s['pkt_index_error'])
        self.assertEqual(0, s['pattern_error'])
