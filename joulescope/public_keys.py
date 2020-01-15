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
Joulescope uses public key cryptography to sign calibrations and
firmware update.  This file contains the public keys.
"""

from binascii import unhexlify as uhx

CALIBRATION_SIGNING = uhx(b'3b3aee3fb2ac984fd37040aa52092e2ad88371455285002a6eef568ef152232f')
RECALIBRATION_01_SIGNING = uhx(b'75693bf67e3a8d783789e831e16a8df40fee55b363e787d2212866b1a18cf64d')
FIRMWARE_SIGNING = uhx(b'32fe2bed04bbc42fe1b382e0371ba95ec2947045e8d919e49fdef601e24c105e')
FIRMWARE_DISTRIBUTION = uhx(b'83fd0b085bda32c041644b7c299859b1101fbccc334f9ba6223a9d94a083a9d0')
JOULESCOPE_ROOT = uhx(b'eb385a7ff9c1315deba7cd5fca29630a1a0c66b4bbb0deac5c1e18aa68715ae4')
