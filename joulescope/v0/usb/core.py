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

from ctypes import Structure, c_ubyte, c_ushort


# USB PIPE TYPE
PIPE_TYPE_CONTROL = 0
PIPE_TYPE_ISO = 1
PIPE_TYPE_BULK = 2
PIPE_TYPE_INTERRUPT = 3


def structure_to_repr(x):
    values = ['\n    %s=%r' % (field, getattr(x, field)) for field, _ in x._fields_]
    return "%s(%s)" % (x.__class__.__name__, ','.join(values))


def buffer_to_str(x):
    a = ', '.join(['0x%02x' % z for z in x])
    return '[%s]' % a


class RequestType:  # from USB 2.0 spec chapter 9
    DIRECTION_OUT = (0 << 7)  # Host to device
    DIRECTION_IN = (1 << 7)  # Device to host
    TYPE_STANDARD = (0 << 5)  # Standard request
    TYPE_CLASS = (1 << 5)  # Class-specific request
    TYPE_VENDOR = (2 << 5)  # Vendor-specific request
    TYPE_RESERVED = (3 << 5)  # reserved request
    RECIPIENT_DEVICE = (0 << 0)  # Recipient device
    RECIPIENT_INTERFACE = (1 << 0)  # Recipient interface
    RECIPIENT_ENDPOINT = (2 << 0)  # Recipient endpoint
    RECIPIENT_OTHER = (3 << 0)  # Recipient other

    _DIRECTIONS = ['out', 'in']
    _TYPES = ['standard', 'class', 'vendor']
    _RECIPIENTS = ['device', 'interface', 'endpoint', 'other']

    def __init__(self, value=None, direction=None, type_=None, recipient=None):
        self._directions_s2v = dict([(v, getattr(self, 'DIRECTION_' + v.upper())) for v in RequestType._DIRECTIONS])
        self._directions_v2s = dict([(getattr(self, 'DIRECTION_' + v.upper()), v) for v in RequestType._DIRECTIONS])
        self._types_s2v = dict([(v, getattr(self, 'TYPE_' + v.upper())) for v in RequestType._TYPES])
        self._types_v2s = dict([(getattr(self, 'TYPE_' + v.upper()), v) for v in RequestType._TYPES])
        self._recipients_s2v = dict([(v, getattr(self, 'RECIPIENT_' + v.upper())) for v in RequestType._RECIPIENTS])
        self._recipients_v2s = dict([(getattr(self, 'RECIPIENT_' + v.upper()), v) for v in RequestType._RECIPIENTS])

        if value is not None:
            self.direction = self._directions_v2s[value & 0x80]
            self.type = self._types_v2s[value & 0x60]
            self.recipients = self._recipients_v2s[value & 0x1f]
        else:
            self.direction = direction.lower()
            self.type = type_.lower()
            self.recipient = recipient.lower()
            assert (self.direction in self._directions_s2v)
            assert (self.type in self._types_s2v)
            assert (self.recipient in self._recipients_s2v)

    @property
    def u8(self):
        return self._directions_s2v[self.direction] | self._types_s2v[self.type] | self._recipients_s2v[self.recipient]


class Request:
    """Standard USB requests."""
    GET_STATUS = 0
    CLEAR_FEATURE = 1
    SET_FEATURE = 3
    SET_ADDRESS = 5
    GET_DESCRIPTOR = 6
    SET_DESCRIPTOR = 7
    GET_CONFIGURATION = 8
    SET_CONFIGURATION = 9
    GET_INTERFACE = 10
    SET_INTERFACE = 11
    SYNCH_FRAME = 12


class SetupPacket(Structure):
    _fields_ = [
        ('request_type', c_ubyte),
        ('request', c_ubyte),
        ('value', c_ushort),
        ('index', c_ushort),
        ('length', c_ushort),
    ]

    def __repr__(self):
        return structure_to_repr(self)


class InterfaceDescriptor(Structure):
    _fields_ = [
        ('length', c_ubyte),
        ('descriptor_type', c_ubyte),
        ('interface_number', c_ubyte),
        ('alternate_setting', c_ubyte),
        ('num_endpoints', c_ubyte),
        ('interface_class', c_ubyte),
        ('interface_sub_class', c_ubyte),
        ('interface_protocol', c_ubyte),
        ('interface', c_ubyte)
    ]

    def __repr__(self):
        return structure_to_repr(self)


class ControlTransferResponse:

    def __init__(self, setup_packet, result, data=None):
        self.setup_packet = setup_packet
        self.result = result
        self.data = data

    def __str__(self):
        if self.data is None:
            return 'ControlTransferResponse(result=%r)' % (self.result,)
        else:
            return 'ControlTransferResponse(result=%r, data=%s)' % (
                self.result, buffer_to_str(self.data))

    def __repr__(self):
        if self.data is not None:
            return 'ControlTransferResponse(setup_packet=%r, result=%r)' % (
                self.setup_packet, self.result)
        else:
            return 'ControlTransferResponse(setup_packet=%r, result=%r, data=%s)' % (
                self.setup_packet, self.result, buffer_to_str(self.data))
