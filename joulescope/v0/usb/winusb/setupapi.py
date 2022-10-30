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

# https://msdn.microsoft.com/en-us/library/windows/hardware/ff550897(v=vs.85).aspx
from ctypes import windll, Structure, sizeof, resize, byref, wstring_at, \
    c_byte, c_ulong, c_wchar_p, c_void_p, POINTER
from ctypes.wintypes import DWORD, WORD, WCHAR, HANDLE, BOOL
import uuid

_setupapi = windll.SetupApi


# SetupDiGetClassDevs flags
DIGCF_DEFAULT = 0x00000001
DIGCF_PRESENT = 0x00000002
DIGCF_ALLCLASSES = 0x00000004
DIGCF_PROFILE = 0x00000008
DIGCF_DEVICE_INTERFACE = 0x00000010


class GUID(Structure):
    _fields_ = [
        ('data1', DWORD), 
        ('data2', WORD),
        ('data3', WORD), 
        ('data4', c_byte * 8),
    ]
    
    def __init__(self, guid=None):
        Structure.__init__(self)
        if guid:
            x = uuid.UUID(guid)
            f = x.fields
            self.data1 = DWORD(f[0])
            self.data2 = WORD(f[1])
            self.data3 = WORD(f[2])
            data4_cast = c_byte * 8
            self.data4 = data4_cast(*x.bytes[8:])


class SpDevinfoData(Structure):
    _fields_ = [
        ('cb_size', DWORD), 
        ('class_guid', GUID),
        ('dev_inst', DWORD), 
        ('reserved', POINTER(c_ulong)),
    ]

    def __init__(self):
        Structure.__init__(self)
        self.cb_size = sizeof(self)


class SpDeviceInterfaceData(Structure):
    _fields_ = [
        ('cb_size', DWORD), 
        ('interface_class_guid', GUID),
        ('flags', DWORD), 
        ('reserved', POINTER(c_ulong)),
    ]
    
    def __init__(self):
        Structure.__init__(self)
        self.cb_size = sizeof(self)


class SpDeviceInterfaceDetailData(Structure):
    _fields_ = [
        ("cb_size", DWORD), 
        ("device_path", WCHAR * 1),  # variable size character array
    ]

    def __init__(self):
        Structure.__init__(self)
        self.cb_size = sizeof(self)
    

# HDEVINFO SetupDiGetClassDevs(
#   _In_opt_ const GUID   *ClassGuid,
#   _In_opt_       PCTSTR Enumerator,
#   _In_opt_       HWND   hwndParent,
#   _In_           DWORD  Flags);
SetupDiGetClassDevs = _setupapi.SetupDiGetClassDevsW
SetupDiGetClassDevs.restype = HANDLE
SetupDiGetClassDevs.argtypes = [POINTER(GUID), c_wchar_p, HANDLE, DWORD]

# BOOL SetupDiEnumDeviceInterfaces(
#   _In_           HDEVINFO                  DeviceInfoSet,
#   _In_opt_       PSP_DEVINFO_DATA          DeviceInfoData,
#   _In_     const GUID                      *InterfaceClassGuid,
#   _In_           DWORD                     MemberIndex,
#   _Out_          PSP_DEVICE_INTERFACE_DATA DeviceInterfaceData);
SetupDiEnumDeviceInterfaces = _setupapi.SetupDiEnumDeviceInterfaces
SetupDiEnumDeviceInterfaces.restype = BOOL
SetupDiEnumDeviceInterfaces.argtypes = [c_void_p, POINTER(SpDevinfoData), POINTER(GUID), DWORD, POINTER(SpDeviceInterfaceData)]

# BOOL SetupDiGetDeviceInterfaceDetail(
#   _In_      HDEVINFO                         DeviceInfoSet,
#   _In_      PSP_DEVICE_INTERFACE_DATA        DeviceInterfaceData,
#   _Out_opt_ PSP_DEVICE_INTERFACE_DETAIL_DATA DeviceInterfaceDetailData,
#   _In_      DWORD                            DeviceInterfaceDetailDataSize,
#   _Out_opt_ PDWORD                           RequiredSize,
#   _Out_opt_ PSP_DEVINFO_DATA                 DeviceInfoData);
SetupDiGetDeviceInterfaceDetail = _setupapi.SetupDiGetDeviceInterfaceDetailW
SetupDiGetDeviceInterfaceDetail.restype = BOOL
SetupDiGetDeviceInterfaceDetail.argtypes = [c_void_p, POINTER(SpDeviceInterfaceData), POINTER(SpDeviceInterfaceDetailData), DWORD, POINTER(DWORD), POINTER(SpDevinfoData)]

# BOOL SetupDiDestroyDeviceInfoList(
#   _In_ HDEVINFO DeviceInfoSet);
SetupDiDestroyDeviceInfoList = _setupapi.SetupDiDestroyDeviceInfoList
SetupDiDestroyDeviceInfoList.restype = BOOL
SetupDiDestroyDeviceInfoList.argtypes = [c_void_p]


def device_interface_guid_to_paths(guid):
    """Find the connected devices that match the provided GUID.
    
    :param guid: The deviceInterfaceGUID to match.
    :return: The list of path strings.
    """
    paths = []
    flags = DWORD(DIGCF_PRESENT | DIGCF_DEVICE_INTERFACE)
    guid = GUID(guid)
    handle = SetupDiGetClassDevs(byref(guid), None, None, flags)
    if handle is None:
        raise RuntimeError('SetupDiGetClassDevs invalid')

    sp_device_interface_data = SpDeviceInterfaceData()
    sp_device_info_data = SpDevinfoData()
    member_index = 0

    while SetupDiEnumDeviceInterfaces(handle, None, byref(guid), DWORD(member_index), byref(sp_device_interface_data)):
        required_size = c_ulong(0)
        bResult = SetupDiGetDeviceInterfaceDetail(handle, byref(sp_device_interface_data), None, 0, byref(required_size), None)
        sp_device_interface_detail_data = SpDeviceInterfaceDetailData()
        resize(sp_device_interface_detail_data, required_size.value)
        sp_device_interface_detail_data.cb_size = sizeof(SpDeviceInterfaceDetailData)
        bResult = SetupDiGetDeviceInterfaceDetail(
            handle, byref(sp_device_interface_data), 
            byref(sp_device_interface_detail_data), required_size, 
            byref(required_size), byref(sp_device_info_data))
        if bResult:
            path = wstring_at(byref(sp_device_interface_detail_data, sizeof(DWORD)))
            paths.append(path)
        member_index += 1
    SetupDiDestroyDeviceInfoList(handle)
    return paths
