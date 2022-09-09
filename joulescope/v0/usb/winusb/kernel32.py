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

from ctypes import windll, Structure, Union, \
    POINTER, pointer, byref, \
    c_void_p, c_wchar_p, memset, sizeof
from ctypes.wintypes import DWORD, HANDLE, BOOL, LPVOID, LPWSTR


_kernel32 = windll.kernel32


# CreateFile dwDesiredAccess values
GENERIC_WRITE = 1073741824
GENERIC_READ = -2147483648

# CreateFile dwShareMode flags
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
FILE_SHARE_DELETE = 4

# CreateFile dwCreationDisposition flags
CREATE_ALWAYS = 2
CREATE_NEW = 1
OPEN_ALWAYS = 4
OPEN_EXISTING = 3
TRUNCATE_EXISTING = 5

FILE_ATTRIBUTE_ARCHIVE = 0x20
FILE_ATTRIBUTE_ENCRYPTED = 0x4000
FILE_ATTRIBUTE_HIDDEN = 0x2
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_ATTRIBUTE_OFFLINE = 0x1000
FILE_ATTRIBUTE_READONLY = 0x1
FILE_ATTRIBUTE_SYSTEM = 0x4
FILE_ATTRIBUTE_TEMPORARY = 0x100

FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_DELETE_ON_CLOSE = 0x04000000
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_OPEN_NO_RECALL = 0x00100000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_FLAG_OVERLAPPED = 0x40000000
FILE_FLAG_POSIX_SEMANTICS = 0x0100000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_FLAG_SESSION_AWARE = 0x00800000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_FLAG_WRITE_THROUGH = 0x80000000

INVALID_HANDLE_VALUE = HANDLE(-1)

# Common IO errors
ERROR_FILE_NOT_FOUND = 2  # The system cannot find the file specified.
ERROR_INVALID_HANDLE = 6  # Invalid handle
ERROR_BAD_COMMAND = 22  # The device does not recognize the command.
ERROR_GEN_FAILURE = 31  # A device attached to the system is not functioning.
ERROR_IO_OPERATION_ABORTED = 995
ERROR_IO_INCOMPLETE = 996
ERROR_IO_PENDING = 997
ERROR_DEVICE_NOT_CONNECTED = 1167

# NTSTATUS value
STATUS_TIMEOUT = 0x00000102
STATUS_PENDING = 0x00000103

WAIT_ABANDONED = 0x00000080
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF

MAXIMUM_WAIT_OBJECTS = 64


class _OverlappedOffset(Structure):
    _fields_ = [
        ('Offset', DWORD),
        ('OffsetHigh', DWORD),
    ]


class _OverlappedUnion(Union):
    _fields_ = [
        ('o', _OverlappedOffset),
        ('Pointer', LPVOID),
    ]


class Overlapped(Structure):
    _fields_ = [
        ('Internal', LPVOID),
        ('InternalHigh', LPVOID),
        ('u', _OverlappedUnion),
        ('hEvent', HANDLE),
    ]

    def __init__(self, hEvent):
        memset(pointer(self), 0, sizeof(self))
        self.hEvent = hEvent

    def reset(self):
        hEvent = self.hEvent
        memset(pointer(self), 0, sizeof(self))
        self.hEvent = hEvent


class SecurityAttributes(Structure):
    _fields_ = [
        ('n_length', DWORD),
        ('lp_security_descriptor', c_void_p),
        ('b_inherit_handle',BOOL)
    ]


# HANDLE WINAPI CreateFile(
#   _In_     LPCTSTR               lpFileName,
#   _In_     DWORD                 dwDesiredAccess,
#   _In_     DWORD                 dwShareMode,
#   _In_opt_ LPSECURITY_ATTRIBUTES lpSecurityAttributes,
#   _In_     DWORD                 dwCreationDisposition,
#   _In_     DWORD                 dwFlagsAndAttributes,
#   _In_opt_ HANDLE                hTemplateFile);
CreateFile = _kernel32.CreateFileW
CreateFile.restype = BOOL
CreateFile.argtypes = [c_wchar_p, DWORD, DWORD, POINTER(SecurityAttributes), DWORD, DWORD, HANDLE]

# BOOL WINAPI CloseHandle(_In_ HANDLE hObject);
CloseHandle = _kernel32.CloseHandle
CloseHandle.restype = BOOL
CloseHandle.argtypes = [HANDLE]

# BOOL WINAPI ReadFile(
#   _In_        HANDLE       hFile,
#   _Out_       LPVOID       lpBuffer,
#   _In_        DWORD        nNumberOfBytesToRead,
#   _Out_opt_   LPDWORD      lpNumberOfBytesRead,
#   _Inout_opt_ LPOVERLAPPED lpOverlapped);
ReadFile = _kernel32.ReadFile
ReadFile.restype = BOOL
ReadFile.argtypes = [HANDLE, c_void_p, DWORD, POINTER(DWORD), POINTER(Overlapped)]

# BOOL WINAPI WriteFile(
#   _In_        HANDLE       hFile,
#   _In_        LPCVOID      lpBuffer,
#   _In_        DWORD        nNumberOfBytesToWrite,
#   _Out_opt_   LPDWORD      lpNumberOfBytesWritten,
#   _Inout_opt_ LPOVERLAPPED lpOverlapped);
WriteFile = _kernel32.WriteFile
WriteFile.restype = BOOL
WriteFile.argtypes = [HANDLE, c_void_p, DWORD, POINTER(DWORD), POINTER(Overlapped)]

# BOOL WINAPI CancelIo(_In_ HANDLE hFile);
CancelIo = _kernel32.CancelIo
CancelIo.restype = BOOL
CancelIo.argtypes = [HANDLE]

#BOOL WINAPI CancelIoEx(
#  _In_     HANDLE       hFile,
#  _In_opt_ LPOVERLAPPED lpOverlapped);
CancelIoEx = _kernel32.CancelIoEx
CancelIoEx.restype = BOOL
CancelIoEx.argtypes = [HANDLE, POINTER(Overlapped)]

# HANDLE WINAPI CreateEvent(
#   _In_opt_ LPSECURITY_ATTRIBUTES lpEventAttributes,
#   _In_     BOOL                  bManualReset,
#   _In_     BOOL                  bInitialState,
#   _In_opt_ LPCTSTR               lpName);
CreateEvent = _kernel32.CreateEventW
CreateEvent.restype = HANDLE
CreateEvent.argtypes = [POINTER(SecurityAttributes), BOOL, BOOL, c_wchar_p]

# BOOL WINAPI SetEvent(_In_ HANDLE hEvent);
SetEvent = _kernel32.SetEvent
SetEvent.restype = BOOL
SetEvent.argtypes = [HANDLE]

# BOOL WINAPI ResetEvent(_In_ HANDLE hEvent);
ResetEvent = _kernel32.ResetEvent
ResetEvent.restype = BOOL
ResetEvent.argtypes = [HANDLE]

# DWORD WINAPI WaitForSingleObject(
#   _In_ HANDLE hHandle,
#   _In_ DWORD  dwMilliseconds);
WaitForSingleObject = _kernel32.WaitForSingleObject
WaitForSingleObject.restype = DWORD
WaitForSingleObject.argtypes = [HANDLE, DWORD]

# DWORD WaitForMultipleObjects(
#   DWORD        nCount,
#   const HANDLE *lpHandles,
#   BOOL         bWaitAll,
#   DWORD        dwMilliseconds);
WaitForMultipleObjects = _kernel32.WaitForMultipleObjects
WaitForMultipleObjects.restype = DWORD
WaitForMultipleObjects.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]

# DWORD WINAPI GetLastError(void);
GetLastError = _kernel32.GetLastError
GetLastError.restype = DWORD
GetLastError.argtypes = []

# FormatMessage dwFlags
FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x00000100
FORMAT_MESSAGE_ARGUMENT_ARRAY = 0x00002000
FORMAT_MESSAGE_FROM_HMODULE = 0x00000800
FORMAT_MESSAGE_FROM_STRING = 0x00000400
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200
FORMAT_MESSAGE_MAX_WIDTH_MASK = 0x000000FF

# DWORD WINAPI FormatMessage(
#   _In_     DWORD   dwFlags,
#   _In_opt_ LPCVOID lpSource,
#   _In_     DWORD   dwMessageId,
#   _In_     DWORD   dwLanguageId,
#   _Out_    LPTSTR  lpBuffer,  -- With dwFlags ALLOCATE_BUFFER, actually POINTER to LPTSTR
#   _In_     DWORD   nSize,     -- 0 with dwFlags ALLOCATE_BUFFER
#   _In_opt_ va_list *Arguments);
FormatMessage = _kernel32.FormatMessageW
FormatMessage.restype = DWORD
FormatMessage.argtypes = [DWORD, c_void_p, DWORD, DWORD, POINTER(c_wchar_p), DWORD, c_void_p]


def get_error_str(error_code):
    if error_code == 0:
        return '[0] No error'
    bufptr = LPWSTR()
    FormatMessage(
        (FORMAT_MESSAGE_FROM_SYSTEM |
         FORMAT_MESSAGE_IGNORE_INSERTS |
         FORMAT_MESSAGE_ARGUMENT_ARRAY |
         FORMAT_MESSAGE_ALLOCATE_BUFFER),
        0,
        error_code,
        0,
        byref(bufptr),
        0,
        0)

    if bufptr.value is None:
        s = '(None)'
    else:
        s = bufptr.value.strip()
    return '[%d] %s' % (error_code, s)


def get_last_error():
    dLastError = GetLastError()
    return get_error_str(dLastError)
