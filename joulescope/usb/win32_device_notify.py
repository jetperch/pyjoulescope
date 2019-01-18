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

"""Detect USB device insertion and removal under Windows."""

# See https://github.com/mhammond/pywin32
# Referenced example win32/Demos/win32gui_devicenotify.py

import time
import win32gui
import win32con
import win32gui_struct
import threading
import logging

log = logging.getLogger(__name__)

# These device GUIDs are from Ioevent.h in the Windows SDK.  Ideally they
# could be collected somewhere for pywin32...
GUID_DEVINTERFACE_USB_DEVICE = "{A5DCBF10-6530-11D2-901F-00C04FB951ED}"


class DeviceNotify:

    def __init__(self, cbk):
        """Start device insertion/removal notification.

        :param cbk: The function called on device insertion or removal.  The
            arguments are (inserted, info).  "inserted" is True on insertion
            and False on removal.  "info" contains platform-specific details
            about the device.  In general, the application should rescan for
            relevant devices.
        """

        self._event = None
        self._thread = None
        self._window_thread = None
        self._hwnd = None
        self._cbk = cbk
        self.open()

    def window_callback(self, hwnd, nMsg, wParam, lParam):
        if nMsg == win32con.WM_CLOSE:
            log.debug('WM_CLOSE')
            win32gui.DestroyWindow(hwnd)
        elif nMsg == win32con.WM_DESTROY:
            log.debug('WM_DESTROY')
            win32gui.PostQuitMessage(0)
        elif nMsg == win32con.WM_DEVICECHANGE:
            log.debug('WM_DEVICECHANGE')
            if wParam in [win32con.DBT_DEVICEARRIVAL, win32con.DBT_DEVICEREMOVECOMPLETE]:
                # Unpack the 'lp' into the appropriate DEV_BROADCAST_* structure,
                # using the self-identifying data inside the DEV_BROADCAST_HDR.
                try:
                    info = win32gui_struct.UnpackDEV_BROADCAST(lParam)
                except NotImplementedError:
                    info = None
                log.debug("Device change notification: nMsg=%s, wParam=%s, info=%s:"
                          % (nMsg, wParam, str(info)))
                inserted = True if wParam == win32con.DBT_DEVICEARRIVAL else False
                self._cbk(inserted, info)
            return True
        else:
            log.debug('other message nMsg = %d' % nMsg)
        return win32gui.DefWindowProc(hwnd, nMsg, wParam, lParam)

    def open(self):
        self.close()
        log.info('open')
        self._window_thread = threading.Thread(name='device_notify', target=self._run_window)
        self._window_thread.start()

    def close(self):
        if self._window_thread:
            log.info('close')
            win32gui.PostMessage(self._hwnd, win32con.WM_CLOSE, 0, 0)
            self._window_thread.join()
            self._window_thread = None

    def _run_window(self):
        # Create hidden window
        log.debug('_run_window start')
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = 'devicenotify'
        wc.style = win32con.CS_GLOBALCLASS | win32con.CS_VREDRAW | win32con.CS_HREDRAW
        wc.hbrBackground = win32con.COLOR_WINDOW + 1
        wc.lpfnWndProc = self.window_callback
        class_atom = win32gui.RegisterClass(wc)
        if not class_atom:
            log.error('window class not created')
            return
        self._hwnd = win32gui.CreateWindow(
            wc.lpszClassName,
            'devicenotify',
            win32con.WS_CAPTION,
            100, 100, 900, 900, 0, 0, 0, None)
        if not self._hwnd:
            log.error('window not created')
            return

        # Watch for all USB device notifications
        devfilt = win32gui_struct.PackDEV_BROADCAST_DEVICEINTERFACE(
            GUID_DEVINTERFACE_USB_DEVICE)
        self._event = win32gui.RegisterDeviceNotification(
            self._hwnd, devfilt, win32con.DEVICE_NOTIFY_WINDOW_HANDLE)

        while 1:
            b, msg = win32gui.GetMessage(None, 0, 0)
            log.debug('win32_device_notify message')
            if not b or not msg:
                break
            win32gui.TranslateMessage(msg)
            win32gui.DispatchMessage(msg)
        win32gui.UnregisterDeviceNotification(self._event)
        win32gui.UnregisterClass(wc.lpszClassName, None)
        self._hwnd = None
        log.debug('_run_window done')


if __name__ == '__main__':

    def callback_fn(inserted, info):
        print('callback %s %s' % (inserted, info))

    dn = DeviceNotify(callback_fn)
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        dn.close()
