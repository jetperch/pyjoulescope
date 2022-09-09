# Copyright 2022 Jetperch LLC
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


class Device:

    def __init__(self, driver, device_path):
        self._driver = driver
        while device_path.endswith('/'):
            device_path = device_path[:-1]
        self._path = device_path

    def __str__(self):
        return str(self._path)

    def _topic_make(self, topic):
        if topic[0] != '/':
            topic = '/' + topic
        return self._path + topic

    def publish(self, topic, value):
        return self._driver.publish(self._topic_make(topic), value)

    def query(self, topic):
        return self._driver.query(self._topic_make(topic))

    def subscribe(self, topic, flags, fn, timeout=None):
        """Subscribe to receive topic updates.

        :param self: The driver instance.
        :param topic: Subscribe to this topic string.
        :param flags: The flags or list of flags for this subscription.
            The flags can be int32 jsdrv_subscribe_flag_e or string
            mnemonics, which are:
            - pub: Subscribe to normal values
            - pub_retain: Subscribe to normal values and immediately publish
              all matching retained values.  With timeout, this function does
              not return successfully until all retained values have been
              published.
            - metadata_req: Subscribe to metadata requests (not normally useful).
            - metadata_rsp: Subscribe to metadata updates.
            - metadata_rsp_retain: Subscribe to metadata updates and immediately
              publish all matching retained metadata values.
            - query_req: Subscribe to all query requests (not normally useful).
            - query_rsp: Subscribe to all query responses.
            - return_code: Subscribe to all return code responses.
        :param fn: The function to call on each publish.  Note that python
            dynamically constructs bound methods.  To unsubscribe a method,
            provide the exact  same bound method instance to unsubscribe.
        :param timeout: The timeout in float seconds to wait for this operation
            to complete.  None waits the default amount.
            0 does not wait and subscription will occur asynchronously.
        :raise RuntimeError: on subscribe failure.
        """
        return self._driver.subscribe(self._topic_make(topic), flags, fn, timeout)

    def unsubscribe(self, topic, fn, timeout=None):
        return self._driver.unsubscribe(self._topic_make(topic), fn, timeout)

    def unsubscribe_all(self, fn, timeout=None):
        return self._driver.unsubscribe(fn, timeout)

    def open(self, mode=None, timeout=None):
        """Open this device.

        :param mode: The open mode which is one of:
            * 'defaults': Reconfigure the device for default operation.
            * 'restore': Update our state with the current device state.
            * 'raw': Open the device in raw mode for development or firmware update.
            * None: equivalent to 'defaults'.
        :param timeout: The timeout in seconds.  None uses the default timeout.
        """
        return self._driver.open(self._path, mode, timeout)

    def close(self, timeout=None):
        return self._driver.close(self._path, timeout)

    @property
    def model(self):
        return self._path.split('/')[1]

    @property
    def serial_number(self):
        return self._path.split('/')[-1]

    @property
    def device_serial_number(self):
        return self.serial_number

    def info(self):
        raise NotImplementedError()

    def __enter__(self):
        """Device context manager, automatically open."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Device context manager, automatically close."""
        self.close()
