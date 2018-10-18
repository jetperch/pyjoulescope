# Copyright 2018 Jetperch LLC

"""
The USB backend which must be implemented for each platform type.

This module defines the USB backend.  Each target platform
(such as Windows, Mac OS/X and Linux), must implement backend that conforms
to this API.
"""

from .core import ControlTransferResponse


class DeviceDriverApi:
    """The device driver API."""

    def __str__(self):
        """Get the user-friendly device string.

        :return: f'{product_id_str}:{serial_number_str}'
        :raise IOError: On failure.

        WARNING: This function must correctly identify the device BEFORE it
        is opened.  Therefore, it must only use the information available
        from USB enumeration.
        """
        raise NotImplementedError()

    def open(self):
        """Open the USB device.

        :raise IOError: On failure.
        """
        raise NotImplementedError()

    def close(self):
        """Close the USB device."""
        raise NotImplementedError()

    def control_transfer_out(self, recipient, type_, request, value=0, index=0, data=None) -> ControlTransferResponse:
        """Perform a control transfer with data from host to device.

        :param recipient: The recipient which is one of ['device', 'interface', 'endpoint', 'other']
        :param type_: The type which is one of ['standard', 'class', 'vendor'].
        :param request: The bRequest value.
        :param value: The wValue value.
        :param index: The wIndex value.
        :param data: The optional data to transfer from host to device.
            None (default) skips the data phase.

        :return: A :class:`ControlTransferResponse` instance.
        """
        raise NotImplementedError()

    def control_transfer_in(self, recipient, type_, request, value, index, length) -> ControlTransferResponse:
        """Perform a control transfer with data from device to host.

        :param recipient: The recipient which is one of ['device', 'interface', 'endpoint', 'other']
        :param type_: The type which is one of ['standard', 'class', 'vendor'].
        :param request: The bRequest value.
        :param value: The wValue value.
        :param index: The wIndex value.
        :param length: The maximum number of bytes to transfer from device to host.

        :return: A :class:`ControlTransferResponse` instance.
        """
        raise NotImplementedError()

    def read_stream_start(self, endpoint_id, transfers, block_size, data_fn, process_fn):
        """Read a stream of data using non-blocking (overlapped) IO.

        :param endpoint_id: The target endpoint address.
        :param transfers: The number of overlapped transfers to use,
            each of block_size bytes.
        :param block_size: The length of each block in bytes which must be
            a multiple of the maximum packet size for the endpoint.
        :param data_fn: The function(data) to call on each block
            of data.  The data is an np.ndarray(dtype=uint8) containing
            the raw bytes received for each USB transaction.
            The length of data is normally block_size.
            Any value less than block_size is the last transfer
            in the transaction.

            When the device stops, it calls data_fn(None).  The
            device can stop "automatically" through errors or when data_fn
            returns True.  Call :meth:`read_stream_stop` to stop from
            the caller.

            This function will be called from the device's thread.  The
            data_fn must return quickly to ensure that the USB stream
            is not starved.

            In all cases, data_fn should return None or False to continue
            streaming.  data_fn can return True to stop the transmission.

            Most implementations use some form of non-blocking IO with
            multiple queue (overlapped) transactions that are pended
            early.  On stop, additional data may be read before the
            transaction fully stops.
        :param process_fn: The function process_fn() to call after all
            USB endpoints have been recently serviced and data_fn was
            called at least once.  The function should still be quick,
            but it can have more latency than data_fn.

        Use :meth:`read_stream_stop` to stop.
        """
        raise NotImplementedError()

    def read_stream_stop(self, endpoint_id):
        """Stop a read stream.

        :param endpoint_id: The target endpoint address.

        Use :meth:`read_stream_start` to start.
        """
        raise NotImplementedError()

    def status(self):
        """Get the current device status.

        :return: A dict containing the following structure:

            endpoints: {
                pipe_id: { name: {value: v, units: u}, ...}
                ...
            }
        """
        raise NotImplementedError()

    def signal(self):
        """Signal that an external event occurred.

        This method allows another thread to cause the wait in process
        to activate.
        """
        raise NotImplementedError()

    def process(self, timeout=None):
        """Process any pending events.

        :param timeout: The timeout in float seconds.

        This method uses the operating-system specific method to wait on
        pending events, such select and WaitForMultipleObjects.
        """
        raise NotImplementedError()


def scan(name: str=None):
    """Scan for attached devices.

    :param name: The case-insensitive name of the device to scan.
    :return: The list of attached backend :class:`Device` instances.
    """
    raise NotImplementedError()
