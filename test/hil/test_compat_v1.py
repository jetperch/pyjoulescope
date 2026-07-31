# Copyright 2026 Jetperch LLC
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

"""Backwards-compatibility contract for the joulescope package v1 API.

Every test here exercises the public API exactly as pre-1.6 application
code does.  This suite must pass on JS110, JS220, and JS320 before and
after every stage of the JS220/JS320 feature work.  See
doc/plans/js220_js320_feature_support.md.
"""

import numpy as np
import os
import pytest
import queue
import time
import joulescope


LEGACY_FIELDS = ['current', 'voltage', 'power', 'current_range',
                 'current_lsb', 'voltage_lsb']

# The output sampling frequency each device reports with default config.
DEFAULT_FS = {
    'js110': 2000000,
    'js220': 1000000,
    'js320': 1000000,
}


def test_scan_lists_devices():
    devices = joulescope.scan(config='auto')
    assert len(devices) >= 1
    for d in devices:
        model, sn = str(d).split('-')
        assert model in ['JS110', 'JS220', 'JS320']
        assert d.serial_number == sn


def test_str_and_properties(device):
    s = str(device)
    assert s.startswith(device.model.upper())
    assert device.device_path.startswith('u/')
    assert device.device_serial_number == device.serial_number


def test_read_calibrated(device):
    data = device.read(contiguous_duration=0.05)
    fs = device.output_sampling_frequency
    assert fs == DEFAULT_FS[device.model]
    assert data.shape == (int(0.05 * fs), 2)
    assert data.dtype == np.float32
    assert np.all(np.isfinite(data))


def test_read_samples_get_legacy_fields(device):
    data = device.read(contiguous_duration=0.05, out_format='samples_get')
    signals = data['signals']
    for field in LEGACY_FIELDS:
        assert field in signals, f'missing {field}'
        assert len(signals[field]['value']) > 0
    assert signals['current']['units'] == 'A'
    assert signals['voltage']['units'] == 'V'
    assert signals['power']['units'] == 'W'
    for field in ['current_lsb', 'voltage_lsb']:
        v = signals[field]['value']
        assert np.all((v == 0) | (v == 1))


def test_statistics_callback_format(device):
    q = queue.Queue()
    device.statistics_callback_register(q.put)
    try:
        value = q.get(timeout=5.0)
    finally:
        device.statistics_callback_unregister(q.put)
    assert 'time' in value
    assert 'signals' in value
    assert 'accumulators' in value
    t = value['time']
    assert 'range' in t and t['range']['units'] == 's'
    assert 'delta' in t
    for name in ['current', 'voltage', 'power']:
        s = value['signals'][name]
        for stat in ['µ', 'σ2', 'min', 'max', 'p2p']:
            assert stat in s, f'{name} missing {stat}'
    for name in ['charge', 'energy']:
        assert 'value' in value['accumulators'][name]
    # second callback: accumulators advance from the offset-adjusted start
    value2 = None
    device.statistics_callback_register(q.put)
    try:
        value2 = q.get(timeout=5.0)
    finally:
        device.statistics_callback_unregister(q.put)
    assert value2['time']['range']['value'][0] >= 0


def test_parameter_i_range(device):
    for value in ['auto', '10 A', 'auto']:
        device.parameter_set('i_range', value)
        assert device.parameter_get('i_range') == value


def test_parameter_v_range(device):
    device.parameter_set('v_range', '15V')
    assert device.parameter_get('v_range') == '15V'


def test_parameter_sampling_frequency(device):
    device.parameter_set('sampling_frequency', 100000)
    try:
        assert device.output_sampling_frequency == 100000
        data = device.read(contiguous_duration=0.05)
        assert data.shape[0] == 5000
    finally:
        device.parameter_set('sampling_frequency', 1000000)


def test_parameter_sampling_frequency_500khz(device):
    """500 kHz works on JS110/JS220; the JS320 gateware cannot (2->4)."""
    if device.model == 'js320':
        with pytest.raises(ValueError):
            device.parameter_set('sampling_frequency', 500000)
        return
    device.parameter_set('sampling_frequency', 500000)
    try:
        assert device.output_sampling_frequency == 500000
    finally:
        device.parameter_set('sampling_frequency',
                             min(DEFAULT_FS[device.model], 1000000))


def test_parameter_legacy_2mhz_accepted(device):
    """Legacy code sets 2 MHz; JS220/JS320 clamp to 1 MHz."""
    device.parameter_set('sampling_frequency', 2000000)
    try:
        expected = 2000000 if device.model == 'js110' else 1000000
        assert device.output_sampling_frequency == expected
    finally:
        device.parameter_set('sampling_frequency', DEFAULT_FS[device.model])


def test_parameters_list(device):
    params = device.parameters()
    names = [p.name for p in params]
    for name in ['i_range', 'v_range', 'sampling_frequency',
                 'buffer_duration', 'reduction_frequency']:
        assert name in names


def test_extio_status(device):
    status = device.extio_status()
    for key in ['gpo0', 'gpo1', 'gpi_value', 'io_voltage']:
        assert key in status
        assert 'value' in status[key]
        assert status[key]['name'] == key


def test_gpo_set_clear(device):
    for value in ['1', '0']:
        device.parameter_set('gpo0', value)
        device.parameter_set('gpo1', value)
        assert device.parameter_get('gpo0') == value
        assert device.parameter_get('gpo1') == value


def test_info(device):
    info = device.info()
    assert info['model'] == device.model
    if device.model in ['js220', 'js320']:
        assert info['serial_number'] == device.serial_number
        assert 'ver' in info['ctl']['fw']
        assert 'ver' in info['sensor']['fpga']


def test_statistics_accumulators_clear(device):
    device.statistics_accumulators_clear()


def test_jls_writer(device, tmp_path):
    from joulescope import JlsWriter
    from pyjls import Reader
    path = str(tmp_path / 'test.jls')
    with JlsWriter(device, path, signals='current,voltage') as wr:
        device.stream_process_register(wr)
        try:
            device.read(contiguous_duration=0.1)
        finally:
            device.stream_process_unregister(wr)
    with Reader(path) as r:
        signals = [s.name for s in r.signals.values()]
        assert 'current' in signals
        assert 'voltage' in signals
        for s in r.signals.values():
            if s.name in ['current', 'voltage']:
                assert s.length > 0
                assert s.sample_rate == device.output_sampling_frequency


def test_capture_entry_point(device_closed, tmp_path):
    """The capture entry point must produce a readable file.

    Records the joulescope capture format contract.  Before stage 4 of
    doc/plans/js220_js320_feature_support.md this is the legacy v0 JLS
    format; stage 4 switches the default to JLS v2 (pyjls).
    """
    from joulescope.entry_points.capture import run
    path = str(tmp_path / 'capture.jls')
    rv = run(device_closed, path, contiguous_duration=0.1)
    assert rv == 0
    assert os.path.getsize(path) > 0
    _assert_capture_readable(path, device_closed)


def _assert_capture_readable(path, device):
    from joulescope.data_recorder import DataReader
    r = DataReader()
    r.open(path)
    try:
        assert r.footer['size'] > 0
        assert r.duration > 0
    finally:
        r.close()


def test_open_close_reopen(device_closed):
    d = device_closed
    for _ in range(2):
        d.open()
        assert d.is_open
        d.close()
        assert not d.is_open


def test_context_manager(device_closed):
    with device_closed as d:
        assert d.is_open
    assert not device_closed.is_open


def test_view_factory_statistics(device):
    """Exercise the View path used by legacy GUI/scripting code."""
    view = device.view_factory()
    view.open()
    try:
        device.start()
        time.sleep(0.5)
        view.refresh()
        time.sleep(0.25)
    finally:
        device.stop()
        view.close()
