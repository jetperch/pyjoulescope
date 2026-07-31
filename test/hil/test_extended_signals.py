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

"""Extended signal streaming (gpi2, gpi3, trigger_in) on JS220/JS320."""

import numpy as np
import pytest


DEFAULT_SIGNALS = 'i,v,p,r,0,1'

# The JS220 delivers no current_range data with 8+ concurrent streams
# (HIL-characterized 2026-07-31), so its maximum selection omits r.
ALL_SIGNALS = {
    'js220': 'i,v,p,0,1,2,3,T',
    'js320': 'i,v,p,r,0,1,2,3,T',
}


@pytest.fixture()
def xdevice(device):
    """The opened device, extended signals supported, defaults restored."""
    if device.model == 'js110':
        pytest.skip('extended signals require JS220/JS320')
    yield device
    device.parameter_set('signals', DEFAULT_SIGNALS)


def test_signals_parameter_roundtrip(xdevice):
    signals = ALL_SIGNALS[xdevice.model]
    xdevice.parameter_set('signals', signals)
    assert xdevice.parameter_get('signals') == signals


def test_stream_all_signals(xdevice):
    signals = ALL_SIGNALS[xdevice.model]
    xdevice.parameter_set('signals', signals)
    fields = ['current' if s == 'i' else
              'voltage' if s == 'v' else
              'power' if s == 'p' else
              'current_range' if s == 'r' else s
              for s in signals.split(',')]
    data = xdevice.read(contiguous_duration=0.05, out_format='samples_get',
                        fields=fields)
    signals_out = data['signals']
    n = None
    for field in fields:
        assert field in signals_out, f'missing {field}'
        v = signals_out[field]['value']
        if n is None:
            n = len(v)
        assert len(v) == n, f'{field}: {len(v)} != {n}'
    for field in ['0', '1', '2', '3', 'T']:
        v = signals_out[field]['value']
        assert np.all((v == 0) | (v == 1)), f'{field} not binary'


def test_js220_stream_limit_rejected(xdevice):
    if xdevice.model != 'js220':
        pytest.skip('JS220-specific limitation')
    with pytest.raises(ValueError):
        xdevice.parameter_set('signals', 'i,v,p,r,0,1,2,3,T')
    # 7 concurrent signals with current_range works
    xdevice.parameter_set('signals', 'i,v,p,r,0,1,T')
    data = xdevice.read(contiguous_duration=0.05, out_format='samples_get',
                        fields=['current_range', 'T'])
    assert len(data['signals']['current_range']['value']) > 0


def test_stream_subset(xdevice):
    xdevice.parameter_set('signals', 'i,v')
    data = xdevice.read(contiguous_duration=0.05)
    assert data.shape[1] == 2
    assert np.all(np.isfinite(data))
    # unselected legacy fields present as NaN in samples_get default
    s = xdevice.read(duration=0.05, out_format='samples_get')
    assert np.all(np.isnan(s['signals']['power']['value']))


def test_legacy_default_unchanged(xdevice):
    """With no signals parameter change, behavior matches pre-1.6."""
    data = xdevice.read(contiguous_duration=0.05, out_format='samples_get')
    signals = data['signals']
    assert list(signals.keys()) == ['current', 'voltage', 'power',
                                    'current_range', 'current_lsb',
                                    'voltage_lsb']
    for v in signals.values():
        assert np.all(np.isfinite(v['value'].astype(np.float64)))


def test_downsampled_extended_signals(xdevice):
    """Extended signals track the output rate when downsampling."""
    xdevice.parameter_set('signals', ALL_SIGNALS[xdevice.model])
    xdevice.parameter_set('sampling_frequency', 100000)
    try:
        data = xdevice.read(contiguous_duration=0.05,
                            out_format='samples_get',
                            fields=['current', '2', 'T'])
        n = len(data['signals']['current']['value'])
        assert n == len(data['signals']['2']['value'])
        assert n == len(data['signals']['T']['value'])
    finally:
        xdevice.parameter_set('sampling_frequency', 1000000)


def test_capture_jls2_extended_signals(xdevice, tmp_path):
    """Capture the full signal set to JLS v2 and verify with pyjls."""
    from joulescope.entry_points.capture import run
    from pyjls import Reader
    signals = ALL_SIGNALS[xdevice.model]
    path = str(tmp_path / 'extended.jls')
    xdevice.close()
    try:
        rv = run(xdevice, path, contiguous_duration=0.2, signals=signals)
        assert rv == 0
    finally:
        xdevice.open()
    expect = {'i': 'current', 'v': 'voltage', 'p': 'power',
              'r': 'current_range', '0': 'gpi[0]', '1': 'gpi[1]',
              '2': 'gpi[2]', '3': 'gpi[3]', 'T': 'trigger_in'}
    expect_names = [expect[s] for s in signals.split(',')]
    with Reader(path) as r:
        names = [s.name for s in r.signals.values() if s.name != 'global_annotation_signal']
        for name in expect_names:
            assert name in names, f'missing {name}'
        # i/v/p follow h/fs; JS220 gpi/current_range stream at the
        # native 2 Msps rate.  JLS v2 stores per-signal rates.
        out_rate = xdevice.output_sampling_frequency
        for s in r.signals.values():
            if s.name not in expect_names:
                continue
            assert s.length > 0, f'{s.name} has no samples'
            if s.name in ['current', 'voltage', 'power']:
                assert s.sample_rate == out_rate, s.name
            else:
                assert s.sample_rate >= out_rate, s.name
