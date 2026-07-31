# joulescope package: full JS220 / JS320 feature support

Status: PROPOSED
Date: 2026-07-31

## Objective

Extend the `joulescope` package (simple, synchronous API) to support the
full JS220 / JS320 signal set and JLS v2 recording while guaranteeing that
existing application code continues to work unmodified.  New features may
require the new package version; old code must not break.

## Investigation summary (2026-07-31)

Baseline: v1 backend basic path verified on hardware for all three devices
(JS110-000578, JS220-001707, JS320-X2VJ): scan, open, `read()`,
`samples_get()` with legacy field names, statistics.

### Missing features

1. **Signals**: v1 `StreamBuffer` only buffers
   `(1,0) i, (2,0) v, (3,0) p, (4,0) r, (5,0) gpi0, (5,1) gpi1`, with gpi0/1
   exposed only under the legacy names `current_lsb` / `voltage_lsb`.
   The JS220 and JS320 both stream `s/gpi/2`, `s/gpi/3`, and trigger
   `s/gpi/7` (field_id=5, index=2/3/7).  The JS320 additionally streams
   voltage range `s/v/range/!data` (field_id=4, index=1).  None are
   reachable through the joulescope package.
2. **JLS v2 recording**: `entry_points/capture.py` uses `DataRecorder`
   (legacy v0 tag format) even for JS220/JS320.  `JlsWriter` (pyjls) is
   exported but unused by any entry point.
   `pyjoulescope_driver.record.Record` already implements JLS v2 recording
   for i, v, p, r, gpi[0..3], trigger_in (short names `i v p r 0 1 2 3 T`)
   and works for both JS220 and JS320.
3. **Parameters**: `parameters_v1.py` is JS110-shaped:
   `v_range` options are 15V/5V (JS220/JS320 2 V handled by special-case),
   `sampling_frequency` advertises 2 MHz / 500 kHz which the JS320 rejects
   (max 1 MHz, no 500 kHz: gateware promotes factors 2/3 to 4).
   `parameters()` does not reflect per-device capability.
4. **JS320-only device features** with no joulescope-package access
   (available via `device.publish()/query()` passthrough): downsample
   filter `s/dwnN/mode` (sinc1/2/3), autorange limits + predict, per-ADC
   select/order, analog out `s/aout/*`, fuses, `c/trigger/dir`,
   `c/trigger/filt`.

### Defects found (must fix; affect backwards compatibility today)

1. `v1/stream_buffer.py:405-410` `samples_get()`: inactive-buffer branch
   assigns `{'value': out}` where `out` is undefined or the previous
   field's array (should be `d`).  Latent `UnboundLocalError`.
2. `v1/sample_buffer.py:138,142`: duplicate/overlap math uses the
   `decimate_factor` *argument* (may be `None`) instead of
   `self._incoming_decimate`.
3. `joulescope/__init__.py:38-41` `__all__` lists
   `bootloaders_run_application` / `bootloader_go`, undefined under the
   default v1 backend, so `from joulescope import *` fails.
4. `v1/device.py`: `extio_status()` defined twice (first definition dead).
5. `v1/stream_buffer.py` docstring advertises `raw`, `raw_current`,
   `raw_voltage`, `bits`, `current_voltage` fields that v1 does not
   implement (docs published via Sphinx autoclass).
6. Upstream `pyjoulescope_driver/record.py`: (a) `current_range` written
   as JLS U4 but the binding delivers unpacked u8, doubling the sample
   count with wrong packing; (b) with multiple devices, subscribe/enable
   loops nest incorrectly and repeat per device.
7. GPI / current_range gap fill is 0 (indistinguishable from real data);
   acceptable, but document it.

### Test coverage

`joulescope/v1/test/` holds only 7 `sample_buffer` tests.  No tests exist
for v1 `stream_buffer`, `device`, `driver`, or the device subclasses.
`test_data_recorder*` / `test_jls_v2_writer` exercise only the v0 buffer.

## Non-goals

- No change to the v0 backend or the legacy JLS v1 (`datafile`) format.
- `statistics_get()` / `data_get()` keep the 6-column layout
  (i, v, p, r, gpi0, gpi1); `View` and the v0 stats dtype depend on it.
- No parameter-per-feature for deep JS320 controls (aout, fuses, ADC
  select).  The `publish()/query()` passthrough remains the documented
  escape hatch, keeping the package simple.

## Design decisions

- **New field names**: extend `FIELDS` in `v1/stream_buffer.py` with
  `gpi0..gpi3`, `trigger_in`, `voltage_range`, plus short aliases
  `'0' '1' '2' '3' 'T'` matching `pyjoulescope_driver.record`.
  `current_lsb` / `voltage_lsb` remain aliases of gpi0 / gpi1 forever.
  The `samples_get(fields=None)` default list is unchanged.
- **Stream enable**: new signals are OFF by default.  Add a `signals`
  parameter (comma-separated short names, default `i,v,p,r,0,1` == current
  behavior).  `start()` subscribes/enables only selected streams.
  JS110 ignores names it does not support.
- **Capture backend**: `joulescope capture` writes JLS v2 via
  `pyjoulescope_driver.record.Record` when running the v1 backend
  (`Device` gains a documented way to reach the underlying driver +
  device_path).  `--format jls1` keeps the `DataRecorder` path; v0
  backend keeps the old behavior.  New `--signals` option accepts the
  short names incl. `0 1 2 3 T`.  `run()` keeps its signature.
- **Per-device parameters**: device subclasses override the option list
  for `sampling_frequency` (and `v_range`) so `parameters()` /
  `parameter_set()` validation reflects the actual device.  Values that
  worked before keep working (JS220/JS320 already clamp 2 MHz to 1 MHz).

## Stages

Each stage is a separate reviewable change, <= 3 files of production code
plus tests.

### Stage 1: backwards-compatibility regression suite (before any change)

- Unit: replay canned `!data` message sequences (JS110 2 Msps shape,
  JS220 2 Msps decimated, JS320 sample_rate=16 M / decimate_factor>=16,
  packed u1 GPI, u4 range, gaps, duplicates) into v1 `StreamBuffer`;
  assert `samples_get` legacy fields, `statistics_get`, `data_get`.
- HIL (`test/hil/`, skipped without hardware): for each of
  JS110/JS220/JS320: scan variants, open modes, `read()` shape/dtype,
  legacy `samples_get` fields, statistics callback format per
  `docs/api/statistics.rst`, `parameter_set` for i_range / v_range /
  sampling_frequency, extio_status, `JlsWriter`, capture entry point
  produces a readable file, View via `stream_test` path.
- This suite is the compatibility contract; it must pass unchanged after
  every later stage.

### Stage 2: defect fixes

- Fix defects 1-5 above (stream_buffer, sample_buffer, `__all__`,
  duplicate `extio_status`, docstring).  Unit tests for each.
- Upstream: fix `record.py` defects 6a/6b in joulescope_driver repo
  (separate change there; prerequisite for Stage 4).

### Stage 3: extended signals

- `v1/stream_buffer.py`: add buffers (5,2), (5,3), (5,7), (4,1); extend
  `FIELDS` + aliases; only active buffers participate in
  `sample_id_range` intersection.
- `v1/device.py` + subclasses: `signals` parameter; `_stream_topics`
  derived from selection; JS320 adds `s/v/range/` via `s/i/range/src`
  guidance documented (v/range stream rides the shared range channel).
- Unit tests with canned messages; HIL test streams `0,1,2,3,T` on JS220
  and JS320 and verifies data alignment (GPO loopback where wired).

### Stage 4: JLS v2 capture

- Rework `entry_points/capture.py` onto `Record` per design above.
- HIL: capture on all three devices; read back with `pyjls.Reader`;
  verify sample rate, signal set, UTC presence; `--format jls1`
  round-trips via `DataReader` (baseline parity).

### Stage 5: per-device parameters + docs

- Subclass parameter option overrides (`sampling_frequency`, `v_range`).
- Docs: JS320 column in `transition_to_v1.csv`, statistics/stream_buffer
  doc updates, README + CHANGELOG.
- HIL: `parameters()` matches accepted `parameter_set` values per device.

### Stage 6: hardware validation sweep

- Full suite (Stage 1 + new-feature tests) on JS110 + JS220 + JS320.
- Long-duration capture (>= 60 s) on JS320 at 1 Msps and a downsampled
  rate; verify no sample drops and JLS v2 integrity.
