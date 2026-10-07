# v1 backend: deduplicate parameter_set

Status: PROPOSED
Date: 2026-10-07

## Problem

`DeviceJs110.parameter_set()` (`joulescope/v1/js110.py`) and
`DeviceJs220.parameter_set()` (`joulescope/v1/js220.py`) share the same
flow: look up the Parameter, reject read-only, `name_to_value` with
validator fallback, validate `signals`, store, queue while closed, then
dispatch through `_param_map`.  They differ only in:

* JS110 handles `current_ranging` by splitting it into sub-parameters.
* JS220 normalizes `v_range` values ('2V', '5 V') before lookup.
* JS110 `_param_map` values may be topic strings (published directly);
  JS220 values are always callables.

Found while moving the v1 backend onto pyjoulescope_driver helpers
(feature/api_refresh).

## Plan

1. Move the shared flow into `Device.parameter_set()`, with two hooks:
   `_parameter_value_normalize(name, value)` (JS220 v_range) and an
   early-return hook for composite parameters (JS110 current_ranging).
2. Dispatch both str (publish) and callable `_param_map` entries in the
   base class.
3. Delete both subclass `parameter_set()` overrides.

Existing `TestVRangeCompat`, `TestSignalsParameter` and
`TestParametersOverride` cover the behavior; add a JS110
`current_ranging` split test before refactoring.

## Also noted

`joulescope/entry_points/statistics.py --compare` registers two
callbacks for sensor and stream_buffer statistics.  The v1 backend has
one statistics source per device, so both print the same values.
Remove the option or restrict it to the v0 backend.
