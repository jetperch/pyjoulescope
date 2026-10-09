# JLS UTC cadence deduplication

`joulescope.jls_v2_writer.JlsWriter._utc_update()` repeats the UTC entry
cadence in `pyjoulescope_driver.record.Record._on_data()`: write the first
sample_id/UTC pair, one pair per minute, and the latest pair at close.

The copy is about 15 lines.  Sharing it now would make pyjoulescope depend
on `Record` internals, so it was accepted for issue #37.

## Plan

1. Add a small public helper to pyjoulescope_driver, such as
   `pyjoulescope_driver.record.UtcCadence(writer, signal_ids)` with
   `update(sample_id, utc)` and `close()`.
2. Use it from `Record`, release pyjoulescope_driver.
3. Bump the pyjoulescope dependency and replace `JlsWriter._utc_update()`.
