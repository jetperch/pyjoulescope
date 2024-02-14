
# CHANGELOG

This file contains the list of changes made to pyjoulescope.


## 1.1.12

2023 Feb 14

* Fixed regex string in units  #35
* Bumped dependency versions
  * pyjls from 0.9.1 to 0.9.2.
  * pyjoulescope_driver version from 1.4.6 to 1.4.7


## 1.1.11

2023 Dec 15

* Deferred v1 singleton driver initialization.
* Bumped pyjls and pyjoulescope_driver dependencies.


## 1.1.10

2023 Oct 17

* Fixed JS220 handling of `sample_frequency` parameter set to 2000000.


## 1.1.9

2023 Jul 24

* Added "noexcept" to python callbacks. 
  Cython 3.0 deprecates implicit noexcept.
* Bumped dependency versions
  * pyjls from 0.7.2 to 0.7.3.
  * pyjoulescope_driver version from 1.3.17 to 1.3.18.


## 1.1.8

2023 Jul 11

* Fixed JlsWriter for JS220.
* Bumped pyjoulescope_driver version from 1.3.14 to 1.3.17.


## 1.1.7

2023 Jun 8

* Added GitHub Actions build.
* Bumped minimum python version from 3.8 to 3.9.


## 1.1.6

2023 May 31

* Bumped dependency revisions.


## 1.1.5

2023 May 24

* Added JS110 on-instrument (sensor) statistics option to v1 backend.


## 1.1.4

2023 Apr 28

* Bumped revisions for pyjls and pyjoulescope_driver to latest.
  Many improvements and fixes.  See projects for details.


## 1.1.3

2023 Apr 13

* Fixed single field output for stream_buffer v1.


## 1.1.2

2023 Mar 20

* Modified JOULESCOPE_BACKEND to skip backend autoimport if 'None' specified. 


## 1.1.1

2023 Mar 16

* Bumped to latest 1.2.1 pyjoulescope_driver.


## 1.1.0

2023 Mar 10

* Bumped to latest 1.2.0 pyjoulescope_driver.
* Removed broken v0 backend regression test.
* Fixed JLS v2 writer sample rate validation.


## 1.0.17

2023 Feb 15

* Fixed bit order unpacking for general-purpose input signals,
  which fixes "glitches" on signal transitions.


## 1.0.16

2023 Jan 25

* Implemented v1 extio_status for both JS110 and JS220.


## 1.0.15

2022 Dec 20

* Fixed joulescope.v1.device.Device.statistics_callback.
  If other callbacks were registered, it was re-registering
  the last processed callback, not the one requested.
* Improved error handling for statistics callback registration.
* Fixed JS110 statistics computation with pyjoulescope_driver 1.1.2.


## 1.0.14

2022 Nov 30

* Skip v1 sample_buffer insert when no new data.
* Fixed build dependencies.


## 1.0.13

2022 Nov 21

* Improved info method.
* Fixed JLS v2 writer for JS110.
* Added "pyproject.toml" and removed ignored setup_requires in setup.py.


## 1.0.12

2022 Nov 14

* Added read method to v1 backend.


## 1.0.11

2022 Nov 10

* Modified v1 to output decimate sample id as needed.
* Fixed v1 sample buffering on resets.
* Added local sample buffer decimation for JS220 integer quantities.


## 1.0.10

2022 Nov 8

* Added missing c & h files to tar.gz source distribution. 
* Improved JS220 parameter support.
  * Added v_range "2 V" support.
  * Fixed default output frequency (was 2 MHz, now 1 MHz).


## 1.0.9

2022 Nov 1

* Updated to pyjoulescope_driver 1.0.5
  * Fixed JS110 current range processing for window N and M.
  * Fixed JS110 sample alignment.
  * Fixed JS110 statistics generation time and rate.
* Fixed v1 JS110 config=auto.
* Fixed v1 JS220 voltage to use 15V manual range by default.
* Modified v1 stats to skip NaN values.


## 1.0.8

2022 Oct 30

* Added JS220 GPO support.
* Fixed entry point imports for deployment.


## 1.0.7

2022 Oct 24
 
* Fixed v1 linux support.
* Fixed min/max limits on v1 statistics.
* Updated to pyjoulescope_driver 1.0.3 for improved JS110 support.


## 1.0.6

2022 Oct 12

* Improved v1 stream buffer performance.
  This improvement uses native cython code to still perform brute
  force computation.  Reductions could further improve performance.
* Coalesced duplicate view requests to reduce processing. 


## 1.0.5

2022 Oct 8

* Added JS220 support for current range streaming and fixed
  output sample rate at 1 Msps for all channels.
* Added JLS v1 viewer scale for GPI and current range signals.


## 1.0.4

2022 Oct 6

* Added optional timeouts to v1 device publish and query.
* Fixed v1 device status stub return value to be compatible with UI. 
* Added v1 scan support for bootloader. 
* Enabled JS220 gpi0 and gpi1 streaming by default.
* Ignore view get_samples exception for short-term fix.
  Will be addressed long-term by upcoming memory buffer reimplementation.


## 1.0.3

2022 Oct 4

* Updated udev rules for linux.
* Improved signal support for v1 driver.
  * Add host-side power computation to JS220.
  * Added current range, GPI0, and GPI1 to JS110.
  * JS220 produces empty current_range, GPI0, GPI1.
* Fixed v1 variance computation.
* Added JS220 reduction_frequency support.
* Simplified v1 device name string.
* Added reduction_frequency support to JS110 v1.


## 1.0.2

2022 Sep 27

* Added pyjoulescope_driver dependency.
* Increased pyjls version dependency to >=0.4.2.


## 1.0.1

2022 Sep 24

* Fixed JS220 stream callback.
* Fixed entry points to work with selected backend (not just v0).
* Removed capture_usb entry point, which is no longer valid.
* Fixed JS220 device remove.
* Fixed JS220 scan_for_changes.


## 1.0.0

2022 Sep 9

* JS220 support
  * Refactored code for v0 & v1 support.
  * Migrated to joulescope_driver as default backend.


## 0.9.12

2022 May 31

* Fixed DownsamplingStreamBuffer to use reductions correctly #26


## 0.9.11

2022 Feb 22

* Fixed Mac crashes due to incorrect libusb usage.
* Changed mac libusb search to use signed, packaged libs.


## 0.9.7

2021 Aug 18

*   Added jls_writer for JLS v2 write support.
*   Added pyjls dependency.  


## 0.9.6

2021 Apr 13

*   Updated macOS libusb distribution files for homebrew and arm64 support.


## 0.9.4

2021 Mar 9

*   Added user footer data to JLS v1 data files.
*   Reduced exception catching from all to Exception in scan() #20.
*   Fixed running statistics variance computation.
*   Added numpy requirement to >= 1.17 and Windows from 1.19 to 1.20.
*   Officially dropped Python 3.6 support.
*   Officially dropped Win 7, 8, 8.1 support.


## 0.9.3

2020 Nov 20

*   Updated documentation: added Statistics API and View.
*   Updated initial documentation examples to show float64 accumulator.
*   Fixed "calibrated" option to device read().
*   Clear stream buffer callback on device close.
*   Modifed raw_processor suppress_mode to match current_ranging_type defaults.
*   Improved JLS read performance for downsampled files.
*   Fixed dependencies for numpy 1.19.4 which does not work on Windows.
*   Added UI import traceback print when applicable.


## 0.9.1

2020 Aug 12

*   Fixed PatternBuffer.__len__, which was breaking the unit test in Python 3.8.
*   Eliminated None control transfer reference when closing device under Windows.
*   Improved statistics entry point to display both statistics sources.
*   Updated parameter current_ranging value "n" to be more conservative.
*   Added parameter current_ranging value "m" with original "n" values.
*   Simplified current range filtering.
*   Bumped firmware revision to 1.3.1 which fixes sensor-based statistics computation.
*   Included CREDITS.html and CHANGELOG.md with package.


## 0.9.0

2020 Aug 2

*   Updated installation instructions.
*   Clearly display matplotlib not installed error for recordings entry point.
*   Added numeric option to buffer_duration and reduction_frequency parameters.
*   Clarified documentation stating that USB API is NOT thread-safe.
*   Added install troubleshooting section to documentation.
*   Updated documentation to sphinx 3 and recommonmark (m2r not maintained).
*   Improved installation instructions.
*   Added support for sensor statistics (modified USB status message).
*   Added new driver API methods:
    * statistics_callback_register, statistics_callback_unregister
    * statistics_accumulators_clear
*   Added JOULESCOPE_LOG_LEVEL environment variable.
*   Reduced default runner log level to WARNING.
*   Added program "upgrade" option for easy, complete firmware update.


## 0.8.14

2020 May 8

*   Added encoding='utf-8' to setup.py to fix package install on macOS.
*   Fixed JLS load to better handle truncated files.


## 0.8.13

2020 Apr 29

*   Fixed momentary power OUT power glitch when reconnecting using 'auto' #11.


## 0.8.12

2020 Apr 27

*   Added numpy import_array() to stream_buffer and pattern_buffer.
*   Invalidate statistics on zero length for file reader and stream buffer.


## 0.8.11

2020 Apr 10

*   Added View thread exception handling.
*   Fixed View.samples_get to correctly supply default start and stop.
*   Fixed API calls passing numpy.float32/64 rather than python float.


## 0.8.8

2020 Mar 23

*   Updated sampling_frequency parameter to accept integer values.
*   Fixed downsampling_stream_buffer to support current_voltage for samples_get.
*   Added out_format='samples_get' option to Driver.read().
*   Added "info" command.
*   Added documentation.
*   Updated setup.py
    *   Added "joulescope" entry_point (joulescope_cmd was broken).
    *   Improved "classifiers".
    *   Added "docs" option to build the documents.
    *   Import settings from joulescope.version and check python version.
*   Reduced control transfer timeout.  Prevents crash due to transfer
    completing after device closed.
*   Fixed device thread join so that only happens once.
*   Added "bootloader_go" command.
*   Added parameter_set command-line option which addresses UI issue 27.
*   Fixed libusb backend to wait for transfer cancellation #9.
*   Cleanly close and join the view and device threads.
*   Fixed span set length beyond maximum to set to max and not throw exception.
*   Fixed DownsamplingStreamBuffer.samples_get zero length handling.
*   Improved handling of empty JLS files and zero length data.
*   Modified statistics_get to constrain range and return actual range.
    NOTICE: direct users of StreamBuffer.statistics_get or 
    DownsamplingStreamBuffer.samples_get will have to update their programs.
*   Improved statistics_get handling for missing samples.


## 0.8.7

2020 Feb 28

*   Added missing UsbBulkProcessor.reset() on StreamBuffer.reset().


## 0.8.6

2020 Feb 26

*   API CHANGE to samples_get return value.  Now consistent format.  Affects:
    *   joulescope.stream_buffer.StreamBuffer.samples_get()
    *   joulescope.stream_buffer.DownsamplingStreamBuffer.samples_get()
    *   joulescope.view.View.samples_get()
    *   joulescope.data_recorder.DataReader.samples_get()
*   Added joulescope.data_recorder.DataRecorder.insert() method that accepts
    samples_get formatted data.
*   Fixed capture.py entry point script.
*   Improved non-string support for parameter value aliases through the API.
*   Unified streamed and downsampled APIs.
    *   Converted to µ (micro sign \u00B5), not μ = small greek mu \u03BC.
    *   Correctly populate field when None.



## 0.8.3

2020 Feb 19

*   Fixed MacOS compiler warnings for Cython.


## 0.8.0

2020 Feb 18

*   Unified statistics API data structure.  Code using this structure will
    need to be updated!
*   Restructured DataReader methods to conform to other classes. 
    Applications using DataRead will need to be updated!
    Use methods samples_get, data_get, statistics_get.
*   Removed Device.stream_buffer_duration.  Use "buffer_duration" property.
*   Removed Device.reduction_frequency.  Use "reduction_frequency" property.
*   Refactored StreamBuffer to split out UsbBulkProcessor.
*   Improved parameter definition for UI integration.
*   Added running statistics computation.
*   Added downsampling filter implementation.
*   Removed StreamBuffer.raw_get.  Use StreamBuffer.samples_get.
*   Changed StreamBuffer.__init__() to take duration rather than sample length.
    Allows future compatibility with downsampling buffer length specification.
*   Added ['time']['sample_range'] to StreamBuffer statistics callback data
    to allow for least-squares time fitting on host computer.
*   Fixed ['time']['range'] in StreamBuffer statistics callback data.
*   Added arbitrary JSON-serializable user data storage to JLS files.
*   Added memory check before allocating streaming buffer.
*   Updated version storage.  Just use joulescope.version file.


## 0.7.0

2019 Dec 4

*   Renamed "command" to "entry_point" to prevent confusion with UI "commands".
*   Added "statistics_get_multiple" to view, which allows for markers to be
    fetched together.
*   Changed libusb device name to match Windows device name: Joulescope:xxxxxx.
*   Added support for setting the reduction frequency, which is normally used
    for statistics display, such as the UI multimeter.
*   Forced StreamBuffer array allocations using np.full.  np.empty and np.zeros
    appear to defer allocation which can degrade performance.
*   Removed joulescope.paths which is no longer necessary.
    joulescope.data_recorder.construct_record_filename no longer includes path.


## 0.6.10

2019 Oct 23

*   Fixed current range glitch filter using invalid sample data.
    The glitch filter could occasionally use one sample of invalid data during
    the computation of the "pre" mean component.  The underlying cause was 
    that the pre mean value was computed over a FIFO that was rolling over 1 
    sample too late.  This injected up to one sample of undefined data. 
    For a length n pre value, this error occurred on roughly (n - 1) / 8 
    current range transitions.  Testing shows that we were lucky on 
    Win10 and the data was not a huge floating point number.
    Added unit test and fixed.


## 0.6.8

2019 Oct 15

*   Fixed data-dependent single NaN sample insertion. Only occurred when
    i_range was 3 or 7 and current LSBs was saturated.
    Affects 0.6.0 through 0.6.7.
*   Added customizable current range switching filter using parameters
    suppress_type, suppress_samples_pre, suppress_samples_window, 
    suppress_samples_post.
*   Changed default current range switch filter from mean_0_3_1 to mean_1_n_1,
    which significantly reduces glitches due to current ranging.
*   Added travis-ci build status and Joulescope logo to README.
*   Updated README.


## 0.6.7

2019 Oct 14

*   Improved joulescope.driver.Device.read() method.
    *   Added duration checking.
    *   Slightly improved memory footprint.
*   Added missing hasattr for StreamProcessing object close().
*   Added (fake) missing length to PatternBuffer.
*   Added StreamBuffer.samples_get() to make getting the per sample data 
    simpler, more flexible, and faster.  


## 0.6.5

2019 Oct 9

*   Added optional callback to View.statistics_get for non-blocking operation.
*   Fixed data_recorder.DataReader.raw() to gracefully handle end of file.


## 0.6.4

2019 Oct 3

*   Added configurable stream_buffer_duration (was fixed at 30 seconds).
*   Removed logging.setLevel commands used in debugging.
*   Upgraded to pymonocypher 0.1.3.
*   Added general-purpose input (GPI) support.
    *   Refactored steam_buffer to expose STATS_FIELDS, STATS_VALUES
    *   Report additional signals: current_range, GPI0, GPI1.
    *   Updated file recording to store reductions.  Handle old & new.
*   Defer view update until requested (improve performance).
*   Fixed recordings to apply glitch filtering on current range switching. Now 
    behaves sames as live streaming.


## 0.6.3

2019 Sep 22

*   Modified "recording" command to display reduction and samples.
*   Fixed data recorder statistics computation when length < 1 reduction.


## 0.6.2

2019 Sep 20

*   Added parameter aliases for increased API flexibility.


## 0.6.0

2019 Sep 16

*   Fixed contiguous_stop log message, which was not correctly displayed.
*   Modified capture command to use config='auto'.
*   Reduced span.conform_discrete log level from info to debug.
*   Fixed macOS crash on Joulescope removal (Issue #5).
*   Added digital data path pattern test.  Was intentionally broken long ago.
*   Modified View.samples_get() to have 'signals' like Driver.statistics_get().
*   Fixed invalid data surrounding dropped samples.
*   Reimplemented View to run in separate thread.
    *   Moved Device.statistics_get to View.statistics_get.
    *   Removed Device.view and added Device.view_factory().
    *   Added StreamProcessApi.  Refactored View and DataReader.
    *   Modified StreamBuffer to be aware of time and sampling frequency.
*   Removed Device.recording_start() and Device.recording_stop().
    *   Update DataRecorder to implement StreamProcessApi.
    *   Updated joulescope/command/capture.py.  See for new usage example.


## 0.5.1

2019 Aug 11

*   Added INF for joulescope_bootloader for old Win7 support.
*   Immediately use calibrated data during "off" to "on" transitions.
*   Added "raw" field to View samples_get return value.
*   Added Travis-CI integration.
*   Fixed issue #6: setup.py imports numpy.
*   Caught event_callback_fn exceptions in WinUSB implementation.
*   Fixed sensor programming when unprogrammed.
    *   Increased sensor timeout.
    *   Removed unnecessary normal mode power on.


## 0.5.0

2019 Jul 26

*   Added bootloader_go to get matching device.
*   Added support for FW 1.1.0 JSON-only info format.
*   Added "config" option to Device initialization.
*   Added context manager to Device class.
*   Added file_replace module.
*   Added runtime check for Python 3.6+ and Python bits matching OS bits.
*   Moved current range change glitch suppression from FPGA to stream_buffer.
*   Improved macOS & Linux USB reliability, also be nice to libusb on removal.
*   Improved USB device error handling.
*   Removed NaN injection source.
*   Corrected invalid data possible on first 2 samples.
*   Added timeouts to bootloader / application transitions.
*   Added firmware_manager for more controller firmware update.
*   Made Device and Bootloader safe to open() when already open.


## 0.4.6

2019 Jul 15

*   Added optional application-specific metadata to datafile collection start.
*   Added support for voltage range to datafile: save/load/process correctly.
*   Improved missing sample (NaN) handling robustness.
*   Modified Driver.statistics_callback to match statistics_get format.
*   Compute total charge in addition to energy.
*   Fixed stream_buffer int/uint comparison warning.


## 0.4.5

2019 Jul 2

*   Added joulescope.inf to manually force Win 7 machines without WCID update 
    to recognize Joulescopes.
*   Fixed divide by zero error in stream_buffer.stats_compute_end.
*   Correctly close thread on device_thread open error.
*   Fixed potential 32 bit overflow issue in stream_buffer.
*   Fixed unchecked None data when not streaming in driver.View.


## 0.4.4

2019 Jun 28

*   Added "Quick Start" to README.md.
*   Added parameter "source" alias "on" for "raw".


## 0.4.3

2019 Jun 28

*   Added GPO alias values [0, 1].
*   Added GPO demo example.
*   Updated udev rules with instructions for improved security.
*   Increased Joulescope open timeout again, now 10.0 seconds from 6.0.
*   Fixed Joulescope Bootloader.go() to always close the bootloader.
*   Refactored so that "import joulescope" is useful.
*   Added "bootloaders_run_application".
*   Improved Device.bootloader error handling.
*   Removed libusb IOError if platform does not support hotplug.
*   Added no exception guarantee to scan() and bootloaders_run_application().
*   Fixed Cython compiler warnings.


## 0.4.2

2019 Jun 24

*   Improved error handling on Linux/Mac (libusb) device open.
*   Increase device open timeout.
*   Improved "capture" command logging and error handling.
*   Fixed string descriptor parsing for Linux/Mac (libusb).


## 0.4.1

2019 Jun 20

*   Added CREDITS.html file.


## 0.4.0

2019 Jun 20

*   Create a new ControlTransfer instance each time WinUsbDevice is opened.
*   Added StreamBuffer.stats_get to explicitly compute stats over a range.
*   Migrated to cython language_level=3.
*   Fixed error with statistics computation on NaN data, visible in UI as 
    min/max not being displayed correctly at some zoom levels.
*   Refactored statistics and added statistics_get to DataReader.
*   Unified View API for the physical device and recordings for UI.


## 0.3.1

2019 Jun 3

*   Added log messages for troubleshooting robustness issues.
*   Improved device thread error handling.
*   Eliminated small WinUsb memory leak in normal disconnect case.
*   Added progress callbacks for programming operations.
*   Fixed INFO record processing.
*   Added event_callback_fn and added support to win32 driver.
*   Improved win32 driver error handling & recovery.
*   Promoted driver._info to driver.info.
*   Improved documentation.
*   Keep x_max in view range whenever streaming is active.
*   Added view_time_to_sample_id.


## 0.3.0

2019 Apr 27

*   Added asynchronous control transfers so streaming continues correctly.
*   Improved robustness and recovery on Joulescope fw/hw issues.
*   Added GPI value read (IN) for compliance testing.


## 0.2.7

2019 Mar 2

*   Improved USB device error handling.
*   Allow data_recorder raw defaults to fetch entire file.
*   Added VERSION and __version__ members.
*   Added support for older Mac OS versions when packaged.


## 0.2.6

2019 Feb 16

*   Fixed incorrect column index for "power"


## 0.2.4

2019 Feb 10

*   Fixed incorrect formatting in three_sig_figs for negative numbers.


## 0.2.3

2019 Feb 8

*   Modified span scaling to use pivot point rather than force to center,
    which results in more intuitive UI behavior.
*   Suppress glitches (up to 2 samples) which occur on current range switches.


## 0.2.2

2019 Jan 27

*   Added data file read & data file write support (file format changed).
*   Added command-line tool "recording" for dealing the ".jls" files.


## 0.2.1

2019 Jan 25

*   Added linux support using libusb
*   Added Mac OS X support using libusb


## 0.1.5

2018 Dec 5

*   Check status during sensor firmware programming (was presuming success!).
*   Fixed max range when value is always negative.


## 0.1.0

2018 Oct 9

*   Initial public release.
