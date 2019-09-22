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


from joulescope.data_recorder import DataReader
import numpy as np


def parser_config(p):
    """Inspect recordings"""
    p.add_argument('filename',
                   help='The JLS filename to process.')
    p.add_argument('--plot-reduction',
                   action='store_true',
                   help='Plot the stored data reduction preview of the captured data.')
    p.add_argument('--plot',
                   action='store_true',
                   help='Plot the calibrated, captured data.')
    p.add_argument('--plot-raw',
                   help='Plot the raw data with list of plots.'
                        "'i'=current, 'v'=voltage, 'r'=current range. "
                        'Time range must < 2,000,000 samples')
    p.add_argument('--export',
                   help='Filename for export data.  ".csv" and ".npy" supported.')
    p.add_argument('--start',
                   default=0,
                   type=int,
                   help='Starting sample index.  Use 0 by default')
    p.add_argument('--stop',
                   default=-1,
                   type=int,
                   help='Stopping sample index.  Use last available by default')
    return on_cmd


def on_cmd(args):
    r = DataReader().open(args.filename)
    print(r.summary_string())
    start = args.start
    stop = args.stop
    if stop < 0:
        stop = r.sample_id_range[1] + 1 + stop

    if args.export is not None:
        i, v = r.get_calibrated(start, stop, units='samples')
        data = np.hstack((i.reshape((-1, 1)), (v.reshape((-1, 1)))))
        if args.export.endswith('npy'):
            np.save(args.export, data)
        else:
            np.savetxt(args.export, data, fmt='%.5g', delimiter=',')

    if args.plot_reduction:
        import matplotlib.pyplot as plt
        y = r.get_reduction(start, stop)
        x = np.arange(len(y)) * (r.config['samples_per_reduction'] / r.config['sampling_frequency'])
        f = plt.figure()
        for axis in range(3):
            ax = f.add_subplot(3, 1, axis + 1)
            ax.plot(x, y[:, axis, 0], color='blue')
            ax.plot(x, y[:, axis, 2], color='red')
            ax.plot(x, y[:, axis, 3], color='red')

        plt.show()
        plt.close(f)

    if args.plot:
        import matplotlib.pyplot as plt
        i, v = r.get_calibrated(start, stop)
        x = np.arange(len(i)) * (1.0 / r.config['sampling_frequency'])
        f = plt.figure()

        ax_i = f.add_subplot(2, 1, 1)
        ax_i.plot(x, i)
        ax_i.set_ylabel('Current (A)')
        ax_i.grid(True)

        ax_v = f.add_subplot(2, 1, 2, sharex=ax_i)
        ax_v.plot(x, v)
        ax_v.set_ylabel('Voltage (V)')
        ax_v.grid(True)

        ax_v.set_xlabel('Time (s)')

        plt.show()
        plt.close(f)

    if args.plot_raw:
        import matplotlib.pyplot as plt
        if stop - start > 2000000:
            print('Time range too long, cannot --plot-raw')
        else:
            plot_idx_total = len(args.plot_raw)
            link_axis = None
            plot_idx = 1
            d = r.raw(start=start, stop=stop, calibrated=False)
            i_raw = np.right_shift(d[:, 0], 2)
            v_raw = np.right_shift(d[:, 1], 2)
            x = np.arange(len(i_raw)) * (1.0 / r.config['sampling_frequency'])

            i_sel = np.bitwise_and(d[:, 0], 0x0003)
            i_sel_tmp = np.bitwise_and(d[:, 1], 0x0001)
            np.left_shift(i_sel_tmp, 2, out=i_sel_tmp)
            np.bitwise_or(i_sel, i_sel_tmp, out=i_sel)
            del i_sel_tmp
            i_sel = i_sel.astype(np.uint8)

            f = plt.figure()
            f.suptitle('Joulescope Raw Data')

            for c in args.plot_raw:
                if c == 'i':
                    ax = f.add_subplot(plot_idx_total, 1, plot_idx, sharex=link_axis)
                    ax.plot(x, i_raw)
                    ax.set_ylabel('Current (LSBs)')
                elif c == 'v':
                    ax = f.add_subplot(plot_idx_total, 1, plot_idx, sharex=link_axis)
                    ax.plot(x, v_raw)
                    ax.set_ylabel('Voltage (LSBs)')
                elif c == 'r':
                    ax = f.add_subplot(plot_idx_total, 1, plot_idx, sharex=link_axis)
                    ax.plot(x, i_sel)
                    ax.set_ylabel('Current Range')
                else:
                    raise ValueError('unsupported plot: %s' % c)

                ax.grid(True)
                if link_axis is None:
                    link_axis = ax
                plot_idx += 1

            # plt.tight_layout()
            ax.set_xlabel('Time (s)')
            plt.show()
            plt.close(f)

    r.close()
    return 0
