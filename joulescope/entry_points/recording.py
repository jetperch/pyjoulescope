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
        k = r.samples_get(start, stop, units='samples', fields=['current', 'voltage'])
        i = k['signals']['current']['value']
        v = k['signals']['voltage']['value']
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
        fields = r.config['reduction_fields']
        for axis, name in enumerate(fields):
            ax = f.add_subplot(len(fields), 1, axis + 1)
            ax.fill_between(x, y[:, axis]['min'], y[:, axis]['max'], color=(0.5, 0.5, 1.0, 0.5))
            ax.plot(x, y[:, axis]['mean'], color='blue')
            ax.set_ylabel(name)
            ax.grid(True)

        plt.show()
        plt.close(f)

    if args.plot:
        import matplotlib.pyplot as plt
        k = r.samples_get(start, stop, units='samples', fields=['current', 'voltage'])
        i = k['signals']['current']['value']
        v = k['signals']['voltage']['value']
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
            rv = r.samples_get(start=start, stop=stop, units='samples', fields=['raw', 'current_range'])
            d_raw = rv['signals']['raw']['value']
            i_sel = rv['signals']['current_range']['value']
            i_raw = np.right_shift(d_raw[:, 0], 2)
            v_raw = np.right_shift(d_raw[:, 1], 2)
            x = np.arange(len(i_raw)) * (1.0 / r.config['sampling_frequency'])

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
