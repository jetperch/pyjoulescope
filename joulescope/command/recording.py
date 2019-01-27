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
                   help='The capture duration in seconds.')
    p.add_argument('--plot',
                   action='store_true',
                   help='Plot the captured data (data reduction preview only).')
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
        i, v = r.get_calibrated(start, stop)
        data = np.hstack((i.reshape((-1, 1)), (v.reshape((-1, 1)))))
        if args.export.endswith('npy'):
            np.save(args.export, data)
        else:
            np.savetxt(args.export, data, fmt='%.5g', delimiter=',')

    if args.plot:
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

    r.close()
    return 0
