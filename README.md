<!--
# SPDX-FileCopyrightText: Copyright 2018-2026 Jetperch LLC
# SPDX-License-Identifier: Apache-2.0
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
-->

# ![Joulescope](https://download.joulescope.com/press/joulescope_logo-PNG-Transparent-Exact-Small.png "Joulescope Logo")

[![Packaging](https://github.com/jetperch/pyjoulescope/actions/workflows/packaging.yml/badge.svg)](https://github.com/jetperch/pyjoulescope/actions/workflows/packaging.yml)
[![Docs Status](https://readthedocs.org/projects/joulescope/badge/?version=latest)](https://joulescope.readthedocs.io/)

Welcome to the Joulescope™ python driver!  
[Joulescope](https://www.joulescope.com) is an affordable, precision DC energy 
analyzer that enables you to build better products. 

This pyjoulescope python package enables you to
automate Joulescope operation and easily measure current, voltage, power and
energy within your own Python programs.
With the Joulescope driver, controlling your Joulescope is easy.  The following
example captures 0.1 seconds of data and then prints the average current
and voltage:

    import joulescope
    import numpy as np
    with joulescope.scan_require_one(config='auto') as js:
        data = js.read(contiguous_duration=0.1)
    current, voltage = np.mean(data, axis=0, dtype=np.float64)
    print(f'{current} A, {voltage} V')

This package also installs the "joulescope" command line tool:

    joulescope --help

Most Joulescope users will run the graphical user interface which is in the 
[pyjoulescope_ui](https://github.com/jetperch/pyjoulescope_ui) package and
available for [download](https://www.joulescope.com/download).


## Documentation

Visit the [documentation](https://joulescope.readthedocs.io) for details on
installing and using this joulescope package.


## License

All pyjoulescope code is released under the permissive Apache 2.0 license.
See the [License File](LICENSE.txt) for details.
