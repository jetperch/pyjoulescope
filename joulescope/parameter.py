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


class Parameter:
    """Define a single parameter accessible by the application.

    :param name: The parameter name.
    :param permission: The parameter permission which is either
        'r' (read-only) or 'rw' (read-write).
    :param path: None if the parameter is only changed at the streaming start.
        'setting' if the parameter is part of the settings packet.
    :param default: The default value.
    :param values: The list of acceptable values as ('name', value) tuples.
    :param units: The units string for the parameter.
    """
    def __init__(self, name, permission, path, default, values, units=None):
        self.name = name
        self.permission = permission
        self.default = default
        self.path = path
        self.values = values
        self.units = units
        assert(permission in ['rw', 'r'])
        for value in values:
            assert(len(value) == 2)
            assert(isinstance(value[0], str))
        assert(default in [x[0] for x in values])
