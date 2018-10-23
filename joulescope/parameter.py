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
    :param values: The list of acceptable values as ('name', value, ['alias1', ...]) tuples.
    :param units: The units string for the parameter.
    """
    def __init__(self, name, permission, path, default, values, units=None):
        self.name = name
        self.permission = permission
        self.default = default
        self.path = path
        self.values = []
        self.str_to_value = {}
        self.units = units
        if permission not in ['rw', 'r']:
            raise ValueError('Parameter %s, invalid permission %r' % (name, permission))
        for idx, value in enumerate(values):
            if len(value) < 2 or len(value) > 3:
                raise ValueError('Parameter %s, value %d invalid: %r' % (name, idx, value))
            vname = value[0]
            vvalue = value[1]
            if not isinstance(vname, str):
                raise ValueError('Parameter %s, value %d invalid name %r' % (name, idx, vname))
            if len(value) == 2:
                value = (vname, vvalue, [])
            self.values.append(value)
            self._insert(vname, vvalue)
            for alias in value[2]:
                self._insert(alias, vvalue)

        if default not in [x[0] for x in values]:
            raise ValueError('Parameter %s, default %r not in ' % (name, permission))

    def _insert(self, key, value):
        if key in self.str_to_value:
            raise ValueError('Parameter %s: key %s already exists' % (self.name, key))
        self.str_to_value[key] = value
