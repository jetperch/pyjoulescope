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
    :param path: None if the parameter is only changed at the streaming start.
        'setting' if the parameter is part of the settings packet.
    :param default: The default value.
    :param options: The list of acceptable values as ('name', value, ['alias1', ...]) tuples.
    :param units: The units string for the parameter.
    :param brief: The brief user-meaningful description.
    :param detail: The detailed user-meaningful description.
    :param flags: The list of flags which include:
        * developer: This parameter is intended for developers only, not end users.
        * read_only: This parameter can only be read (no write permission).
    :param validator: Function called to validate the parameter value.
    """
    def __init__(self, name, path, default=None, options=None, units=None,
                 brief=None, detail=None,
                 flags=None, validator=None):
        self.name = name
        self.default = default
        self.path = path
        self.options = []
        self.str_to_value = {}
        self.units = units
        self.brief = brief
        self.detail = detail
        self.flags = [] if flags is None else flags
        self.validator = validator
        if options is None:
            return
        for idx, option in enumerate(options):
            if len(option) < 2 or len(option) > 3:
                raise ValueError('Parameter %s, value %d invalid: %r' % (name, idx, option))
            vname = option[0]
            vvalue = option[1]
            if not isinstance(vname, str):
                raise ValueError('Parameter %s, value %d invalid name %r' % (name, idx, vname))
            if len(option) == 2:
                option = (vname, vvalue, [])
            self.options.append(option)
            self._insert(vname, vvalue)
            for alias in option[2]:
                self._insert(alias, vvalue)

        if default not in [x[0] for x in options]:
            raise ValueError('Parameter %s, default %r not found' % (name, default))

    def __str__(self):
        return f'Parameter({self.name})'

    def _insert(self, key, value):
        if key in self.str_to_value:
            raise ValueError('Parameter %s: key %s already exists' % (self.name, key))
        self.str_to_value[key] = value
