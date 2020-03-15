#!/usr/bin/env python3
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

"""Joulescope command-line utility."""


import sys
import argparse
import logging
from joulescope.entry_points import bootloader_go, capture, capture_usb, gpo_demo, info, program, recording


entry_points = [bootloader_go, capture, capture_usb, gpo_demo, info, program, recording]
"""This list of available command modules.  Each module must contain a 
parser_config(subparser) function.  The function must return the callable(args)
that will be executed for the command."""


try:
    from joulescope_ui.entry_points import ui as ui_entry_point
    entry_points.insert(0, ui_entry_point)
except ImportError:
    ui_entry_point = None


def get_parser():
    parser = argparse.ArgumentParser(description='Joulescope™ command line tools.')
    subparsers = parser.add_subparsers(
        dest='subparser_name',
        help='The command to execute')

    for entry_point in entry_points:
        default_name = entry_point.__name__.split('.')[-1]
        name = getattr(entry_point, 'NAME', default_name)
        cfg_fn = entry_point.parser_config
        p = subparsers.add_parser(name, help=cfg_fn.__doc__)
        cmd_fn = cfg_fn(p)
        if not callable(cmd_fn):
            raise ValueError(f'Invalid command function for {name}')
        p.set_defaults(func=cmd_fn)

    subparsers.add_parser('help', help='Display the command help. Use [command] --help to display help for a specific command.')

    return parser


def run():
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    if args.subparser_name is None:
        if ui_entry_point is not None:
            return ui_entry_point.run()
        else:
            print('No command provided and user interface not found.  Please specify a command.')
            parser.print_help()
            parser.exit()
    elif args.subparser_name.lower() in ['help']:
        parser.print_help()
        parser.exit()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(run())
