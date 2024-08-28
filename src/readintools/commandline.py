#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of UVWXYZ
#
# Copyright (c) 2022-2024 Andrea Beck
#
# UVWXYZ is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# UVWXYZ is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# UVWXYZ. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import os
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import argparse
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
STD_LENGTH = 79  # Standard length for output to console
PAR_LENGTH = 32
DEF_LENGTH = 20
# ==================================================================================================================================


def info(string, newline=False):
    if newline:
        print('\n!', string)
    else:
        print('!', string)


def separator(length=STD_LENGTH):
    return '!' + '-'*(length-1)


class CommandLine:
    """ Parse command line arguments, both explicit [*.ini] and flags [--]
    """
    def __init__(self, argv, name, version, commit):
        # Local imports ----------------------------------------
        import src.config.config as config
        from src.output.output import Colors
        # ------------------------------------------------------

        # Read the command line arguments and store everything
        self.argv    = argv
        self.name    = name
        self.version = version
        self.commit  = commit
        # self.help    = ''

        # Print the header
        self.help = (Colors.BANNERA + '!' + '='*(STD_LENGTH-1))
        self.helpjoin('! {} version {} [commit {}]'.format(name, version, commit))
        self.helpjoin(Colors.BANNERA + '!' + '='*(STD_LENGTH-1) + Colors.END)

        # Assemble the help output
        for key in config.prms:
            # Check if we encountered a section
            if config.prms[key]['type'] == 'section':
                self.helpjoin(separator())
                self.helpjoin('! {}'.format(key))
                self.helpjoin(separator())
                continue

            if config.prms[key]['default']:
                default = config.prms[key]['default']
            else:
                default = ''

            if config.prms[key]['help']:
                help    = config.prms[key]['help']
            else:
                help    = ''

            self.helpjoin(f'{key:<{PAR_LENGTH}} = {default:>{DEF_LENGTH}} ! {help}')

        return None

    def __enter__(self):
        # Setup an argument parser and add know arguments
        parser = argparse.ArgumentParser(prog=self.name,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         epilog=self.help)
        parser.add_argument('-V', '--version',
                            action='store_true',
                            help='display the version number and exit')
        parser.add_argument('parameter',
                            nargs='?',
                            metavar='<parameter.ini>',
                            help='HOPR parameter file')
        # Parse known arguments and return other flags for further processing
        args, argv = parser.parse_known_args()
        return args, argv

    def __exit__(self, *args: object) -> None:
        return None


    def helpjoin(self, end):
            self.help = os.linesep.join([self.help, end])
