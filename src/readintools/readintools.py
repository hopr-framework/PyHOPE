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
import sys
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from collections import OrderedDict
from configparser import ConfigParser
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


class MultiOrderedDict(OrderedDict):
    """ Add option to repeat the same key multiple times

        Standard ConfigParser only supports one value per key,
        thus overload the ConfigParser with this new dict_type
    """
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super().__setitem__(key, value)


def strtobool(val: int | bool | str) -> bool:  # From distutils.util.strtobool() [Python 3.11.2]
    """ Convert a string representation of truth to True or False.
        True values  are 'y', 'yes', 't', 'true', 'on', and '1';
        False values are 'n', 'no' , 'f', 'false', 'off', and '0'.
        Raises ValueError if 'val' is anything else.
    """
    if type(val) is bool:
        return val
    if type(val) is int:
        val = str(val)
    if type(val) is str:
        val = val.lower()

    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError('invalid truth value %r' % (val,))


# ==================================================================================================================================
class DefineConfig:
    """ Provide routines to define all HOPR parameters
    """
    def __init__(self) -> None:
        # Create an empty config dictionary
        self.dict = dict()
        return None

    def __enter__(self) -> dict:
        return self.dict

    def __exit__(self, *args: object) -> None:
        return None


# ==================================================================================================================================
def CheckDefined(name, multiple=False, init=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------

    # Check if already defined which is allowed if
    # - we are not in init
    # - multiple parameter
    if init:
        if name in config.prms and not multiple:
            hopout.warning('Parameter "{}" already define and not a multiple option, exiting...'.format(name))
            traceback.print_stack(file=sys.stdout)
            sys.exit()
    else:
        if name not in config.prms:
            hopout.warning('Parameter "{}" is not defined, exiting...'.format(name))
            traceback.print_stack(file=sys.stdout)
            sys.exit()


def CheckUsed(name) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['counter'] > 1 and not config.prms[name]['multiple']:
        hopout.warning('Parameter "{}" already used and not a multiple option, exiting...'.format(name))
        traceback.print_stack(file=sys.stdout)
        sys.exit()


def CheckType(name, calltype) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['type'] is not calltype:
        hopout.warning('Call type of parameter "{}" does not match definition, exiting...'.format(name))
        traceback.print_stack(file=sys.stdout)
        sys.exit()


def CheckDimension(name, result) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['number'] != result:
        hopout.warning('Parameter "{}" has array length mismatch, exiting...'.format(name))
        traceback.print_stack(file=sys.stdout)
        sys.exit()


def CreateSection(string) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple=False, init=True)
    config.prms[string] = dict(type='section', name=string)


def CreateStr(string, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='str',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateInt(string, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='int',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateLogical(string, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='bool',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntFromString(string, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='int2str',
                               name=string,
                               mapping=dict(),
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntOption(string, name, number) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    multiple = config.prms[string]['multiple']
    CheckDefined(string, multiple=multiple, init=False)
    config.prms[string]['mapping'].update({number: name})


def CreateRealArray(string, nReals, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='realarray',
                               number=nReals,
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntArray(string, nInts, help=None, default=None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import src.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='intarray',
                               number=nInts,
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


# ==================================================================================================================================
def CountOption(string) -> int:
    # Local imports ----------------------------------------
    import src.config.config as config
    from configparser import NoOptionError
    # ------------------------------------------------------

    CheckDefined(string)

    try:
        counter = len([s for s in config.params.get('general', string).split('\n') if s != ''])
    except NoOptionError:
        counter = 0
    return counter


def GetParam(name, calltype, default=None, number=None):
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------

    CheckDefined(name)
    config.prms[name]['counter'] += 1
    CheckUsed(name)
    CheckType(name, calltype)

    if config.params.has_option('general', name):
        if config.prms[name]['multiple']:
            # We can request specific indices
            if number is None: number = config.prms[name]['counter']-1
            else:              number = number

            value = [s for s in config.params.get('general', name).split('\n') if s != ''][number]
        else:
            value = config.params.get('general', name)

        # int2str has custom output
        if calltype != 'int2str':
            if calltype == 'bool':
                hopout.printoption(name, '{0:}'.format(value), '*CUSTOM')
            else:
                hopout.printoption(name, value               , '*CUSTOM')
    else:
        if default:
            value = default
        else:
            if config.prms[name]['default'] is not None:
                value = config.prms[name]['default']

                # int2str has custom output
                if calltype != 'int2str':
                    if calltype == 'bool':
                        hopout.printoption(name, '{0:}'.format(value), 'DEFAULT')
                    else:
                        hopout.printoption(name, value               , 'DEFAULT')
            else:
                hopout.warning('Keyword "{}" not found in file and no default given, exiting...'
                               .format(name))
                traceback.print_stack(file=sys.stdout)
                sys.exit()
    return value


def GetStr(name, default=None, number=None) -> str:
    value = GetParam(name=name, default=default, number=number, calltype='str')
    return value


def GetInt(name, default=None, number=None) -> int:
    value = GetParam(name=name, default=default, number=number, calltype='int')
    return int(value)


def GetLogical(name, default=None, number=None) -> bool:
    value = GetParam(name=name, default=default, number=number, calltype='bool')
    return strtobool(value)


def GetIntFromStr(name, default=None, number=None) -> int:
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    # ------------------------------------------------------
    value = GetParam(name=name, default=default, number=number, calltype='int2str')
    # Check if we already received the int. Otherwise, get the value from the
    #  mapping
    mapping = config.prms[name]['mapping']
    if type(value) is int:
        value = value
        hopout.printoption(name, '{} [{}]'.format(value, mapping[value]), 'DEFAULT')
    else:
        if not value.isdigit():
            value = [s for s, v in mapping.items() if v.lower() == value.lower()]
            if len(value) == 0:
                hopout.warning('Unknown value for parameter {}, exiting...'.format(name))
                traceback.print_stack(file=sys.stdout)
                sys.exit()
            else:
                value = int(value[0])
                hopout.printoption(name, '{} [{}]'.format(value, mapping[value]), '*CUSTOM')

    return int(value)


def GetRealArray(name, default=None, number=None) -> np.ndarray:
    value = GetParam(name=name, default=default, number=number, calltype='realarray')

    # Split the array definitiosn
    value = value.split('(/')[1]
    value = value.split('/)')[0]

    # Commas separate 1st dimension, double commas separate 2nd dimension
    if ',,' in value:
        value = [s.split(',') for s in value.split(',,')]
        value = np.array(value).astype(float)
    else:
        value = value.split(',')
        value = np.array(value).astype(float)
    CheckDimension(name, value.size)
    return value


def GetIntArray(name, default=None, number=None) -> np.ndarray:
    value = GetParam(name=name, default=default, number=number, calltype='intarray')

    # Split the array definitiosn
    value = value.split('(/')[1]
    value = value.split('/)')[0]

    # Commas separate 1st dimension, double commas separate 2nd dimension
    value = [s.split(',') for s in value.split(',,')]
    value = np.array(value).astype(int)
    # Reduce dimensions
    value = np.concatenate(value).ravel()
    CheckDimension(name, value.size)
    return value


# ==================================================================================================================================
class ReadConfig():
    """ Read an HOPR parameter file

        This file is meant to remain compatible to the HOPR parameter file
        format, so we need some hacks around the INI file format
    """

    def __init__(self, parameter) -> None:
        self.parameter = parameter
        return None

    def __enter__(self) -> ConfigParser:
        # Local imports ----------------------------------------
        import src.config.config as config
        # ------------------------------------------------------

        parser = ConfigParser(strict=False,
                              comment_prefixes=('#', ';', '!'),
                              inline_comment_prefixes=('#', ';', '!'),
                              dict_type=MultiOrderedDict
                              )

        # HOPR does not use conventional sections, so prepend a
        # fake section header
        with open(self.parameter) as stream:
            parser.read_string('[general]\n' + stream.read())
        if False:
            print(dict(parser.items('general')))

        config.std_length = max(len(s) for s in config.prms.keys())
        config.std_length = max(32, config.std_length+1)

        # Loop over all objects and check if they are provided
        # for key, value in config.prms.items():
        #     if value['type'] == 'section':
        #         hopout.separator()
        #         hopout.info(key)
        #         hopout.separator()
        #         continue
        #
        #     # Check if the key is given in the parameter file
        #     if parser.has_option('general', key):
        #         # Check if the value can be converted
        #         match value['type']:
        #             case 'int':
        #                 try:
        #                     str_int = int(parser.get('general', key))
        #                 except ValueError:
        #                     hopout.warning('Keywords {} cannot be converted to integer'.format(key))
        #
        #         hopout.printoption(key, parser.get('general', key),
        #                            '*CUSTOM', std_length)
        #     # Check if a default option is given
        #     else:
        #         if value['default']:
        #             hopout.printoption(key, value['default'],
        #                                'DEFAULT', std_length)
        #         else:
        #             hopout.warning('Keyword "{}" not found in file, exiting...'
        #                            .format(key))

        return parser

    def __exit__(self, *args: object) -> None:
        return None
