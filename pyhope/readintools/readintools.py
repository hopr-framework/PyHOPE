#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
#
# PyHOPE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyHOPE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyHOPE. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
import os
import subprocess
from typing import Optional, Union, cast, final
from typing_extensions import override
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
import numpy as np
from collections import OrderedDict
from configparser import ConfigParser
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@final
class MultiOrderedDict(OrderedDict):
    """ Add option to repeat the same key multiple times

        Standard ConfigParser only supports one value per key,
        thus overload the ConfigParser with this new dict_type
    """
    @override
    def __setitem__(self, key, value) -> None:
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super().__setitem__(key, value)


def strtobool(val: Union[int, bool, str]) -> bool:  # From distutils.util.strtobool() [Python 3.11.2]
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
@final
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
def CheckDefined(name: str, multiple: bool = False, init: bool = False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    # Check if already defined which is allowed if
    # - we are not in init
    # - multiple parameter
    if init:
        if name in config.prms and not multiple:
            hopout.error('Parameter "{}" already define and not a multiple option, exiting...'.format(name), traceback=True)
    else:
        if name not in config.prms:
            hopout.error('Parameter "{}" is not defined, exiting...'.format(name), traceback=True)


def CheckUsed(name: str) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['counter'] > 1 and not config.prms[name]['multiple']:
        hopout.error('Parameter "{}" already used and not a multiple option, exiting...'.format(name), traceback=True)


def CheckType(name: str, calltype: str) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['type'] is not calltype:
        hopout.error('Call type of parameter "{}" does not match definition, exiting...'.format(name), traceback=True)


def CheckDimension(name: str, result: int) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    if config.prms[name]['number'] != result:
        hopout.error('Parameter "{}" has array length mismatch, exiting...'.format(name), traceback=True)


def CreateSection(string: str) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple=False, init=True)
    config.prms[string] = dict(type='section', name=string)


def CreateStr(string: str, help: Optional[str] = None, default: Optional[str] = None, multiple: bool = False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='str',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateReal(string: str, help: Optional[str] = None, default: Optional[float] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='real',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateInt(string: str, help: Optional[str] = None, default: Optional[int] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='int',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateLogical(string: str, help: Optional[str] = None, default: Optional[bool] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='bool',
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntFromString(string: str, help: Optional[str] = None, default: Optional[str] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='int2str',
                               name=string,
                               mapping=dict(),
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntOption(string: str, name, number) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    multiple = config.prms[string]['multiple']
    CheckDefined(string, multiple=multiple, init=False)
    config.prms[string]['mapping'].update({number: name})


def CreateRealArray(string: str, nReals, help: Optional[str] = None, default: Optional[str] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    # ------------------------------------------------------

    CheckDefined(string, multiple, init=True)
    config.prms[string] = dict(type='realarray',
                               number=nReals,
                               name=string,
                               help=help,
                               default=default,
                               counter=0,
                               multiple=multiple)


def CreateIntArray(string: str, nInts, help: Optional[str] = None, default: Optional[str] = None, multiple=False) -> None:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
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
def CountOption(string: str) -> int:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    from configparser import NoOptionError
    # ------------------------------------------------------

    CheckDefined(string)

    try:
        counter = len([s for s in config.params.get('general', string).split('\n') if s != ''])
    except NoOptionError:
        counter = 0
    return counter


def GetParam(name: str, calltype: str, default: Optional[str] = None, number: Optional[int] = None):
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    CheckDefined(name)
    config.prms[name]['counter'] += 1
    CheckUsed(name)
    CheckType(name, calltype)

    if config.params.has_option('general', name):
        if config.prms[name]['multiple']:
            # We can request specific indices
            if number is None: num = config.prms[name]['counter']-1  # noqa: E701
            else:              num = number                          # noqa: E701

            input = [s for s in config.params.get('general', name).split('\n') if s != '']
            if num >= len(input):
                hopout.error(f'Index {num+1} is out of range for option "{name}"', traceback=False)
            value = input[num]
        else:
            value = config.params.get('general', name)
            # Single values cannot contain spaces
            if '\n' in value:
                hopout.error(f'Option "{name}" is already set, but is not a multiple option!', traceback=False)

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
                hopout.error(f'Keyword "{name}" not found in file and no default given, exiting...', traceback=False)
    return value


def GetStr(name: str, default: Optional[str] = None, number: Optional[int] = None) -> str:
    value = GetParam(name=name, default=default, number=number, calltype='str')
    return value


def GetReal(name: str, default: Optional[str] = None, number: Optional[int] = None) -> float:
    value = GetParam(name=name, default=default, number=number, calltype='real')
    return float(value)


def GetInt(name: str, default: Optional[str] = None, number: Optional[int] = None) -> int:
    value = GetParam(name=name, default=default, number=number, calltype='int')
    return int(value)


def GetLogical(name: str, default: Optional[str] = None, number: Optional[int] = None) -> bool:
    value = GetParam(name=name, default=default, number=number, calltype='bool')
    return strtobool(value)


def GetIntFromStr(name: str, default: Optional[str] = None, number: Optional[int] = None) -> int:
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    value  = GetParam(name=name, default=default, number=number, calltype='int2str')
    source = 'DEFAULT' if config.prms[name]['counter'] == 0 else '*CUSTOM'

    # Check if we already received the int. Otherwise, get the value from the mapping
    mapping = config.prms[name]['mapping']
    if type(value) is int:
        hopout.printoption(name, '{} [{}]'.format(value, mapping[value]), source)
    else:
        if not value.isdigit():
            value = [s for s, v in mapping.items() if v.lower() == value.lower()]
            if len(value) == 0:
                hopout.error('Unknown value for parameter {}, exiting...'.format(name), traceback=True)
            else:
                value = int(value[0])
                hopout.printoption(name, '{} [{}]'.format(value, mapping[value]), source)
        else:
            hopout.printoption(name, '{} [{}]'.format(value, mapping[int(value)]), source)

    return int(value)


def GetRealArray(name: str, default: Optional[str] = None, number: Optional[int] = None) -> np.ndarray:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    value = GetParam(name=name, default=default, number=number, calltype='realarray')

    # Split the array definitiosn
    value = value.split('(/')[1]
    value = value.split('/)')[0]

    # Commas separate 1st dimension, double commas separate 2nd dimension
    if ',,' in value:
        value = [s.split(',') for s in value.split(',,')]
        try:
            value = np.array(value).astype(float)
        except ValueError as e:
            print()
            print(hopout.warn(f'{e}'))
            hopout.error(f'Failed to read "{name}" array, possibly malformed comma-separated data. Exiting...')
    else:
        value = value.split(',')
        try:
            value = np.array(value).astype(float)
        except ValueError as e:
            print()
            print(hopout.warn(f'{e}'))
            hopout.error(f'Failed to read "{name}" array, possibly malformed comma-separated data. Exiting...')
    CheckDimension(name, value.size)
    return value


def GetIntArray(name: str, default: Optional[str] = None, number: Optional[int] = None) -> np.ndarray:
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
@final
class ReadConfig():
    """ Read an HOPR parameter file

        This file is meant to remain compatible to the HOPR parameter file
        format, so we need some hacks around the INI file format
    """

    def __init__(self, input: str) -> None:
        self.input     = input
        self.parameter = ''
        self.mesh      = ''

        # define allowed comments
        self.sym_comm = ('#', ';', '!')
        return None

    def _read_file(self) -> list:
        """ Read the parameter file and replace DEFVAR variables
        """
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        processed_lines = []
        variables       = {}

        with open(self.parameter, 'r', encoding='utf-8') as stream:
            for line in stream:
                # Remove all whitespaces
                line = ''.join(line.split())

                # HOPR supported inline comments as prefix before '%'
                # For legacy reasons also support such comment constructs
                if '%' in line:
                    line = line.split('%', 1)[1].strip()

                # Split of [#, ;, !] comments
                for symbol in self.sym_comm:
                    if symbol in line:
                        line = line.split(symbol, 1)[0].strip()

                # HOPR supported inline variable definitions with prefix 'DEFVAR='
                # For legacy reasons also support such variable definition constructs
                if line.strip().startswith('DEFVAR='):
                    if ':' not in line:
                        hopout.error('DEFVAR= syntax error while parsing parameter file. Missing ":"')
                    parts = line.split(':')

                    var_type_part = parts[0].replace('DEFVAR=', '').strip()
                    var_def_part  = parts[1].strip()

                    # Check if comment is in value part
                    for symbol in self.sym_comm:
                        if symbol in var_def_part:
                            var_def_part = var_def_part.split(symbol, 1)[0].strip()  # Take the part before the symbol
                            break  # Stop at the first symbol found

                    # Extract variable type and optional array size
                    if '~' in var_type_part:
                        # Vector
                        _, size_part = var_type_part.split('~')
                        arr_size = int(size_part.strip(')'))  # Convert size to int
                    else:
                        # Scalar
                        arr_size = None

                    # Extract variable name and value (handling spaces around `=`)
                    if '=' in var_def_part:
                        var_name, var_value = [s.strip() for s in var_def_part.split('=', 1)]
                    else:
                        hopout.error(f'DEFVAR= syntax error while parsing "{var_def_part}"')

                    # Ensure unique variable names
                    for existing_var in variables:
                        if var_name in set(existing_var):
                            hopout.error(f'Variable "{var_name}" is ambiguous')

                    # Convert values to proper types
                    if arr_size:  # Handle array
                        values = [float(v) if '.' in v else int(v) for v in var_value.split(',')]
                        if len(values) != arr_size:
                            hopout.error(f'Expected {arr_size} values for array "{var_name}", got {len(values)}')
                        variables[var_name] = values
                    else:  # Single value
                        variables[var_name] = float(var_value) if '.' in var_value else int(var_value)

                    # We have to sort the variables according to the length of the keys in order to avoid
                    # substring replacement in the parameter file. This way it can be assured that long strings
                    # get replaced first.
                    # variables = sorted(variables.items(), key=lambda item: len(item[0]), reverse=True)
                    variables = dict(sorted(variables.items(), key=lambda item: len(item[0]), reverse=True))

                    continue  # Skip adding this line to config

                # Replace variables in the parameter file
                for var, value in variables.items():
                    if isinstance(value, list):  # Convert arrays to string format
                        replacement = f'(/{",".join(map(str, value))}/)'
                    else:
                        replacement = str(value)

                    # Ensure exact match replacement (avoiding substring issues)
                    if '=' in line:
                        tmp = line.split('=')
                        if var in tmp[1]:
                            tmp[1] = tmp[1].replace(var, replacement)
                        line = '='.join(tmp)

                processed_lines.append(line)

        return processed_lines

    def __enter__(self) -> ConfigParser:
        # Local imports ----------------------------------------
        from pyhope.common.common_vars import Common
        import pyhope.config.config as config
        import pyhope.output.output as hopout
        # ------------------------------------------------------

        parser = ConfigParser(strict=False,
                              comment_prefixes=self.sym_comm,
                              inline_comment_prefixes=self.sym_comm,
                              dict_type=MultiOrderedDict
                              )

        # Check if the file exists in argv
        if not self.input:
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE,
                                                                                             stderr=subprocess.DEVNULL)
            common  = Common()
            program = common.program
            version = common.version
            commit  = process.communicate()[0].strip().decode('ascii')

            hopout.header(program, version, commit)
            hopout.error('No parameter or mesh file given')

        # Check if file exists on drive
        if not os.path.isfile(self.input):
            process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE)
            common  = Common()
            program = common.program
            version = common.version
            commit  = process.communicate()[0].strip().decode('ascii')

            hopout.header(program, version, commit)
            hopout.error('Parameter or mesh file [󰇘]/{} does not exist'.format(os.path.basename(self.input)))

        # Check if input is mesh or parameter file
        parameter_mode = False
        mesh_mode      = False

        # Check whether given input is a valid mesh or a parameter file
        if h5py.is_hdf5(self.input):
            mesh_mode = True
        else:
            try:
                with open(self.input, 'r', encoding='utf-8') as f:
                    f.read()
                parameter_mode = True
            except UnicodeDecodeError:
                hopout.error('Parameter or mesh file [󰇘]/{} are of unknown type'.format(os.path.basename(self.input)))

        # Handle parameter data
        if parameter_mode:
            self.parameter = self.input

            # Sore full path of the parameter file
            config.prmfile = os.path.abspath(self.parameter)

            # HOPR does not use conventional sections, so prepend a fake section header
            parser.read_string('[general]\n' + '\n'.join(self._read_file()))

        # Handle mesh data
        if mesh_mode:
            # Set the prmfile to an empty string as it is required for searching for the mesh later in script.
            # As the mesh is however explicitly givenb, this is not required in mesh_mode
            config.prmfile = ''

            # In this mode we need to create a dummy parameter file for processing the mesh.
            # For basic processing as calculating the jacobians and performing the checks only
            # little parameters have to be defined and extracted from the file
            mesh_params = [
                '[general]',
                f'ProjectName = {os.path.splitext(os.path.basename(self.input))[0]}',
                f'Filename    = {os.path.abspath(self.input)}',
                'OutputFormat = HDF5',
                'DebugVisu    = F',
                'Mode         = external',
            ]

            # Get geometric order and boundary conditions
            with h5py.File(self.input, 'r') as f:
                # Here we use item for legacy reasons as HOPR stores scalars as arrays with one element
                NGeo    = cast(int, f.attrs['Ngeo'].item())
                BCNames = [s.decode('utf-8').strip() for s in cast(h5py.Dataset, f['BCNames'])[:]]
                BCType  = cast(h5py.Dataset, f['BCType'])[:]

            # Write geometric order info to file
            mesh_params.append(f'NGeo = {NGeo}')
            mesh_params.append(f'MeshIsAlreadyCurved = {"T" if NGeo > 1 else "F"}')

            # Setup boundary conditions
            for iBC, BC in enumerate(BCNames):
                mesh_params.append(f'BoundaryName = {BC}')
                mesh_params.append(f'BoundaryType = (/{", ".join(map(str, BCType[iBC]))}/)')

            # Join lines into a single string
            mesh_param = '\n'.join(mesh_params)

            # Parse dummy parameters
            parser.read_string(mesh_param)

        # Parse configation file either from read in parameter file or
        # recovered from a given mesh file
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
        #                     hopout.error('Keywords {} cannot be converted to integer'.format(key))
        #
        #         hopout.printoption(key, parser.get('general', key),
        #                            '*CUSTOM', std_length)
        #     # Check if a default option is given
        #     else:
        #         if value['default']:
        #             hopout.printoption(key, value['default'],
        #                                'DEFAULT', std_length)
        #         else:
        #             hopout.error('Keyword "{}" not found in file, exiting...'
        #                            .format(key))

        return parser

    def __exit__(self, *args: object) -> None:
        return None
