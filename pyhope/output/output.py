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
# ----------------------------------------------------------------------------------------------------------------------------------
import sys
from dataclasses import dataclass
from typing import Final, Optional, NoReturn
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
STD_LENGTH: Final[int] = 79  # Standard length for output to console
# ==================================================================================================================================


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class Symbols:
    OK:      Final[str] = '✅ OK'
    WARN:    Final[str] = '⚠️ WARNING'
    ERR:     Final[str] = '❌ ERROR'


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class Colors:
    """ Define colors used throughout this framework

        Attributes:
            WARN    (str): Defines color for warnings
            END     (str): Defines end of color in string
    """
    BANNERA: Final[str] = '\033[93m'
    BANNERB: Final[str] = '\033[94m'
    WARN:    Final[str] = '\033[91m'
    END:     Final[str] = '\033[0m'


def header(program: str, version: str, commit: Optional[str], length: int = STD_LENGTH) -> None:
    """ Print big header with program name and logo to console

        Args:
            length (int): Number of characters used within each line
    """
    # string = 'Parametric Exploration and Control Engine'
    print(Colors.BANNERA + '┏' + '━'*(length-1))
    # print(Colors.BANNERA + '┃')
    print(Colors.BANNERA + '┃' + ' P y H O P E — Python High-Order Preprocessing Environment')
    # print(Colors.BANNERA + '┃' + ' {}'.format(string))
    print(f'{Colors.BANNERA}┃{Colors.END} {program} version {version}' + (f' [commit {commit}]' if commit else ''))
    print(Colors.BANNERA + '┡' + '━'*(length-1) + Colors.END)


# def banner(string: str, length: int = STD_LENGTH) -> None:
#     """ Print the input `string` in a banner-like output
#
#         Args:
#             string (str): String to be printed in banner
#             length (int): (Optional.) Number of characters in each line
#     """
#     print(Colors.BANNERA + '\n' + '='*length)
#     print(Colors.BANNERA + ' '+string)
#     print(Colors.BANNERA + '='*length + Colors.END)


def small_banner(string: str, length: int = STD_LENGTH) -> None:
    """ Print the input `string` in a small banner-like output

        Args:
            string (str): String to be printed in banner
            length (int): (Optional.) Number of characters in each line
    """
    print(Colors.BANNERB + '├' + '─'*(length-1))
    print(Colors.BANNERB + '│ '+string)
    print(Colors.BANNERB + '├' + '─'*(length-1) + Colors.END)


def warn(string:   str,
         length:   int  = STD_LENGTH,
         prefix:   str  = Colors.WARN + '│  WARNING  ┃ '  + Colors.END,
         warnonce: bool = False) -> str:
    """ Format the input `string` as a warning with the corresponding color

        Args:
                string (str): String to be printed in banner
                length (int): (Optional.) Number of characters in each line
    """
    # Standard libraries -----------------------------------
    import re
    import textwrap
    # ------------------------------------------------------
    # Remove ANSI escape codes for accurate visible-length calculation
    ansiEscape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    lprefix    = len(ansiEscape.sub('', prefix))
    # Wrap the message to account for the visible prefix width
    wrap_msg   = textwrap.fill(string, width=length - lprefix)

    # Split into lines and format
    lines = wrap_msg.splitlines()
    if not lines:
        # If there's nothing to print, return just the prefix (trim trailing spaces)
        return prefix.rstrip()

    if not warnonce:
        # Add prefix to every wrapped line
        formatted_lines = [f'{prefix}{line}' for line in lines]
    else:
        # Add full prefix only to the first line, pad subsequent lines with spaces
        formatted_lines = [f'{prefix}{lines[0]}']
        indent = '│' + ' ' * (lprefix-1)
        formatted_lines.extend(f'{indent}{line}' for line in lines[1:])

    return '\n'.join(formatted_lines)


def warning(string: str, file=sys.stdout) -> None:
    """ Print the input `string` as a warning with the corresponding color

        Args:
            string (str): String to be printed in banner
            file (TextIO): Output unit of the message
    """
    print(Colors.WARN + '\n !! '+string+' !! \n' + Colors.END, flush=True, file=file)


def error(string: str, traceback=False, file=sys.stderr) ->  NoReturn:
    """ Print the input `string` as a error with the corresponding color

        Args:
            string (str): String to be printed in banner
            traceback (bool): Print traceback information
            file (TextIO): Output unit of the message
    """
    # Local imports ----------------------------------------
    from traceback import print_stack
    # ------------------------------------------------------
    print(Colors.WARN + '\n !! '+string+' !! \n' + Colors.END, flush=True, file=file)
    if traceback:
        print_stack(file=file)
    sys.exit(1)


def sep(length: int = 5) -> None:
    print('├' + '─'*(length-1))


def separator(length: int = 46) -> None:
    print('├' + '─'*(length-1))


def end(program: str, time: float, length: int = STD_LENGTH) -> None:
    print('┢' + '━'*(length-1))
    print('┃ {} completed in [{:.2f} sec]'.format(program, time))
    print('┗' + '━'*(length-1))


def info(string: str, newline: bool = False, end: Optional[str] = None) -> None:
    """ Print the input `string` as generic output without special formatting

        Args:
            string (str): String to be printed in banner
            length (int): (Optional.) Number of characters in each line
    """
    if newline:
        print('\n│ '+ string, end=end)
    else:
        print('│ '  + string, end=end)


def routine(string: str, newline: bool = False) -> None:
    """ Print the input `string` as generic output without special formatting

        Args:
            string (str): String to be printed in banner
            length (int): (Optional.) Number of characters in each line
    """
    if newline:
        print('\n├── ' + string)
    else:
        print('├── '   + string)


def printoption(option: str, value: str, status: str, length: int = 31) -> None:
    """ Print the input `string` as option string

        Args:
            string (str): String to be printed in banner
            length (int): (Optional.) Number of characters in each line
    """
    try:
        if len(value) > length:
            pvalue = '{}...'.format(value[:(length-3)])
        else:
            pvalue = value
    except TypeError:
        pvalue = value
    print(f'│ {option:>{length}} │ {pvalue:<{length}} │ {status} │')
