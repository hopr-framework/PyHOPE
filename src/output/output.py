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
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
STD_LENGTH = 79  # Standard length for output to console
# ==================================================================================================================================


class Colors:
    # Define colors used throughout this framework
    #
    # Attributes:
    #     WARN    (str): Defines color for warnings.
    #     END     (str): Defines end of color in string.
    BANNERA = '\033[93m'
    BANNERB = '\033[94m'
    WARN = '\033[91m'
    END = '\033[0m'


def header(program: str, version: str, commit: str, length: int = STD_LENGTH) -> None:
    """ Print big header with program name and logo to console.

        Args:
            length (int): Number of characters used within each line.
    """
    # string = 'Parametric Exploration and Control Engine'
    print(Colors.BANNERA + '┏' + '━'*(length-1))
    # print(Colors.BANNERA + '┃')
    print(Colors.BANNERA + '┃' + ' U V W X Y Z ')
    # print(Colors.BANNERA + '┃' + ' {}'.format(string))
    print(Colors.BANNERA + '┃' + Colors.END + ' {} version {} [commit {}]'.format(program, version, commit))
    print(Colors.BANNERA + '┡' + '━'*(length-1) + Colors.END)


def banner(string: str, length: int = STD_LENGTH) -> None:
    """ Print the input `string` in a banner-like output.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.BANNERA + '\n' + '='*length)
    print(Colors.BANNERA + ' '+string)
    print(Colors.BANNERA + '='*length + Colors.END)


def small_banner(string: str, length: int = STD_LENGTH) -> None:
    """ Print the input `string` in a small banner-like output.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.BANNERB + '\n' + '-'*length)
    print(Colors.BANNERB + ' '+string)
    print(Colors.BANNERB + '-'*length + Colors.END)


def warning(string: str) -> None:
    """ Print the input `string` as a warning with the corresponding color.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.WARN + '\n !! '+string+' !! \n'+Colors.END)


def sep(length: int = 5) -> None:
    print('├' + '─'*(length-1))


def separator(length: int = 46) -> None:
    print('├' + '─'*(length-1))


def end(time: float, length: int = STD_LENGTH) -> None:
    print('┢' + '━'*(length-1))
    print('┃ UVWXYZ completed in [{:.2f} sec]'.format(time))
    print('┗' + '━'*(length-1))


def info(string: str, newline=False) -> None:
    """ Print the input `string` as generic output without special formatting.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    if newline:
        print('\n│', string)
    else:
        print('│', string)


def routine(string: str, newline=False) -> None:
    """ Print the input `string` as generic output without special formatting.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    if newline:
        print('\n├──', string)
    else:
        print('├──', string)


def printoption(option: str, value: str, status: str, length: int = 31) -> None:
    """ Print the input `string` as option string

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    try:
        if len(value) > length:
            pvalue = '{}...'.format(value[:(length-3)])
        else:
            pvalue = value
    except TypeError:
        pvalue = value
    print(f'│ {option:>{length}} │ {pvalue:<{length}} │ {status} │')
