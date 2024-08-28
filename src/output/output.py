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
import sys
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# sys.path.append(os.path.dirname(sys.path[0]))
# import src.config.config as config
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


def header(program, version, commit, length=STD_LENGTH):
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


def banner(string, length=STD_LENGTH):
    """ Print the input `string` in a banner-like output.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.BANNERA + '\n' + '='*length)
    print(Colors.BANNERA + ' '+string)
    print(Colors.BANNERA + '='*length + Colors.END)


def small_banner(string, length=STD_LENGTH):
    """ Print the input `string` in a small banner-like output.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.BANNERB + '\n' + '-'*length)
    print(Colors.BANNERB + ' '+string)
    print(Colors.BANNERB + '-'*length + Colors.END)


def warning(string):
    """ Print the input `string` as a warning with the corresponding color.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    print(Colors.WARN + '\n !! '+string+' !! \n'+Colors.END)


def sep(length=5):
    print('├' + '─'*(length-1))


def separator(length=46):
    print('├' + '─'*(length-1))


def end(time, length=STD_LENGTH):
    print('┢' + '━'*(length-1))
    print('┃ UVWXYZ completed in [{:.2f} sec]'.format(time))
    print('┗' + '━'*(length-1))


def info(string, newline=False):
    """ Print the input `string` as generic output without special formatting.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    if newline:
        print('\n│', string)
    else:
        print('│', string)


def routine(string, newline=False):
    """ Print the input `string` as generic output without special formatting.

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    if newline:
        print('\n├──', string)
    else:
        print('├──', string)


def printoption(option, value, status, length=31):
    """ Print the input `string` as option string

        Args:
            string (str): String to be printed in banner.
            length (int): (Optional.) Number of characters in each line.
    """
    if len(value) > length:
        pvalue = '{}...'.format(value[:(length-3)])
    else:
        pvalue = value
    print(f'│ {option:>{length}} │ {pvalue:<{length}} │ {status} │')
