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
import os
from io import TextIOWrapper
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def DefineCommon() -> None:
    """ Define general options for the entire program
    """
    # Local imports ----------------------------------------
    from pyhope.readintools.readintools import CreateInt, CreateSection
    # ------------------------------------------------------

    # Check the number of available threads
    try:
        np_aff = len(os.sched_getaffinity(0))
    except AttributeError:
        np_aff = os.cpu_count() or 1
    # Reserve two threads for the operating system and the main thread
    np_aff -= 2

    CreateSection('Common')
    CreateInt(      'nThreads',        default=np_aff,     help='Number of threads for multiprocessing')


def InitCommon() -> None:
    """ Readin general option for the entire program
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.common.common_vars as common_vars
    from pyhope.gmsh.gmsh_install import PkgsCheckGmsh
    from pyhope.readintools.readintools import GetInt
    # ------------------------------------------------------

    hopout.info('INIT PROGRAM...')

    # Check the number of available threads
    np_req = GetInt('nThreads')
    match np_req:
        case -1 | 0:  # All available cores / no multiprocessing
            np_mtp = np_req
        case _:       # Check if the number of requested processes can be provided
            # os.affinity is Linux only
            try:
                np_aff = len(os.sched_getaffinity(0))
            except AttributeError:
                np_aff = os.cpu_count() or 1
            np_mtp = min(np_req, np_aff)

    # If running under debugger, multiprocessing is not available
    if DebugEnabled():
        print(hopout.warn('Debugger detected, disabling multiprocessing!'))
        np_mtp = 0

    # Actually overwrite the global value
    common_vars.np_mtp = np_mtp

    # Check if we are using the NRG Gmsh version and install it if not
    PkgsCheckGmsh()

    # hopout.info('INIT PROGRAM DONE!')


def DebugEnabled() -> bool:
    """ Check if program runs with debugger attached
        > https://stackoverflow.com/a/77627075/23851165
    """
    # Standard libraries -----------------------------------
    import sys
    # ------------------------------------------------------
    try:
        if sys.gettrace() is not None:
            return True
    except AttributeError:
        pass

    try:
        if sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None:
            return True
    except AttributeError:
        pass

    return False


def IsInteractive() -> bool:
    """ Check if the program is running in an interactive terminal
    """
    # Standard libraries -----------------------------------
    import sys
    # ------------------------------------------------------
    return cast(TextIOWrapper, sys.__stdin__).isatty() and cast(TextIOWrapper, sys.__stdout__).isatty()


def IsDisplay() -> bool:
    """ Check if the program is running in a display environment
    """
    # Standard libraries -----------------------------------
    import sys
    # ------------------------------------------------------
    # Check if running on Linux, otherwise assume a display
    if not sys.platform.startswith('linux'):
        return True

    # Check for environment variables that indicate a graphical display
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


# > https://stackoverflow.com/a/5419576/23851165
# def object_meth(object) -> list:
#     methods = [method_name for method_name in dir(object)
#                if '__' not in method_name]
#     return methods


def find_key(dict: dict[int, str], item) -> int | None:
    """ Find the first occurrence of a key in dictionary
    """
    if type(item) is np.ndarray:
        for key, val in dict.items():
            if np.all(val == item):
                return key
    else:
        for key, val in dict.items():
            if        val == item :  # noqa: E271
                return key
    return None


def find_keys(dict: dict[int, str], item) -> tuple[int, ...] | None:
    """ Find all occurrence of a key in dictionary
    """
    if type(item) is np.ndarray:
        keys = tuple(key for key, val in dict.items() if np.all(val == item))
        if len(keys) > 0:
            return keys
    else:
        keys = tuple(key for key, val in dict.items() if        val == item )  # noqa: E271
        if len(keys) > 0:
            return keys
    return None


# def find_value(dict, item):
#     """ Find key by value in dictionary
#     """
#     return dict.keys()[dict.values().index(item)]


def find_index(seq, item) -> int:
    """ Find the first occurrences of a key in a list
    """
    if type(seq) is np.ndarray:
        seq = seq.tolist()

    if type(item) is np.ndarray:
        for index, val in enumerate(seq):
            if np.all(val == item):
                return index
    else:
        for index, val in enumerate(seq):
            if        val == item :  # noqa: E271
                return index
    return -1


def find_indices(seq, item) -> tuple[int, ...]:
    """ Find all occurrences of a key in a list
    """
    if type(seq) is np.ndarray:
        seq = seq.tolist()

    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return tuple(locs)


def lines_that_equal(     string: str, fp: list, start_idx=0) -> list[int]:
    """ Find all occurrences of a string in a file-like object
    """
    return [num for num, line in enumerate(fp[start_idx:]) if line.strip() == string]


def lines_that_contain(   string: str, fp: list, start_idx=0) -> list[int]:
    """ Find all occurrences of a string in a file-like object
    """
    return [num for num, line in enumerate(fp[start_idx:], start=start_idx) if string in line]


def lines_that_start_with(string: str, fp: list, start_idx=0) -> list[int]:
    """ Find all occurrences of a string at the start of a line in a file-like object
    """
    return [num for num, line in enumerate(fp[start_idx:]) if line.startswith(string)]


def lines_that_end_with(  string: str, fp: list, start_idx=0) -> list[int]:
    """ Find all occurrences of a string at the end of a line in a file-like object
    """
    return [num for num, line in enumerate(fp[start_idx:]) if line.rstrip().endswith(string)]
