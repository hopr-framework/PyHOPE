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
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def DefineCommon():
    """ Define general options for the entire program
    """
    # Local imports ----------------------------------------
    from src.readintools.readintools import CreateInt, CreateSection
    # ------------------------------------------------------
    CreateSection('Common')
    CreateInt(      'nThreads',        default=4,     help='Number of threads for multiprocessing')


def InitCommon():
    """ Readin general option for the entire program
    """
    # Local imports ----------------------------------------
    import src.output.output as hopout
    from src.common.common_vars import np_mtp
    from src.readintools.readintools import GetInt
    # ------------------------------------------------------

    hopout.separator()
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
                np_aff = 1
            np_mtp = min(np_req, np_aff)

    hopout.info('INIT PROGRAM DONE!')


# > https://stackoverflow.com/a/5419576/23851165
def object_meth(object):
    methods = [method_name for method_name in dir(object)
               if '__' not in method_name]
    return methods


def find_key(seq, item):
    """ Find the first occurance of a a key in dictionary
    """
    if type(item) is np.ndarray:
        for key, val in seq.items():
            if np.all(val == item): return key
    else:
        for key, val in seq.items():
            if        val == item : return key
    return None


def find_keys(seq, item):
    """ Find all occurance of a a key in dictionary
    """
    if type(item) is np.ndarray:
        keys = [key for key, val in seq.items() if np.all(val == item)]
        if len(keys) > 0: return keys
    else:
        keys = [key for key, val in seq.items() if        val == item ]
        if len(keys) > 0: return keys
    return None


def find_index(seq, item) -> int:
    """ Find the first occurances of a a key in a list
    """
    if type(seq) is np.ndarray:
        seq = seq.tolist()

    if type(item) is np.ndarray:
        for index, val in enumerate(seq):
            if np.all(val == item): return index
    else:
        for index, val in enumerate(seq):
            if        val == item : return index
    return -1


def find_indices(seq, item):
    """ Find all occurances of a a key in a list
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
    return locs
