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
import importlib
import subprocess
import sys
from typing import Union
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
    from src.readintools.readintools import CreateInt, CreateSection
    # ------------------------------------------------------
    CreateSection('Common')
    CreateInt(      'nThreads',        default=4,     help='Number of threads for multiprocessing')


def InitCommon() -> None:
    """ Readin general option for the entire program
    """
    # Local imports ----------------------------------------
    import src.output.output as hopout
    import src.common.common_vars as common_vars
    from src.readintools.readintools import GetInt
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('INIT PROGRAM...')

    # Check the number of available threads
    np_mtp = 4
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
        hopout.info('Debugger detected, disabling multiprocessing!')
        np_mtp = 0

    # Actually overwrite the global value
    common_vars.np_mtp = np_mtp

    # Check if we are using the NRG Gmsh version
    if not PkgsMetaData('gmsh', 'Intended Audience: NRG'):
        if IsInteractive():
            hopout.info('Detected non-NRG Gmsh version, exiting...')
            response = input('Do you want to install the correct package version? (Y/n): ')
            if response.lower() in ['yes', 'y', '']:
                PkgsInstallGmsh()
        else:
            hopout.warning('Detected non-NRG Gmsh version, exiting...')
            sys.exit(1)

    hopout.info('INIT PROGRAM DONE!')


def DebugEnabled() -> bool:
    """ Check if program runs with debugger attached
        > https://stackoverflow.com/a/77627075/23851165
    """
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


def IsInteractive():
    return sys.__stdin__.isatty()


def PkgsMetaData(pkgs, classifier) -> Union[bool, None]:
    """ Check if the package contains a given classifier
    """
    try:
        metadata    = importlib.metadata.metadata(pkgs)
        classifiers = metadata.get_all('Classifier', [])
        return classifier in classifiers

    except importlib.metadata.PackageNotFoundError:
        return None


def PkgsInstallGmsh():
    # Local imports ----------------------------------------
    import hashlib
    import tempfile
    import src.output.output as hopout
    from src.common.common_vars import Gitlab
    # ------------------------------------------------------

    # Gitlab "python-gmsh" access
    lfs = 'yes'
    lib = 'gmsh-{}-py3-none-linux_x86_64.whl'.format(Gitlab.LIB_VERSION)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as path:
        pkgs = os.path.join(path, lib)
        curl = [f'curl https://{Gitlab.LIB_GITLAB}/api/v4/projects/{Gitlab.LIB_PROJECT}/repository/files/{lib}/raw?lfs={lfs} --output {pkgs}']  # noqa: E501
        subprocess.run(curl, check=True, shell=True)

        # Compare the hash
        # > Initialize a new sha256 hash
        sha256 = hashlib.sha256()
        with open(pkgs, 'rb') as f:
            # Read and update hash string value in blocks of 4K
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        if sha256.hexdigest() == Gitlab.LIB_HASH:
            hopout.info('Hash matches, installing Gmsh wheel...')
        else:
            hopout.warning('Hash mismatch, exiting...')
            sys.exit(1)

        # Install the package in the current environment
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkgs], check=True)


# > https://stackoverflow.com/a/5419576/23851165
# def object_meth(object) -> list:
#     methods = [method_name for method_name in dir(object)
#                if '__' not in method_name]
#     return methods


def find_key(dict: dict, item) -> int | None:
    """ Find the first occurrence of a key in dictionary
    """
    if type(item) is np.ndarray:
        for key, val in dict.items():
            if np.all(val == item):
                return key
    else:
        for key, val in dict.items():
            if        val == item :
                return key
    return None


def find_keys(dict: dict, item) -> list | None:
    """ Find all occurrence of a key in dictionary
    """
    if type(item) is np.ndarray:
        keys = [key for key, val in dict.items() if np.all(val == item)]
        if len(keys) > 0:
            return keys
    else:
        keys = [key for key, val in dict.items() if        val == item ]
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
            if        val == item :
                return index
    return -1


def find_indices(seq, item) -> list:
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
    return locs
