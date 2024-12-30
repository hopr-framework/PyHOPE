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
import platform
import subprocess
import sys
from importlib import metadata
from io import TextIOWrapper
from packaging.version import Version
from typing import Optional, cast
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
    from pyhope.readintools.readintools import GetInt
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
                np_aff = os.cpu_count() or 1
            np_mtp = min(np_req, np_aff)

    # If running under debugger, multiprocessing is not available
    if DebugEnabled():
        print('│ '  + hopout.warn('Debugger detected, disabling multiprocessing!'))
        np_mtp = 0

    # Actually overwrite the global value
    common_vars.np_mtp = np_mtp

    # Check if we are using the NRG Gmsh version and install it if not
    PkgsCheckGmsh()

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
    return cast(TextIOWrapper, sys.__stdin__).isatty()


def PkgsMetaData(pkgs, classifier) -> Optional[bool]:
    """ Check if the package contains a given classifier
    """
    try:
        meta = metadata.metadata(pkgs)
        classifiers = meta.get_all('Classifier', [])
        return classifier in classifiers

    except metadata.PackageNotFoundError:
        return None


def PkgsMetaVersion(pkgs) -> Optional[str]:
    """ Check the package version
    """
    try:
        version = metadata.version(pkgs)
        return version

    except metadata.PackageNotFoundError:
        return None


def PkgsCheckGmsh() -> None:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Gitlab
    # ------------------------------------------------------

    # Check the current platform
    system = platform.system().lower()
    arch   = platform.machine().lower()

    gmsh_version  = PkgsMetaVersion('gmsh')
    if gmsh_version is None:
        # Gmsh is not installed
        if IsInteractive():
            if system in Gitlab.LIB_SUPPORT and arch in Gitlab.LIB_SUPPORT[system]:
                warning = 'Gmsh is not installed. For compatibility, the NRG Gmsh version will be installed. Continue? (Y/n):'
                response = input('\n' + hopout.warn(warning) + '\n')
                if response.lower() in ['yes', 'y', '']:
                    PkgsInstallGmsh(system, arch, version='nrg')
                    return None
            else:
                warning = 'Gmsh is not installed. As NRG does not provide a compatible Gmsh version,' + \
                          'the PyPI Gmsh version will be installed. Continue? (Y/n):'
                response = input('\n' + hopout.warn(warning) + '\n')
                if response.lower() in ['yes', 'y', '']:
                    PkgsInstallGmsh(system, arch, version='pypi')
                    return None
        else:
            hopout.warning('Gmsh is not installed, exiting...')
            sys.exit(1)

    gmsh_version = cast(str, gmsh_version)
    # Assume that newer versions have updated CGNS
    gmsh_expected = '4.14'
    if Version(gmsh_version) > Version(gmsh_expected):
        return None

    # Check if the installed version is the NRG version
    if PkgsMetaData('gmsh', 'Intended Audience :: NRG'):
        return None

    if system not in Gitlab.LIB_SUPPORT or arch not in Gitlab.LIB_SUPPORT[system]:
        warning = hopout.warn(f'Detected non-NRG Gmsh version on unsupported platform [{system}/{arch}]. ' +
                              'Functionality may be limited.')
        print(warning)
        return None

    if not PkgsMetaData('gmsh', 'Intended Audience :: NRG'):
        if IsInteractive():
            warning  = 'Detected Gmsh package uses an outdated CGNS (v3.4). For compatibility, ' + \
                       'the package will be uninstalled and replaced with the updated NRG GMSH ' + \
                       'version. Continue? (Y/n):'
            response = input('\n' + hopout.warn(warning) + '\n')
            if response.lower() in ['yes', 'y', '']:
                PkgsInstallGmsh(system, arch, version='nrg')
                return None
        else:
            warning = hopout.warn('Detected Gmsh package uses an outdated CGNS (v3.4). Functionality may be limited.')
            print(warning)
            return None


def PkgsInstallGmsh(system: str, arch: str, version: str):
    # Local imports ----------------------------------------
    import hashlib
    import tempfile
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Gitlab
    # ------------------------------------------------------
    # Get our package manager
    # > Check if 'uv' is available
    command = None
    try:
        subprocess.run(['uv', '--version'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = ['uv', 'pip']
    except subprocess.CalledProcessError:
        pass

    # > Check if 'python -m pip' is available
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        command = [sys.executable, '-m', 'pip']
    except subprocess.CalledProcessError:
        pass

    if command is None:
        hopout.warning('No package manager found, you are on your own...')
        return None

    if version == 'nrg':
        # Gitlab "python-gmsh" access
        lfs = 'yes'
        lib = 'gmsh-{}-py3-none-{}_{}.whl'.format(Gitlab.LIB_VERSION, system, arch)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as path:
            # On macOS add major version string to filename an rename darwin to macosx in whl filename
            if system == 'darwin':
                mac_ver = platform.mac_ver()[0].split('.')[0]
                lib  = lib.replace('darwin', 'macosx')
                pkgs = os.path.join(path, lib.replace('macosx_', f'macosx_{mac_ver}_0_'))
            else:
                pkgs = os.path.join(path, lib)

            curl = [f'curl https://{Gitlab.LIB_GITLAB}/api/v4/projects/{Gitlab.LIB_PROJECT}/repository/files/{lib}/raw?lfs={lfs} --output {pkgs}']  # noqa: E501
            _ = subprocess.run(curl, check=True, shell=True)

            # Compare the hash
            # > Initialize a new sha256 hash
            sha256 = hashlib.sha256()
            with open(pkgs, 'rb') as f:
                # Read and update hash string value in blocks of 4K
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)

            if sha256.hexdigest() == Gitlab.LIB_SUPPORT[system][arch]:
                hopout.info('Hash matches, installing Gmsh wheel...')
            else:
                hopout.warning('Hash mismatch, exiting...')
                sys.exit(1)

            # Remove the old version
            try:
                meta = metadata.metadata('gmsh')
                if meta is not None:
                    _ = subprocess.run(command + ['uninstall'] + (['-y'] if sys.executable in command else []) + ['gmsh'], check=True)  # noqa: E501

            except metadata.PackageNotFoundError:
                pass

            # Install the package in the current environment
            _ = subprocess.run(command + ['install'] + (['-y'] if sys.executable in command else []) + [pkgs], check=True)
    else:
        # Install the package in the current environment
        _ = subprocess.run(command + ['install'] + (['-y'] if sys.executable in command else []) + ['gmsh'], check=True)


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


def find_keys(dict: dict[int, str], item) -> list[int] | None:
    """ Find all occurrence of a key in dictionary
    """
    if type(item) is np.ndarray:
        keys = [key for key, val in dict.items() if np.all(val == item)]
        if len(keys) > 0:
            return keys
    else:
        keys = [key for key, val in dict.items() if        val == item ]  # noqa: E271
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


def find_indices(seq, item) -> list[int]:
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
