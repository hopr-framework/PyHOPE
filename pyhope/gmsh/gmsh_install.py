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
import shutil
import platform
import subprocess
from importlib import metadata
from packaging.version import Version
from typing import Optional, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


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
    from pyhope.common.common import IsInteractive
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
            hopout.error('Gmsh is not installed, exiting...')

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


def PkgsInstallGmsh(system: str, arch: str, version: str) -> None:
    # Standard libraries -----------------------------------
    import sys
    import hashlib
    import tempfile
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Gitlab
    # ------------------------------------------------------
    # Get our package manager
    # > Check if 'uv' is available
    command = None
    if shutil.which('uv') is not None:
        command = ['uv', 'pip']

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
                hopout.error('Hash mismatch, exiting...')

            # Remove the old version
            try:
                meta = metadata.metadata('gmsh')
                if meta is not None:
                    _ = subprocess.run(command + ['uninstall'] + ['gmsh'], check=True)  # noqa: E501

            except metadata.PackageNotFoundError:
                pass

            # Install the package in the current environment
            _ = subprocess.run(command + ['install'] + [pkgs], check=True)
    else:
        # Install the package in the current environment
        _ = subprocess.run(command + ['install'] + ['gmsh'], check=True)
