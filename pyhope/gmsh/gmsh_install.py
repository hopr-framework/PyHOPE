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
import shutil
from importlib import metadata
from packaging.version import Version
from typing import Final, Optional, cast
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


def PkgsMetaData(pkgs: str, classifier: str) -> Optional[bool]:
    """ Check if the package contains a given classifier
    """
    try:
        meta = metadata.metadata(pkgs)
        classifiers = meta.get_all('Classifier', [])
    except metadata.PackageNotFoundError:
        return None
    else:
        return classifier in classifiers


def PkgsMetaVersion(pkgs: str) -> Optional[str]:
    """ Check the package version
    """
    try:
        return metadata.version(pkgs)

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
                print(hopout.warn('\n'))
                warning = 'Gmsh is not installed. For compatibility, the NRG Gmsh version will be installed. Continue? (Y/n):'
                response = input(hopout.warn(warning) + '\n')
                if response.lower().strip() in ('yes', 'y', '',):
                    PkgsInstallGmsh(system, arch, version='nrg')
                    return None
            else:
                print(hopout.warn('\n'))
                warning = 'Gmsh is not installed. As NRG does not provide a compatible Gmsh version,' + \
                          'the PyPI Gmsh version will be installed. Continue? (Y/n):'
                response = input(hopout.warn(warning) + '\n')
                if response.lower().strip() in ('yes', 'y', '',):
                    PkgsInstallGmsh(system, arch, version='pypi')
                    return None
        else:
            hopout.error('Gmsh is not installed, exiting...')

    # Assume that newer versions have updated CGNS
    gmsh_version  = cast(str, gmsh_version)
    gmsh_expected = '5.0'
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
            print(hopout.warn('\n'))
            warning  = 'Detected Gmsh package uses an outdated CGNS (v3.4). For compatibility, ' + \
                       'the package will be uninstalled and replaced with the updated NRG GMSH ' + \
                       'version. Continue? (Y/n):'
            response = input(hopout.warn(warning) + '\n')
            if response.lower().strip() in ('yes', 'y', '',):
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
    import subprocess
    import tempfile
    import requests
    # Local imports ----------------------------------------
    from pyhope.common.common_vars import Common
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import Gitlab
    from pyhope.common.common_progress import ProgressBar
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
        lib = f'gmsh-{Gitlab.LIB_VERSION[system][arch]}-py3-none-{system}_{arch}.whl'

        noPing: Final[int] = 1  # Number  of pings
        toPing: Final[int] = 5  # Timeout of pings

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as path:
            # On macOS add major version string to filename an rename darwin to macosx in whl filename
            if system == 'darwin':
                mac_ver = platform.mac_ver()[0].split('.')[0]
                lib  = lib.replace('darwin', 'macosx')
                pkgs = os.path.join(path, lib.replace('macosx_', f'macosx_{mac_ver}_0_'))
            else:
                pkgs = os.path.join(path, lib)

            download_success = False

            # Try primary download from GitHub release assets first
            common = Common()
            github_url = f'{common.url}/releases/download/v{common.version}/{lib}'
            try:
                request = requests.get(github_url, stream=True, timeout=toPing)
                if request.status_code == 200:
                    size = int(request.headers.get('content-length', 0))
                    bar  = ProgressBar(title     = f'│ Downloading Gmsh [v{Gitlab.LIB_VERSION[system][arch]}] from GitHub',  # noqa: E251
                                       value     = int(size/1024),                                                           # noqa: E251
                                       threshold = 0)                                                                        # noqa: E251

                    with open(pkgs, 'wb') as f:
                        for chunk in request.iter_content(chunk_size=1024*50):
                            if chunk:
                                bar.step(int(len(chunk)/1024))
                                f.write(chunk)

                    bar.close()

                    sha256 = hashlib.sha256()
                    with open(pkgs, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256.update(chunk)

                    if sha256.hexdigest() == Gitlab.LIB_SUPPORT[system][arch]:
                        hopout.info('Hash matches, installing Gmsh wheel...')
                        download_success = True
                    else:
                        hopout.info('GitHub asset hash mismatch, falling back to GitLab...')
            except Exception:
                hopout.info('GitHub asset download failed, falling back to GitLab...')

            if not download_success:
                # Try the fallback hosts in order, we generally expect the first host to be up and up-to-date
                for idGit, (gitURL, gitID) in enumerate(zip(Gitlab.LIB_HOST, Gitlab.LIB_PROJECT, strict=True)):
                    try:
                        subprocess.check_call(('ping', f'-c{noPing}', f'-w{toPing}', gitURL),
                                              stdout  = subprocess.DEVNULL,  # noqa: E251
                                              stderr  = subprocess.DEVNULL,  # noqa: E251
                                              timeout = toPing)              # noqa: E251
                    except subprocess.CalledProcessError:  # noqa: PERF203
                        continue

                    # Host is responding, attempt to get the file
                    # urllib.request.urlretrieve(url      = f'https://{gitURL}/api/v4/projects/{Gitlab.LIB_PROJECT}/repository/files/{lib}/raw?lfs={lfs}',  # noqa: E251
                    #                            filename = pkgs                                                                                            # noqa: E251
                    #                           )
                    request = requests.get(f'https://{gitURL}/api/v4/projects/{gitID}/repository/files/{lib}/raw?lfs={lfs}',
                                     stream=True)
                    size    = int(request.headers['content-length'])
                    bar     = ProgressBar(title     = f'│ Downloading Gmsh [v{Gitlab.LIB_VERSION[system][arch]}] from {gitURL}',  # noqa: E251
                                          value     = int(size/1024),                                                             # noqa: E251
                                          threshold = 0)                                                                          # noqa: E251

                    with open(pkgs, 'wb') as f:
                        for chunk in request.iter_content(chunk_size=1024*50):
                            if chunk:  # filter out keep-alive new chunks
                                bar.step(int(len(chunk)/1024))
                                f.write(chunk)

                    bar.close()

                    # Compare the hash
                    # > Initialize a new sha256 hash
                    sha256 = hashlib.sha256()
                    with open(pkgs, 'rb') as f:
                        # Read and update hash string value in blocks of 4K
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256.update(chunk)

                    if sha256.hexdigest() == Gitlab.LIB_SUPPORT[system][arch]:
                        hopout.info('Hash matches, installing Gmsh wheel...')
                        download_success = True
                        break

                    if idGit < len(Gitlab.LIB_HOST) - 1:
                        hopout.info('Gitlab asset hash mismatch, trying next mirror...')

                    hopout.warning('Gitlab asset hash mismatch, not updating Gmsh...')

                if not download_success:
                    return None

            # Remove the old version
            try:
                meta = metadata.metadata('gmsh')
                if meta is not None:
                    try:
                        _ = subprocess.run([*command, 'uninstall', 'gmsh'], check=True)  # noqa: E501
                    except subprocess.CalledProcessError:
                        print(hopout.warn( 'Failed to uninstall system Gmsh. Please run the following commands manually:'))
                        print(hopout.warn( '$ python -m pip uninstall gmsh'))
                        print(hopout.warn(f'$ python -m pip install {pkgs}'))
                        return None

            except metadata.PackageNotFoundError:
                pass

            # Install the package in the current environment
            _ = subprocess.run([*command, 'install', pkgs], check=True)
    else:
        # Install the package in the current environment
        _ = subprocess.run([*command, 'install', 'gmsh'], check=True)
