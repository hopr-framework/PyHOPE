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
from __future__ import annotations
import os
import platform
import subprocess
import sys
import tempfile
from enum import Enum, unique
from packaging.version import Version
from typing import Final, Optional
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
import numpy as np
import pathlib
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
user  : Final[str] = 'hopr-framework'
repo  : Final[str] = 'PyHOPE'
branch: Final[str] = 'main'
# ==================================================================================================================================


@unique
class lfsFlag(Enum):
    lfsOkay = 0
    lfsWarn = 1
    lfsErr  = 2


def hdf5Stats(obj: h5py.Dataset) -> Optional[dict[str, float]]:
    """ Calculate mean, min/max and standard deviation of a given HDF5 dataset
    """
    stats = {}

    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[()]
            if np.issubdtype(data.dtype, np.number):
                minv    = float(np.min(data))
                maxv    = float(np.max(data))
                meanv   = float(np.mean(data))
                stddevv = float(np.std(data))
                stats = {'min': minv, 'max': maxv, 'mean': meanv, 'stddev': stddevv}
        except Exception:
            # raise ValueError('Failed to parse generated HDF5 file')
            return None

    return stats


def CheckInstall(path: Optional[str] = None) -> None:
    """ Verify the installation by comparing against known results
    """
    # Third-party libraries --------------------------------
    import h5py
    import re
    import urllib.error
    from contextlib import redirect_stdout
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.common.common_git import findGitRoot, downloadGitDir, downloadGitFile
    from pyhope.common.common_git import isLFSFile
    from pyhope.readintools.readintools import ReadConfig
    # ------------------------------------------------------

    # Standard library since Python 3.11
    if Version(platform.python_version()) >= Version('3.11'):
        import tomllib  # ty: ignore[unresolved-import]
    else:
        tomllib = None

    testDir: Final[str]   = 'tutorials'
    testArr: Final[tuple] = ('ElemInfo', 'GlobalNodeIDs', 'NodeCoords', 'SideInfo')
    testLen: Final[int]   = max(len(s) for s in testArr)
    tmpDir  = None
    lfsWarn = lfsFlag.lfsOkay

    hopout.small_banner('Verifying installation')

    # Path is None, search for the tutorials directory relative to the git root
    if not path:
        root = findGitRoot()

        if root:
            path = os.path.join(root, testDir)

    # Path is None, search for the tutorials directory relative to the git root
    if not path or not os.path.isdir(path):
        cwd  = os.path.join(os.getcwd(), testDir)
        path = os.path.join(cwd, testDir)

    # Directory not exist, download to temporary path
    if not path or not os.path.isdir(path):
        # --- Token Discovery ---
        token = os.environ.get('GITHUB_TOKEN')
        if not token:
            try:
                # Try to get the token from the gh-cli
                token = subprocess.check_output(['gh', 'auth', 'token']).strip().decode()
                hopout.info('Discovered GitHub token, will use it for authentification')
                hopout.sep()
            except (FileNotFoundError, subprocess.CalledProcessError):
                # gh-cli not found or not logged in
                token = None
                print(hopout.warn('No GitHub token found. Using unauthenticated requests which are severely rate-limited.'))
                hopout.sep()

        tmpDir = tempfile.TemporaryDirectory()
        downloadGitDir(user, repo, testDir, tmpDir.name, token)
        path = tmpDir.name
        hopout.sep()

    # Final check, are there tutorials at the given path
    if not path:
        return None

    # Loop over the given path
    if os.path.isdir(path):
        tutorials = sorted(os.listdir(path))

        # Exlude all tutorials with index 5 or higher
        # > These are only used for internal testing
        tutorials = tuple(s for s in tutorials if s[0] in '1234')

        tsuccess  = np.ones(len(tutorials), dtype=bool)

        # Enter each tutorial and look for the parameter.ini
        bar = ProgressBar(value        = len(tutorials),                       # noqa: E251
                          title        = '│                   Running tests',  # noqa: E251
                          length       = 33,                                   # noqa: E251
                          threshold    = 1,                                    # noqa: E251
                          enrich_print = False)                                # noqa: E251
        for tNum, tutorial in enumerate(tutorials):
            tutorialPath = os.path.join(path, tutorial)

            # Assemble the parameter path
            parameter = os.path.join(tutorialPath, 'parameter.ini')

            # Skip the test if the parameter file is missing
            if not os.path.isfile(parameter):
                bar.step()
                continue

            # Suppress output to standard output
            try:
                # All code that should have silent stdout here
                with open(os.devnull, 'w') as null, redirect_stdout(null), ReadConfig(parameter) as rc:
                    params = rc
            except Exception:
                # Config read failed
                bar.step()
                continue

            # Validate expected config fields
            try:
                projectName = params['general']['projectname']
            except Exception:
                bar.step()
                continue

            # Ensure the mesh file is valid as Git LFS might leave dangling pointers
            meshTmp = None
            meshOK  = True
            if 'filename' in (s.lower() for s in params['general']):
                # Check if any mesh file is actually an LFS pointer
                meshNames = tuple(s for s in params['general']['filename'].splitlines() if s)
                needsMesh = any(isLFSFile(os.path.join(tutorialPath, s)) for s in meshNames)

                # Only perform mesh replacements if we actually need to
                if needsMesh:
                    meshTmp = tempfile.TemporaryDirectory()
                    replIni = pathlib.Path(parameter).read_text()

                    for meshFile in meshNames:
                        # Skip mesh files that are already valid
                        if not isLFSFile(os.path.join(tutorialPath, meshFile)):
                            continue

                        # Try to re-download the missing LFS meshes
                        try:
                            bar.title('│             Re-downloading mesh')
                            downloadGitFile(user, repo, meshTmp.name, {'name'        : meshFile,
                                                                       'type'        : 'file',
                                                                       'download_url': f'https://media.githubusercontent.com/media/{user}/{repo}/{branch}/tutorials/{tutorial}/{meshFile}'
                                                                      })

                            # Replace only the next FileName occurrence in the parameter file
                            old = re.escape(meshFile)
                            new = f"FileName = {os.path.join(meshTmp.name, meshFile)}"
                            replIni, _ = re.subn(rf'^[ \t]*FileName[ \t]*=[ \t]*{old}[ \t]*.*$', new, replIni, count=1, flags=re.IGNORECASE | re.MULTILINE)  # noqa: E501
                            bar.title('│                   Running tests')

                            if lfsWarn is lfsFlag.lfsOkay:
                                lfsWarn = lfsFlag.lfsWarn

                        except urllib.error.HTTPError:  # noqa: PERF203
                            lfsWarn = lfsWarn.lfsErr
                            meshOK  = False
                            hopout.info(f'{hopout.Symbols.WARN } Failed obtaining mesh for test "{tutorial}"')
                            break

                    if meshOK:
                        parameter = os.path.join(meshTmp.name, 'parameter.ini')  # ty:ignore[no-matching-overload]
                        pathlib.Path(parameter).write_text(replIni)

                    if not meshOK:
                        # Cleanup temporary directory
                        if  meshTmp is not None             \
                        and os.path.isdir(meshTmp.name)     \
                        and meshTmp.name.startswith('/tmp'):  # noqa: E271
                            meshTmp.cleanup()

                        bar.step()
                        continue

            # Run pyhope to generate mesh; trap errors and timeouts
            try:
                subprocess.run([sys.executable, '-m', 'pyhope', parameter, '--skip-checks'],
                                cwd     = tutorialPath,        # noqa: E251
                                check   = True,                # noqa: E251
                                stdout  = subprocess.DEVNULL,  # noqa: E251
                                stderr  = subprocess.DEVNULL,  # noqa: E251
                                timeout = 300)                 # noqa: E251 # timeout in seconds
                # Alternative implementation with Popen
                # p = subprocess.Popen([sys.executable, '-m', 'pyhope', 'parameter.ini', '--skip-checks'],
                #                 cwd     = tutorialPath,        # noqa: E251
                #                 stdout  = subprocess.PIPE,     # noqa: E251
                #                 stderr  = subprocess.DEVNULL)  # noqa: E251
                # try:
                #     # if this returns, the process completed
                #     p.wait(timeout=300)
                # except subprocess.TimeoutExpired:
                #     p.terminate()
            except subprocess.CalledProcessError as exc:
                tsuccess[tNum] = False
                hopout.info(f'{hopout.Symbols.ERR } PyHOPE failed for "{tutorial}": ' +
                               f'Return code = {getattr(exc, "returncode", "")}')
                bar.step()
                continue
            except subprocess.TimeoutExpired:
                tsuccess[tNum] = False
                hopout.info(f'{hopout.Symbols.ERR } PyHOPE timed out for "{tutorial}"')
                bar.step()
                continue
            except Exception as exc:
                tsuccess[tNum] = False
                hopout.info(f'{hopout.Symbols.ERR } Unexpected error running PyHOPE for "{tutorial}": {exc}')
                bar.step()
                continue

            # Clean the temporary mesh directory
            if  meshTmp is not None             \
            and os.path.isdir(meshTmp.name)     \
            and meshTmp.name.startswith('/tmp'):  # noqa: E271
                meshTmp.cleanup()

            # Check if PyHOPE generated an output file
            if not os.path.isfile(os.path.join(tutorialPath, f'{projectName}_mesh.h5')):
                # raise ExecError(f'PyHOPE failed to generate mesh for {tutorial}')
                tsuccess[tNum] = False
                hopout.info(f'{hopout.Symbols.ERR } PyHOPE did not produce {projectName}_mesh.h5 for "{tutorial}"')
                bar.step()
                continue

            # Open the TOML file
            tomlData  = None
            toml_path = os.path.join(tutorialPath, 'analyze.toml')
            if os.path.isfile(toml_path):
                try:
                    with open(toml_path, mode='rb') as f:
                        tomlData = tomllib.load(f)
                except Exception:
                    # If TOML is present but invalid,
                    # Python 3.11+: Skip the tutorial
                    # Python 3.10-: Mark the tutorial as successful
                    bar.step()
                    if Version(platform.python_version()) < Version('3.11'):
                        hopout.info(f'{hopout.Symbols.OK  } Successfully completed test (w/o verification) "{tutorial}"')
                        tsuccess[tNum] = True
                    continue

            # Open the mesh file and compare against the reference
            with h5py.File(os.path.join(tutorialPath, f'{projectName}_mesh.h5'), mode='r') as h5:
                for key, val in h5.items():
                    # Calculate the padding length
                    padLen = max(0, testLen - len(key))

                    # Only relevant datasets
                    if key not in testArr:
                        continue

                    # Load the stats from the HDF5 file
                    try:
                        h5stats = hdf5Stats(val)
                    except Exception as exc:
                        tsuccess[tNum] = False
                        hopout.routine(f'{hopout.Symbols.ERR } Failed computing stats for {key} in "{tutorial}": {exc}')
                        # Continue to next dataset
                        continue

                    # Load the stats from the TOML file
                    if  h5stats  is not None \
                    and tomlData is not None \
                    and key in tomlData:  # noqa: E271, E272
                        # Fallback tolerances:
                        if 'GlobalNodeIDs' in key:
                            # GlobalNodeIDs are susceptible to rounding issues
                            rtol = tomlData.get('_defaults', {}).get('rtol', 1E-6)
                            atol = tomlData.get('_defaults', {}).get('atol', 1E-1)
                        else:
                            rtol = tomlData.get('_defaults', {}).get('rtol', 1E-6)
                            atol = tomlData.get('_defaults', {}).get('atol', 1E-8)
                        # If dataset-specific tolerances exist, use them
                        ds_defaults = tomlData.get(key, {})
                        rtol = ds_defaults.get('_rtol', rtol)
                        atol = ds_defaults.get('_atol', atol)

                        for skey, sval in tomlData[key].items():
                            if skey.startswith('_'):
                                # skip metadata keys starting with underscore
                                continue

                            # Ensure the stat exists in h5stats
                            if skey not in h5stats:
                                tsuccess[tNum] = False
                                hopout.routine(f'{hopout.Symbols.ERR } Missing stat "{skey}" for {key} in "{tutorial}"')
                                continue

                            hval = h5stats[skey]
                            # Finally, compare the values
                            try:
                                # If both are numeric scalars
                                if np.isscalar(hval) and np.isscalar(sval):
                                    # GlobalNodeIDs are susceptible to rounding issues
                                    if not np.isclose(hval, sval, rtol=rtol, atol=atol, equal_nan=True) \
                                       and key != 'GlobalNodeIDs' and skey not in ('mean', 'stddev'):
                                        tsuccess[tNum] = False
                                        hopout.routine(f'{hopout.Symbols.ERR } Stat mismatch for "{tutorial}" {key}.{skey}: ' +
                                                    f'h5={hval} toml={sval}')
                                else:
                                    # Try comparing as arrays
                                    h_arr = np.asarray(hval)
                                    s_arr = np.asarray(sval)

                                    # Shapes should match
                                    if h_arr.shape != s_arr.shape:
                                        tsuccess[tNum] = False
                                        hopout.routine(f'{hopout.Symbols.ERR } Shape mismatch for "{tutorial}" {key}.{skey}: ' +
                                                    f'h5={h_arr.shape} toml={s_arr.shape}')
                                        continue

                                    # Values should match
                                    if not np.allclose(h_arr, s_arr, rtol=rtol, atol=atol, equal_nan=True):
                                        tsuccess[tNum] = False
                                        hopout.routine(f'{hopout.Symbols.ERR } Array stat mismatch for "{tutorial}" {key}.{skey}')

                            except Exception as exc:
                                tsuccess[tNum] = False
                                hopout.routine(f'{hopout.Symbols.ERR } Error comparing stat "{skey}" for "{tutorial}": {exc}')
                                continue

                    else:
                        # No TOML entry for this dataset
                        tsuccess[tNum] = False
                        hopout.routine(f'{hopout.Symbols.ERR } No TOML stats for dataset "{key}"{" " * padLen} in "{tutorial}"')

            bar.step()

            if tsuccess[tNum]:
                hopout.info(f'{hopout.Symbols.OK  } Successfully completed test "{tutorial}"')
            else:
                hopout.info(f'{hopout.Symbols.ERR } Failed test "{tutorial}"')

        # Cleanup temporary directory
        if  tmpDir is not None             \
        and os.path.isdir(tmpDir.name)     \
        and tmpDir.name.startswith('/tmp'):  # noqa: E271
            tmpDir.cleanup()

        # Close the progress bar
        if bar is not None:
            hopout.sep()
            bar.title('│          Finished running tests')
            bar.close()

        match lfsWarn:
            case lfsFlag.lfsOkay:
                # hopout.info('')
                pass
            case lfsFlag.lfsWarn | lfsFlag.lfsErr:
                hopout.sep()
                print(hopout.warn('Encountered missing mesh files. Please ensure Git LFS [https://git-lfs.com] is set-up correctly'))  # noqa: E501

        # Print the final output
        hopout.small_banner('Verification summary')
        if all(tsuccess):
            hopout.info(f'{hopout.Symbols.OK  } Successfully completed {np.sum(tsuccess)}/{len(tutorials)} tests')
        else:
            hopout.info(f'{hopout.Symbols.ERR } Failed {np.sum(~tsuccess)}/{len(tutorials)} tests')
        hopout.separator(length=79)

        # Final exit code
        if not all(tsuccess):
            sys.exit(1)
