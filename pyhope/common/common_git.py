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
import pathlib
import subprocess
from typing import Optional
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    from urllib.response import addinfourl
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
from pyhope.common.common_progress import ProgressBar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def findGitRoot() -> Optional[str]:
    """ Attempt to find the git root
    """
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def isLFSFile(path: str) -> bool:
    """ Small helper to check if a file is a Git LFS pointer
    """
    # Read the first 4KB
    with open(path, 'rb') as f:
        chunk = f.read(min(os.path.getsize(path), 4096))
        # Return if the file is a Git LFS File
        return chunk.startswith(b'version https://git-lfs.github.com/spec/v1')


# Helper to manage API requests and rate limiting
def makeAPIRequest(url  : str,
                  token: Optional[str]         = None,
                  # base : Optional[str]         = None,
                  bar  : Optional[ProgressBar] = None) -> addinfourl:
    # Standard libraries -----------------------------------
    import time
    import urllib.request
    from urllib.error import HTTPError
    # ------------------------------------------------------

    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    req = urllib.request.Request(url, headers=headers)
    # print(url)

    while True:
        try:
            return urllib.request.urlopen(req)
        except HTTPError as e:  # noqa: PERF203
            # Check for rate-limiting error
            if  e.code == 403                        \
            and 'X-RateLimit-Remaining' in e.headers \
            and int(e.headers['X-RateLimit-Remaining']) == 0:  # noqa: E271
                timeReset = int(e.headers['X-RateLimit-Reset'])
                timeWait  = max(timeReset - time.time(), 1)
                if bar is not None:
                    bar.title(f'│ Rate limited, waiting {timeWait} sec')
                time.sleep(timeWait)
                if bar is not None:
                    bar.title( '│               Downloading tests')
                # Retry the request
                continue
            # Re-raise other HTTP errors
            raise


def downloadGitDir(user    : str,
                   repo    : str,
                   path    : str,
                   target  : str,
                   token   : Optional[str]  = None,
                   branch  : str            = 'main',
                   progress: Optional[bool] = True) -> None:
    # Standard libraries -----------------------------------
    import json
    # ------------------------------------------------------

    apiURL = f'https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}'

    with makeAPIRequest(apiURL, token) as u:
        contents = json.loads(u.read().decode())

    # Exlude all tutorials with index 5 or higher
    # > These are only used for internal testing
    if progress:
        contents = tuple(s for s in contents if s['name'][0] in '1234')

    # If we are in a subdirectory, create it
    os.makedirs(target, exist_ok=True)

    bar = None
    if progress:
        bar = ProgressBar(value=len(contents), title='│               Downloading tests', length=33, threshold=1)

    for item in contents:
        downloadGitFile(user, repo, target, item, branch, bar)
        if progress:
            bar.step()

    if progress:
        bar.close()


def downloadGitFile(user    : str,
                    repo    : str,
                    target  : str,
                    item    : dict,
                    branch  : str = 'main',
                    bar : Optional[ProgressBar] = None
                    ) -> None:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    name     = item['name']
    subPath  = os.path.join(target, name)
    itemType = item.get('type')

    match itemType:
        case 'file':
            # Initially, download the file content
            # > This might be the actual file or an LFS pointer
            with makeAPIRequest(url=item['download_url'], bar=bar) as u:
                content = u.read()

            # Check if the content is a Git LFS pointer
            if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
                # If it's an LFS pointer, the actual file needs to be downloaded
                # from the media URL, which is constructed from the file's path
                # print(f'Downloading LFS file: {item["path"]}...')
                lfs_url = f'https://media.githubusercontent.com/media/{user}/{repo}/{branch}/{item["path"]}'
                with makeAPIRequest(url=lfs_url, bar=bar) as lfs_u:
                    content = lfs_u.read()

            # Write the final content (either regular file or LFS file) to disk
            pathlib.Path(subPath).write_bytes(content)

        case 'dir':
            # Recursively call the function for subdirectories
            # > Progress is disabled for sub-calls to issues with duplicate progressBar
            downloadGitDir(user, repo, item['path'], subPath, branch=branch, progress=False)

        case _:
            print(hopout.warn(f'Unknown item type "{itemType}" for item "{name}". Skipping.'))
