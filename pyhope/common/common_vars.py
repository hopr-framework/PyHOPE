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
import importlib.metadata
import pathlib
import re
from typing import Self
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from packaging.version import Version
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# PEP 318 – Decorators for Functions and Methods
# > https://peps.python.org/pep-0318/
def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance


@singleton
class Common():
    def __init__(self: Self) -> None:
        self._version = self.__version__
        self._program = self.__program__

    @property
    def __version__(self):
        # Retrieve version from package metadata
        try:
            package = pathlib.Path(__file__).parent.name
            version = importlib.metadata.version(package)
        # Fallback to pyproject.toml
        except importlib.metadata.PackageNotFoundError:
            pyproject = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
            if not pyproject.exists():
                raise FileNotFoundError(f'pyproject.toml not found at {pyproject}')

            with pyproject.open('r') as p:
                match = re.search(r'version\s*=\s*["\'](.+?)["\']', p.read())
            if not match:
                raise ValueError('Version not found in pyproject.toml')
            version = match.group(1)

        return Version(version)

    @property
    def __program__(self):
        return 'PyHOPE'

    @property
    def program(self):
        return str(self._program)

    @property
    def version(self):
        return str(self._version)


class Gitlab():
    # Gitlab "python-gmsh" access
    LIB_GITLAB:  str = 'gitlab.iag.uni-stuttgart.de'
    # LIB_PROJECT  = 'libs/python-gmsh'
    LIB_PROJECT: str = '797'
    LIB_VERSION: str = '4.13.1'
    LIB_HASH:    str = '82f655d3a7d97da5b4de06fd0af2198a95b8098a10b3f89edc99764f635c5d4d'


np_mtp : int  # Number of threads for multiprocessing
