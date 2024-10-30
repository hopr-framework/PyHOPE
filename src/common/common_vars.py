#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2022-2024 Andrea Beck
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


class Common():
    __version__ = Version('0.9.0')
    __program__ = 'PyHOPE'

    program: str = str(__program__)
    version: str = str(__version__)


class Gitlab():
    # Gitlab "python-gmsh" access
    LIB_GITLAB:  str = 'gitlab.iag.uni-stuttgart.de'
    # LIB_PROJECT  = 'libs/python-gmsh'
    LIB_PROJECT: str = '797'
    LIB_VERSION: str = '4.13.1'
    LIB_HASH:    str = '82f655d3a7d97da5b4de06fd0af2198a95b8098a10b3f89edc99764f635c5d4d'


np_mtp : int                                     # Number of threads for multiprocessing
