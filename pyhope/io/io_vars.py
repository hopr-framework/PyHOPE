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
from dataclasses import dataclass
from functools import cache
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
projectname  : str                               # Name of output files
outputformat : int                               # Mesh output format

debugvisu    : bool                              # Enable and show debug output / visualization


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class MeshFormat:
    FORMAT_HDF5: int = 0
    FORMAT_VTK:  int = 1
    FORMAT_GMSH: int = 2


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class ELEM:
    INFOSIZE:  int = 6
    TYPE:      int = 0
    ZONE:      int = 1
    FIRSTSIDE: int = 2
    LASTSIDE:  int = 3
    FIRSTNODE: int = 4
    LASTNODE:  int = 5

    TYPES: tuple[int, ...] = (104, 204, 105, 115, 205, 106, 116, 206, 108, 118, 208)


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class SIDE:
    INFOSIZE: int = 5
    TYPE:     int = 0
    ID:       int = 1
    NBELEMID: int = 2
    NBLOCSIDE_FLIP: int = 3
    BCID:     int = 4


@cache
def ELEMTYPE(elemType: int) -> str:
    """ Name of a given element type
    """
    match elemType:
        case 104:
            return ' Straight-edge Tetrahedra '
        case 204:
            return '        Curved Tetrahedra '
        case 105:
            return '  Planar-faced Pyramids   '
        case 115:
            return ' Straight-edge Pyramids   '
        case 205:
            return '        Curved Pyramids   '
        case 106:
            return '  Planar-faced Prisms     '
        case 116:
            return ' Straight-edge Prisms     '
        case 206:
            return '        Curved Prisms     '
        case 108:
            return '  Planar-faced Hexahedra  '
        case 118:
            return ' Straight-edge Hexahedra  '
        case 208:
            return '        Curved Hexahedra  '
        case _:  # Default
            import pyhope.output.output as hopout
            hopout.error('Error in ELEMTYPE, unknown elemType')
