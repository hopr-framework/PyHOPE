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
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Dict, Final, Optional, Union, Tuple, final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================
mode     : int                                    # Mesh generation mode (1 - Internal, 2 - External (MeshIO))
mesh     : meshio.Mesh                            # MeshIO object holding the mesh
nGeo     : int                                    # Order of spline-reconstruction for curved surfaces
sortIJK  : bool                                   # Flag if mesh should be I,J,K sorted

bcs      : list[type | None]                      # [list of dict] - Boundary conditions
vvs      : list                                   # [list of dict] - Periodic vectors

nZones   : int       = 1                          # Number of zones
elemTypes: list[int] = []                         # Element types per zone
elems    : list[type | None]                      # [list of list] - Element nodes
sides    : list[type | None]                      # [list of list] - Side    nodes

# Periodic nodes
periNodes: dict                                   # Mapping from the periodic nodes to the master nodes

# Mesh curving
already_curved: bool                              # Flag if mesh is already curved

# Mesh connectitivity
doMortars: bool                                   # Flag if mortars are enabled
doPeriodicCorrect: bool                           # Flag if displacement between periodic elements should be corrected

# Internal variables
tolInternal: Final[float] = 1.E-10                # Tolerance for mesh connect (internal sides)
tolExternal: Final[float] = 1.E-8                 # Tolerance for mesh connect (external sides)
tolPeriodic: Final[float] = 5.E-2                 # Tolerance for mesh connect (periodic sides)


@dataclass(init=False, repr=False, eq=False, slots=False, frozen=True)
class MeshMode:
    MODE_INT:    int = 1
    MODE_EXT:    int = 3


@dataclass(init=False, repr=False, eq=False, slots=False)
class CGNS:
    regenerate_BCs: bool = False                  # Flag if CGNS needs BC regeneration


@dataclass(init=True, repr=False, eq=False, slots=True)
class SIDE:
    # Explicitly declare data members
    # __slots__ = ('elemID', 'sideID', 'locSide', 'face', 'corners', 'sideType',
    #              # Sorting
    #              'globalSideID',
    #              # Connection
    #              'MS', 'connection', 'flip', 'nbLocSide',
    #              # Boundary Conditions
    #              'bcid',
    #              # Mortar
    #              'locMortar')

    # def __init__(self,
    #              elemID      : Optional[int] = None,
    #              sideID      : Optional[int] = None,
    #              locSide     : Optional[int] = None,
    #              face        : Optional[str] = None,
    #              corners     : Optional[np.ndarray] = None,
    #              sideType    : Optional[int] = None,
    #              # Sorting
    #              globalSideID: Optional[int] = None,
    #              # Connection
    #              MS          : Optional[int] = None,
    #              connection  : Optional[int] = None,
    #              flip        : Optional[int] = None,
    #              nbLocSide   : Optional[int] = None,
    #              # Boundary Conditions
    #              bcid        : Optional[int] = None,
    #              # Mortar
    #              locMortar   : Optional[int] = None,
    #             ):
    #     self.elemID      : Optional[int] = elemID
    #     self.sideID      : Optional[int] = sideID
    #     self.locSide     : Optional[int] = locSide
    #     self.face        : Optional[str] = face
    #     self.corners     : Optional[np.ndarray] = corners
    #     self.sideType    : Optional[int] = sideType
    #     # Sorting
    #     self.globalSideID: Optional[int] = globalSideID
    #     # Connection
    #     self.MS          : Optional[int] = MS
    #     self.connection  : Optional[int] = connection
    #     self.flip        : Optional[int] = flip
    #     self.nbLocSide   : Optional[int] = nbLocSide
    #     # Boundary Conditions
    #     self.bcid        : Optional[int] = bcid
    #     # Mortar
    #     self.locMortar   : Optional[int] = locMortar
    elemID      : Optional[int] = None
    sideID      : Optional[int] = None
    locSide     : Optional[int] = None
    face        : Optional[str] = None
    corners     : Optional[np.ndarray] = None
    sideType    : Optional[int] = None
    # Sorting
    globalSideID: Optional[int] = None
    # Connection
    MS          : Optional[int] = None
    connection  : Optional[int] = None
    flip        : Optional[int] = None
    nbLocSide   : Optional[int] = None
    # Boundary Conditions
    bcid        : Optional[int] = None
    # Mortar
    locMortar   : Optional[int] = None

    # def update(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)

    # def dict(self):
    #     """Return a dictionary of the SIDE object
    #     """
    #     return {key: value for key, value in self.__dict__.items() if value is not None}

    # Comparison operator for bisect
    def __lt__(self, other) -> bool:
        return self.sideID < other.sideID


@dataclass(init=True, repr=False, eq=False, slots=True)
class ELEM:
    # Explicitly declare data members
    # __slots__ = ('type', 'elemID', 'sides', 'nodes')

    # def __init__(self,
    #              type        : Optional[int]  = None,
    #              elemID      : Optional[int]  = None,
    #              sides       : Optional[list] = None,
    #              nodes       : Optional[list] = None,
    #             ):
    #     self.type        : Optional[int]  = type
    #     self.elemID      : Optional[int]  = elemID
    #     self.sides       : Optional[list] = sides
    #     self.nodes       : Optional[list] = nodes
    type        : Optional[int]  = None
    elemID      : Optional[int]  = None
    sides       : Optional[Union[list, np.ndarray]] = None
    nodes       : Optional[            np.ndarray]  = None
    # FEM connectivity
    edgeInfo    : Optional[Dict[int,                    # locEdgeIdx
                                Tuple[int,              # locEdge
                                      int | None,       # globalEdge
                                      Tuple[int, ...],  # FEMVertexID
                                      Tuple[int, ...]   # NodeID
                                     ]
                                ]] = None
    vertexInfo  : Optional[Dict[int,                    # locNodeIdx
                                Tuple[int,              # FEMVertexID
                                      Tuple[int, ...]   # Vertex connectivity
                                     ]
                                ]] = None

    # def update(self, **kwargs):
    #     for key, value in kwargs.items():
    #         setattr(self, key, value)

    # def dict(self):
    #     """Return a dictionary of the ELEM object
    #     """
    #     return {key: value for key, value in self.__dict__.items() if value is not None}

    # Comparison operator for bisect
    def __lt__(self, other) -> bool:
        return self.elemID < other.elemID


@dataclass(init=True, repr=False, eq=False, slots=True)
class BC:
    # Explicitly declare data members
    # __slots__ = ('name', 'bcid', 'type', 'dir')

    # def __init__(self,
    #              name        : Optional[str]  = None,
    #              bcid        : Optional[int]  = None,
    #              type        : Optional[list] = None,
    #              dir         : Optional[list] = None,
    #              ):
    #     self.name        : Optional[str]  = name
    #     self.bcid        : Optional[int]  = bcid
    #     self.type        : Optional[list] = type
    #     self.dir         : Optional[list] = dir
    name        : Optional[str]        = None
    bcid        : Optional[int]        = None
    type        : Optional[np.ndarray] = None
    dir         : Optional[list]       = None

    # def update(self, **kwargs):
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             raise AttributeError(f'"BC" object has no attribute "{key}"')
    #
    # def dict(self):
    #     """Return a dictionary of the BC object
    #     """
    #     return {key: value for key, value in self.__dict__.items() if value is not None}


@final
class ELEMTYPE:
    type = {'tetra'     : 4,
            'pyramid'   : 5,
            'wedge'     : 5,
            'hexahedron': 6}
    name = {'tetra'     : 104, 'tetra10'      : 204, 'tetra20'       : 204, 'tetra35'       : 204, 'tetra56'       : 204,
                               'tetra84'      : 204, 'tetra120'      : 204, 'tetra165'      : 204, 'tetra220'      : 204,
                               'tetra286'     : 204,
            'pyramid'   : 105, 'pyramid13'    : 205, 'pyramid14'     : 205, 'pyramid30'     : 205, 'pyramid55'     : 205,
            'wedge'     : 106, 'wedge15'      : 206, 'wedge18'       : 206, 'wedge40'       : 206, 'wedge75'       : 206,
                               'wedge126'     : 206, 'wedge196'      : 206, 'wedge288'      : 206, 'wedge405'      : 206,
                               'wedge550'     : 206,
            'hexahedron': 108, 'hexahedron20' : 208, 'hexahedron24'  : 208, 'hexahedron27'  : 208, 'hexahedron64'  : 208,
                               'hexahedron125': 208, 'hexahedron216' : 208, 'hexahedron343' : 208, 'hexahedron512' : 208,
                               'hexahedron729': 208, 'hexahedron1000': 208, 'hexahedron1331': 208}
    inam = defaultdict(list)
    for key, value in name.items():
        inam[value].append(key)


# @final
# @cache
# class SIDETYPE:
#     type = {'tri'      : 3,
#             'quad'     : 4}


@cache
def ELEMMAP(meshioType: str) -> int:
    elemMap = {  # Linear or curved tetrahedron
                 'tetra'     : (104, 204),
                 # Linear or curved pyramid
                 'pyramid'   : (105, 205),
                 # Linear or curved wedge / prism
                 'wedge'     : (106, 206),
                 # Linear or curved hexahedron
                 'hexahedron': (108, 208)}

    for key, (linear, curved) in elemMap.items():
        if key in meshioType:
            return linear if meshioType == key else curved
    raise ValueError(f'Unknown element type {meshioType}')
