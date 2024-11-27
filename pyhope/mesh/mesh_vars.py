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
from typing import Optional
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
mode   : int                                      # Mesh generation mode (1 - Internal, 2 - External (MeshIO))
mesh   : meshio._mesh.Mesh                        # MeshIO object holding the mesh
nGeo   : int                                      # Order of spline-reconstruction for curved surfaces
sortIJK: bool                                     # Flag if mesh should be I,J,K sorted

bcs    : list                                     # [list of dict] - Boundary conditions
vvs    : list                                     # [list of dict] - Periodic vectors

elems  : list                                     # [list of list] - Element nodes
sides  : list                                     # [list of list] - Side    nodes

HEXMAP : np.ndarray                               # CGNS <-> IJK ordering for high-order hexahedrons
already_curved: bool                              # Flag if mesh is already curved


class CGNS:
    regenerate_BCs: bool                          # Flag if CGNS needs BC regeneration


class SIDE:
    def __init__(self,
                 elemID      : Optional[int] = None,
                 sideID      : Optional[int] = None,
                 locSide     : Optional[int] = None,
                 face        : Optional[str] = None,
                 corners     : Optional[np.ndarray] = None,
                 sideType    : Optional[int] = None,
                 # Sorting
                 globalSideID: Optional[int] = None,
                 # Connection
                 MS          : Optional[int] = None,
                 connection  : Optional[int] = None,
                 flip        : Optional[int] = None,
                 nbLocSide   : Optional[int] = None,
                 # Boundary Conditions
                 bcid        : Optional[int] = None,
                ):
        self.elemID      : Optional[int] = elemID
        self.sideID      : Optional[int] = sideID
        self.locSide     : Optional[int] = locSide
        self.face        : Optional[str] = face
        self.corners     : Optional[np.ndarray] = corners
        self.sideType    : Optional[int] = sideType
        # Sorting
        self.globalSideID: Optional[int] = globalSideID
        # Connection
        self.MS          : Optional[int] = MS
        self.connection  : Optional[int] = connection
        self.flip        : Optional[int] = flip
        self.nbLocSide   : Optional[int] = nbLocSide
        # Boundary Conditions
        self.bcid        : Optional[int] = bcid

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'"SIDE" object has no attribute "{key}"')

    def dict(self):
        """Return a dictionary of the SIDE object
        """
        return {key: value for key, value in self.__dict__.items() if value is not None}


class ELEM:
    def __init__(self,
                 type        : Optional[int]  = None,
                 elemID      : Optional[int]  = None,
                 sides       : Optional[list] = None,
                 nodes       : Optional[list] = None,
                ):
        self.type        : Optional[int]  = type
        self.elemID      : Optional[int]  = elemID
        self.sides       : Optional[list] = sides
        self.nodes       : Optional[list] = nodes

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'"ELEM" object has no attribute "{key}"')

    def dict(self):
        """Return a dictionary of the ELEM object
        """
        return {key: value for key, value in self.__dict__.items() if value is not None}


class BC:
    def __init__(self,
                 name        : Optional[str]  = None,
                 bcid        : Optional[int]  = None,
                 type        : Optional[list] = None,
                 dir         : Optional[list] = None,
                 ):
        self.name        : Optional[str]  = name
        self.bcid        : Optional[int]  = bcid
        self.type        : Optional[list] = type
        self.dir         : Optional[list] = dir

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'"BC" object has no attribute "{key}"')

    def dict(self):
        """Return a dictionary of the BC object
        """
        return {key: value for key, value in self.__dict__.items() if value is not None}


class ELEMTYPE:
    type = {'tetra'     : 4,
            'pyramid'   : 5,
            'wedge'     : 5,
            'hexahedron': 6}
    name = {'tetra'     : 104, 'tetra10'      : 204, 'tetra20'       : 204, 'tetra35'       : 204, 'tetra56'       : 204,
                               'tetra84'      : 204, 'tetra120'      : 204, 'tetra165'      : 204, 'tetra220'      : 204,
                               'tetra286'     : 204,
            'pyramid'   : 105, 'pyramid13'    : 205, 'pyramid14'     : 205,
            'wedge'     : 106, 'wedge15'      : 206, 'wedge18'       : 206, 'wedge40'       : 206, 'wedge75'       : 206,
                               'wedge126'     : 206, 'wedge196'      : 206, 'wedge288'      : 206, 'wedge405'      : 206,
                               'wedge550'     : 206,
            'hexahedron': 108, 'hexahedron20' : 208, 'hexahedron24'  : 208, 'hexahedron27'  : 208, 'hexahedron64'  : 208,
                               'hexahedron125': 208, 'hexahedron216' : 208, 'hexahedron343' : 208, 'hexahedron512' : 208,
                               'hexahedron729': 208, 'hexahedron1000': 208, 'hexahedron1331': 208}


def ELEMMAP(meshioType: str) -> int:
    # Local imports ----------------------------------------
    import sys
    # ------------------------------------------------------
    # Linear or curved tetrahedron
    if 'tetra' in meshioType:
        if 'tetra' == meshioType:
            return 104
        else:
            return 204
    # Linear or curved pyramid
    elif 'pyramid' in meshioType:
        if 'pyramid' == meshioType:
            return 105
        else:
            return 205
    # Linear or curved wedge / prism
    elif 'wedge' in meshioType:
        if 'wedge' == meshioType:
            return 106
        else:
            return 206
    # Linear or curved hexahedron
    elif 'hexahedron' in meshioType:
        if 'hexahedron' == meshioType:
            return 108
        else:
            return 208
    else:
        print('Unknown element type {}'.format(meshioType))
        sys.exit(1)
