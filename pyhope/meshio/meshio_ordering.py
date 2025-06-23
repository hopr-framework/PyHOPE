#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (C) 2022 Nico Schlömer
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
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, cast
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


@dataclass
class NodeOrdering:
    """
    A dataclass that stores the converstion between the node ordering of meshIO and Gmsh.
    """
    # Dictionary for translation of  meshio types to gmsh codes
    # http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format-version-2
    _gmsh_typing: Dict[int, str] = field(
            default_factory=lambda: { 1  : 'line'          , 2  : 'triangle'      , 3  : 'quad'          , 4  : 'tetra'         ,
                                      5  : 'hexahedron'    , 6  : 'wedge'         , 7  : 'pyramid'       , 8  : 'line3'         ,
                                      9  : 'triangle6'     , 10 : 'quad9'         , 11 : 'tetra10'       , 12 : 'hexahedron27'  ,
                                      13 : 'wedge18'       , 14 : 'pyramid14'     , 15 : 'vertex'        , 16 : 'quad8'         ,
                                      17 : 'hexahedron20'  , 18 : 'wedge15'       , 19 : 'pyramid13'     , 21 : 'triangle10'    ,
                                      23 : 'triangle15'    , 25 : 'triangle21'    , 26 : 'line4'         , 27 : 'line5'         ,
                                      28 : 'line6'         , 29 : 'tetra20'       , 30 : 'tetra35'       , 31 : 'tetra56'       ,
                                      36 : 'quad16'        , 37 : 'quad25'        , 38 : 'quad36'        , 42 : 'triangle28'    ,
                                      43 : 'triangle36'    , 44 : 'triangle45'    , 45 : 'triangle55'    , 46 : 'triangle66'    ,
                                      47 : 'quad49'        , 48 : 'quad64'        , 49 : 'quad81'        , 50 : 'quad100'       ,
                                      51 : 'quad121'       , 62 : 'line7'         , 63: 'line8'          , 64 : 'line9'         ,
                                      65 : 'line10'        , 66 : 'line11'        , 71 : 'tetra84'       , 72 : 'tetra120'      ,
                                      73 : 'tetra165'      , 74 : 'tetra220'      , 75 : 'tetra286'      , 90 : 'wedge40'       ,
                                      91 : 'wedge75'       , 92 : 'hexahedron64'  , 93 : 'hexahedron125' , 94 : 'hexahedron216' ,
                                      95 : 'hexahedron343' , 96 : 'hexahedron512' , 97 : 'hexahedron729' , 98 : 'hexahedron1000',
                                      106: 'wedge126'      , 107: 'wedge196'      , 108: 'wedge288'      , 109: 'wedge405'      ,
                                      110: 'wedge550'
                                    }
    )

    # Dictionary for conversion Gmsh to meshIO
    # > TODO: IMPLEMENT RECURSIVE MAPPING USING IO_MESHIO/IO_GMSH
    _meshio_ordering: Dict[str, List[int]] = field(
            default_factory=lambda: {  # 0D elements
                                       # > Vertex
                                       # 'vertex'      : [ 0 ],
                                       # 1D elements
                                       # > Line
                                       # 'line'        : [ 0, 1 ],
                                       # 2D elements
                                       # > Triangle
                                       # 'triangle'    : [ 0, 1, 2 ],
                                       # > Quadrilateral
                                       # 'quad'        : [ 0, 1, 2, 3 ],
                                       # 3D elements
                                       # > Tetrahedron
                                       'tetra'       : [ 0, 1, 2, 3 ],
                                       'tetra10'     : [ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 ],
                                       # > Wedge
                                       'wedge'       : [ 0, 1, 2, 3, 4, 5],
                                       'wedge15'     : [ 0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11 ],
                                       'wedge18'     : [ 0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 16, 17 ],
                                       # > Pyramid
                                       'pyramid'     : [ 0, 1, 2, 3, 4],
                                       'pyramid13'   : [ 0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12 ],
                                       # > Hexahedron: for all hexahedron, we now use analytics
                                    }
    )

    # # Dictionary for conversion meshIO to Gmsh
    # # > TODO: IMPLEMENT RECURSIVE MAPPING USING IO_MESHIO/IO_GMSH
    # _gmsh_ordering: Dict[str, List[int]] = field(
    #         default_factory=lambda: { 'tetra10'     : [ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 ],
    #                                   'hexahedron20': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14 ],
    #                                   'hexahedron27': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14, 24,
    #                                                     22, 20, 21, 23, 25, 26 ],
    #                                   'wedge15'     : [ 0, 1, 2, 3, 4, 5, 6, 8, 12, 7, 13, 14, 9, 11, 10 ],
    #                                   'pyramid13'   : [ 0, 1, 2, 3, 4, 5, 8, 9, 6, 10, 7, 11, 12 ],
    #                                 }
    # )

    # Dictionary for translation of  gambit types to gmsh codes
    _gambit_typing: Dict[int, str] = field(
            default_factory=lambda: { 1  : 'line'          , 2  : 'quad'          , 3  : 'triangle'      , 4  : 'hexahedron'    ,
                                      5  : 'wedge'         , 6  : 'tetrahedron'   , 7  : 'pyramid'                              ,
                                    }
    )

    # Dictionary for conversion of Gambit to meshIO
    _gambit_ordering: Dict[str, List[int]] = field(
            default_factory=lambda: {  # 0D elements
                                       # 1D elements
                                       # 2D elements
                                       # 3D elements
                                       # > Hexahedron
                                       'hexahedron':   [0, 1, 3, 2, 4, 5, 7, 6],
                                    }
    )

    def ordering_gmsh_to_meshio(self, elemType: Union[int, str, np.uint], idx: np.ndarray) -> np.ndarray:
        """
        Return the meshIO node ordering for a given element type.
        """

        if isinstance(elemType, (int, np.integer)):
            elemType = self._gmsh_typing[int(elemType)]

        # 0D/1D/2D elements
        if elemType.startswith(('vertex', 'line', 'triangle', 'quad')):
            return cast(np.ndarray, idx)

        # Check if we have a fixed ordering
        if elemType in self._meshio_ordering:
            return cast(np.ndarray, idx[:, self._meshio_ordering[elemType]])

        # Check if we are requesting higher-order simplices than currently implemented
        if not elemType.startswith('hexahedron'):
            raise ValueError(f'Unknown element type {elemType}')

        # For hexahedrons with analytic ordering
        nNodes = 8 if elemType.partition('hexahedron')[2] == '' else int(elemType.partition('hexahedron')[2])

        if self.deviation(nNodes ** (1/3) - 1) < self.deviation((nNodes-8)/12 + 1):
            nGeo = round(nNodes ** (1/3) - 1)
            incomplete = False
        else:
            nGeo = round((nNodes-8)/12 + 1)
            incomplete = True

        ordering = self.HEXREORDER(nGeo, incomplete=incomplete)
        return idx[:, ordering]

    def deviation(self, x: float) -> float:
        return abs(x - round(x))

    def HEXREORDER(self, order: int, incomplete: Optional[bool] = False) -> tuple[int]:
        """ Converts node ordering from gmsh to meshio format
        """
        EDGEMAP   = (  0,  3,  5,  1,  8, 10, 11,  9,  2,  4,  6,  7)
        FACEMAP   = (  2,  3,  1,  4,  0,  5)

        order    += 1
        nNodes    = 8 + 12*(order - 2) if incomplete else order**3
        map: List = [None for _ in range(nNodes)]

        count = 0
        # Recursively build the mapping
        for iOrder in range(np.floor(order/2).astype(int)):
            # Vertices
            map[count:count+8] = list(range(count, count+8))
            count += 8

            pNodes = (order-2*(iOrder+1))

            # Edges
            for iEdge in range(12):
                iSlice = slice(count + pNodes   *iEdge                , count + pNodes    *(iEdge+1))
                map[iSlice] = [count + pNodes   *(EDGEMAP[iEdge])+iNode for iNode in range(pNodes   )]
            count += pNodes*12

            # Only vertices and edges of the outermost shell required for incomplete elements
            if incomplete:
                return tuple(map)

            # Faces
            for iFace in range(6):
                iSlice = slice(count + pNodes**2*iFace                , count + pNodes**2*(iFace+1))
                map[iSlice] = [count + pNodes**2*(FACEMAP[iFace])+iNode for iNode in range(pNodes**2)]
            count += pNodes**2*6

        if order % 2 != 0:
            map[count] = count

        return tuple(map)

    # INFO: Alternative implementation
    # def _compute_hexahedron_meshio_order(self, p: int, recursive: Optional[bool] = False) -> List[int]:
    #     # 1) Corner nodes
    #     mapping = list(range(8))
    #     if p == 1:
    #         return mapping
    #
    #     # Permutation for the 12 edge blocks in meshio ordering.
    #     GmshToMeshioEdgePerm = [0, 3, 5, 1, 8, 10, 11, 9, 2, 4, 6, 7]
    #     # Permutation for the 6 face blocks.
    #     GmshToMeshioFacePerm = [2, 3, 1, 4, 0, 5]
    #
    #     # 2) Edge nodes
    #     nNodeEdgeEdge   = p - 1
    #     nNodeEdgeTotal  = 12 * nNodeEdgeEdge
    #     gmshEdgeNodes   = list(range(8, 8 + nNodeEdgeTotal))
    #     # Partition edge nodes into 12 blocks.
    #     blockEdge       = [gmshEdgeNodes[i * nNodeEdgeEdge : (i + 1) * nNodeEdgeEdge] for i in range(12)]
    #     # Permute edge blocks to align with meshio order
    #     blockEdgeOrient = [blockEdge[i] for i in GmshToMeshioEdgePerm]
    #     blockEdgeOrient = [node for block in blockEdgeOrient for node in block]
    #     mapping.extend(blockEdgeOrient)
    #
    #     # 3) Face nodes
    #     nNodeFaceFace   = (p - 1) ** 2
    #     nNodeFaceTotal  = 6 * nNodeFaceFace
    #     startFaceNode   = 8 + nNodeEdgeTotal
    #     gmshFaceNodes   = list(range(startFaceNode, startFaceNode + nNodeFaceTotal))
    #     # Partition face nodes into 6 blocks.
    #     blockFace       = [gmshFaceNodes[i * nNodeFaceFace : (i + 1) * nNodeFaceFace] for i in range(6)]
    #     # Permuted face blocks to align with meshio order
    #     blockFaceOrient = [blockFace[i] for i in GmshToMeshioFacePerm]
    #     blockFaceOrient = [node for block in blockFaceOrient for node in block]
    #     mapping.extend(blockFaceOrient)
    #
    #     # 4) Interior nodes.
    #     # -- Interior nodes (recursive approach for p >= 4) --
    #     nNodeInterior = (p - 1) ** 3
    #     start_interior = startFaceNode + nNodeFaceTotal
    #
    #     if p <= 3:
    #         # For p <= 3, just append in natural order
    #         interior_nodes = list(range(start_interior, start_interior + nNodeInterior))
    #         mapping.extend(interior_nodes)
    #     # If we are the outermost call, we need to handle the recursive case
    #     elif not recursive:
    #         # General chunk-based approach for p >= 4
    #         remainder      = p - 2
    #         subcubeIndices = []
    #         currentOffset  = start_interior
    #
    #         # Repeatedly carve out sub-cubes (of order=3) until remainder used up
    #         while remainder >= 2:
    #             currenMap = self._compute_hexahedron_meshio_order(remainder, recursive=True)
    #             offsetMap = [currentOffset + node for node in currenMap]
    #             subcubeIndices.extend(offsetMap)
    #             currentOffset += len(currenMap)
    #             remainder     -= 2
    #
    #         mapping.extend(subcubeIndices)
    #
    #     return mapping

    def typing_gambit_to_meshio(self, elemType: Union[int, str, np.uint]) -> str:
        """
        Return the meshIO element type for a given Gambit element type
        """
        if isinstance(elemType, (int, np.integer)):
            return self._gambit_typing[int(elemType)]

        raise ValueError(f'Unknown element type {elemType}')

    def ordering_gambit_to_meshio(self, elemType: Union[int, str, np.uint], idx: np.ndarray) -> np.ndarray:
        """
        Return the meshIO node ordering for a given element type
        """
        if isinstance(elemType, (int, np.integer)):
            elemType = self._gambit_typing[int(elemType)]

        # 0D/1D/2D elements
        if elemType.startswith(('vertex', 'line', 'triangle', 'quad')):
            return cast(np.ndarray, idx)

        # Check if we have a fixed ordering
        if elemType in self._gambit_ordering:
            return cast(np.ndarray, idx[self._gambit_ordering[elemType]])

        raise ValueError(f'Unknown element type {elemType}')
