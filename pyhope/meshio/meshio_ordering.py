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
from typing import Dict, List, Union, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from numpy.typing import ArrayLike
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
            default_factory=lambda: { 'tetra10'     : [ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 ],
                                      'hexahedron20': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15 ],
                                      'hexahedron27': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 22,
                                                        23, 21, 24, 20, 25, 26 ],
                                      'hexahedron64': [ 0, 1, 2, 3, 4, 5, 6, 7,  # Vertices

                                                         8,  9,                  # Edge  1, x-, z- from (y-,y+)
                                                        14, 15,                  # Edge  2, y+, z- from (x-,x+)
                                                        18, 19,                  # Edge  3, x+, z+ from (y+,y-)
                                                        10, 11,                  # Edge  4, y-, z+ from (x+,x-)

                                                        24, 25,                  # Edge  5, x-, z+ from (y-,y+)
                                                        28, 29,                  # Edge  6, y+, z+ from (x-,x+)
                                                        30, 31,                  # Edge  7, x+, z- from (y+,y-)
                                                        26, 27,                  # Edge  8, y-, z- from (x+,x-)

                                                        12, 13,                  # Edge  9, x-, y- from (z-,z+)
                                                        16, 17,                  # Edge 10, x+, y+ from (z-,z+)
                                                        20, 21,                  # Edge 11, y-, z- from (x-,x+)
                                                        22, 23,                  # Edge 12, y+, z+ from (x-,x+)

                                                        40, 41, 42, 43,          # Face at x+
                                                        44, 45, 46, 47,          # Face at y+
                                                        36, 37, 38, 39,          # Face at y-
                                                        48, 49, 50, 51,          # Face at x-
                                                        32, 33, 34, 35,          # Face at z-
                                                        52, 53, 54, 55,          # Face at z+

                                                        # Interior nodes
                                                        56, 57, 58, 59, 60, 61, 62, 63
                                                    ],
                                      'hexahedron125': [ 0, 1, 2, 3, 4, 5, 6, 7,  # Vertices

                                                         8,  9, 10,               # Edge  1, x-, z- from (y-,y+)
                                                        17, 18, 19,               # Edge  2, y+, z- from (x-,x+)
                                                        23, 24, 25,               # Edge  3, x+, z+ from (y+,y-)
                                                        11, 12, 13,               # Edge  4, y-, z+ from (x+,x-)

                                                        32, 33, 34,               # Edge  5, x-, z+ from (y-,y+)
                                                        38, 39, 40,               # Edge  6, y+, z+ from (x-,x+)
                                                        41, 42, 43,               # Edge  7, x+, z- from (y+,y-)
                                                        35, 36, 37,               # Edge  8, y-, z- from (x+,x-)

                                                        14, 15, 16,               # Edge  9, x-, y- from (z-,z+)
                                                        20, 21, 22,               # Edge 10, x+, y+ from (z-,z+)
                                                        26, 27, 28,               # Edge 11, y-, z- from (x-,x+)
                                                        29, 30, 31,               # Edge 12, y+, z+ from (x-,x+)

                                                        62, 63, 64, 65, 66, 67, 68, 69, 70,  # Face at x+
                                                        71, 72, 73, 74, 75, 76, 77, 78, 79,  # Face at y+
                                                        53, 54, 55, 56, 57, 58, 59, 60, 61,  # Face at z+
                                                        80, 81, 82, 83, 84, 85, 86, 87, 88,  # Face at x-
                                                        44, 45, 46, 47, 48, 49, 50, 51, 52,  # Face at y-
                                                        89, 90, 91, 92, 93, 94, 95, 96, 97,  # Face at z-

                                                        # Interior nodes
                                                        98,  99, 100, 101, 102, 103, 104, 105,                       # 1st,  8 corner nodes  # noqa: E501
                                                        106, 109, 111, 107, 114, 116, 117, 115, 108, 110, 112, 113,  # 2nd, 12 edge   nodes  # noqa: E501
                                                        120, 121, 119, 122, 118, 123, 124                            # 3rd,  6 face   nodes  # noqa: E501
                                                    ],
                                      'wedge15'     : [ 0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11 ],
                                      'pyramid13'   : [ 0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12 ],
                                    }
    )

    # Dictionary for conversion meshIO to Gmsh
    # > TODO: IMPLEMENT RECURSIVE MAPPING USING IO_MESHIO/IO_GMSH
    _gmsh_ordering: Dict[str, List[int]] = field(
            default_factory=lambda: { 'tetra10'     : [ 0, 1, 2, 3, 4, 5, 6, 7, 9, 8 ],
                                      'hexahedron20': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14 ],
                                      'hexahedron27': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14, 24,
                                                        22, 20, 21, 23, 25, 26 ],
                                      'wedge15'     : [ 0, 1, 2, 3, 4, 5, 6, 8, 12, 7, 13, 14, 9, 11, 10 ],
                                      'pyramid13'   : [ 0, 1, 2, 3, 4, 5, 8, 9, 6, 10, 7, 11, 12 ],
                                    }
    )

    def gmsh_to_meshio(self, elemType: Union[int, str, np.uint], idx: ArrayLike) -> np.ndarray:
        """
        Return the meshIO node ordering for a given element type.
        """
        if isinstance(elemType, (int, np.integer)):
            elemType = self._gmsh_typing[int(elemType)]

        if elemType not in self._meshio_ordering:
            return cast(np.ndarray, idx)
        return idx[:, self._meshio_ordering[elemType]]

