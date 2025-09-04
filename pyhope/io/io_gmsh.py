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
from typing import List, Tuple, Union, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from meshio import CellBlock
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@dataclass(init=True, repr=False, eq=False, slots=True)
class GMSHCELLTYPES:
    ''' Gmsh cell type definitions

        Reference: http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    '''
    cellTypes3D = [ 'tetra'   , 'hexahedron'    ,                                 'wedge'  ,            'pyramid',                 # NGeo =  1 # noqa: E501
                    'tetra10' , 'hexahedron20'  , 'hexahedron24', 'hexahedron27', 'wedge15', 'wedge18', 'pyramid13', 'pyramid14',  # NGeo =  2 # noqa: E501
                    'tetra20' , 'hexahedron64'  ,                                 'wedge40',                                       # NGeo =  3 # noqa: E501
                    'tetra35' , 'hexahedron125' ,                                 'wedge75',                                       # NGeo =  4 # noqa: E501
                    'tetra56' , 'hexahedron216' ,                                 'wedge126',                                      # NGeo =  5 # noqa: E501
                    'tetra84' , 'hexahedron343' ,                                 'wedge196',                                      # NGeo =  6 # noqa: E501
                    'tetra120', 'hexahedron512' ,                                 'wedge288',                                      # NGeo =  7 # noqa: E501
                    'tetra165', 'hexahedron729' ,                                 'wedge405',                                      # NGeo =  8 # noqa: E501
                    'tetra220', 'hexahedron1000',                                 'wedge550',                                      # NGeo =  9 # noqa: E501
                    'tetra286', 'hexahedron1331'                                                                                   # NGeo = 10 # noqa: E501
                  ]
    cellTypes2D = [ 'triangle'  , 'quad'   ,                                                                                       # NGeo =  1 # noqa: E501
                    'triangle6' , 'quad8'  , 'quad9',                                                                              # NGeo =  2 # noqa: E501
                    'triangle10', 'quad16' ,                                                                                       # NGeo =  3 # noqa: E501
                    'triangle15', 'quad25' ,                                                                                       # NGeo =  4 # noqa: E501
                    'triangle21', 'quad36' ,                                                                                       # NGeo =  5 # noqa: E501
                    'triangle28', 'quad49' ,                                                                                       # NGeo =  6 # noqa: E501
                    'triangle36', 'quad64' ,                                                                                       # NGeo =  7 # noqa: E501
                    'triangle45', 'quad81' ,                                                                                       # NGeo =  8 # noqa: E501
                    'triangle55', 'quad100',                                                                                       # NGeo =  9 # noqa: E501
                    'triangle66', 'quad121'                                                                                        # NGeo = 10 # noqa: E501
                   ]


def gmshEntityFromBlock(points:         np.ndarray,
                        block:          CellBlock,
                        dim:            int,
                        geom_id:        int,
                        zones: Union[int, np.ndarray],
                        uptPoints:      List[np.ndarray],
                        uptDimTag:      List[List[int]],
) -> Tuple[CellBlock, np.ndarray, np.ndarray]:
    ''' Accepts one meshio CellBlock and builds one Gmsh entity from it

        Steps:
            - 1) Deduplicate nodes within the block
            - 2) Append points/dim_tags to global arrays
            - 3) Remap connectivity and apply node reordering

        Returns:
            Tuple: (new_cell_block, physical_tags, geometrical_tags)
    '''

    # 1) Unique point indices used by this block
    old_ids = np.unique(block.data).astype(np.int64)
    start   = sum(len(chunk) for chunk in uptPoints)

    # 2) Append points/dim_tags to global arrays
    uptPoints.append(points[old_ids])
    uptDimTag.extend([[dim, geom_id]] * len(old_ids))

    # Build mapping only for ids present in this block to avoid large dense arrays
    mapping = {int(o): int(start + i) for i, o in enumerate(old_ids)}

    # 3) Remap connectivity and apply node reordering
    flat_old: np.ndarray = cast(np.ndarray, block.data).ravel()
    flat_new: np.ndarray = np.fromiter((mapping[int(i)] for i in flat_old), dtype=np.int64, count=flat_old.size)
    remapped: np.ndarray = flat_new.reshape(cast(np.ndarray, block.data).shape)

    # Create CellBlock and tags
    cell_block = CellBlock(block.type, remapped)
    nCellBlock = remapped.shape[0]
    tagGeom    = np.full(nCellBlock, int(geom_id), dtype=int)

    if isinstance(zones, np.ndarray):
        tagPhys = zones
        if len(tagPhys) != nCellBlock:
            raise ValueError('Length of physical_value array must equal number of cells in the block.')
    else:
        tagPhys = np.full(nCellBlock, int(zones), dtype=int)

    return cell_block, tagPhys, tagGeom

