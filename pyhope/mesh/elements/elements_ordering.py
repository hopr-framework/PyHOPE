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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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


@dataclass
class ElementInfo:
    """
    A dataclass that stores face definitions and mesh ordering for the element types. Both faces and orderings are retrieved via the
    subsequent methods that take the element type as a parameter.
    """

    # Dictionary of dictionaries for face data, keyed by element type.
    # Each entry is a dictionary of face-name to node indices.
    _faces_map: Dict[str, Dict[str, List[int]]] = field(
        default_factory=lambda: { 'pyramid13':  { 'z-': [ 0, 1, 2, 3,  5,   6, 7, 8],
                                                  'y-': [ 0, 1, 4, 5, 10,   9],
                                                  'y+': [ 2, 3, 4, 7, 12,  11],
                                                  'x+': [ 1, 2, 4, 6, 11,  10],
                                                  'x-': [ 3, 0, 4, 8,  9,  12],
                                                 },
                                  'wedge15':    { 'z-': [ 0, 1, 2, 6,  7,  8],
                                                  'z+': [ 3, 4, 5, 9, 10, 11],
                                                  'y-': [ 0, 1, 4, 3,  6, 13,  9, 12],
                                                  'x+': [ 1, 2, 5, 4,  7, 14, 10, 13],
                                                  'x-': [ 2, 0, 3, 5,  8, 12, 11, 14],
                                                 },
                                  'hexahedron20': { 'z-': [ 0,  1,  2,  3,  8,  9, 10, 11],
                                                    'z+': [ 4,  5,  6,  7, 12, 13, 14, 15],
                                                    'y-': [ 0,  1,  5,  4,  8, 17, 12, 16],
                                                    'x+': [ 1,  2,  6,  5,  9, 18, 13, 17],
                                                    'y+': [ 2,  3,  7,  6, 10, 19, 14, 18],
                                                    'x-': [ 3,  0,  4,  7, 11, 16, 15, 19],
                                                   },
                                  'hexahedron27': { 'z-': [ 0,  1,  2,  3,  8,  9, 10, 11, 24],
                                                    'z+': [ 4,  5,  6,  7, 12, 13, 14, 15, 25],
                                                    'y-': [ 0,  1,  5,  4,  8, 17, 12, 16, 22],
                                                    'x+': [ 1,  2,  6,  5,  9, 18, 13, 17, 21],
                                                    'y+': [ 2,  3,  7,  6, 10, 19, 14, 18, 23],
                                                    'x-': [ 3,  0,  4,  7, 11, 16, 15, 19, 20],
                                                   }
                                })

    _params_map: Dict[str, Dict[str, Tuple[float, float, float]]] = field(
        default_factory=lambda: { 'hexahedron20': { 'z-': ( 0.,  0., -1.),
                                                    'y-': ( 0., -1.,  0.),
                                                    'x-': (-1.,  0.,  0.),
                                                    'x+': ( 1.,  0.,  0.),
                                                    'y+': ( 0.,  1.,  0.),
                                                    'z+': ( 0.,  0.,  1.),
                                                   },
                                  'hexahedron27': { 'z-': ( 0.,  0., -1.),
                                                    'y-': ( 0., -1.,  0.),
                                                    'x-': (-1.,  0.,  0.),
                                                    'x+': ( 1.,  0.,  0.),
                                                    'y+': ( 0.,  1.,  0.),
                                                    'z+': ( 0.,  0.,  1.),
                                                   }
                                 })

    def faces_to_nodes(self, elemType: str) -> Dict[str, List[int]]:
        """
        Retrieves the face definitions for the specified element type.

        Args:
            elemType (str): The type of element
        """
        if elemType not in self._faces_map:
            raise ValueError(f'No faces defined for element type "{elemType}".')

        return self._faces_map[elemType]

    def faces_to_params(self, elemType: str) -> Dict[str, Tuple[float, float, float]]:
        """
        Retrieves the face parameters for the specified element type.

        Args:
            elemType (str): The type of element
        """
        if elemType not in self._params_map:
            raise ValueError(f'No face parameters defined for element type "{elemType}".')

        return self._params_map[elemType]
