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
from typing import Dict
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


# Monkey-patching MeshIO
# See <https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells> for the node ordering.
@dataclass(repr=False, eq=False, slots=True, frozen=True)
class NumNodesPerCell:
    _data: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_data', {
            # NGeo = 1
            'vertex'         : 1,
            'line'           : 2,
            'triangle'       : 3,
            'quad'           : 4,
            'tetra'          : 4,
            'hexahedron'     : 8,
            'wedge'          : 6,
            'pyramid'        : 5,
            # NGeo = 2
            'line3'          : 3,
            'triangle6'      : 6,
            'quad8'          : 8,
            'quad9'          : 9,
            'tetra10'        : 10,
            'hexahedron20'   : 20,
            'hexahedron24'   : 24,
            'hexahedron27'   : 27,
            'wedge15'        : 15,
            'wedge18'        : 18,
            'pyramid13'      : 13,
            'pyramid14'      : 14,
            # NGeo = 3
            'line4'          : 4,
            'triangle10'     : 10,
            'quad16'         : 16,
            'tetra20'        : 20,
            'wedge40'        : 40,
            'hexahedron64'   : 64,
            # NGeo = 4
            'line5'          : 5,
            'triangle15'     : 15,
            'quad25'         : 25,
            'tetra35'        : 35,
            'wedge75'        : 75,
            'hexahedron125'  : 125,
            # NGeo = 5
            'line6'          : 6,
            'triangle21'     : 21,
            'quad36'         : 36,
            'tetra56'        : 56,
            'wedge126'       : 126,
            'hexahedron216'  : 216,
            # NGeo = 6
            'line7'          : 7,
            'triangle28'     : 28,
            'quad49'         : 49,
            'tetra84'        : 84,
            'wedge196'       : 196,
            'hexahedron343'  : 343,
            # NGeo = 7
            'line8'          : 8,
            'triangle36'     : 36,
            'quad64'         : 64,
            'tetra120'       : 120,
            'wedge288'       : 288,
            'hexahedron512'  : 512,
            # NGeo = 8
            'line9'          : 9,
            'triangle45'     : 45,
            'quad81'         : 81,
            'tetra165'       : 165,
            'wedge405'       : 405,
            'hexahedron729'  : 729,
            # NGeo = 9
            'line10'         : 10,
            'triangle55'     : 55,
            'quad100'        : 100,
            'tetra220'       : 220,
            'wedge550'       : 550,
            'hexahedron1000' : 1000,
            # NGeo = 10
            'line11'         : 11,
            'triangle66'     : 66,
            'quad121'        : 121,
            'tetra286'       : 286,
            # 'wedge715'       : 715,
            'hexahedron1331' : 1331,
        })

    def __getitem__(self, key: str) -> int:
        return self._data[key]

    def __len__(self) -> int:  # pragma: no cover
        return len(self._data)
