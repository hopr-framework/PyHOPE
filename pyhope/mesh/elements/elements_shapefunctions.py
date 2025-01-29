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
from typing import Callable, Dict, List
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
class ShapeFunctions:
    """
    A dataclass that holds shape functions for different element types.

    Each entry of shape_functions maps an element type (e.g., 'hexa20', 'tetra10')
    to a list of lambda functions. These functions compute the shape-function value
    at a given parametric coordinate (xi, eta, zeta) for the corresponding node.
    """
    shape_functions: Dict[str, List[Callable[[float, float, float], float]]] = field(
        default_factory=lambda: {
            # 20-Node Hexahedron (Hexa20)
            'hexahedron20': [
                # Corner nodes (indices 0..7)
                lambda xi, eta, zeta: 0.125 * (1 - xi) * (1 - eta) * (1 - zeta) * (-xi - eta - zeta - 2),  # index  0
                lambda xi, eta, zeta: 0.125 * (1 + xi) * (1 - eta) * (1 - zeta) * ( xi - eta - zeta - 2),  # index  1
                lambda xi, eta, zeta: 0.125 * (1 + xi) * (1 + eta) * (1 - zeta) * ( xi + eta - zeta - 2),  # index  2
                lambda xi, eta, zeta: 0.125 * (1 - xi) * (1 + eta) * (1 - zeta) * (-xi + eta - zeta - 2),  # index  3
                lambda xi, eta, zeta: 0.125 * (1 - xi) * (1 - eta) * (1 + zeta) * (-xi - eta + zeta - 2),  # index  4
                lambda xi, eta, zeta: 0.125 * (1 + xi) * (1 - eta) * (1 + zeta) * ( xi - eta + zeta - 2),  # index  5
                lambda xi, eta, zeta: 0.125 * (1 + xi) * (1 + eta) * (1 + zeta) * ( xi + eta + zeta - 2),  # index  6
                lambda xi, eta, zeta: 0.125 * (1 - xi) * (1 + eta) * (1 + zeta) * (-xi + eta + zeta - 2),  # index  7

                # Edge/face-mid nodes (indices 8..19) in meshio ordering
                lambda xi, eta, zeta: 0.25  * (1 - xi**2) * (1 - eta)    * (1 - zeta),
                lambda xi, eta, zeta: 0.25  * (1 + xi)    * (1 - eta**2) * (1 - zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi**2) * (1 + eta)    * (1 - zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi)    * (1 - eta**2) * (1 - zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi**2) * (1 - eta)    * (1 + zeta),
                lambda xi, eta, zeta: 0.25  * (1 + xi)    * (1 - eta**2) * (1 + zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi**2) * (1 + eta)    * (1 + zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi)    * (1 - eta**2) * (1 + zeta),
                lambda xi, eta, zeta: 0.25  * (1 - xi)    * (1 - eta)    * (1 - zeta**2),
                lambda xi, eta, zeta: 0.25  * (1 + xi)    * (1 - eta)    * (1 - zeta**2),
                lambda xi, eta, zeta: 0.25  * (1 + xi)    * (1 + eta)    * (1 - zeta**2),
                lambda xi, eta, zeta: 0.25  * (1 - xi)    * (1 + eta)    * (1 - zeta**2),
            ],

            # 15-Node Wedge (Wedge15)
            'wedge15': [
                lambda xi, eta, zeta: 0.5 * ((1 - xi - eta) * (2. * (1 - xi - eta) - 1) * (1 - zeta) - (1 - xi - eta ) * (1 - zeta**2)),     # index 0
                lambda xi, eta, zeta: 0.5 * (xi * (2.* xi - 1) * (1 - zeta) - xi * (1 - zeta**2)),                                           # index 1
                lambda xi, eta, zeta: 0.5 * (eta * (2.* eta - 1) * (1 - zeta) - eta * (1 - zeta**2)),                                        # index 2
                lambda xi, eta, zeta: 0.5 * ((1 - xi - eta) * (2. * (1 - xi - eta) - 1) * (1 + zeta) - (1 - xi -eta) * (1 - zeta**2)),       # index 3
                lambda xi, eta, zeta: 0.5 * (xi * (2. * xi - 1) * (1 + zeta) - xi * (1 - zeta**2)),                                          # index 4
                lambda xi, eta, zeta: 0.5 * (eta * (2. * eta - 1) * (1 + zeta) - eta * (1 - zeta**2)),                                       # index 5
                lambda xi, eta, zeta: 2. * (1 - xi - eta) * xi * (1 - zeta),                                                                 # index 6
                lambda xi, eta, zeta: 2. * xi * eta * (1 - zeta),                                                                            # index 7
                lambda xi, eta, zeta: 2. * eta * (1 - xi - eta) * (1 - zeta),                                                                # index 8
                lambda xi, eta, zeta: 2. * (1 - xi - eta) * xi * (1 + zeta),                                                                 # index 9
                lambda xi, eta, zeta: 2. * xi * eta * (1 + zeta),                                                                            # index 10
                lambda xi, eta, zeta: 2. * eta * (1 - xi - eta) * (1 + zeta),                                                                # index 11
                lambda xi, eta, zeta: (1 - xi - eta) * (1 - zeta**2),                                                                        # index 12
                lambda xi, eta, zeta: xi * (1 - zeta**2),                                                                                    # index 13
                lambda xi, eta, zeta: eta * (1 - zeta**2),                                                                                   # index 14
            ],
        }
    )

    def evaluate(self, elemType: str, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        Evaluate the shape functions for a given element type at the specified
        parametric coordinates (xi, eta, zeta).

        Args:
            elemType (str): The type of element
            xi (float)    : Parametric coordinate along the x-direction
            eta (float)   : Parametric coordinate along the y-direction
            zeta (float)  : Parametric coordinate along the z-direction

        Returns:
            np.ndarray: A NumPy array of evaluated shape function values
        """
        if elemType not in self.shape_functions:
            raise ValueError(f'Unsupported element type: {elemType}')

        funcs = self.shape_functions[elemType]
        return np.array([func(xi, eta, zeta) for func in funcs])
