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
from __future__ import annotations
from typing import Final
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    import numpy.typing as npt
    from pyhope.mesh.mesh_vars import ELEM, SIDE, BC
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def RebuildMortarGeometry() -> None:
    # Local imports ----------------------------------------
    from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde
    from pyhope.mesh.mesh_common import LINTEN
    from pyhope.mesh.mesh_common import sidetovol2
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    if not hasattr(mesh_vars, 'hasMortars') or not mesh_vars.hasMortars:
        return None

    nGeo:   Final[int]        = mesh_vars.nGeo
    elems:  Final[list[ELEM]] = mesh_vars.elems
    sides:  Final[list[SIDE]] = mesh_vars.sides
    points: Final[npt.NDArray[np.float64]] = mesh_vars.mesh.points

    # Periodic sides need displacement
    bcs:    Final[list[BC  ]] = mesh_vars.bcs
    vvs:    Final[list      ] = mesh_vars.vvs

    # Rebuilding mortar geometries is only supported for hexahedral meshes
    if any([s != 8 for s in set([e.type % 100 for e in elems])]):
        return None

    hopout.sep()
    hopout.routine('Rebuilding mortar interfaces')

    # Compute the equidistant point set used by meshIO
    xEq:     Final[npt.NDArray[np.float64]] = np.linspace(-1., 1., nGeo+1)
    wBaryEq: Final[npt.NDArray[np.float64]] = barycentric_weights( nGeo+1, xEq)

    # Get the mortar Vandermonde
    mortarVdm = tuple(calc_vandermonde(nGeo+1, nGeo+1, wBaryEq, xEq, 0.5*(xEq+i)).reshape((nGeo+1, nGeo+1), order='F').transpose() for i in range(-1, 3, 2))  # noqa: E501

    elemOrder: Final[int] = 100 if mesh_vars.nGeo == 1 else 200
    elemType:  Final[int] = elemOrder + 8
    _, mapLin = LINTEN(elemType, order=nGeo)
    mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))

    # Loop over all big mortar sides
    for side in (s for s in sides if s.connection is not None and s.connection < 0):

        elem = elems[side.elemID]
        if elem is None or elem.nodes is None:
            raise LookupError('Malformed element nodes')

        # Fill the NodeCoords
        bigNodes = elem.nodes[sidetovol2(nGeo, 0, side.face, elemType)]
        xGeo     = points[bigNodes].reshape((nGeo+1, nGeo+1, 3), order='F').transpose(2, 0, 1)

        # Correct for periodic sides
        bcID     = side.bcid
        if bcID is not None and cast(npt.NDArray, bcs[bcID].type)[0] == 1:
            iVV    = cast(npt.NDArray, bcs[bcID].type)[3]
            VV     = vvs[np.abs(iVV)-1]['Dir'] * np.sign(iVV)
            # Shift the center in periodic direction
            xGeo  += VV

        # Loop over the small mortar sides
        match abs(side.connection):
            case 1:  # 1->4
                mortarSides = tuple(sides[sides[side.sideID+i].connection] for i in range(1, 5))
                mortarElems = tuple(elems[s.elemID]                        for s in mortarSides)  # noqa: E272

                mortarNodes = tuple(e.nodes[sidetovol2(nGeo, s.flip, s.face, elemType)] for e, s in zip(mortarElems, mortarSides))  # noqa: E271, E272
                mortarGeo   = tuple(s.reshape((nGeo+1, nGeo+1), order='F') for s in mortarNodes)  # noqa: E271, E272

                # Interpolate big mortar side to small mortar sides
                mortarSmall = tuple(np.empty((nGeo+1, nGeo+1, 3), dtype=np.float64) for _ in range(2))

                # > First in eta
                # INFO: Explicit loop matching HOPR
                # for q in range(nGeo+1):
                #     for p in range(nGeo+1):
                #         mortarSmall[0][p, q] = mortarVdm[0][0, q] * xGeo[:, p, 0]
                #         mortarSmall[1][p, q] = mortarVdm[1][0, q] * xGeo[:, p, 0]
                #
                #         for ll in range(1, nGeo+1):
                #             mortarSmall[0][p, q] += mortarVdm[0][ll, q] * xGeo[:, p, ll]
                #             mortarSmall[1][p, q] += mortarVdm[1][ll, q] * xGeo[:, p, ll]
                # INFO: Same as above but using np.einsum
                mortarSmall[0][:] = np.einsum('cpk,kq->pqc', xGeo, mortarVdm[0])
                mortarSmall[1][:] = np.einsum('cpk,kq->pqc', xGeo, mortarVdm[1])

                # > Then in xi
                # INFO: Explicit loop matching HOPR
                # for q in range(nGeo+1):
                #     for p in range(nGeo+1):
                #         points[mortarGeo[0][p, q]] = mortarVdm[0][0, p] * mortarSmall[0][0, q]
                #         points[mortarGeo[1][p, q]] = mortarVdm[1][0, p] * mortarSmall[0][0, q]
                #         points[mortarGeo[2][p, q]] = mortarVdm[0][0, p] * mortarSmall[1][0, q]
                #         points[mortarGeo[3][p, q]] = mortarVdm[1][0, p] * mortarSmall[1][0, q]
                #
                #         for ll in range(1, nGeo+1):
                #             points[mortarGeo[0][p, q]] += mortarVdm[0][ll, p] * mortarSmall[0][ll, q]
                #             points[mortarGeo[1][p, q]] += mortarVdm[1][ll, p] * mortarSmall[0][ll, q]
                #             points[mortarGeo[2][p, q]] += mortarVdm[0][ll, p] * mortarSmall[1][ll, q]
                #             points[mortarGeo[3][p, q]] += mortarVdm[1][ll, p] * mortarSmall[1][ll, q]
                # INFO: Same as above but using np.einsum
                for i, (m_idx, vdm_idx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                    result = np.einsum('kqc,kp->pqc', mortarSmall[m_idx], mortarVdm[vdm_idx])
                    for q in range(nGeo+1):
                        for p in range(nGeo+1):
                            points[mortarGeo[i][p, q]] = result[p, q]

            case 2:  # 1->2 in eta
                mortarSides = tuple(sides[sides[side.sideID+i].connection] for i in range(1, 3))
                mortarElems = tuple(elems[s.elemID]                        for s in mortarSides)  # noqa: E272

                mortarNodes = tuple(e.nodes[sidetovol2(nGeo, s.flip, s.face, elemType)] for e, s in zip(mortarElems, mortarSides))  # noqa: E271, E272
                mortarGeo   = tuple(s.reshape((nGeo+1, nGeo+1), order='F')              for    s in     mortarNodes)  # noqa: E271, E272

                # Interpolate big mortar side to small mortar sides
                # INFO: Explicit loop matching HOPR
                # for q in range(nGeo+1):
                #     for p in range(nGeo+1):
                #         points[mortarGeo[0][p, q]] = mortarVdm[0][0, q] * xGeo[:, p, 0]
                #         points[mortarGeo[1][p, q]] = mortarVdm[1][0, q] * xGeo[:, p, 0]
                #
                #         for ll in range(1, nGeo+1):
                #             points[mortarGeo[0][p, q]] += mortarVdm[0][ll, q] * xGeo[:, p, ll]
                #             points[mortarGeo[1][p, q]] += mortarVdm[1][ll, q] * xGeo[:, p, ll]
                # INFO: Same as above but using np.einsum
                for i in range(2):
                    result = np.einsum('cpk,kq->pqc', xGeo, mortarVdm[i])
                    for q in range(nGeo+1):
                        for p in range(nGeo+1):
                            points[mortarGeo[i][p, q]] = result[p, q]

            case 3:  # 1->2 in xi
                mortarSides = tuple(sides[sides[side.sideID+i].connection] for i in range(1, 3))
                mortarElems = tuple(elems[s.elemID]                        for s in mortarSides)  # noqa: E272

                mortarNodes = tuple(e.nodes[sidetovol2(nGeo, s.flip, s.face, elemType)] for e, s in zip(mortarElems, mortarSides))  # noqa: E271, E272
                mortarGeo   = tuple(s.reshape((nGeo+1, nGeo+1), order='F')              for    s in     mortarNodes)  # noqa: E271, E272

                # Interpolate big mortar side to small mortar sides
                # INFO: Explicit loop matching HOPR
                # for q in range(nGeo+1):
                #     for p in range(nGeo+1):
                #         points[mortarGeo[0][p, q]] = mortarVdm[0][0, p] * xGeo[:, 0, q]
                #         points[mortarGeo[1][p, q]] = mortarVdm[1][0, p] * xGeo[:, 0, q]
                #
                #         for ll in range(1, nGeo+1):
                #             points[mortarGeo[0][p, q]] += mortarVdm[0][ll, p] * xGeo[:, ll, q]
                #             points[mortarGeo[1][p, q]] += mortarVdm[1][ll, p] * xGeo[:, ll, q]
                # INFO: Same as above but using np.einsum
                for i in range(2):
                    result = np.einsum('ckq,kp->pqc', xGeo, mortarVdm[i])
                    for q in range(nGeo+1):
                        for p in range(nGeo+1):
                            points[mortarGeo[i][p, q]] = result[p, q]
