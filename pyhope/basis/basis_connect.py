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
import re
import sys
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.mesh.mesh_common import sidetovol2
from pyhope.mesh.mesh_common import face_to_nodes
# ==================================================================================================================================


def check_sides(elem,
                ) -> list[bool | int | np.ndarray]:
    results = []
    elems:  Final[list]  = mesh_vars.elems
    sides:  Final[list]  = mesh_vars.sides
    nGeo:   Final[int]   = mesh_vars.nGeo
    bcs:    Final[list]  = mesh_vars.bcs
    # Tolerance for physical comparison
    tol:    Final[float] = mesh_vars.tolPeriodic
    vvs:    Final[list]  = mesh_vars.vvs
    points: Final[npt.NDArray] = mesh_vars.mesh.points

    for SideID in elem.sides:
        master = sides[SideID]
        # Only connected sides that are master sides and not small mortar sides
        if master.connection is None \
        or master.connection < 0     \
        or master.MS != 1            \
        or master.sideType < 0:  # noqa: E271
            continue

        side   = (master, sides[master.connection])

        # Only actual element sides
        if side[0].face is None      \
        or side[1].face is None:  # noqa: E271
            continue

        # Sanity check the flip with the other nodes
        elem0  = elems[side[0].elemID]
        elem1  = elems[side[1].elemID]
        if elem1.type % 100 != 8:
            continue

        # Map the meshio nodes to the tensor-product nodes
        elemType = elem0.type
        nodes    = elem0.nodes[sidetovol2(nGeo, 0           , side[0].face, elemType)]
        nbNodes  = elem1.nodes[sidetovol2(nGeo, side[1].flip, side[1].face, elemType)]

        # INFO: THIS CURRENTLY MIGHT NOT WORK SINCE WE POTENTIALLY ONLY HAVE THE CORNER NODES AVAILABLE
        try:
            # Translate to periodic nodes if required
            if side[0].bcid is not None and side[1].bcid is not None and bcs[side[1].bcid].type[0] == 1:
                nbNodes = np.vectorize(lambda s: mesh_vars.periNodes[(s, bcs[side[1].bcid].name)], otypes=[int])(nbNodes)
            # Check if the node IDs match
            success = np.array_equal(nodes, nbNodes)
        # Fallback to comparison of physical coordinates
        except KeyError:
            # Check if periodic vector matches using vectorized np.allclose
            iVV = bcs[side[0].bcid].type[3]
            vv  = vvs[np.abs(iVV) - 1]['Dir'] * np.sign(iVV)
            success = np.allclose(points[nodes] + vv, points[nbNodes], rtol=tol, atol=tol)

        results.append((success, SideID))
    return results


def process_chunk(chunk) -> np.ndarray:
    """Process a chunk of elements by checking surface normal orientation
    """
    chunk_results    = np.empty(len(chunk), dtype=object)
    chunk_results[:] = [check_sides(elem_data) for elem_data in chunk]
    return chunk_results


def CheckConnect() -> None:
    """ Check if the mesh is correctly connected
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common_parallel import run_in_parallel
    from pyhope.common.common_vars import np_mtp
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('CHECK CONNECTIVITY...')
    hopout.sep()

    checkConnectivity  = GetLogical('CheckConnectivity')
    if not checkConnectivity:
        return None

    # Check all sides
    elems:     Final[list] = mesh_vars.elems

    # Only consider hexahedrons
    if any(e.type % 100 != 8 for e in elems):
        elemTypes = list(set([e.type for e in elems if e.type % 100 != 8]))
        print(hopout.warn('Ignored element type: {}'.format(
            [re.sub(r"\d+$", "", mesh_vars.ELEMTYPE.inam[e][0]) for e in elemTypes]
        )))
        return

    # Prepare elements for parallel processing
    if np_mtp > 0:
        tasks  = tuple((elem)
                        for elem in elems)
        # Run in parallel with a chunk size
        # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
        res    = run_in_parallel(process_chunk, tasks, chunk_size=max(1, min(1000, max(10, int(len(tasks)/(40.*np_mtp))))))
    else:
        res    = np.empty(len(elems), dtype=object)
        res[:] = [check_sides(elem) for elem in elems]

    results = tuple(tuple(result for r in res for result in r if not bool(result[0]))
)
    if len(results) > 0:
        nGeo:      Final[int]        = mesh_vars.nGeo
        sides:     Final[list]       = mesh_vars.sides
        points:    Final[np.ndarray] = mesh_vars.mesh.points

        nconn = len(tuple(tuple(result for r in res for result in r)))

        for result in results:
            # Unpack the results
            side    = sides[result[1]]
            elem    = elems[side.elemID]
            nbside  = sides[side.connection]
            nbelem  = elems[nbside.elemID]

            nodes   =   elem.nodes[face_to_nodes(  side.face,   elem.type, nGeo)]
            nbnodes = nbelem.nodes[face_to_nodes(nbside.face, nbelem.type, nGeo)]

            print()
            # Check if side is oriented inwards
            errStr = 'Side connectivity does not match the calculated neighbour side'
            print(hopout.warn(errStr, length=len(errStr)+16))

            # Print the information
            strLen  = max(len(str(side.sideID+1)), len(str(nbside.sideID+1)))
            print(hopout.warn(f'> Element {  elem.elemID+1:>{strLen}}, Side {  side.face}, Side {  side.sideID+1:>{strLen}}'))  # noqa: E501
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[ 0,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[ 0, -1]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[-1,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[  nodes[-1, -1]]) + ']'))    # noqa: E271
            # print()
            print(hopout.warn(f'> Element {nbelem.elemID+1:>{strLen}}, Side {nbside.face}, Side {nbside.sideID+1:>{strLen}}'))  # noqa: E501
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[ 0,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[ 0, -1]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[-1,  0]]) + ']'))    # noqa: E271
            print(hopout.warn('- Coordinates  : [' + ' '.join('{:12.3f}'.format(s) for s in points[nbnodes[-1, -1]]) + ']'))    # noqa: E271

        hopout.warning(f'Connectivity check failed for {len(results)} / {nconn} connections!')
        sys.exit(1)
