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
from typing import Final, Optional, cast
from collections.abc import Iterable
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
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
from pyhope.mesh.mesh_common import sidetovol2
from pyhope.mesh.mesh_common import face_to_nodes
# ==================================================================================================================================


def check_sides(elem,
                failed_only: bool = False,
               ) -> Optional[list[tuple]]:
    """ Check if connected sides have matching corner nodes
    """
    results = None
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

        # If requested, only return errors
        if failed_only and success:
            continue

        # Lazily initialize results on first failure
        if results is None:
            results = []
        results.append((success, SideID))

    # Avoid creating empty lists on elem_results
    if results is None:
        return None if failed_only else []

    return results


def process_chunk(chunk) -> list:
    """Process a chunk of elements by checking surface normal orientation
    """
    # Only keep failures to reduce memory and avoid building large arrays of successes
    chunk_results = []
    for elem in chunk:
        elem_results = check_sides(elem,
                                   failed_only=True)
        # Append a lightweight sentinel (None) for successes, actual failure list otherwise
        chunk_results.append(elem_results)
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
    if any(cast(int, e.type) % 100 != 8 for e in elems):
        elemTypes = list(set([e.type for e in elems if e.type % 100 != 8]))
        print(hopout.warn('Ignored element type: {}'.format(
            [re.sub(r"\d+$", "", mesh_vars.ELEMTYPE.inam[e][0]) for e in elemTypes]
        )))
        return

    # Prepare elements for parallel processing
    if np_mtp > 0:
        # Run in parallel with a chunk size
        # > Dispatch the tasks to the workers, minimum 10 tasks per worker, maximum 1000 tasks per worker
        res     = run_in_parallel(process_chunk,
                                  elems,
                                  chunk_size=max(1, min(1000, max(10, int(len(elems)/(40.*np_mtp))))),
                                 )
    else:
        res     = [elem for elem in elems if check_sides(elem, failed_only=True)]

    if len(res) > 0:
        # Flatten per-element results (skip None placeholders)
        results = tuple(result for elem_results in res if isinstance(elem_results, Iterable) and elem_results is not None
                               for result       in elem_results)  # noqa: E272

        nGeo:      Final[int]        = mesh_vars.nGeo
        sides:     Final[list]       = mesh_vars.sides
        points:    Final[np.ndarray] = mesh_vars.mesh.points

        # Compute total number of checked connections without materializing all results
        nconn = 0
        for SideID, side in enumerate(sides):
            # Only connected sides and not small mortar sides
            if side.connection is None or side.sideType < 0:
                continue
            # Big mortar side is counted once
            elif side.connection < 0:
                nconn += 1
            # Internal side: only count the canonical representative and ignore virtual mortar sides
            elif side.connection >= 0:
                if SideID > side.connection:
                    continue
                if side.locMortar is not None:
                    continue
                nconn += 1

        for result in cast(tuple[tuple], results):
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
