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

# import copy
import gc
from os import abort
import sys
from collections import defaultdict
from typing import Final, Optional, cast

# from multiprocessing import Pool
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing

from pyhope.readintools.readintools import CountOption, GetRealArray, GetInt, GetLogical, GetReal, GetIntArray

if typing.TYPE_CHECKING:
    import meshio
    import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.output.output as hopout


# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================
def InitSM() -> None:
    from pyhope.mesh.mesh_vars import SM
    import pyhope.mesh.mesh_vars as mesh_vars

    # get sliding mesh interface and mark stat./rot. elements
    mesh_vars.doSlidingMesh = GetLogical('doSlidingMesh')
    if mesh_vars.doSlidingMesh == False:
        return

    hopout.sep()
    hopout.routine('Init sliding mesh')

    nSlidingMeshPartitions = CountOption('SlidingMeshType')
    if nSlidingMeshPartitions == 0:
        return None

    mesh_vars.smPartitions = [SM() for _ in range(nSlidingMeshPartitions)]
    SMPartitions = mesh_vars.smPartitions
    for iSM, SM in enumerate(SMPartitions):
        SM.type = GetInt('SlidingMeshType')
        SM.center = GetRealArray('SlidingMeshCenter', number=iSM)
        SM.radius = GetReal('SlidingMeshRadius', number=iSM)
        SM.axis = GetInt('SlidingMeshAxis', number=iSM)

        if SM.axis != 3:
            hopout.error('Sliding Mesh axis must be alligned with z!')

        SM.interval = GetRealArray('SlidingMeshInterval', number=iSM)
        SM.norm = GetInt('SlidingMeshNormal', number=iSM)
        SM.dir = GetInt('SlidingMeshDirection', number=iSM)

        if SM.dir == SM.norm:
            hopout.error(
                'SlidingMeshDirection and SlidingMeshNormal must be orthogonal! (SlidingMeshDirection != SlidingMeshNormal)'
            )

        tmp = GetIntArray('SlidingMeshBCID', number=iSM)

        match SM.type:
            case 1:
                SM.n_interfaces = 1
                # Initialize with SMInterface default objects
                SM.interfaces = [mesh_vars.SMInterface() for _ in range(SM.n_interfaces)]
                SM.interfaces[0].bcid = tmp
            case 2:
                SM.n_interfaces = 2
                # Initialize with SMInterface default objects
                SM.interfaces = [mesh_vars.SMInterface() for _ in range(SM.n_interfaces)]
                SM.interfaces[0].bcid = tmp[0]
                SM.interfaces[1].bcid = tmp[1]
            case 3:
                SM.n_interfaces = 2
                # Initialize with SMInterface default objects
                SM.interfaces = [mesh_vars.SMInterface() for _ in range(SM.n_interfaces)]
                SM.interfaces[0].bcid = tmp[0]
                SM.interfaces[1].bcid = tmp[1]


def compute_smcoords(x, SM):
    """Computes sliding mesh surface coordinates (smcoords) based on SM type.

    Parameters
    ----------
    x : array-like of shape (3,)
        Centroid coordinates of the side/element [x0, x1, x2].
    SM : object
        Sliding mesh configuration object with attributes:
        - type (int): 1 (Annulus), 2 (Planar), or 3 (Axial)
        - center (array-like, optional): Center point for Annulus type
        - dir (int, optional): Sliding direction (1-indexed: 1, 2, or 3)
        - norm (int, optional): Normal direction (1-indexed: 1, 2, or 3)

    Returns
    -------
    list of float
        [coord_0, coord_1] representing the local 2D sliding mesh surface
        coordinates.
    """
    x = np.asarray(x, dtype=float)

    match SM.type:
        case 1:  # Annulus interface
            x_rel = x - np.asarray(SM.center)
            return [x_rel[2], np.atan2(-x_rel[1], -x_rel[0])]

        case 2:  # Planar interface
            # 1-indexed to 0-indexed direction calculation
            # Layer direction is the remaining axis: 6 - SM.dir - SM.norm (1-based)
            # converting to 0-based: (6 - SM.dir - SM.norm) - 1 = 5 - SM.dir - SM.norm
            layer_dir = 5 - SM.dir - SM.norm
            slide_dir = SM.dir - 1
            return [x[layer_dir], x[slide_dir]]

        case 3:  # Axial / Cylindrical interface
            r = np.sqrt(x[0] * x[0] + x[1] * x[1])
            theta = np.atan2(-x[1], -x[0])
            return [r, theta]

        case _:
            raise ValueError(f'Unsupported Sliding Mesh Type: {SM.type}. Expected 1, 2, or 3.')


def prepareSlidingMesh() -> None:
    from pyhope.mesh.mesh_vars import SM
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.io.io_vars import SIDE

    hopout.sep()
    hopout.routine('Prepare sliding mesh')

    mesh = mesh_vars.mesh
    points = mesh.points
    elems = mesh_vars.elems
    sides = mesh_vars.sides

    # Save all BC sides which are potential SM interfaces
    sides_sm_all = [s for s in sides if s.bcid is not None and mesh_vars.bcs[s.bcid].type[0] == -100]

    SMPartitions = mesh_vars.smPartitions
    for iSM, SM in enumerate(SMPartitions):
        for iSMInt, SMInt in enumerate(SM.interfaces):
            SMInt.height = [1.0e13, -1.0e13]
            SMInt.boundaries = [1.0e13, -1.0e13]

            # Extract sides matching the sliding mesh boundary condition
            sides_sm_raw = [s for s in sides_sm_all if mesh_vars.bcs[s.bcid].type[3] == SMInt.bcid]

            seen_side_ids = set()
            sides_sm = []
            for s in sides_sm_raw:
                # Use frozenset so order doesn't result in duplicate entries
                corners_key = frozenset(s.corners)
                if corners_key not in seen_side_ids:
                    seen_side_ids.add(corners_key)
                    sides_sm.append(s)

            for s in sides_sm:
                # get SM coords
                x = 0.0
                for i in s.corners:
                    x += points[i]
                x = x / s.corners.shape[0]

                s.smcoords = compute_smcoords(x, SM)

                if SM.type == 2:
                    # Find begin and end of sm interface in moving direction for modified periodic BCs
                    # Also find begin and end of sm interface in z-direction
                    for i in s.corners:
                        point = points[i]
                        SMInt.boundaries = [min(SMInt.boundaries[0], point[SM.dir]), max(SMInt.boundaries[1], point[SM.dir])]
                        SMInt.height = [min(SMInt.height[0], point[2]), max(SMInt.height[1], point[2])]

            if SM.type in [1, 3]:
                SMInt.boundaries = [0.0, 2 * np.pi]

            # 1. Bin discovery with tolerance matching Fortran loop
            nLA = [0, 0]
            eps = 1.0e-12
            LAMaxBins = [[], []]

            for s in sides_sm:
                for dir_idx in range(2):
                    coord_val = s.smcoords[dir_idx]
                    found = False
                    for bin_val in LAMaxBins[dir_idx]:
                        if abs(coord_val - bin_val) < eps:
                            found = True
                            break
                    if not found:
                        nLA[dir_idx] += 1
                        LAMaxBins[dir_idx].append(coord_val)

            nSMSides = len(sides_sm)
            if np.prod(nLA) != nSMSides:
                hopout.error('Error while sorting SM sides into bins')

            # 2. Determine MasterOrient using 1/3 and 2/3 interior side points
            first_side = sides_sm[0]
            p1 = points[first_side.corners[0]]
            p2 = points[first_side.corners[1]]

            X1 = (2.0 / 3.0) * p1 + (1.0 / 3.0) * p2
            X2 = (1.0 / 3.0) * p1 + (2.0 / 3.0) * p2

            Eta1 = compute_smcoords(X1, SM)
            Eta2 = compute_smcoords(X2, SM)

            # Sign of azimuthal difference
            diff = Eta2[1] - Eta1[1]
            MasterOrient = 1 if diff >= 0 else -1
            SMInt.masterOrient = MasterOrient

            # 3. Sort bin coordinates by layer and azimuth (orientation-aware)
            unique_coords_0 = sorted(LAMaxBins[0])
            unique_coords_1 = sorted(LAMaxBins[1], key=lambda val: MasterOrient * val)

            # Convert back if MasterOrient is negative to match Fortran orientation scaling
            if MasterOrient < 0:
                unique_coords_1 = [MasterOrient * val for val in sorted([MasterOrient * val for val in LAMaxBins[1]])]

            # 4. Allocate 3D matrix (3, nAzimuthal, nLayer) and initialize with -1
            SMInt.n_layer = nLA[0]
            SMInt.n_azimuthal_sides_per_layer = nLA[1]
            SMInt.sides = -np.ones((3, nLA[1], nLA[0]), dtype=int)

            # 5. Map sides into 3D bins
            for s in sides_sm:
                iLA_0 = next((i for i, val in enumerate(unique_coords_0) if abs(s.smcoords[0] - val) < 2 * eps), None)
                iLA_1 = next((i for i, val in enumerate(unique_coords_1) if abs(s.smcoords[1] - val) < 2 * eps), None)

                if iLA_0 is None or iLA_1 is None:
                    hopout.error('bin sorting of SM sides failed')

                if SMInt.sides[0, iLA_1, iLA_0] > 0:
                    hopout.error('double entry in SlidingMeshInfo')

                # Assign Side Index, Element Index, and Connected Element Index
                SMInt.sides[0, iLA_1, iLA_0] = s.globalSideID
                SMInt.sides[1, iLA_1, iLA_0] = s.elemID + 1
                SMInt.sides[2, iLA_1, iLA_0] = sides[s.connection].elemID + 1

            if np.min(SMInt.sides) < 0:
                hopout.error('not all entries found in SlidingMeshInfo')


def prepareSMSFC(mesh_vars, sfc_type: int, np_mtp: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sorts elements using Space Filling Curves (SFC) into stationary and rotating partitions.

    Returns:
        IDList (np.ndarray): Permutation array of reordered 0-based element indices.
        RotatingElem (np.ndarray): Partition flag array indicating stationary (0) or
                                   sliding partition ID (>0) for each reordered element.
    """
    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_common import calc_elem_bary
    from pyhope.mesh.mesh_sort import compute_sfc_distances

    # ------------------------------------------------------
    #
    mesh = mesh_vars.mesh
    elems = mesh_vars.elems
    points = mesh.points
    SMPartitions = mesh_vars.smPartitions

    nElems = len(elems)
    nSMPartitions = len(SMPartitions)

    # 1. Vectorized Element Barycenter Calculation
    elem_bary = calc_elem_bary(elems)  # Shape: (nElems, 3)

    # Track classification: 0 = stationary, >0 = rotating partition index (1-based)
    elem_partition = np.zeros(nElems, dtype=int)

    # 2. Classify elements into rotating partitions or stationary domain
    for iSM, SM in enumerate(SMPartitions, start=1):
        SM.nrotelems = 0

        match SM.type:
            case 1:  # Annulus
                radius = [-1.0, SM.radius]
                radius_elem = np.sqrt((elem_bary[:, 0] - SM.center[0]) ** 2 + (elem_bary[:, 1] - SM.center[1]) ** 2)

            case 2:  # Planar (SM.dir is 1-based: 1, 2, 3)
                radius = SM.interval
                radius_elem = elem_bary[:, SM.norm - 1]

            case 3:  # Axial
                radius = SM.interval
                radius_elem = elem_bary[:, SM.axis - 1]

        # Condition matching: radius_elem > radius[0] AND radius_elem < radius[1]
        is_in_sm = (radius_elem > radius[0]) & (radius_elem < radius[1])

        # Assign only elements that haven't been assigned to an earlier partition
        unassigned_mask = elem_partition == 0
        rot_mask = is_in_sm & unassigned_mask

        elem_partition[rot_mask] = iSM
        SM.nrotelems = int(np.sum(rot_mask))

    # Identify stationary vs rotating elements
    stat_indices = np.where(elem_partition == 0)[0]
    nStatElems = len(stat_indices)

    # 3. Sort Stationary Elements along SFC
    if nStatElems > 0:
        stat_bary = elem_bary[stat_indices]
        stat_distances = compute_sfc_distances(stat_bary, points, sfc_type, np_mtp)
        stat_sort_order = np.argsort(stat_distances)
        sorted_stat_ids = stat_indices[stat_sort_order]
    else:
        sorted_stat_ids = np.array([], dtype=int)

    # 4. Sort Rotating Elements independently for each Partition
    sorted_rot_ids_list = []
    current_first_rot = nStatElems

    for iSM, SM in enumerate(SMPartitions, start=1):
        # SM.FirstRotElem = current_first_rot + 1  # 1-based indexing for Fortran
        rot_indices = np.where(elem_partition == iSM)[0]

        if len(rot_indices) > 0:
            rot_bary = elem_bary[rot_indices]
            rot_distances = compute_sfc_distances(rot_bary, points, sfc_type, np_mtp)
            rot_sort_order = np.argsort(rot_distances)
            sorted_rot_ids = rot_indices[rot_sort_order]
        else:
            sorted_rot_ids = np.array([], dtype=int)

        sorted_rot_ids_list.append(sorted_rot_ids)
        current_first_rot += SM.nrotelems

    # 5. Construct Global Permutation Array (IDList)
    if sorted_rot_ids_list:
        all_rot_ids = np.concatenate(sorted_rot_ids_list)
        IDList = np.concatenate([sorted_stat_ids, all_rot_ids])
    else:
        IDList = sorted_stat_ids

    # 6. Construct RotatingElem Array mapping sorted elements to partition flags
    RotatingElem = np.zeros(nElems, dtype=int)
    offset = nStatElems
    for iSM, SM in enumerate(SMPartitions, start=1):
        n_rot = SM.nrotelems
        RotatingElem[offset : offset + n_rot] = iSM
        offset += n_rot

    return IDList, RotatingElem, nStatElems
