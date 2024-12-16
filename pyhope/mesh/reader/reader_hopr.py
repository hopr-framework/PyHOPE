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
# import copy
import os
# import subprocess
import sys
# import tempfile
# import time
# import traceback
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# import gmsh
import h5py
import meshio
import numpy as np
# import pygmsh
# from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def ReadHOPR(fnames: list) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.mesh.mesh_common import LINTEN
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points = np.zeros((0, 3), dtype=np.float64)
    cells  = dict()

    offsetnNodes = 0

    for fname in fnames:
        # Check if the file is using HDF5 format internally
        if not h5py.is_hdf5(fname):
            hopout.warning('File [󰇘]/{} is not in HDF5 format, exiting...'.format(os.path.basename(fname)))
            sys.exit(1)

        with h5py.File(fname, mode='r') as f:
            # Check if file contains the Hopr version
            if 'HoprVersion' not in f.attrs:
                hopout.warning('File [󰇘]/{} does not contain the Hopr version, exiting...'.format(os.path.basename(fname)))
                sys.exit(1)

            # Read the nodeCoords
            nodeCoords = np.array(f['NodeCoords'])
            points     = np.append(points, nodeCoords, axis=0)
            offsetnNodes += nodeCoords.shape[0]

            # Read nGeo
            nGeo       = cast(int, f.attrs['Ngeo'])
            if nGeo != mesh_vars.nGeo:
                # TODO: FIX THIS WITH A CHANGEBASIS
                filename = os.path.basename(fname)
                hopout.warning('File [󰇘]/{} has different polynomial order than the current mesh, exiting...'.format(filename))
                sys.exit(1)

            # Read the elemInfo
            elemInfo   = np.array(f['ElemInfo'])
            for elem in elemInfo:
                # Obtain the element type
                elemType = ELEMTYPE.inam[elem[0]]
                if len(elemType) > 1:
                    elemType = elemType[nGeo-2]
                else:
                    elemType = elemType[0]

                linMap    = LINTEN(elem[0], order=nGeo)
                # mapLin    = np.array(list({k: v for v, k in enumerate(linMap)}))
                elemNodes = np.arange(elem[4], elem[5])
                elemNodes = np.expand_dims(elemNodes[linMap], axis=0)

                if elemType in cells:
                    cells[elemType] = np.append(cells[elemType], elemNodes, axis=1)
                else:
                    cells[elemType] = elemNodes

    print(cells)
    # sys.exit()
    mesh = meshio.Mesh(points, cells)

    return mesh
