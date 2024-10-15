#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of UVWXYZ
#
# Copyright (c) 2022-2024 Andrea Beck
#
# UVWXYZ is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# UVWXYZ is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# UVWXYZ. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import copy
import os
import subprocess
import sys
import tempfile
import time
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import gmsh
import h5py
import meshio
import numpy as np
import pygmsh
from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def MeshExternal() -> meshio._mesh.Mesh:
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    from src.io.io_vars import debugvisu
    from src.readintools.readintools import CountOption, GetIntArray, GetRealArray, GetStr, GetLogical
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('LOADING EXTERNAL MESH')

    gmsh.initialize()
    # gmsh.option.setString('SetFactory', 'OpenCascade')
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)

    # Set default value
    nBCs_CGNS = 0

    hopout.sep()
    hopout.routine('Setting boundary conditions')
    hopout.sep()
    nBCs = CountOption('BoundaryName')
    mesh_vars.bcs = [dict() for _ in range(nBCs)]
    bcs = mesh_vars.bcs

    # Check if the number of BCs matches
    # if nBCs_CGNS is not nBCs:
    #     hopout.warning(    'Different number of boundary conditions between CGNS and parameter file\n'
    #                    ' !! Possibly see upstream issue, https://gitlab.onelab.info/gmsh/gmsh/-/issues/2727\n'
    #                    ' !! {} is now exiting ...'.format(Common.program))
    #     sys.exit(1)

    for iBC, bc in enumerate(bcs):
        bc['Name'] = GetStr('BoundaryName', number=iBC)
        bc['BCID'] = iBC + 1
        bc['Type'] = GetIntArray('BoundaryType', number=iBC)

    nVVs = CountOption('vv')
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    if len(vvs) > 0:
        hopout.sep()
    for iVV, _ in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)

    mesh_vars.CGNS.regenerate_BCs = False

    hopout.sep()
    fnames = CountOption('Filename')
    for iName in range(fnames):
        fname = GetStr('Filename')
        fname = os.path.join(os.getcwd(), fname)

        # check if the file exists
        if not os.path.isfile(os.path.join(os.getcwd(), fname)):
            hopout.warning('File [󰇘]/{} does not exist'.format(os.path.basename(fname)))
            sys.exit(1)

        # get file extension
        _, ext = os.path.splitext(fname)

        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        # if not GMSH format convert
        if ext == '.cgns':
            # Setup GMSH to import required data
            # gmsh.option.setNumber('Mesh.SaveAll', 1)
            gmsh.option.setNumber('Mesh.CgnsImportIgnoreBC', 0)
            gmsh.option.setNumber('Mesh.CgnsImportIgnoreSolution', 1)

        # Enable agglomeration
        mesh_vars.already_curved = GetLogical('MeshIsAlreadyCurved')
        hopout.sep()
        if mesh_vars.already_curved and mesh_vars.nGeo > 1:
            if ext == '.cgns':
                gmsh.option.setNumber('Mesh.CgnsImportOrder', mesh_vars.nGeo)
            # Set the element order
            # > Technically, this is only required in generate_mesh but let's be precise here
            gmsh.model.mesh.setOrder(mesh_vars.nGeo)

        gmsh.merge(fname)

        entities  = gmsh.model.getEntities()
        nBCs_CGNS = len([s for s in entities if s[0] == 2])

        # Check if GMSH read all BCs
        # > This will only work if the CGNS file identifies elementary entities by CGNS "families" and by "BC" structures
        # > Possibly see upstream issue, https://gitlab.onelab.info/gmsh/gmsh/-/issues/2727\n'
        if ext == '.cgns':
            if nBCs_CGNS is nBCs:
                for entDim, entTag in entities:
                    # Surfaces are dim-1
                    if entDim == 3:
                        continue

                    entName = gmsh.model.get_entity_name(dim=entDim, tag=entTag)
                    gmsh.model.addPhysicalGroup(entDim, [entTag], name=entName)
            else:
                mesh_vars.CGNS.regenerate_BCs = True

        gmsh.model.geo.synchronize()
        # gmsh.model.occ.synchronize()

    # PyGMSH returns a meshio.mesh datatype
    mesh = pygmsh.geo.Geometry().generate_mesh(dim=3, order=mesh_vars.nGeo)
    # mesh = pygmsh.occ.Geometry().generate_mesh(dim=3, order=mesh_vars.nGeo)

    if debugvisu:
        gmsh.fltk.run()

    # Finally done with GMSH, finalize
    gmsh.finalize()
    hopout.info('LOADING EXTERNAL MESH DONE!')
    hopout.separator()
    return mesh


def BCCGNS() -> meshio._mesh.Mesh:
    """ Some CGNS files setup their boundary conditions in a different way than gmsh expects
        > Add them here manually to the meshIO object
    """
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    # from src.common.common import find_index
    # from src.common.common_vars import Common
    from src.io.io_cgns import ElemTypes
    from src.readintools.readintools import CountOption, GetStr
    # ------------------------------------------------------

    mesh    = mesh_vars.mesh
    points  = mesh_vars.mesh.points
    cells   = mesh_vars.mesh.cells
    # elems   = mesh_vars.elems
    # sides   = mesh_vars.sides
    # bcs     = mesh_vars.bcs

    # cell_sets contain the face IDs [dim=2]
    # > Offset is calculated with entities from [dim=0, dim=1]
    offsetcs = 0
    for key, value in mesh.cells_dict.items():
        if 'vertex' in key:
            offsetcs += value.shape[0]
        elif 'line' in key:
            offsetcs += value.shape[0]
        # elif 'hexahedron' in key:  # FIXME: Support non-hexahedral meshes
        #     offsetcs += value.shape[0]

    # All non-connected sides (technically all) are potential BC sides
    # nConnSide = [s for s in sides if 'Connection' not in s and 'BCID' not in s]
    nConnSide = [value for key, value in mesh.cells_dict.items() if 'quad' in key][0]
    nConnType = [key   for key, _     in mesh.cells_dict.items() if 'quad' in key][0]  # FIXME: Support mixed LO/HO meshes
    nConnNum  = list(mesh.cells_dict).index(nConnType)
    nConnLen  = len(list(mesh.cells_dict))

    # Collapse all opposing corner nodes into an [:, 12] array
    # nbCorners  = [s['Corners'] for s in nConnSide]
    nbCorners  = [s[0:4] for s in nConnSide]
    nbPoints   = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
    nbPoints   = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
    del nbCorners

    # Build a k-dimensional tree of all points on the opposing side
    stree = spatial.KDTree(nbPoints)

    # TODO: SET ANOTHER TOLERANCE
    tol = 1.E-10

    # Now set the missing CGNS boundaries
    fnames = CountOption('Filename')
    for iName in range(fnames):
        fname = GetStr('Filename', number=iName)
        fname = os.path.join(os.getcwd(), fname)

        # Check if the file is using HDF5 format internally
        tfile = None
        # Try to convert the file automatically
        if not h5py.is_hdf5(fname):
            # Create a temporary directory and keep it existing until manually cleaned
            tfile = tempfile.NamedTemporaryFile(delete=False, delete_on_close=False)
            tname = tfile.name

            hopout.sep()
            hopout.info('File {} is not in HDF5 CGNS format, converting ...'.format(os.path.basename(fname)))
            tStart = time.time()
            subprocess.run([f'adf2hdf {fname} {tname}'], check=True, shell=True, stdout=subprocess.DEVNULL)
            tEnd   = time.time()
            hopout.info('File {} converted HDF5 CGNS format [{:.2f} sec]'.format(os.path.basename(fname), tEnd - tStart))
            hopout.sep()

            # Rest of this code operates on the converted file
            fname = tname

        with h5py.File(fname, mode='r') as f:
            if 'CGNSLibraryVersion' not in f.keys():
                hopout.warning('CGNS file does not contain library version header')
                sys.exit(1)

            try:
                base = f['Base']
            except KeyError:
                hopout.warning('Object [Base] does not exist in CGNS file')
                sys.exit(1)

            for baseNum, baseZone in enumerate(base.keys()):
                # Ignore the base dataset
                if baseZone.strip() == 'data':
                    continue

                zone = base[baseZone]
                # Check if the zone contains BCs
                if 'ZoneBC' not in zone.keys():
                    continue

                # Load the CGNS points
                nPoints = int(zone[' data'][0])
                bpoints  = [np.zeros(3, dtype='f8') for _ in range(nPoints)]

                for pointNum, point in enumerate(bpoints):
                    point[0] = float(zone['GridCoordinates']['CoordinateX'][' data'][pointNum])
                    point[1] = float(zone['GridCoordinates']['CoordinateY'][' data'][pointNum])
                    point[2] = float(zone['GridCoordinates']['CoordinateZ'][' data'][pointNum])

                # Loop over all BCs
                zoneBCs = [s for s in zone['ZoneBC'].keys() if s.strip() != 'innerfaces']

                for zoneBC in zoneBCs:
                    # bcName = zoneBC[3:]
                    # bcID   = find_index([s['Name'] for s in bcs], bcName)

                    cgnsBC = zone[zoneBC]['ElementConnectivity'][' data']

                    # Read the surface elements, one at a time
                    count  = 0

                    # Loop over all elements and get the type
                    cellsets = mesh.cell_sets
                    while count < cgnsBC.shape[0]:

                        elemType = ElemTypes(cgnsBC[count])

                        # Map the unique quad sides to our non-unique elem sides
                        corners  = cgnsBC[count+1:count+elemType['Nodes']+1]
                        # BCpoints = copy.copy(bpoints[corners])
                        BCpoints = [bpoints[s-1] for s in corners]
                        BCpoints = np.sort(BCpoints, axis=0)
                        BCpoints = BCpoints.flatten()

                        # Query the try for the opposing side
                        trSide = copy.copy(stree.query(BCpoints))
                        del BCpoints

                        # trSide contains the Euclidean distance and the index of the
                        # opposing side in the nbFaceSet
                        if trSide[0] > tol:
                            hopout.warning('Could not find a periodic side within tolerance {}, exiting...'.format(tol))
                            traceback.print_stack(file=sys.stdout)
                            sys.exit(1)

                        sideID   = int(trSide[1]) + offsetcs
                        # For the first side on the BC, the dict does not exist
                        try:
                            prevSides = cellsets[zoneBC]
                            prevSides[nConnNum] = np.append(prevSides[nConnNum], sideID)
                        except KeyError:
                            # FIXME: WE ASSUME THERE IS ONLY ONE FACE TYPE
                            prevSides = [None for _ in range(nConnLen)]
                            prevSides[nConnNum] = np.asarray([sideID]).astype(np.uint64)
                            cellsets.update({zoneBC: prevSides})

                        # Move to the next element
                        count += int(elemType['Nodes']) + 1

                    mesh   = meshio.Mesh(points=points,
                                         cells=cells,
                                         cell_sets=cellsets)

                    mesh_vars.mesh = mesh

        # Cleanup temporary file
        if tfile is not None:
            os.unlink(tfile.name)

    return mesh
