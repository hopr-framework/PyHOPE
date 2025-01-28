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
import copy
import gc
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import gmsh
import h5py
import meshio
import numpy as np
from scipy import spatial
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Monkey-patching MeshIO
meshio._mesh.topological_dimension['wedge15'] = 3
# ==================================================================================================================================


def compatibleGMSH(file: str) -> bool:
    ioFormat = {1 : '.msh',
                2 : '.unv',
                # 10: 'auto',
                16: '.vtk',
                19: '.vrml',
                21: '.mail',
                # 26: 'pos stat',
                27: '.stl',
                28: '.p3d',
                30: '.mesh',
                31: '.bdf',
                32: '.cgns',
                33: '.med',
                34: '.diff',
                38: '.ir3',
                39: '.inp',
                40: '.ply2',
                41: '.celum',
                42: '.su2',
                47: '.tochnog',
                49: '.neu',
                50: '.matlab'}

    # get file extension
    _, ext = os.path.splitext(file)

    if ext in ioFormat.values():
        return True
    else:
        return False


def ReadGMSH(fnames: list) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.meshio.meshio_convert import gmsh_to_meshio
    import pyhope.output.output as hopout
    from pyhope.io.io_vars import debugvisu
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.sep()
    gmsh.initialize()
    # gmsh.option.setString('SetFactory', 'OpenCascade')
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Geometry.Tolerance'         , 1e-12)  # default: 1e-6
        gmsh.option.setNumber('Geometry.MatchMeshTolerance', 1e-09)  # default: 1e-8

    for fname in fnames:
        # get file extension
        _, ext = os.path.splitext(fname)

        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        # gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 0)

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

        # Explicitly load the OpenCASCADE kernel
        gmsh.model.occ.synchronize()

        # entities  = gmsh.model.getEntities()
        # nBCs_CGNS = len([s for s in entities if s[0] == 2])

        # Check if GMSH read all BCs
        # > This will only work if the CGNS file identifies elementary entities by CGNS "families" and by "BC" structures
        # > Possibly see upstream issue, https://gitlab.onelab.info/gmsh/gmsh/-/issues/2727\n'
        if ext == '.cgns':
            # WARNING: THIS PROBABLY NEVER WORKS, SO JUST USE OUR OWN APPROACH
            # if nBCs_CGNS == len(mesh_vars.bcs):
            #     for entDim, entTag in entities:
            #         # Surfaces are dim-1
            #         if entDim == 3:
            #             continue
            #
            #         entName = gmsh.model.get_entity_name(dim=entDim, tag=entTag)
            #         gmsh.model.addPhysicalGroup(entDim, [entTag], name=entName)
            # else:
            #     mesh_vars.CGNS.regenerate_BCs = True
            mesh_vars.CGNS.regenerate_BCs = True

        # gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        # Optimize the high-order mesh
        # gmsh.model.mesh.optimize(method='Relocate3D', force=True)
        # gmsh.model.occ.synchronize()

    # Reclassify the nodes to ensure correct node ordering
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.occ.synchronize()

    if debugvisu:
        gmsh.fltk.run()

    # Convert Gmsh object to meshio object
    mesh = gmsh_to_meshio(gmsh)

    # Check whether the mesh contains high-order elements and nGeo is set to 1
    if not mesh_vars.already_curved or mesh_vars.nGeo == 1:
        for elemtype in mesh.cells_dict.keys():
            if elemtype in mesh_vars.ELEMTYPE.name and mesh_vars.ELEMTYPE.name[elemtype] > 200:
                hopout.warning('High-order elements detected in the mesh but MeshIsAlreadyCurved=F or nGeo is set to 1, exiting...')
                sys.exit(1)

    # If the mesh contains second-order incomplete elements, fix them
    mesh = fixSecondOrderIncomplete(mesh)

    # Finally done with GMSH, finalize
    gmsh.finalize()

    # Run garbage collector to release memory
    gc.collect()

    return mesh


def fixSecondOrderIncomplete(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Some GMSH files contain incomplete elements, fix them here
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.mesh.elements.elements_shapefunctions import ShapeFunctions
    from pyhope.mesh.elements.elements_ordering import ElementInfo
    # ------------------------------------------------------

    # Check the mesh contains second-order incomplete elements
    secondOrderIncomplete = ['triangle6', 'quad8', 'tetra10', 'hexahedron20', 'wedge15', 'pyramid13']
    if not any(s for s in mesh.cells_dict.keys() if s in secondOrderIncomplete):
        return mesh

    # Copy original points
    points    = mesh.points.copy()

    # Prepare new cell blocks
    elems_new = {}
    shapefunctions = ShapeFunctions()
    elementinfo    = ElementInfo()

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        # Valid cell type
        if ctype not in secondOrderIncomplete:
            elems_new[ctype] = cdata
            continue

        match ctype:
            # Remove surface elements
            case 'triangle6' | 'quad8':
                continue

            case 'tetra10':
                hopout.warning('Tetra10 not supported yet')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

            case 'hexahedron20':
                faces = ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']

                # Loop over all hexahedrons
                for elem in cdata:
                    # Create the 6 face mid-points
                    for face in faces:
                        # Face parameters are the same as for the 27-node hexahedron
                        xi, eta, zeta = elementinfo.faces_to_params('hexahedron27')[face]
                        faceNodes     = elementinfo.faces_to_nodes( 'hexahedron27')[face]

                        # Evaluate the quadratic shape function at the face center
                        N      = shapefunctions.evaluate(ctype, xi, eta, zeta)
                        center = np.dot(N, mesh.points[elem])
                        points = np.vstack((points, center))

                        # Take the existing 8 face nodes and append the new center node
                        subFace = elem[faceNodes[:8]].tolist()
                        subFace.append(len(points) - 1)
                        subFace = np.array(subFace)

                        # Create the surface element
                        # > For the first surface element, the key does not exist in the dict
                        try:
                            elems_new['quad9'] = np.append(elems_new['quad9'], np.array([subFace]).astype(np.uint), axis=0)  # noqa: E501
                        except KeyError:
                            elems_new['quad9'] = np.array([subFace]).astype(np.uint)

                    # Evaluate the quadratic shape function at the volume center
                    N      = shapefunctions.evaluate(ctype, 0, 0, 0)
                    center = np.dot(N, mesh.points[elem])
                    points = np.vstack((points, center))

                    # Create the volume element
                    subElem = elem.tolist()

                    # Append the 6 face center and the volume center
                    subElem.extend(i for i in range(len(points)-7, len(points)))
                    subElem = np.array(subElem)

                    # For the first volume element, the key does not exist in the dict
                    try:
                        elems_new['hexahedron27'] = np.append(elems_new['hexahedron27'], np.array([subElem]).astype(np.uint), axis=0)  # noqa: E501
                    except KeyError:
                        elems_new['hexahedron27'] = np.array([subElem]).astype(np.uint)

            # case 'hexahedron27':
            #     # Loop over all hexahedrons
            #     for elem in cdata:
            #         # Fix the meshio type
            #         meshio_elem = elem[meshio_ordering[ctype]]
            #         subElem     = meshio_elem
            #
            #         # For the first volume element, the key does not exist in the dict
            #         try:
            #             elems_new['hexahedron27'] = np.append(elems_new['hexahedron27'], np.array([subElem]).astype(int), axis=0)  # noqa: E501
            #         except KeyError:
            #             elems_new['hexahedron27'] = np.array([subElem]).astype(int)

            case 'wedge15':
                hopout.warning('Wedge15 not supported yet')
                traceback.print_stack(file=sys.stdout)
                sys.exit(1)

    # At this point, the mesh does not contain boundary conditions / cell sets
    # > We add them in the BCCGNS function
    mesh   = meshio.Mesh(points = points,     # noqa: E251
                         cells  = elems_new)  # noqa: E251

    return mesh


def BCCGNS() -> meshio.Mesh:
    """ Some CGNS files setup their boundary conditions in a different way than gmsh expects
        > Add them here manually to the meshIO object
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import CountOption, GetStr
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

    # All non-connected sides (technically all) are potential BC sides
    # nConnSide = [s for s in sides if 'Connection' not in s and 'BCID' not in s]
    nConnSide = [value for key, value in mesh.cells_dict.items() if 'quad' in key][0]
    nConnType = [key   for key, _     in mesh.cells_dict.items() if 'quad' in key][0]  # FIXME: Support mixed LO/HO meshes  # noqa: E272, E501
    nConnNum  = list(mesh.cells_dict).index(nConnType)
    nConnLen  = len(list(mesh.cells_dict))

    # Collapse all opposing corner nodes into an [:, 12] array
    # nbCorners  = [s['Corners'] for s in nConnSide]
    nbCorners = [s[0:4] for s in nConnSide]
    nbPoints  = copy.copy(np.sort(mesh.points[nbCorners], axis=1))
    nbPoints  = nbPoints.reshape(nbPoints.shape[0], nbPoints.shape[1]*nbPoints.shape[2])
    del nbCorners

    # Build a k-dimensional tree of all points on the opposing side
    stree = spatial.KDTree(nbPoints)

    # Now, the same thing for triangular elements
    tConnLen = 0
    tConnNum = 0
    ttree    = spatial.KDTree([[0.0]])
    if len([value for key, value in mesh.cells_dict.items() if 'triangle' in key]) > 0:
        tConnSide = [value for key, value in mesh.cells_dict.items() if 'triangle' in key][0]
        tConnType = [key   for key, _     in mesh.cells_dict.items() if 'triangle' in key][0]  # FIXME: Support mixed LO/HO meshes  # noqa: E272, E501
        tConnNum  = list(mesh.cells_dict).index(tConnType)
        tConnLen  = len(list(mesh.cells_dict))

        # Collapse all opposing corner nodes into an [:, 9] array
        tbCorners = [s[0:3] for s in tConnSide]
        tbPoints  = copy.copy(np.sort(mesh.points[tbCorners], axis=1))
        tbPoints  = tbPoints.reshape(tbPoints.shape[0], tbPoints.shape[1]*tbPoints.shape[2])
        del tbCorners

        ttree = spatial.KDTree(tbPoints)

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
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tname = tfile.name

            hopout.sep()
            hopout.info('File {} is not in HDF5 CGNS format, converting ...'.format(os.path.basename(fname)))
            tStart = time.time()
            _ = subprocess.run([f'adf2hdf {fname} {tname}'], check=True, shell=True, stdout=subprocess.DEVNULL)
            tEnd   = time.time()
            hopout.info('File {} converted HDF5 CGNS format [{:.2f} sec]'.format(os.path.basename(fname), tEnd - tStart))
            hopout.sep()

            # Rest of this code operates on the converted file
            fname = tname

        with h5py.File(fname, mode='r') as f:
            if 'CGNSLibraryVersion' not in f.keys():
                hopout.warning('CGNS file does not contain library version header')
                sys.exit(1)

            key = [s for s in f.keys() if "base" in s.lower()]
            if len(key) == 0:
                hopout.warning('Object [Base] does not exist in CGNS file')
                sys.exit(1)
            elif len(key) > 1:
                hopout.warning('More than one object [Base] exists in CGNS file')
                sys.exit(1)

            if not isinstance(f[key[0]], h5py.Group):
                hopout.warning('Object [Base] is not a group in CGNS file')
                sys.exit(1)
            base = cast(h5py.Group, f[key[0]])

            for baseZone in base.keys():
                # Ignore the base dataset
                if baseZone.strip() == 'data':
                    continue

                zone = cast(h5py.Group, base[baseZone])
                # Check if the zone contains BCs
                if 'ZoneBC' not in zone.keys():
                    continue

                zonedata = cast(h5py.Dataset, zone[' data'])
                match len(zonedata[0]):
                    case 1:  # Unstructured mesh, 1D arrays
                        BCCGNS_Unstructured(mesh, points, cells, cast(spatial.KDTree, stree), zone, tol, offsetcs, nConnNum, nConnLen,
                                            # Support for triangular elements
                                            cast(spatial.KDTree, ttree), tConnNum, tConnLen)
                    case 3:  # Structured 3D mesh, 3D arrays
                        # Structured grid can only contain tensor-product elements
                        BCCGNS_Structured(mesh, points, cells, cast(spatial.KDTree, stree), zone, tol, offsetcs, nConnNum, nConnLen)
                    case _:  # Unsupported number of dimensions
                        # raise ValueError('Unsupported number of dimensions')
                        hopout.warning('Unsupported number of dimensions')
                        sys.exit(1)

        # Cleanup temporary file
        if tfile is not None:
            os.unlink(tfile.name)

    # Run garbage collector to release memory
    gc.collect()

    return mesh


def BCCGNS_SetBC(BCpoints: np.ndarray,
                 cellsets,
                 nConnLen: int,
                 nConnNum: int,
                 offsetcs: int,
                 stree:    spatial.KDTree,
                 tol:      float,
                 BCName:   str) -> dict:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    # Query the tree for the opposing side
    trSide = copy.copy(stree.query(BCpoints))

    # trSide contains the Euclidean distance and the index of the
    # opposing side in the nbFaceSet
    if trSide[0] > tol:
        hopout.warning('Could not find a boundary side within tolerance {}, exiting...'.format(tol))
        traceback.print_stack(file=sys.stdout)
        sys.exit(1)

    sideID   = int(trSide[1]) + offsetcs
    # For the first side on the BC, the dict does not exist
    try:
        prevSides = cellsets[BCName]
        prevSides[nConnNum] = np.append(prevSides[nConnNum], sideID)
    except KeyError:
        prevSides = [np.empty((0,), dtype=int) for _ in range(nConnLen)]
        prevSides[nConnNum] = np.asarray([sideID]).astype(int)
        cellsets.update({BCName: prevSides})
    return cellsets


def BCCGNS_Unstructured(  mesh:     meshio.Mesh,
                          points:   np.ndarray,
                          cells:    list,
                          stree:    spatial.KDTree,
                          zone,     # CGNS zone
                          tol:      float,
                          offsetcs: int,
                          nConnNum: int,
                          nConnLen: int,
                          # Triangular elements
                          ttree:    spatial.KDTree,
                          tConnNum: int,
                          tConnLen: int) -> None:
    """ Set the CGNS boundary conditions for uncurved (unstructured) grids
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.io.io_cgns import ElemTypes
    # ------------------------------------------------------
    # Load the CGNS points
    nPoints = int(zone[' data'][0])
    bpoints = [np.zeros(3, dtype='f8') for _ in range(nPoints)]

    for pointNum, point in enumerate(bpoints):
        point[0] = float(zone['GridCoordinates']['CoordinateX'][' data'][pointNum])
        point[1] = float(zone['GridCoordinates']['CoordinateY'][' data'][pointNum])
        point[2] = float(zone['GridCoordinates']['CoordinateZ'][' data'][pointNum])

    # Loop over all BCs
    cellsets = mesh.cell_sets
    zoneBCs  = [s for s in cast(h5py.Group, zone['ZoneBC']).keys() if s.strip() != 'innerfaces']

    for zoneBC in zoneBCs:
        # bcName = zoneBC[3:]
        # bcID   = find_index([s['Name'] for s in bcs], bcName)
        zoneBC = cast(str, zoneBC)

        # Data given with separate zoneBCs
        if zoneBC in zone:
            cgnsBC = cast(h5py.Dataset, zone[zoneBC]['ElementConnectivity'][' data'])

            # Read the surface elements, one at a time
            count  = 0

            # Loop over all elements and get the type
            while count < cgnsBC.shape[0]:
                elemType = ElemTypes(cgnsBC[count])

                # Map the unique quad sides to our non-unique elem sides
                corners  = cgnsBC[count+1:count+int(elemType['Nodes'])+1]
                # BCpoints = copy.copy(bpoints[corners])
                BCpoints = [bpoints[s-1] for s in corners]
                BCpoints = np.sort(BCpoints, axis=0)
                BCpoints = BCpoints.flatten()
                cellsets = BCCGNS_SetBC(BCpoints, cellsets, nConnLen, nConnNum, offsetcs, stree, tol, zoneBC)
                del BCpoints

                # Move to the next element
                count += int(elemType['Nodes']) + 1

        # Data attached to the zoneBC node
        elif f'{zoneBC}/PointList' in zone['ZoneBC']:
            cgnsBC = cast(h5py.Dataset, zone['ZoneBC'][zoneBC]['PointList'][' data'])
            cgnsBC = sorted(int(s) for s in cgnsBC)

            # Read the shell elements (either from GridShells or SurfaceElements)
            # Check how Surface Elements are stored in the file. This destinguishes
            # for example ANSA unstructured and structured CGNS output
            if 'GridShells' in zone:
                cgnsShells  = cast(h5py.Dataset, zone['GridShells']['ElementConnectivity'][' data'])
                nShells     = cast(int, zone['GridShells']['ElementRange'][' data'][0])
            elif 'SurfaceElements' in zone:
                cgnsShells  = cast(h5py.Dataset, zone['SurfaceElements']['ElementConnectivity'][' data'])
                nShells     = cast(int, zone['SurfaceElements']['ElementRange'][' data'][0])
            else:
                hopout.warning('Format of BC implementation for FaceCenters not recognized, exiting...')
                sys.exit(1)

            # Get the location of the BC faces
            cgnsGridLoc = bytes(zone['ZoneBC'][zoneBC]['GridLocation'][' data']).decode('ascii')

            if cgnsGridLoc == 'Vertex':
                count    = 0
                cgnsSmax = 0
                cgnsSmin = sys.maxsize

                # Precompute the min and max values in cgnsBC for filtering
                cgns_set = set(cgnsBC)

                # Loop over all elements and get the type
                while count < cgnsShells.shape[0]:
                    elemType = ElemTypes(cgnsShells[count])
                    nNodes   = int(elemType['Nodes'])

                    corners  = cgnsShells[count+1:count+nNodes+1]
                    # corners  = sorted(int(s) for s in corners)

                    cgnsSmax = max(np.max(corners), cgnsSmax)
                    cgnsSmin = min(np.min(corners), cgnsSmin)

                    count   += int(elemType['Nodes']) + 1

                # Read the surface elements, one at a time
                count   = 0

                # Loop over all elements and get the type
                while count < cgnsShells.shape[0]:
                    elemType = ElemTypes(cgnsShells[count])
                    nNodes   = int(elemType['Nodes'])

                    corners  = cgnsShells[count+1:count+nNodes+1]
                    corners  = sorted(int(s) for s in corners)

                    # Early rejection based on bounds
                    # if max(corners) > cgnsSmax or min(corners) < cgnsSmin:
                    #     count   += int(elemType['Nodes']) + 1
                    #     continue

                    # Filter cgnsBC to narrow the search space
                    # filtered_cgnsBC = [x for x in cgnsBC if cgnsSmin <= x <= cgnsSmax]

                    # Check if corners can form a subset of cgnsBC
                    corners_set = set(corners)
                    if corners_set.issubset(cgns_set):
                        BCpoints = [bpoints[s-1] for s in corners]
                        BCpoints = np.sort(BCpoints, axis=0)
                        BCpoints = BCpoints.flatten()

                        # Use regex to check if the string ends with _<number> and split accordingly
                        match = re.match(r"(.*)_\d+$", zoneBC)
                        if match:
                            zoneBC = match.group(1)

                        cellsets = BCCGNS_SetBC(BCpoints, cellsets, nConnLen, nConnNum, offsetcs, stree, tol, zoneBC)
                        del BCpoints

                    count   += int(elemType['Nodes']) + 1

            elif cgnsGridLoc == 'FaceCenter':
                count    = 0

                # Loop over all elements and get the type
                while count < cgnsShells.shape[0]:
                    elemType = ElemTypes(cgnsShells[count])
                    nNodes   = int(elemType['Nodes'])

                    corners  = cgnsShells[count+1:count+nNodes+1]
                    # corners  = sorted(int(s) for s in corners)

                    if nShells in cgnsBC:
                        BCpoints = [bpoints[s-1] for s in corners]

                        # For high-order elements, we only consider the 3/4 corner nodes
                        match len(BCpoints):
                            case 6:   # triangle6
                                BCpoints = BCpoints[:3]
                            case 8:   # quad8
                                BCpoints = BCpoints[:4]
                            case 9:   # quad9:
                                BCpoints = BCpoints[:4]

                        BCpoints = np.sort(BCpoints, axis=0)
                        BCpoints = BCpoints.flatten()

                        # Use regex to check if the string ends with _<number> and split accordingly
                        match = re.match(r"(.*)_\d+$", zoneBC)
                        if match:
                            zoneBC = match.group(1)

                        match len(BCpoints):
                            case 9:   # triangle
                                cellsets = BCCGNS_SetBC(BCpoints, cellsets, tConnLen, tConnNum, offsetcs, ttree, tol, zoneBC)
                            case 12:  # quad
                                cellsets = BCCGNS_SetBC(BCpoints, cellsets, nConnLen, nConnNum, offsetcs, stree, tol, zoneBC)
                            case _:
                                hopout.warning('Unsupported number of corners for shell elements, exiting...')
                                sys.exit(1)

                        del BCpoints

                    nShells += 1
                    count   += int(elemType['Nodes']) + 1

        mesh   = meshio.Mesh(points    = points,    # noqa: E251
                             cells     = cells,     # noqa: E251
                             cell_sets = cellsets)  # noqa: E251

        mesh_vars.mesh = mesh


def BCCGNS_Structured(mesh:     meshio.Mesh,
                      points:   np.ndarray,
                      cells:    list,
                      stree:    spatial.KDTree,
                      zone,     # CGNS zone
                      tol:      float,
                      offsetcs: int,
                      nConnNum: int,
                      nConnLen: int) -> None:
    """ Set the CGNS boundary conditions for (un)curved (structured) grids
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    # from pyhope.io.io_cgns import ElemTypes
    # ------------------------------------------------------
    # Load the zone BCs
    zoneBCs = zone['ZoneBC']

    for zoneBC in zoneBCs:
        try:
            cgnsBC   = cast(h5py.Dataset, zone['ZoneBC'][zoneBC]['FamilyName'][' data'])
            cgnsName = ''.join(map(chr, cgnsBC))
        except KeyError:
            cgnsName = zoneBC.rpartition('_')[0]

        # Ignore internal DEFAULT BCs
        if 'DEFAULT' in cgnsName:
            continue

        try:
            cgnsPointRange = zone['ZoneBC'][zoneBC]['PointRange'][' data']
            cgnsPointRange = np.array(cgnsPointRange, dtype=int) - 1
            # Sanity check the CGNS point range
            if any(cgnsPointRange[1, :] - cgnsPointRange[0, :] < 0):
                hopout.warning(f'Point range is not monotonically increasing on BC "{cgnsName}", exiting...')
                sys.exit(1)

            # Calculate the ranges of the indices
            iStart, iEnd = cgnsPointRange[:, 0]
            jStart, jEnd = cgnsPointRange[:, 1]
            kStart, kEnd = cgnsPointRange[:, 2]

            # Load the grid coordinates
            iCoords = np.array(zone['GridCoordinates']['CoordinateX'][' data'])
            jCoords = np.array(zone['GridCoordinates']['CoordinateY'][' data'])
            kCoords = np.array(zone['GridCoordinates']['CoordinateZ'][' data'])

            # Slice the grid
            xSurf = iCoords[kStart:kEnd+1, jStart:jEnd+1, iStart:iEnd+1].squeeze()
            ySurf = jCoords[kStart:kEnd+1, jStart:jEnd+1, iStart:iEnd+1].squeeze()
            zSurf = kCoords[kStart:kEnd+1, jStart:jEnd+1, iStart:iEnd+1].squeeze()

            # Dimensions of the surface grid
            iDim, jDim = xSurf.shape

            # Check if the grid dimensions can be sliced
            if (iDim - 1) % mesh_vars.nGeo != 0 or (jDim - 1) % mesh_vars.nGeo != 0:
                raise ValueError(f"Grid dimensions ({iDim}, {jDim}) are not divisible by the agglomeration factor {mesh_vars.nGeo}")

            # Slice the grid for agglomeration
            xSurfNGeo = xSurf[::mesh_vars.nGeo, ::mesh_vars.nGeo]
            ySurfNGeo = ySurf[::mesh_vars.nGeo, ::mesh_vars.nGeo]
            zSurfNGeo = zSurf[::mesh_vars.nGeo, ::mesh_vars.nGeo]

            # Updated dimensions after agglomeration
            iDimNGeo, jDimNGeo = xSurfNGeo.shape

            # Generate quads for the agglomerated grid
            quads = []
            for j in range(iDimNGeo - 1):
                for k in range(jDimNGeo - 1):
                    # Define the quad by its four corner points
                    quads.append([(xSurfNGeo[j    , k    ], ySurfNGeo[j    , k    ], zSurfNGeo[j    , k    ]),
                                  (xSurfNGeo[j + 1, k    ], ySurfNGeo[j + 1, k    ], zSurfNGeo[j + 1, k    ]),
                                  (xSurfNGeo[j + 1, k + 1], ySurfNGeo[j + 1, k + 1], zSurfNGeo[j + 1, k + 1]),
                                  (xSurfNGeo[j    , k + 1], ySurfNGeo[j    , k + 1], zSurfNGeo[j    , k + 1]),])
        except KeyError:
            hopout.warning(f'ZoneBC "{zoneBC}" does not have a PointRange. PointLists are currently not supported.')
            sys.exit(1)

        # Convert to numpy array if needed
        quads = np.array(quads)

        # Loop over all elements
        cellsets = mesh.cell_sets
        for quad in quads:
            # elemType = ElemTypes(cgnsBC[count])

            # Map the unique quad sides to our non-unique elem sides
            BCpoints = quad
            BCpoints = np.sort(BCpoints, axis=0)
            BCpoints = BCpoints.flatten()
            cellsets = BCCGNS_SetBC(BCpoints, cellsets, nConnLen, nConnNum, offsetcs, stree, tol, cgnsName)

            # Move to the next element
            # count += int(elemType['Nodes']) + 1

        mesh   = meshio.Mesh(points    = points,    # noqa: E251
                             cells     = cells,     # noqa: E251
                             cell_sets = cellsets)  # noqa: E251

        mesh_vars.mesh = mesh
