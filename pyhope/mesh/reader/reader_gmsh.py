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
import gc
import os
import re
import shutil
import subprocess
import time
from typing import Final, Optional, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
import meshio
import numpy as np
from scipy.spatial import KDTree
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Monkey-patching MeshIO
meshio._mesh.topological_dimension.update({'wedge15'   : 3,  # ty: ignore [unresolved-attribute]
                                           'pyramid13' : 3,
                                           'pyramid55' : 3})
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
                # 49: '.neu',   # Cubit/Gambit reader is broken beyond repair
                50: '.matlab'}

    # get file extension
    _, ext = os.path.splitext(file)
    return ext in ioFormat.values()


def ReadGMSH(fnames: list) -> meshio.Mesh:
    # Third-party libraries --------------------------------
    import gmsh
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common import IsDisplay
    from pyhope.common.common_vars import np_mtp
    from pyhope.io.io_vars import debugvisu
    from pyhope.mesh.topology.mesh_serendipity import convertSerendipityToFullLagrange
    from pyhope.meshio.meshio_convert import gmsh_to_meshio
    from pyhope.readintools.readintools import GetLogical
    # ------------------------------------------------------

    hopout.sep()
    gmsh.initialize()

    # Setup multiprocessing
    numThreads = np_mtp if np_mtp > 0 else 1
    gmsh.option.setNumber('General.NumThreads',   numThreads)
    gmsh.option.setNumber('Geometry.OCCParallel', 1 if np_mtp > 0 else 0)

    # Setup mesh factory
    # gmsh.option.setString('SetFactory', 'OpenCascade')

    # Setup debug visualization
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('Geometry.Tolerance'         , 1e-12)  # default: 1e-6
        gmsh.option.setNumber('Geometry.MatchMeshTolerance', 1e-09)  # default: 1e-8

    for fname in fnames:
        # Get file extension
        _, ext = os.path.splitext(fname)

        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        # gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 0)

        # If not GMSH format convert
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

    if debugvisu and IsDisplay():
        gmsh.fltk.run()

    # Sanity check if the mesh contains volume elements
    # > User might have modified the mesh inside the FLTK GUI
    gmsh_elems = np.asarray((gmsh.option.getNumber('Mesh.NbTetrahedra'),
                             gmsh.option.getNumber('Mesh.NbPrisms'    ),
                             gmsh.option.getNumber('Mesh.NbPyramids'  ),
                             gmsh.option.getNumber('Mesh.NbHexahedra')), dtype=int)
    if not np.any(gmsh_elems):
        hopout.error('Generated mesh does not contain volume elements, exiting...')

    # Convert Gmsh object to meshio object
    mesh = gmsh_to_meshio(gmsh)

    # Check whether the mesh contains high-order elements and nGeo is set to 1
    if not mesh_vars.already_curved or mesh_vars.nGeo == 1:
        for elemtype in mesh.cells_dict.keys():
            if elemtype in mesh_vars.ELEMTYPE.name and mesh_vars.ELEMTYPE.name[elemtype] > 200:
                hopout.error('High-order elements detected in the mesh but MeshIsAlreadyCurved=F or nGeo is set to 1, exiting...')

    # If the mesh contains second-order incomplete elements, fix them
    mesh = convertSerendipityToFullLagrange(mesh)

    # Finally done with GMSH, finalize
    gmsh.finalize()

    # Convert BC names to lower case
    mesh.cell_sets = {k.lower(): v for k, v in mesh.cell_sets.items()}

    # Run garbage collector to release memory
    gc.collect()

    return mesh


def BCCGNS(mesh: meshio.Mesh, fnames: list) -> meshio.Mesh:
    """ Some CGNS files setup their boundary conditions in a different way than gmsh expects
        > Add them here manually to the meshIO object
    """
    # Standard libraries -----------------------------------
    import tempfile
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    # ------------------------------------------------------

    hopout.routine('Applying boundary conditions')
    # hopout.sep()

    points  = mesh.points
    cells   = mesh.cells
    # elems   = mesh_vars.elems
    # sides   = mesh_vars.sides
    # bcs     = mesh_vars.bcs

    # All non-connected sides (technically all) are potential BC sides
    # nConnSide = [s for s in sides if 'Connection' not in s and 'BCID' not in s]
    cells_lst = tuple(mesh.cells_dict)

    # Now, for quadrilateral elements
    nConnLen = 0
    nConnNum = 0
    stree    = None

    if any('quad' in key for key in mesh.cells_dict):
        nConnSide = [value for key, value in mesh.cells_dict.items() if 'quad' in key][0]
        nConnType = [key   for key, _     in mesh.cells_dict.items() if 'quad' in key][0]  # noqa: E272, E501
        nConnNum  = cells_lst.index(nConnType)
        nConnLen  = len(cells_lst)

        # Collapse all opposing corner nodes into an [:, 12] array
        # nbCorners  = [s['Corners'] for s in nConnSide]
        nbCorners = [s[0:4] for s in nConnSide]
        # Calculate the centroid for each face (3D point)
        nbCenters = np.mean(mesh.points[nbCorners], axis=1)
        del nbCorners

        # Build a k-dimensional tree of all face centroids on the opposing side
        stree = KDTree(nbCenters)

    # Now, the same thing for triangular elements
    tConnLen  = 0
    tConnNum  = 0
    ttree     = None

    if any('triangle' in key for key in mesh.cells_dict):
        tConnSide = [value for key, value in mesh.cells_dict.items() if 'triangle' in key][0]
        tConnType = [key   for key, _     in mesh.cells_dict.items() if 'triangle' in key][0]  # FIXME: Support mixed LO/HO meshes  # noqa: E272, E501
        tConnNum  = cells_lst.index(tConnType)
        tConnLen  = len(cells_lst)

        # Collapse all opposing corner nodes into an [:, 9] array
        tbCorners = [s[0:3] for s in tConnSide]
        # Calculate the centroid for each face (3D point)
        tbCenters = np.mean(mesh.points[tbCorners], axis=1)
        del tbCorners

        # Build a k-dimensional tree of all face centroids
        ttree = KDTree(tbCenters)

    tol: Final[float] = mesh_vars.tolExternal

    # Now set the missing CGNS boundaries
    for fname in fnames:

        # Create a temporary directory and keep it existing until manually cleaned
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tname = tfile.name
        # Try to convert the file automatically
        if not h5py.is_hdf5(fname):
            hopout.sep()
            hopout.info('File {} is not in HDF5 CGNS format, converting ...'.format(os.path.basename(fname)))
            tStart = time.time()
            _ = subprocess.run([f'adf2hdf {fname} {tname}'], check=True, shell=True, stdout=subprocess.DEVNULL)
            tEnd   = time.time()
            hopout.info('File {} converted HDF5 CGNS format [{:.2f} sec]'.format(os.path.basename(fname), tEnd - tStart))

            # Rest of this code operates on the converted file
            fname = tname
        else:
            # Alternatively, load the file directly into tmpfs for faster access
            shutil.copyfile(fname, tname)

        with h5py.File(fname, mode='r') as f:
            if 'CGNSLibraryVersion' not in f.keys():
                hopout.error('CGNS file does not contain library version header')

            key = [s for s in f.keys() if "base" in s.lower()]
            match len(key):
                case 0:
                    hopout.error('Object [Base] does not exist in CGNS file')
                case 1:
                    if not isinstance(f[key[0]], h5py.Group):
                        hopout.error('Object [Base] is not a group in CGNS file')
                    base = cast(h5py.Group, f[key[0]])
                case _:
                    hopout.error('More than one object [Base] exists in CGNS file')

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
                        mesh = BCCGNS_Unstructured(mesh, points, cells, stree, zone, tol, nConnNum, nConnLen,  # noqa: E501
                                                   # Support for triangular elements
                                                   ttree, tConnNum, tConnLen)
                    case 3:  # Structured 3D mesh, 3D arrays
                        # Structured grid can only contain tensor-product elements
                        mesh = BCCGNS_Structured(mesh, points, cells, stree, zone, tol, nConnNum, nConnLen)
                    case _:  # Unsupported number of dimensions
                        # raise ValueError('Unsupported number of dimensions')
                        hopout.error('Unsupported number of dimensions')

        # Cleanup temporary file
        if tfile is not None:
            os.unlink(tfile.name)

    # Run garbage collector to release memory
    gc.collect()
    hopout.sep()

    return mesh


def BCCGNS_Unstructured(  mesh:     meshio.Mesh,
                          points:   np.ndarray,
                          cells:    list,
                          stree:    Optional[KDTree],
                          zone,     # CGNS zone
                          tol:      float,
                          nConnNum: int,
                          nConnLen: int,
                          # Triangular elements
                          ttree:    Optional[KDTree],
                          tConnNum: int,
                          tConnLen: int) -> meshio.Mesh:
    """ Set the CGNS boundary conditions for uncurved (unstructured) grids
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.io.formats.cgns import ElemTypes
    # ------------------------------------------------------
    # Load the CGNS points
    bpoints = np.column_stack([zone['GridCoordinates'][f'Coordinate{axis}'][' data'][:].astype(float) for axis in 'XYZ'])

    # Loop over all BCs
    zoneBCs  = [s for s in cast(h5py.Group, zone['ZoneBC']).keys() if s.strip() != 'innerfaces']
    cellsets = mesh.cell_sets
    # Convert the cellsets to a list of lists for easier manipulation
    for k, v in cellsets.items():
        cellsets[k] = list(map(lambda cell: cell.tolist() if isinstance(cell, (np.ndarray, np.generic)) else cell, v))

    for zoneBC in zoneBCs:
        # Lists to collect centroids
        quadCenters = []
        triaCenters = []

        # Data given with separate zoneBCs
        if zoneBC in zone:
            cgnsBC = cast(h5py.Dataset, zone[zoneBC]['ElementConnectivity'][' data'])

            # Read the surface elements, one at a time
            count  = 0

            # Loop over all elements and get the type
            while count < cgnsBC.shape[0]:
                elemType = ElemTypes(cgnsBC[count])

                # Map the unique quad sides to our non-unique elem sides
                nNodes   = int(elemType['Nodes'])
                corners  = cgnsBC[count+1:count+nNodes+1]
                # Calculate centroid from corner points and add to list
                quadCenters.append(np.mean(bpoints[corners - 1], axis=0))

                # Move to the next element
                count += nNodes + 1

        # Data attached to the zoneBC node
        elif f'{zoneBC}/PointList' in zone['ZoneBC']:
            cgnsBC = sorted(int(s) for s in zone['ZoneBC'][zoneBC]['PointList'][' data'])

            # Identify how surface elements are stored
            surface_key = 'GridShells' if 'GridShells' in zone else 'SurfaceElements' if 'SurfaceElements' in zone else None
            if not surface_key:
                hopout.error('Format of BC implementation for FaceCenters not recognized, exiting...')

            cgnsShells  =     zone[surface_key]['ElementConnectivity'][' data']
            nShells     = int(zone[surface_key]['ElementRange'][' data'][0])

            # Get the location of the BC faces
            cgnsGridLoc = bytes(zone['ZoneBC'][zoneBC]['GridLocation'][' data']).decode('ascii')
            cgns_set    = set(cgnsBC)

            # Read the surface elements, one at a time
            count   = 0

            # Loop over all elements and collect centroids
            while count < cgnsShells.shape[0]:
                elemType = ElemTypes(cgnsShells[count])
                nNodes   = int(elemType['Nodes'])

                corners  = cgnsShells[count+1:count+nNodes+1]

                if cgnsGridLoc == 'Vertex':
                    # Check if corners can form a subset of cgnsBC
                    corners_set = set(int(s) for s in corners)
                    if corners_set.issubset(cgns_set):
                        BCpoints = bpoints[[s-1 for s in corners]]
                        quadCenters.append(np.mean(BCpoints, axis=0))
                    count += nNodes + 1

                elif cgnsGridLoc == 'FaceCenter':
                    if nShells in cgnsBC:
                        BCpoints = bpoints[[s-1 for s in corners]]

                        # For high-order elements, we only consider the 3/4 corner nodes for the centroid
                        match len(BCpoints):
                            case 3 | 6:       # triangle, triangle6
                                triaCenters.append(np.mean(BCpoints[:3], axis=0))
                            case 4 | 8 | 9:   # quad, quad8, quad9
                                quadCenters.append(np.mean(BCpoints[:4], axis=0))
                            case _:
                                hopout.error('Unsupported number of corners for shell elements, exiting...')

                    nShells += 1
                    count   += nNodes + 1

        # Use regex to check if the string ends with _<number> and split accordingly
        match  = re.match(r'(.*)_\d+$', zoneBC)
        bcName = match.group(1) if match else zoneBC
        bcName = bcName.lower()

        # Process quads
        if quadCenters and stree is not None:
            distances, indices = cast(tuple[np.ndarray, np.ndarray], stree.query(np.array(quadCenters)))
            if np.any(distances > tol):
                hopout.error(f'Could not find all boundary sides within tolerance {tol} for BC "{bcName}", exiting...',
                             traceback=True)

            if bcName not in cellsets:
                cellsets[bcName] = [[] for _ in range(nConnLen)]
            cast(list, cellsets[bcName][nConnNum]).extend(indices.tolist())

        # Process triangles
        if triaCenters and ttree is not None:
            distances, indices = cast(tuple[np.ndarray, np.ndarray], ttree.query(np.array(triaCenters)))
            if np.any(distances > tol):
                hopout.error(f'Could not find all boundary sides within tolerance {tol} for BC "{bcName}", exiting...',
                             traceback=True)

            if bcName not in cellsets:
                cellsets[bcName] = [[] for _ in range(tConnLen)]
            cast(list, cellsets[bcName][tConnNum]).extend(indices.tolist())

    # Convert the cellsets back to a dictionary
    csets = {}
    for k, v in cellsets.items():
        csets[k] = [np.array(s, dtype=int) for s in v]

    mesh   = meshio.Mesh(points    = points,    # noqa: E251
                         cells     = cells,     # noqa: E251
                         cell_sets = csets)     # noqa: E251

    return mesh


def BCCGNS_Structured(mesh:     meshio.Mesh,
                      points:   np.ndarray,
                      cells:    list,
                      stree:    Optional[KDTree],
                      zone,     # CGNS zone
                      tol:      float,
                      nConnNum: int,
                      nConnLen: int) -> meshio.Mesh:
    """ Set the CGNS boundary conditions for (un)curved (structured) grids
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    # from pyhope.io.io_cgns import ElemTypes
    # ------------------------------------------------------
    # Loop over all BCs
    cellsets = mesh.cell_sets
    # Convert the cellsets to a list of lists for easier manipulation
    for k, v in cellsets.items():
        cellsets[k] = list(map(lambda cell: cell.tolist() if isinstance(cell, (np.ndarray, np.generic)) else cell, v))

    # Load the zone BCs
    for zoneBC, bcData in zone['ZoneBC'].items():
        try:
            cgnsBC   = bcData['FamilyName'][' data']
            cgnsName = ''.join(map(chr, cgnsBC)).lower()
        except KeyError:
            cgnsName = zoneBC.split('_', 1)[0].lower()

        # Ignore internal DEFAULT BCs
        if 'DEFAULT' in cgnsName:
            continue

        try:
            cgnsPointRange = np.array(bcData['PointRange'][' data'], dtype=int) - 1
            # Sanity check the CGNS point range
            if any(cgnsPointRange[1, :] - cgnsPointRange[0, :] < 0):
                hopout.error(f'Point range is not monotonically increasing on BC "{cgnsName}", exiting...')

            # Calculate the ranges of the indices
            iStart, iEnd = cgnsPointRange[:, 0]
            jStart, jEnd = cgnsPointRange[:, 1]
            kStart, kEnd = cgnsPointRange[:, 2]

            # Load the grid coordinates
            coords = {axis: np.array(zone['GridCoordinates'][f'Coordinate{axis}'][' data']) for axis in 'XYZ'}

            # Slice the grid
            slices = (slice(kStart, kEnd + 1), slice(jStart, jEnd + 1), slice(iStart, iEnd + 1))
            xSurf, ySurf, zSurf = (coords[axis][slices].squeeze() for axis in 'XYZ')

            # Dimensions of the surface grid
            iDim, jDim = xSurf.shape

            # Check if the grid dimensions can be sliced
            if (iDim - 1) % mesh_vars.nGeo != 0 or (jDim - 1) % mesh_vars.nGeo != 0:
                raise ValueError(f"Grid dimensions ({iDim}, {jDim}) are not divisible by the agglomeration factor {mesh_vars.nGeo}")

            # Slice the grid for agglomeration
            step = mesh_vars.nGeo
            xSurfNGeo, ySurfNGeo, zSurfNGeo = (arr[::step, ::step] for arr in (xSurf, ySurf, zSurf))

            # Updated dimensions after agglomeration
            iDimNGeo, jDimNGeo = xSurfNGeo.shape

            # Generate quads for the agglomerated grid
            # Define the quad by its four corner points
            quads = np.array([[(xSurfNGeo[j    , k    ], ySurfNGeo[j    , k    ], zSurfNGeo[j    , k    ]),
                               (xSurfNGeo[j + 1, k    ], ySurfNGeo[j + 1, k    ], zSurfNGeo[j + 1, k    ]),
                               (xSurfNGeo[j + 1, k + 1], ySurfNGeo[j + 1, k + 1], zSurfNGeo[j + 1, k + 1]),
                               (xSurfNGeo[j    , k + 1], ySurfNGeo[j    , k + 1], zSurfNGeo[j    , k + 1]) ] for j in range(iDimNGeo - 1)    # noqa: E501
                                                                                                             for k in range(jDimNGeo - 1)])  # noqa: E501

        except KeyError:
            hopout.error(f'ZoneBC "{zoneBC}" does not have a PointRange. PointLists are currently not supported.')

        if quads.size == 0:
            continue

        # Calculate centroids for all quads
        centers = np.mean(quads, axis=1)

        # Query the tree
        distances, indices = cast(tuple[np.ndarray, np.ndarray], stree.query(centers))
        if np.any(distances > tol):
            hopout.error(f'Could not find all boundary sides within tolerance {tol} for BC "{cgnsName}", exiting...',
                         traceback=True)

        # Update cellsets
        bcName = cgnsName.lower()
        if bcName not in cellsets:
            cellsets[bcName] = [[] for _ in range(nConnLen)]

        # Append all found side indices to the correct list
        cast(list, cellsets[bcName][nConnNum]).extend(indices.tolist())

    # Convert the cellsets back to a dictionary
    csets = {}
    for k, v in cellsets.items():
        csets[k] = [np.array(s, dtype=int) for s in v]

    mesh   = meshio.Mesh(points    = points,    # noqa: E251
                         cells     = cells,     # noqa: E251
                         cell_sets = csets)     # noqa: E251

    return mesh
