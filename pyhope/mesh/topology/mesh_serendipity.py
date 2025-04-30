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
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Monkey-patching MeshIO
# ==================================================================================================================================


def convertSerendipityToFullLagrange(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Some GMSH files contain incomplete elements, fix them here
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.common.common_tools import allocate_or_resize
    from pyhope.mesh.elements.elements_shapefunctions import ShapeFunctions
    from pyhope.mesh.elements.elements_ordering import ElementInfo
    # ------------------------------------------------------

    # Check the mesh contains second-order incomplete elements
    serendipityElems = ['quad8', 'hexahedron20', 'wedge15', 'pyramid13']
    if not any(s for s in mesh.cells_dict.keys() if s in serendipityElems):
        return mesh

    hopout.routine('Converting serendipity to full langrange mesh')

    # Copy original points
    points    = mesh.points.copy()

    # Prepare new cell blocks
    elems_new = {}
    shapefunctions = ShapeFunctions()
    elementinfo    = ElementInfo()

    for cell in mesh.cells:
        ctype, cdata = cell.type, cell.data

        # Valid cell type
        if ctype not in serendipityElems:
            elems_new[ctype] = cdata
            continue

        match ctype:
            # Remove surface elements
            case 'quad8':
                continue

            case 'hexahedron20':

                # Get number of hexahedrons which have to be converted
                nHex20    = len(cdata)

                faces     = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')
                nFaces    = len(faces)

                N         = [np.array(()) for _ in range(nFaces + 1)]
                faceNodes = [list()       for _ in faces]  # noqa: E272

                # preallocate the arrays for the new points and elements
                nPoints_old = len(points)
                nNewPoints  = (nFaces + 1) * nHex20
                points      = np.resize(points, (nPoints_old + nNewPoints, 3))

                # Allocate arrays if they do not exist. Else, resize them
                (elems_new, hex27_start) = allocate_or_resize(elems_new, 'hexahedron27', (nHex20,   27))
                (elems_new, quad9_start) = allocate_or_resize(elems_new, 'quad9',        (nHex20*6,  9))

                for iFace, face in enumerate(faces):
                    # Face parameters are the same as for the 27-node hexahedron
                    xi, eta, zeta    = elementinfo.faces_to_params('hexahedron20')[face]
                    faceNodes[iFace] = elementinfo.faces_to_nodes( 'hexahedron20')[face]

                    # Evaluate the quadratic shape function at the face center
                    N[iFace] = shapefunctions.evaluate(ctype, xi, eta, zeta)
                # Append the center node
                N[-1] = shapefunctions.evaluate(ctype, 0, 0, 0)

                # Loop over all hexahedrons
                for iElem, elem in enumerate(cdata):
                    # Create the 6 face mid-points
                    for iFace, face in enumerate(faces):
                        center = np.dot(N[iFace], mesh.points[elem])
                        points[nPoints_old + iFace, :]  = center

                        # Take the existing 8 face nodes and append the new center node
                        subFace = elem[faceNodes[iFace][:8]].tolist()
                        subFace.append(nPoints_old + iFace)
                        elems_new['quad9'][quad9_start + iElem * nFaces + iFace] = np.array(subFace, dtype=np.uint)

                    # Evaluate the quadratic shape function at the volume center
                    center = np.dot(N[-1], mesh.points[elem])
                    points[nPoints_old + nFaces, :] = center

                    # Create the volume element
                    subElem = elem.tolist()

                    # Append the 6 face center and the volume center
                    subElem.extend(range(nPoints_old, nPoints_old + nFaces + 1))
                    elems_new['hexahedron27'][hex27_start + iElem] = np.array(subElem, dtype=np.uint)

                    # Increment counter with number of added points
                    nPoints_old += nFaces + 1

            case 'wedge15':

                # Get number of hexahedrons which have to be converted
                nWed15    = len(cdata)

                faces     = ('y-', 'x+', 'x-')  # square faces of element
                # faces     = ['x+']
                nFaces    = len(faces)

                N         = [np.array(()) for _ in range(nFaces + 1)]
                faceNodes = [list()       for _ in faces]  # noqa: E272

                # preallocate the arrays for the new points and elements
                nPoints_old = len(points)
                nNewPoints  = nFaces * nWed15
                points      = np.resize(points, (nPoints_old + nNewPoints, 3))

                # Allocate arrays if they do not exist. Else, resize them
                (elems_new, wed18_start) = allocate_or_resize(elems_new, 'wedge18',   (nWed15,   18))
                (elems_new, quad9_start) = allocate_or_resize(elems_new, 'quad9',     (nWed15*3,  9))

                for iFace, face in enumerate(faces):
                    if 'z' in face:
                        continue

                    # Face parameters are the same as for the 9-node quad
                    xi, eta, zeta    = (0., 0., 0.)
                    faceNodes[iFace] = elementinfo.faces_to_nodes('wedge15')[face]

                    # Evaluate the quadratic shape function at the face center
                    N[iFace] = shapefunctions.evaluate('quad8', xi, eta, zeta)

                # Loop over all hexahedrons
                for iElem, elem in enumerate(cdata):
                    # Create the 3 face mid-points
                    for iFace, face in enumerate(faces):
                        # 2nd order triangular faces are already present in the mesh
                        # Only for sanity as faces are already excluded from "faces" array
                        if 'z' in face:
                            continue
                        # 2nd order quadrilaterial faces
                        elif len(faceNodes[iFace]) == 8:
                            # Here, we are on the quads and not the actual element
                            center = np.dot(N[iFace], mesh.points[elem[faceNodes[iFace]]])
                            points[nPoints_old + iFace, :]  = center

                            # Take the existing 8 face nodes and append the new center node
                            subFace = elem[faceNodes[iFace][:8]].tolist()
                            subFace.append(nPoints_old + iFace)
                            elems_new['quad9'][quad9_start + iElem * nFaces + iFace] = np.array(subFace, dtype=np.uint)

                    # Create the volume element
                    subElem = elem.tolist()

                    # Append the 3 face center
                    subElem.extend(range(nPoints_old, nPoints_old + nFaces))
                    elems_new['wedge18'][wed18_start + iElem] = np.array(subElem, dtype=np.uint)

                    # Increment counter with number of added points
                    nPoints_old += nFaces

            case 'pyramid13':
                # Get number of hexahedrons which have to be converted
                nPyra13   = len(cdata)

                face      = 'z-'  # square faces of element
                faceNodes = elementinfo.faces_to_nodes('pyramid13')[face]

                # Preallocate the arrays for the new points and elements
                nPoints_old = len(points)
                nNewPoints = nPyra13  # Since there's only one face per element
                points = np.resize(points, (nPoints_old + nNewPoints, 3))

                # Allocate arrays if they do not exist. Else, resize them
                (elems_new, wed18_start) = allocate_or_resize(elems_new, 'pyramid14', (nPyra13, 14))
                (elems_new, quad9_start) = allocate_or_resize(elems_new, 'quad9',     (nPyra13,  9))

                # Face parameters for the 9-node quad
                xi, eta, zeta = (0., 0., 0.)
                N = shapefunctions.evaluate('quad8', xi, eta, zeta)

                # Loop over all hexahedrons
                for iElem, elem in enumerate(cdata):
                    # Create the face mid-point
                    if len(faceNodes) == 8:
                        # Here, we are on the quads and not the actual element
                        center = np.dot(N, mesh.points[elem[faceNodes]])
                        points[nPoints_old, :]  = center

                        # Take the existing 8 face nodes and append the new center node
                        subFace = elem[faceNodes[:8]].tolist()
                        subFace.append(nPoints_old)
                        elems_new['quad9'][quad9_start + iElem] = np.array(subFace, dtype=np.uint)

                    # Create the volume element
                    subElem = elem.tolist()

                    # Append the 3 face center
                    subElem.append(nPoints_old)
                    elems_new['pyramid14'][wed18_start + iElem] = np.array(subElem, dtype=np.uint)

                    # Increment counter with number of added points
                    nPoints_old += 1

    # At this point, the mesh does not contain boundary conditions / cell sets
    # > We add them in the BCCGNS function
    mesh   = meshio.Mesh(points = points,     # noqa: E251
                         cells  = elems_new)  # noqa: E251

    # hopout.sep()
    # hopout.info('CONVERTING SERENDIPITY TO FULL LANGRANGE MESH DONE!')
    # hopout.sep()

    return mesh
