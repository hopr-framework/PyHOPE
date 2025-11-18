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
import pathlib
from typing import Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
from meshio import Mesh
from meshio._common import write_xml
from meshio._exceptions import WriteError
from xml.etree import ElementTree as ET
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# Monkey-patching meshio.xdmf.main.XdmfWriter
# > We want to be able to write multiple grids into the same HDF5 file
def XdmfWriterInit(self,
                   filename        : str,
                   meshes          : Union[Mesh, list[Mesh]],
                   data_format     : str = 'HDF',
                   compression     : str = 'gzip',
                   compression_opts: int = 4) -> None:

    if data_format not in ['XML', 'Binary', 'HDF']:
        raise WriteError(f'Unknown XDMF data format "{data_format}" (use "XML", "Binary", or "HDF")')

    self.filename         = pathlib.Path(filename)
    self.data_format      = data_format
    self.data_counter     = 0
    self.compression      = compression
    self.compression_opts = None if compression is None else compression_opts

    if data_format == 'HDF':
        self.h5_filename = self.filename.with_suffix('.h5')
        self.h5_file     = h5py.File(self.h5_filename, 'w')

    # Create the XDMF file base
    xdmf_file = ET.Element('Xdmf', Version='3.0')

    # Create one domain but multiple grids
    domain = ET.SubElement(xdmf_file, 'Domain')
    # grid   = ET.SubElement(domain   , 'Grid'  , Name='Grid')

    # Convert a singular mesh to a list, so we can loop
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    # Explicit type hint that meshes: list[Mesh]
    meshList: list[Mesh] = meshes

    for mesh in meshList:
        # Assign the correct subgrid name
        if mesh.info is not None and 'name' in mesh.info.keys():
            gridname = mesh.info['name']
        else:
            gridname = 'Grid'

        grid = ET.SubElement(domain, 'Grid', Name=gridname)

        self.write_points(    grid           , mesh.points)
        # self.field_data(      mesh.field_data, information)
        self.write_cells(     mesh.cells     , grid)
        self.write_point_data(mesh.point_data, grid)
        self.write_cell_data( mesh.cell_data , grid)

        ET.register_namespace('xi', 'https://www.w3.org/2001/XInclude/')

        write_xml(filename, xdmf_file)


# @cache
# def xdmfElems(elemType: int, nGeo: int) -> tuple[str, int]:
#     """ XDMF: Get the element type from the HOPR format
#     - Linear
#         - Polyvertex - a group of unconnected points
#         - Polyline - a group of line segments
#         - Polygon
#         - Triangle
#         - Quadrilateral
#         - Tetrahedron
#         - Pyramid
#         - Wedge
#         - Hexahedron
#     - Quadratic
#         - Edge_3 - Quadratic line with 3 nodes
#         - Tri_6
#         - Quad_8
#         - Tet_10
#         - Pyramid_13
#         - Wedge_15
#         - Hex_20
#     - Arbitrary
#         - 1 - POLYVERTEX
#         - 2 - POLYLINE
#         - 3 - POLYGON
#         - 4 - TRIANGLE
#         - 5 - QUADRILATERAL
#         - 6 - TETRAHEDRON
#         - 7 - PYRAMID
#         - 8 - WEDGE
#         - 9 - HEXAHEDRON
#         - 16 - POLYHEDRON
#         - 34 - EDGE_3
#         - 35 - QUADRILATERAL_9
#         - 36 - TRIANGLE_6
#         - 37 - QUADRILATERAL_8
#         - 38 - TETRAHEDRON_10
#         - 39 - PYRAMID_13
#         - 40 - WEDGE_15
#         - 41 - WEDGE_18
#         - 48 - HEXAHEDRON_20
#         - 49 - HEXAHEDRON_24
#         - 50 - HEXAHEDRON_27
#     """
#     elem_map = { 4: 'TETRAHEDRON',
#                  5: 'PYRAMID',
#                  6: 'WEDGE',
#                  8: 'HEXAHEDRON'
#                }
#
#     if elemType % 10 not in elem_map:
#         raise ValueError(f'Error in face_to_edge: elemType {elemType} is not supported')
#
#     match elemType % 10:
#         case 4:  # Tetrahedron
#             ndofs = np.floor((nGeo+1)*(nGeo+2)*(nGeo+3  )/6).astype(int)
#         case 5:  # Pyramid
#             ndofs = np.floor((nGeo+1)*(nGeo+2)*(2*nGeo+3)/6).astype(int)
#         case 6:  # Prism / Wedge
#             ndofs = np.floor((nGeo+1)**2      *(nGeo+2  )/2).astype(int)
#         case 8:  # Hexahedron
#             ndofs = (nGeo+1)**3
#         case _:
#             raise ValueError('Unknown element type')
#
#     if nGeo == 1:
#         return elem_map[elemType % 10]                , int(ndofs)
#     else:
#
#         return elem_map[elemType % 10] + f'_{ndofs:d}', int(ndofs)
#
#
# def xdmfCreate(f:           h5py.File,
#                ElemInfo:    np.ndarray,
#                xdmfConnect: tuple[list[int]]) -> None:
#     # Local imports ----------------------------------------
#     import pyhope.output.output as hopout
#     from pyhope.mesh.mesh_vars import nGeo
#     # ------------------------------------------------------
#
#     # Currently, only first/second order is supported by XDMF
#     if nGeo > 2:
#         print(hopout.warn('XDMF only supports linear/quadratic elements, skipping...'))
#         return None
#
#     hname = f.filename
#     h5nam, _ = os.path.splitext(hname)
#     fname = '{}.xmf'.format(h5nam)
#
#     # Start assembling the XDMF data
#     xdmf = [
#         '<?xml version="1.0" ?>',
#         '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" >',
#         '<Xdmf Version="2.0">',
#         '  <Domain>',
#         '    <Grid Name="Mesh" GridType="Collection" CollectionType="Spatial">'
#     ]
#
#     # Find number of unique element types
#     elemTypes  = np.unique(ElemInfo[:, 0])
#     nElemTypes = np.zeros(elemTypes.size, dtype=int)
#
#     # Currently, pyramids are only supported with first order
#     if any(elemTypes % 10 == 5) and nGeo > 1:
#         print(hopout.warn('XDMF only supports linear pyramidal elements, skipping...'))
#         return None
#
#     # Count the number of elements per type
#     for elem in ElemInfo:
#         elemType = np.where(elemTypes == elem[0])
#         nElemTypes[elemType] += 1
#
#     # Create the arrays for each dataset
#     for elemNum, elemType in enumerate(elemTypes):
#         nElems          = nElemTypes[elemNum]
#         elemName, nDofs = xdmfElems(elemType, nGeo)
#         nNodes          = nElems*nDofs
#
#         xdmf += [
#             f'      <Grid Name="{elemName}" GridType="Uniform">',
#             f'        <Topology TopologyType="{elemName}" NumberOfElements="{nElems}">',
#             f'          <DataItem Format="HDF" Dimensions="{nElems} {nDofs}" NumberType="Int" Precision="4">',
#             f'            {hname}:/ElemConnectivity{elemName}',
#             '           </DataItem>',
#             '         </Topology>',
#             '        <Geometry GeometryType="XYZ">',
#             f'          <DataItem Format="HDF" Dimensions="{nNodes} 3" NumberType="Float" Precision="8">',
#             f'            {hname}:/NodeCoords',
#             '           </DataItem>',
#             '        </Geometry>',
#             '      </Grid>'
#         ]
#
#         # Write the data to the HDF5 file
#         _ = f.create_dataset(f'ElemConnectivity{elemName}'   , data=np.array(xdmfConnect[elemNum], dtype=int))
#
#     xdmf += [
#         '    </Grid>',
#         '  </Domain>',
#         '</Xdmf>'
#     ]
#
#     hopout.routine('Writing XDMF data to "{}"'.format(fname))
#
#     # Write the XDMF file
#     with open(fname, 'w') as x:
#         x.write('\n'.join(xdmf))
