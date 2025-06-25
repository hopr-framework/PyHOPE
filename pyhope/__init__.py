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
from collections import namedtuple
from contextlib import contextmanager
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
from pyhope.basis.basis_basis import legendre_gauss_nodes, legendre_gauss_lobatto_nodes
from pyhope.basis.basis_basis import barycentric_weights, polynomial_derivative_matrix
from pyhope.basis.basis_basis import lagrange_interpolation_polys, calc_vandermonde
from pyhope.basis.basis_basis import change_basis_3D, change_basis_2D
from pyhope.basis.basis_basis import evaluate_jacobian
# ==================================================================================================================================


class Basis:
    """ Basis class to hold all basis related functions and variables
    """
    legendre_gauss_nodes         = staticmethod(legendre_gauss_nodes)
    legendre_gauss_lobatto_nodes = staticmethod(legendre_gauss_lobatto_nodes)
    barycentric_weights          = staticmethod(barycentric_weights)
    polynomial_derivative_matrix = staticmethod(polynomial_derivative_matrix)
    lagrange_interpolation_polys = staticmethod(lagrange_interpolation_polys)
    calc_vandermonde             = staticmethod(calc_vandermonde)
    change_basis_3D              = staticmethod(change_basis_3D)
    change_basis_2D              = staticmethod(change_basis_2D)
    evaluate_jacobian            = staticmethod(evaluate_jacobian)


# Define a named tuple to hold the mesh data
MeshContainer = namedtuple('Mesh',
                          ['mesh',   # The generated mesh object
                           'nGeo',   # Polynomial order
                           'bcs',    # Boundary conditions
                           'elems',  # Elements
                           'sides'   # Sides
                          ])


@contextmanager
def Mesh(*args, stdout=False, stderr=True):
    """ Mesh context manager to generate a mesh from a given file

        Args:
            *args: The mesh file path(s) to be processed
            stdout (bool): If False, standard output is suppressed
            stderr (bool): If False, standard error  is suppressed

        Yields:
            Mesh: An object containing the generated mesh and its properties
    """
    # Standard libraries -----------------------------------
    import os
    # Third-party libraries --------------------------------
    import h5py
    from contextlib import redirect_stdout, redirect_stderr, ExitStack
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common import DefineCommon, InitCommon
    from pyhope.io.io import DefineIO, InitIO
    from pyhope.mesh.connect.connect import ConnectMesh
    from pyhope.mesh.mesh import DefineMesh, InitMesh, GenerateMesh
    from pyhope.mesh.mesh_sides import GenerateSides
    from pyhope.readintools.readintools import DefineConfig, ReadConfig
    # ------------------------------------------------------

    try:
        # Check if the arguments provided are valid mesh files
        if not args:
            raise ValueError('No mesh file provided.')

        for arg in args:
            # Check if the argument is a valid file path
            if not os.path.isfile(arg):
                raise FileNotFoundError(f'Mesh file not found: {arg}')

            # Check if the argument is a valid HDF5 file
            if not h5py.is_hdf5(arg):
                raise ValueError(f'Mesh file not a valid HDF5 file: {arg}')

        # Suppress output to standard output
        with ExitStack() as stack:
            with open(os.devnull, 'w') as null:
                if not stdout:
                    stack.enter_context(redirect_stdout(null))
                if not stderr:
                    stack.enter_context(redirect_stderr(null))

                # Perform the reduced PyHOPE initialization
                with DefineConfig() as dc:
                    config.prms = dc
                    DefineCommon()
                    DefineIO()
                    DefineMesh()

                with ReadConfig(args[0]) as rc:
                    config.params = rc

                # Read-in required parameters
                InitCommon()
                InitIO()
                InitMesh()

                # Generate the actual mesh
                GenerateMesh()

                # Build our data structures
                GenerateSides()
                ConnectMesh()

        # Export mesh variables
        mesh = mesh_vars.mesh

        nGeo  = mesh_vars.nGeo
        bcs   = mesh_vars.bcs

        elems = mesh_vars.elems
        sides = mesh_vars.sides

        yield MeshContainer(mesh  = mesh,   # noqa: E251
                            nGeo  = nGeo,   # noqa: E251
                            bcs   = bcs,    # noqa: E251
                            elems = elems,  # noqa: E251
                            sides = sides   # noqa: E251
                           )

    finally:
        # Cleanup resources after exiting the context
        mesh_vars.mesh  = None
        mesh_vars.nGeo  = None
        mesh_vars.bcs   = None
        mesh_vars.elems = None
        mesh_vars.sides = None
