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
import os
import unittest
import time
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Typing libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import typing
if typing.TYPE_CHECKING:
    from io import StringIO
    from unittest import TestCase
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# from pyhope.basis.basis_basis import *
# ==================================================================================================================================


class TestOutput(unittest.TextTestResult):
    """ Custom unit test output that routes output through hopout
    """
    def __init__(self, stream: StringIO, descriptions: bool, verbosity: int) -> None:
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().__init__(stream, descriptions, verbosity)
        self.hopout = hopout

    def addSuccess(self, test: TestCase) -> None:
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addSuccess(test)
        self.hopout.printtest(test._testMethodName.split("test_")[1], hopout.Symbols.OK)

    def addFailure(self, test: TestCase, err: tuple) -> None:
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addFailure(test, err)
        self.hopout.printtest(test._testMethodName.split("test_")[1], hopout.Symbols.WARN)

    def addError(self, test: TestCase, err: tuple) -> None:
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addError(test, err)
        self.hopout.printtest(test._testMethodName.split("test_")[1], hopout.Symbols.ERR)


class TestLibraryMethods(unittest.TestCase):
    """ Unit test class for library methods
    """
    def test_legendre_gauss_nodes(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import legendre_gauss_nodes
        # ------------------------------------------------------
        nodes, weights = legendre_gauss_nodes(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([-0.77459666924148340    ,  0.                     ,  0.77459666924148340   ]))  # noqa: E501
        np.testing.assert_array_almost_equal(weights, np.array([ 0.55555555555555569    ,  0.88888888888888884    ,  0.55555555555555569   ]))  # noqa: E501

    def test_legendre_gauss_lobatto_nodes(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import legendre_gauss_lobatto_nodes
        # ------------------------------------------------------
        nodes, weights = legendre_gauss_lobatto_nodes(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([-1.                     , -3.06161699786838240e-17, 1.                     ]))  # noqa: E501
        np.testing.assert_array_almost_equal(weights, np.array([ 0.33333333333333331    , 1.33333333333333326     , 0.33333333333333331    ]))  # noqa: E501

    def test_equi_nodes_prism(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_prism
        # ------------------------------------------------------
        nodes = equi_nodes_prism(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. ,  1. , -1. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. , -1. ,  0. ,  0. ,  1. ],    # noqa: E501
                                                                [-1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ]]))  # noqa: E501

    def test_equi_nodes_pyram(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_pyram
        # ------------------------------------------------------
        nodes = equi_nodes_pyram(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. ,  1. , -1. ,  0. ,  1. , -1. ,  0. , -1. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  0. ,  1. ,  1. ,  1. , -1. , -1. ,  0. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  0. ,  1. ]]))  # noqa: E501

    def test_equi_nodes_tetra(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_tetra
        # ------------------------------------------------------
        nodes = equi_nodes_tetra(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. , -1. , -1. ],
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. ,  0. , -1. ],
                                                                [-1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  1. , ]]))

    def test_barycentric_weights(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights
        # ------------------------------------------------------
        weights = barycentric_weights(3, np.linspace(-1, 1, num=3, dtype=np.float64))
        np.testing.assert_array_almost_equal(weights, np.array([ 0.5                    , -1.                     , 0.5                    ]))  # noqa: E501

    def test_polynomial_derivative_matrix(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import polynomial_derivative_matrix
        # ------------------------------------------------------
        deriv = polynomial_derivative_matrix(3,       np.array([-0.77459666924148340    ,  0.                     ,  0.77459666924148340   ]))   # noqa: E501
        np.testing.assert_array_almost_equal(deriv,   np.array([[-1.93649167310370851   ,  2.58198889747161120    , -0.64549722436790280   ],    # noqa: E501
                                                                [-0.64549722436790280   ,  0.                     ,  0.64549722436790280   ],    # noqa: E501
                                                                [ 0.64549722436790280   , -2.58198889747161120    ,  1.93649167310370851   ]]))  # noqa: E501

    def test_change_basis_1D(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_1D
        # ------------------------------------------------------
        # Interpolating f(x) = x^2 from 5 equidistant nodes to 3 nodes must be exact
        xIn    = np.linspace(-1, 1, num=5, dtype=np.float64)
        xOut   = np.linspace(-1, 1, num=3, dtype=np.float64)
        wBary  = barycentric_weights(5, xIn)
        Vdm    = calc_vandermonde(5, 3, wBary, xIn, xOut)
        fOut   = change_basis_1D(Vdm, xIn**2)
        np.testing.assert_array_almost_equal(fOut, xOut**2)

    def test_change_basis_2D(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_2D
        # ------------------------------------------------------
        # Interpolate f(x,y)=x+y from 5x5 grid to 3x3 grid using bilinear mapping
        n_In   = 5
        n_Out  = 3
        xIn    = np.linspace(-1, 1, num=n_In   , dtype=np.float64)
        xOut   = np.linspace(-1, 1, num=n_Out  , dtype=np.float64)
        wBary  = barycentric_weights(n_In, xIn)
        Vdm    = calc_vandermonde(n_In, n_Out, wBary, xIn, xOut)
        # Build f(xi, eta) = xi + eta on the input grid
        XI, ETA = np.meshgrid(xIn, xIn, indexing='ij')              # shape (1, n_In , n_In )
        f_in   = (XI + ETA)[np.newaxis, :, :]                       # shape (1, n_In , n_In )
        f_out  = change_basis_2D(Vdm, f_in)                         # shape (1, n_Out, n_Out)
        XI2, ETA2 = np.meshgrid(xOut, xOut, indexing='ij')
        np.testing.assert_array_almost_equal(f_out[0], XI2 + ETA2)

    def test_change_basis_3D(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_3D
        # ------------------------------------------------------
        # Interpolate f(x,y,z)=x+y+z from 5x5 grid to 3x3 grid using bilinear mapping
        n_In   = 3
        n_Out  = 2
        xIn    = np.linspace(-1, 1, num=n_In   , dtype=np.float64)
        xOut   = np.linspace(-1, 1, num=n_Out  , dtype=np.float64)
        wBary  = barycentric_weights(n_In, xIn)
        Vdm    = calc_vandermonde(n_In, n_Out, wBary, xIn, xOut)
        # Build f(xi, eta, zeta) = xi + eta + zeta on the input grid
        XI, ETA, ZETA = np.meshgrid(xIn, xIn, xIn, indexing='ij')   # shape (1, n_In , n_In , n_In )
        f_in   = (XI + ETA + ZETA)[np.newaxis, :, :, :]             # shape (1, n_In , n_In , n_In )
        f_out  = change_basis_3D(Vdm, f_in)                         # shape (1, n_Out, n_Out, n_Out)
        XI2, ETA2, ZETA2 = np.meshgrid(xOut, xOut, xOut, indexing='ij')
        np.testing.assert_array_almost_equal(f_out[0], XI2 + ETA2 + ZETA2)

    def test_lagrange_interpolation_polys(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights, lagrange_interpolation_polys
        # ------------------------------------------------------
        # Evaluated at the second node, the result must be the unit vector e_1
        xGP    = np.linspace(-1, 1, num=4, dtype=np.float64)
        wBary  = barycentric_weights(4, xGP)
        polys  = lagrange_interpolation_polys(xGP[1], 4, xGP, wBary)
        np.testing.assert_array_almost_equal(polys,   np.array([  0.,  1.,   0.,   0.]))  # noqa: E501

        # Sum of all Lagrange basis functions at an arbitrary point must be 1
        xGP    = np.linspace(-1, 1, num=5, dtype=np.float64)
        wBary  = barycentric_weights(5, xGP)
        polys  = lagrange_interpolation_polys(0.3, 5, xGP, wBary)
        np.testing.assert_almost_equal(np.sum(polys), 1.0)

    # def test_polynomial_derivative_matrix_prism(self):
    #     # Local imports ----------------------------------------
    #     from pyhope.basis.basis_basis import equi_nodes_prism, polynomial_derivative_matrix_prism
    #     # ------------------------------------------------------
    #     order  = 2
    #     xGP    = equi_nodes_prism(order)
    #     D      = polynomial_derivative_matrix_prism(order, xGP)
    #     # D must have shape (3, nDOFs, nDOFs)
    #     self.assertEqual(D.shape[0], 3)
    #     self.assertEqual(D.shape[1], D.shape[2])

    def test_polynomial_derivative_matrix_pyram(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_pyram, polynomial_derivative_matrix_pyram
        # ------------------------------------------------------
        order  = 2
        xGP    = equi_nodes_pyram(order)
        D      = polynomial_derivative_matrix_pyram(order, xGP)
        # D must have shape (3, nDOFs, nDOFs)
        self.assertEqual(D.shape[0], 3)
        self.assertEqual(D.shape[1], D.shape[2])
        np.testing.assert_array_almost_equal(D, np.array([[[-0.500, -0.500,  0.000,  0.000, -0.250,],
                                                           [ 0.500,  0.500,  0.000,  0.000,  0.250,],
                                                           [ 0.000,  0.000, -0.500, -0.500, -0.250,],
                                                           [ 0.000,  0.000,  0.500,  0.500,  0.250,],
                                                           [ 0.000,  0.000,  0.000,  0.000,  0.000,]],
 [                                                         [-0.500,  0.000, -0.500,  0.000, -0.250,],
                                                           [ 0.000, -0.500,  0.000, -0.500, -0.250,],
                                                           [ 0.500,  0.000,  0.500,  0.000,  0.250,],
                                                           [ 0.000,  0.500,  0.000,  0.500,  0.250,],
                                                           [ 0.000,  0.000,  0.000,  0.000,  0.000,]],
 [                                                         [-0.625, -0.375, -0.375, -0.125, -0.375,],
                                                           [ 0.125, -0.125, -0.125, -0.375, -0.125,],
                                                           [ 0.125, -0.125, -0.125, -0.375, -0.125,],
                                                           [-0.125,  0.125,  0.125,  0.375,  0.125,],
                                                           [ 0.500,  0.500,  0.500,  0.500,  0.500,]]]))

    def test_polynomial_derivative_matrix_tetra(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_tetra, polynomial_derivative_matrix_tetra
        # ------------------------------------------------------
        order  = 2
        xGP    = equi_nodes_tetra(order)
        D      = polynomial_derivative_matrix_tetra(order, xGP)
        # D must have shape (3, nDOFs, nDOFs)
        self.assertEqual(D.shape[0], 3)
        self.assertEqual(D.shape[1], D.shape[2])
        np.testing.assert_array_almost_equal(D, np.array([[[-0.500, -0.500, -0.500, -0.500],
                                                           [ 0.500,  0.500,  0.500,  0.500],
                                                           [ 0.000,  0.000,  0.000,  0.000],
                                                           [ 0.000,  0.000,  0.000,  0.000]],
                                                          [[-0.500, -0.500, -0.500, -0.500],
                                                           [-0.000, -0.000, -0.000, -0.000],
                                                           [ 0.500,  0.500,  0.500,  0.500],
                                                           [ 0.000,  0.000,  0.000,  0.000]],
                                                          [[-0.500, -0.500, -0.500, -0.500],
                                                           [-0.000, -0.000, -0.000, -0.000],
                                                           [ 0.000,  0.000,  0.000,  0.000],
                                                           [ 0.500,  0.500,  0.500,  0.500]]]))

    def test_calc_vandermonde(self) -> None:
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde
        # ------------------------------------------------------
        # When in- and out-nodes are identical the Vandermonde matrix must be the identity
        xEq    = np.linspace(-1, 1, num=4     , dtype=np.float64)
        wBary  = barycentric_weights(4, xEq)
        Vdm    = calc_vandermonde(4, 4, wBary, xEq, xEq)
        np.testing.assert_array_almost_equal(Vdm, np.eye(4))

        # Vandermonde built from equidistant nodes to coarser set reproduces coarser polynomial exactly
        xIn    = np.linspace(-1, 1, num=5     , dtype=np.float64)
        xOut   = np.linspace(-1, 1, num=3     , dtype=np.float64)
        wBary  = barycentric_weights(5, xIn)
        Vdm    = calc_vandermonde(5, 3, wBary, xIn, xOut)
        np.testing.assert_array_almost_equal(Vdm @ xIn, xOut)
        # Interpolating from nGeo+1 equidistant HOPR nodes to nGeo_mesh+1 equidistant meshIO nodes
        nGeo   = 3
        mGeo   = 2
        xEqHdf = np.linspace(-1, 1, num=nGeo+1, dtype=np.float64)
        xEqMes = np.linspace(-1, 1, num=mGeo+1, dtype=np.float64)
        wBary  = barycentric_weights(nGeo, xEqHdf)
        Vdm    = calc_vandermonde(nGeo+1, mGeo+1, wBary, xEqHdf, xEqMes)
        # Matrix shape must be (n_Out, n_In)
        self.assertEqual(Vdm.shape, (mGeo+1, nGeo+1))
        # Each row must sum to 1 (partition of unity)
        np.testing.assert_array_almost_equal(Vdm.sum(axis=1), np.ones(mGeo+1))


def CheckUnittest() -> None:
    """ Verify the installation by comparing against known results
    """
    # Third-party libraries --------------------------------
    from contextlib import redirect_stderr
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    hopout.small_banner('Verifying unittests')

    # Suppress output to standard output
    with open(os.devnull, 'w') as null, redirect_stderr(null):
        loader  = unittest.TestLoader()
        suite   = loader.loadTestsFromTestCase(TestLibraryMethods)
        runner  = unittest.TextTestRunner(verbosity=0, resultclass=TestOutput)

        t_start = time.perf_counter()
        result  = runner.run(suite)
        t_end   = time.perf_counter()

    # Summary line consistent with hopout style
    n_run   = result.testsRun
    n_fail  = len(result.failures) + len(result.errors)
    n_pass  = n_run - n_fail

    # Print failure and error details
    for test, traceback in result.failures + result.errors:
        hopout.separator()
        print(hopout.warn(f' {test._testMethodName.split("test_")[1]}'))
        hopout.sep()
        diff = traceback.split('AssertionError:')[-1].strip()
        for line in diff.splitlines():
            hopout.info(line)

    hopout.sep()
    hopout.info(f'Ran {n_run} tests in {t_end - t_start:.3f}s')
    hopout.small_banner(f'Results: {n_pass}/{n_run} passed')
