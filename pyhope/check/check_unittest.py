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
import os
import unittest
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


class TestOutput(unittest.TextTestResult):
    """ Custom unit test output that routes output through hopout
    """
    def __init__(self, stream, descriptions, verbosity):
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().__init__(stream, descriptions, verbosity)
        self.hopout = hopout

    def addSuccess(self, test):
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addSuccess(test)
        self.hopout.info(f'{hopout.Symbols.OK  :<9} │ {test._testMethodName.split("test_")[1]}')

    def addFailure(self, test, err):
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addFailure(test, err)
        self.hopout.info(f'{hopout.Symbols.WARN:<9} │ {test._testMethodName.split("test_")[1]}')

    def addError(self, test, err):
        # Local imports ----------------------------------------
        import pyhope.output.output as hopout
        # ------------------------------------------------------
        super().addError(test, err)
        self.hopout.info(f'{hopout.Symbols.ERR :<9} │ {test._testMethodName.split("test_")[1]}')


class TestLibraryMethods(unittest.TestCase):
    """ Unit test class for library methods
    """
    def test_legendre_gauss_nodes(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import legendre_gauss_nodes
        # ------------------------------------------------------
        nodes, weights = legendre_gauss_nodes(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([-0.77459666924148340    ,  0.                     ,  0.77459666924148340   ]))  # noqa: E501
        np.testing.assert_array_almost_equal(weights, np.array([ 0.55555555555555569    ,  0.88888888888888884    ,  0.55555555555555569   ]))  # noqa: E501

    def test_legendre_gauss_lobatto_nodes(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import legendre_gauss_lobatto_nodes
        # ------------------------------------------------------
        nodes, weights = legendre_gauss_lobatto_nodes(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([-1.                     , -3.06161699786838240e-17, 1.                     ]))  # noqa: E501
        np.testing.assert_array_almost_equal(weights, np.array([ 0.33333333333333331    , 1.33333333333333326     , 0.33333333333333331    ]))  # noqa: E501

    def test_equi_nodes_prism(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_prism
        # ------------------------------------------------------
        nodes = equi_nodes_prism(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. ,  1. , -1. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. , -1. ,  0. ,  0. ,  1. ],    # noqa: E501
                                                                [-1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ]]))  # noqa: E501

    def test_equi_nodes_pyram(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_pyram
        # ------------------------------------------------------
        nodes = equi_nodes_pyram(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. ,  1. , -1. ,  0. ,  1. , -1. ,  0. , -1. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  0. ,  1. ,  1. ,  1. , -1. , -1. ,  0. ,  0. , -1. ],    # noqa: E501
                                                                [-1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  0. ,  1. ]]))  # noqa: E501

    def test_equi_nodes_tetra(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import equi_nodes_tetra
        # ------------------------------------------------------
        nodes = equi_nodes_tetra(3)
        np.testing.assert_array_almost_equal(nodes,   np.array([[-1. ,  0. ,  1. , -1. ,  0. , -1. , -1. ,  0. , -1. , -1. ],
                                                                [-1. , -1. , -1. ,  0. ,  0. ,  1. , -1. , -1. ,  0. , -1. ],
                                                                [-1. , -1. , -1. , -1. , -1. , -1. ,  0. ,  0. ,  0. ,  1. , ]]))

    def test_barycentric_weights(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import barycentric_weights
        # ------------------------------------------------------
        weights = barycentric_weights(3, np.linspace(-1, 1, num=3, dtype=np.float64))
        np.testing.assert_array_almost_equal(weights, np.array([ 0.5                    , -1.                     , 0.5                    ]))  # noqa: E501

    def test_polynomial_derivative_matrix(self):
        # Local imports ----------------------------------------
        from pyhope.basis.basis_basis import polynomial_derivative_matrix
        # ------------------------------------------------------
        deriv = polynomial_derivative_matrix(3,       np.array([-0.77459666924148340    ,  0.                     ,  0.77459666924148340   ]))   # noqa: E501
        np.testing.assert_array_almost_equal(deriv,   np.array([[-1.93649167310370851   ,  2.58198889747161120    , -0.64549722436790280   ],    # noqa: E501
                                                                [-0.64549722436790280   ,  0.                     ,  0.64549722436790280   ],    # noqa: E501
                                                                [ 0.64549722436790280   , -2.58198889747161120    ,  1.93649167310370851   ]]))  # noqa: E501


def CheckUnittest() -> None:
    """ Verify the installation by comparing against known results
    """
    # Third-party libraries --------------------------------
    import time
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    hopout.small_banner('Verifying unittests')

    # Load and run all test methods
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromTestCase(TestLibraryMethods)
    runner  = unittest.TextTestRunner(verbosity=0, resultclass=TestOutput, stream=open(os.devnull, 'w'))

    t_start = time.perf_counter()
    result  = runner.run(suite)
    t_end   = time.perf_counter()

    # Summary line consistent with hopout style
    n_run   = result.testsRun
    n_fail  = len(result.failures) + len(result.errors)
    n_pass  = n_run - n_fail

    hopout.sep()
    hopout.info(f'Ran {n_run} tests in {t_end - t_start:.3f}s')
    hopout.small_banner(f'Results: {n_pass}/{n_run} passed')
