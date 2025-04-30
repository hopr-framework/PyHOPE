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
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def legendre_gauss_lobatto_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """ Return Legendre-Gauss-Lobatto nodes and weights for a given order in 1D
    """
    order -= 1
    # Special cases for small N
    if order == 1:
        return np.array((-1, 1)), np.array((1, 1))

    # Compute the initial guess for the LGL nodes (roots of P'_N)
    nodes = np.cos(np.pi * np.arange(order+1) / order)

    # Initialize the Legendre polynomial and its derivative
    p = np.zeros((order+1, order+1))

    # Iteratively solve for the LGL nodes using Newton's method
    xOld = 2 * np.ones_like(nodes)
    tol = 1e-14
    while np.max(np.abs(nodes - xOld)) > tol:
        xOld = nodes.copy()
        p[:, 0] = 1
        p[:, 1] = nodes
        for k in range(2, order+1):
            p[:, k] = ((2*k-1) * nodes * p[:, k-1] - (k-1) * p[:, k-2]) / k
        nodes -= (nodes * p[:, order] - p[:, order-1]) / (order * (p[:, order]))

    # The LGL nodes
    nodes = np.sort(nodes)

    # Compute the LGL weights
    weights = 2 / (order * (order + 1) * (p[:, order]**2))

    return nodes, weights


def legendre_gauss_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """ Return Legendre-Gauss nodes and weights for a given order in 1D
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def barycentric_weights(_: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the barycentric weights for a given node set
        > Algorithm 30, Kopriva
    """
    # Create a difference matrix (x_i - x_j) for all i, j
    diff_matrix = xGP[:, np.newaxis] - xGP[np.newaxis, :]

    # Set the diagonal to 1 to avoid division by zero (diagonal elements will not be used)
    np.fill_diagonal(diff_matrix, 1.0)

    # Compute the product of all differences for each row (excluding the diagonal)
    wBary = np.prod(diff_matrix, axis=1)

    # Take the reciprocal to get the final barycentric weights
    wBary = 1.0 / wBary

    return wBary


def polynomial_derivative_matrix(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the polynomial derivative matrix for a given node set
        > Algorithm 37, Kopriva
    """
    wBary = barycentric_weights(order, xGP)
    D     = np.zeros((order, order), dtype=float)

    for iLagrange in range(order):
        for iGP in range(order):
            if iLagrange != iGP:
                D[iGP, iLagrange] = wBary[iLagrange]/(wBary[iGP]*(xGP[iGP]-xGP[iLagrange]))
                D[iGP, iGP      ] = D[iGP, iGP] - D[iGP, iLagrange]

    return D


def lagrange_interpolation_polys(x: float, order: int, xGP: np.ndarray, wBary: np.ndarray) -> np.ndarray:
    """ Computes all Lagrange functions evaluated at position x in [-1;1]
        > Algorithm 34, Kopriva
    """
    # Equal points need special treatment
    lagrange = np.zeros(order)
    for iGP in range(order):
        if abs(x - xGP[iGP]) < 1.E-14:
            lagrange[iGP] = 1
            return lagrange

    tmp = 0.
    for iGP in range(order):
        lagrange[iGP] = wBary[iGP] / (x-xGP[iGP])
        tmp   += lagrange[iGP]

    # Normalize
    lagrange = lagrange/tmp
    return lagrange


def calc_vandermonde(n_In: int, n_Out: int, wBary_In: np.ndarray, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
    """ Build a 1D Vandermonde matrix using the Lagrange basis functions of degree N_In,
        evaluated at the interpolation points xi_Out
    """
    Vdm = np.zeros((n_Out, n_In))
    for iXI in range(n_Out):
        Vdm[iXI, :] = lagrange_interpolation_polys(xi_Out[iXI], n_In, xi_In, wBary_In)
    return Vdm


# def change_basis_3D(dim1: int, n_In: int, n_Out: int, Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
def change_basis_3D(Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
    """ Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
        to another 3D tensor product node positions (number of nodes N_out+1)
        defined by (N_out+1) interpolation point  positions xi_Out(0:N_Out)
        xi is defined in the 1DrefElem xi=[-1,1]
    """
    # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x3D_In)
    x3D_Buf1 = np.tensordot(Vdm, x3D_In, axes=(1, 1))
    x3D_Buf1 = np.moveaxis(x3D_Buf1, 0, 1)  # Correct the shape to (dim1, n_Out, n_In, n_In)

    # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x3D_Buf1)
    x3D_Buf2 = np.tensordot(Vdm, x3D_Buf1, axes=(1, 2))
    x3D_Buf2 = np.moveaxis(x3D_Buf2, 0, 2)  # Correct the shape to  (dim1, n_Out, n_Out, n_In)

    # Third contraction along the kN_In axis (axis 1 of Vdm, axis 3 of x3D_Buf2)
    x3D_Out  = np.tensordot(Vdm, x3D_Buf2, axes=(1, 3))
    x3D_Out  = np.moveaxis(x3D_Out , 0, 3)  # Correct the shape to (dim1, n_Out, n_Out, n_Out)

    return x3D_Out


def change_basis_2D(Vdm: np.ndarray, x2D_In: np.ndarray) -> np.ndarray:
    """ Interpolate a 2D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
        to another 2D tensor product node positions (number of nodes N_out+1)
        defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
        xi is defined in the 1D reference element xi=[-1,1]
    """
    # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x2D_In)
    x2D_Buf1 = np.tensordot(Vdm, x2D_In, axes=(1, 1))
    x2D_Buf1 = np.moveaxis(x2D_Buf1, 0, 1)  # Correct the shape to (dim1, n_Out, n_In, n_In)

    # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x2D_Buf1)
    x2D_Out = np.tensordot(Vdm, x2D_Buf1, axes=(1, 2))
    x2D_Out = np.moveaxis(x2D_Out, 0, 2)  # Correct the shape to  (dim1, n_Out, n_Out, n_In)

    return x2D_Out


def evaluate_jacobian(xGeo_In: np.ndarray, VdmGLtoAP: np.ndarray, D_EqToGL: np.ndarray) -> np.ndarray:
    # Perform tensor contraction for the first derivative (Xi direction)
    dXdXiGL   = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 1))
    dXdXiGL   = np.moveaxis(dXdXiGL  , 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Perform tensor contraction for the second derivative (Eta direction)
    dXdEtaGL  = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 2))
    dXdEtaGL  = np.moveaxis(dXdEtaGL , 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Perform tensor contraction for the third derivative (Zeta direction)
    dXdZetaGL = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 3))
    dXdZetaGL = np.moveaxis(dXdZetaGL, 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)

    # Change basis for each direction
    dXdXiAP   = change_basis_3D(VdmGLtoAP, dXdXiGL  )
    dXdEtaAP  = change_basis_3D(VdmGLtoAP, dXdEtaGL )
    dXdZetaAP = change_basis_3D(VdmGLtoAP, dXdZetaGL)

    # Precompute cross products between dXdEtaAP and dXdZetaAP for all points
    cross_eta_zeta = np.cross(dXdEtaAP, dXdZetaAP, axis=0)  # Shape: (3, nGeoRef, nGeoRef, nGeoRef)

    # Fill output Jacobian array
    jacOut = np.einsum('ijkl,ijkl->jkl', dXdXiAP, cross_eta_zeta)

    return jacOut


# INFO: ALTERNATIVE VERSION, CACHING VDM, D
# class JacobianEvaluator:
#     def __init__(self, VdmGLtoAP: np.ndarray, D_EqToGL: np.ndarray) -> None:
#         self.VdmGLtoAP: Final[np.ndarray] = VdmGLtoAP
#         self.D_EqToGL:  Final[np.ndarray] = D_EqToGL
#
#     def evaluate_jacobian(self, xGeo_In: np.ndarray) -> np.ndarray:
#         # Perform tensor contraction for the first derivative (Xi direction)
#         dXdXiGL   = np.tensordot(self.D_EqToGL, xGeo_In, axes=(1, 1))
#         dXdXiGL   = np.moveaxis(dXdXiGL  , 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)
#
#         # Perform tensor contraction for the second derivative (Eta direction)
#         dXdEtaGL  = np.tensordot(self.D_EqToGL, xGeo_In, axes=(1, 2))
#         dXdEtaGL  = np.moveaxis(dXdEtaGL , 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)
#
#         # Perform tensor contraction for the third derivative (Zeta direction)
#         dXdZetaGL = np.tensordot(self.D_EqToGL, xGeo_In, axes=(1, 3))
#         dXdZetaGL = np.moveaxis(dXdZetaGL, 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)
#
#         # Change basis for each direction
#         dXdXiAP   = change_basis_3D(self.VdmGLtoAP, dXdXiGL  )
#         dXdEtaAP  = change_basis_3D(self.VdmGLtoAP, dXdEtaGL )
#         dXdZetaAP = change_basis_3D(self.VdmGLtoAP, dXdZetaGL)
#
#         # Precompute cross products between dXdEtaAP and dXdZetaAP for all points
#         cross_eta_zeta = np.cross(dXdEtaAP, dXdZetaAP, axis=0)  # Shape: (3, nGeoRef, nGeoRef, nGeoRef)
#
#         # Fill output Jacobian array
#         jacOut = np.einsum('ijkl,ijkl->jkl', dXdXiAP, cross_eta_zeta)
#
#         return jacOut
