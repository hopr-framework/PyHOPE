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
from functools import cache
from typing import Final, Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
# from threadpoolctl import ThreadpoolController
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
from pyhope.common.common_numba import NUMBA_AVAILABLE
from pyhope.common.common_numba import jit, types
from pyhope.mesh.mesh_common import NDOFS_ELEM
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


@cache
def equi_nodes_prism(order: int) -> np.ndarray:
    """ Return equidistant nodes on a wedge/prism
    """
    xEq = np.linspace(-1, 1, num=order, dtype=np.float64)
    iZETA, iETA, iXI = np.indices((order, order, order))
    mask = iXI < (order - iETA)
    return np.vstack((xEq[iXI[mask]], xEq[iETA[mask]], xEq[iZETA[mask]]))


@cache
def equi_nodes_pyram(order: int) -> np.ndarray:
    """ Return equidistant nodes on a pyramid
    """
    xEq = np.linspace(-1, 1, num=order, dtype=np.float64)
    iZETA, iETA, iXI = np.indices((order, order, order))
    mask = (iXI < (order - iZETA)) & (iETA < (order - iZETA))
    return np.vstack((xEq[iXI[mask]], xEq[iETA[mask]], xEq[iZETA[mask]]))


@cache
def equi_nodes_tetra(order: int) -> np.ndarray:
    """ Return equidistant nodes on a tetrahedron
    """
    xEq = np.linspace(-1, 1, num=order, dtype=np.float64)
    iZETA, iETA, iXI = np.indices((order, order, order))
    mask = (iXI < (order - iETA - iZETA)) & (iETA < (order - iZETA))
    return np.vstack((xEq[iXI[mask]], xEq[iETA[mask]], xEq[iZETA[mask]]))


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
    # wBary = 1.0 / wBary

    return 1.0 / wBary


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


def lagrange_interpolation_polys(x: Union[float, np.ndarray], order: int, xGP: np.ndarray, wBary: np.ndarray) -> np.ndarray:
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
        tmp += lagrange[iGP]

    # Normalize
    # lagrange = lagrange/tmp

    return lagrange/tmp


def calc_vandermonde(n_In: int, n_Out: int, wBary_In: np.ndarray, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
    """ Build a 1D Vandermonde matrix using the Lagrange basis functions of degree N_In,
        evaluated at the interpolation points xi_Out
    """
    Vdm = np.zeros((n_Out, n_In))
    for iXI in range(n_Out):
        Vdm[iXI, :] = lagrange_interpolation_polys(xi_Out[iXI], n_In, xi_In, wBary_In)
    return Vdm


#  def compute_cols_prism(a, b, c, n):
#      cols = []
#      for iZETA in range(n):
#          fZETA = sp.special.jacobi(iZETA, 0, 0)(c)
#          for iETA in range(n):
#              fETA = sp.special.jacobi(iETA, 0, 0)(b)
#              for iXI in range(n - iETA):
#                  fXI = sp.special.jacobi(iXI, 2*iETA + 1, 0)(a)
#                  col = np.sqrt(2.) * fXI * fETA * fZETA * (1 - a) ** iETA
#                  cols.append(col)
#      return np.array(cols).T
#
#
#  def compute_cols_tetra(a, b, c, n):
#      cols = []
#      for iZETA in range(n):
#          for iETA in range(n - iZETA):
#              for iXI in range(n - iETA - iZETA):
#                  fXI   = sp.special.jacobi(iXI  , 0             , 0)(a)
#                  fETA  = sp.special.jacobi(iETA , 2*iETA + 1    , 0)(b)
#                  fZETA = sp.special.jacobi(iZETA, 2*(iETA+iXI)+2, 0)(c)
#                  col = 2.*np.sqrt(2.) * fXI * fETA * fZETA * (1 - b) ** iXI * (1 - c) ** (iXI+iETA)
#                  cols.append(col)
#      return np.array(cols).T
#
#
#  def calc_vandermonde_prism(n_In: int, n_Out: int, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
#      """ Build a 3D Vandermonde matrix using the PKD basis function,
#          evaluated at the interpolation points xi_Out (build of Jacobi polynomials of degree N_In)
#      """
#      a_in  = xi_In[0, :]
#      b_in  = np.where((1 - xi_In[0, :]) <= 1.e-12, -1.0, 2 * (1 + xi_In[1, :]) / (1. - xi_In[0, :] + 1.e-20) - 1.)
#      c_in  = xi_In[2, :]
#
#      a_out = xi_Out[0, :]
#      b_out = np.where((1 - xi_Out[0, :]) <= 1.e-12, -1.0, 2 * (1 + xi_Out[1, :]) / (1. - xi_Out[0, :] + 1.e-20) - 1.)
#      c_out = xi_Out[2, :]
#
#      if n_In >= n_Out:
#          Vdm_tmp = compute_cols_prism(a_in , b_in , c_in , n_In)
#          Vdm     = compute_cols_prism(a_out, b_out, c_out, n_In)
#          Vdm     = np.linalg.inv(Vdm_tmp) @ Vdm
#      else:
#          Vdm_tmp = compute_cols_prism(a_out, b_out, c_out, n_Out)
#          Vdm     = compute_cols_prism(a_in , b_in , c_in , n_Out)
#          Vdm     = Vdm @ np.linalg.inv(Vdm_tmp)
#
#      return Vdm
#
#
#  def calc_vandermonde_tetra(n_In: int, n_Out: int, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
#      """ Build a 3D Vandermonde matrix using the PKD basis function,
#          evaluated at the interpolation points xi_Out (build of Jacobi polynomials of degree N_In)
#      """
#      a_in  = np.where(np.abs(xi_In[1, :] + xi_In[2, :]) <= 1.e-12, -1.0, 2 * (1 + xi_In[0, :]) / (- xi_In[1, :] - xi_In[2, :] + 1.e-20) - 1.)  # noqa: E501
#      b_in  = np.where((1 - xi_In[2, :]) <= 1.e-12, -1.0, 2 * (1 + xi_In[1, :]) / (1. - xi_In[2, :] + 1.e-20) - 1.)
#      c_in  = xi_In[2, :]
#
#      a_out = np.where(np.abs(xi_Out[1, :] + xi_Out[2, :]) <= 1.e-12, -1.0, 2 * (1 + xi_Out[0, :]) / (- xi_Out[1, :] - xi_Out[2, :] + 1.e-20) - 1.)  # noqa: E501
#      b_out = np.where((1 - xi_Out[2, :]) <= 1.e-12, -1.0, 2 * (1 + xi_Out[1, :]) / (1. - xi_Out[2, :] + 1.e-20) - 1.)
#      c_out = xi_Out[2, :]
#
#      if n_In >= n_Out:
#          Vdm_tmp = compute_cols_tetra(a_in , b_in , c_in , n_In)
#          Vdm     = compute_cols_tetra(a_out, b_out, c_out, n_In)
#          Vdm     = np.linalg.inv(Vdm_tmp) @ Vdm
#      else:
#          Vdm_tmp = compute_cols_tetra(a_out, b_out, c_out, n_Out)
#          Vdm     = compute_cols_tetra(a_in , b_in , c_in , n_Out)
#          Vdm     = Vdm @ np.linalg.inv(Vdm_tmp)
#
#      return Vdm


def polynomial_derivative_matrix_prism(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the polynomial derivative matrix for a prism
    """
    a: Final[np.ndarray]  = xGP[0, :]
    b: Final[np.ndarray]  = np.where((1 - xGP[0, :]) <= 1.e-12, -1.0, 2 * (1 + xGP[1, :]) / (1. - xGP[0, :] + 1.e-20) - 1.)
    c: Final[np.ndarray]  = xGP[2, :]

    nDOFs = NDOFS_ELEM(106, order-1)
    Vdm = np.zeros((   nDOFs, nDOFs))
    Vr  = np.zeros((   nDOFs, nDOFs))
    Vs  = np.zeros((   nDOFs, nDOFs))
    Vt  = np.zeros((   nDOFs, nDOFs))
    D   = np.zeros((3, nDOFs, nDOFs))

    # Precompute required Jacobi polynomials and derivatives
    # fZETA(i) = P_i^(0,0)(c), dfZETA(i) = 0.5*(i+1)*P_{i-1}^{(1,1)}(c)
    fZETA_all  = np.array([          sp.special.eval_jacobi(i  , 0, 0, c) for i in range(order)])
    dfZETA_all = np.array([0.5*(i+1)*sp.special.eval_jacobi(i-1, 1, 1, c) for i in range(order)])
    # fETA(j) = P_j^(0,0)(b), dfETA(j) = 0.5*(j+1)*P_{j-1}^{(1,1)}(b)
    fETA_all   = np.array([          sp.special.eval_jacobi(j  , 0, 0, b) for j in range(order)])
    dfETA_all  = np.array([0.5*(j+1)*sp.special.eval_jacobi(j-1, 1, 1, b) for j in range(order)])

    jacobi_xi_polys  = [[sp.special.jacobi(i, 2*j + 1, 0) for i in range(order - j)] for j in range(order)]
    jacobi_xi_derivs = [[p.deriv() for p in row] for row in jacobi_xi_polys]

    # Precompute constants
    sqrt2    = np.sqrt(2.)
    base_pa  = 0.5*(1 - a)
    base_pb  = 0.5*(1 + b)

    pow1ma   = [np.ones_like(a)]  # (1 - a)**k
    for _ in range(1, order+1):
        pow1ma.append(pow1ma[-1] * (1 - a))

    iX = 0
    for iZETA in range(order):
        fZETA, dfZETA = fZETA_all[iZETA], dfZETA_all[iZETA]
        for iETA in range(order):
            fETA, dfETA = fETA_all[iETA], dfETA_all[iETA]
            # fXI(ix) = P_ix^(2*iETA+1, 0)(a)
            # dfXI(ix)= 0.5*(ix + 2*iETA + 2) * P_{ix-1}^{(2*iETA+2, 1)}(a), with dfXI(0)=0
            for iXI in range(order - iETA):
                fXI  = jacobi_xi_polys[ iETA][iXI](a)
                dfXI = jacobi_xi_derivs[iETA][iXI](a)

                # powfac = (1 - a) ** iETA
                powfac = pow1ma[iETA]

                # Fill Vdm columns
                Vdm[:, iX] = sqrt2 * fXI * fETA * fZETA * powfac

                # Vr term
                Vr[:, iX] = dfETA * fXI * fZETA
                if iETA > 0:
                    Vr[:, iX] *= (base_pa**(iETA-1))

                # Vs term
                Vs[:, iX] = dfETA * (fXI * base_pb) * fZETA
                if iETA > 0:
                    Vs[:, iX] *= (base_pa**(iETA-1))
                tmp = dfXI * (base_pa**iETA)
                if iETA > 0:
                    tmp -= 0.5*iETA*fXI*(base_pa**(iETA-1))
                Vs[:, iX] += fETA*tmp*fZETA

                # Vt term
                Vt[:, iX] = sqrt2*dfZETA*fETA*fXI*powfac

                # Final scaling
                scale = 2.**(iETA+0.5)
                Vr[:, iX] *= scale
                Vs[:, iX] *= scale

                iX += 1

    # PERF: Need to use threadpoolctl here as BLAS has issues with multiprocessing
    #       > Workaround by restricting to one thread during concurrent execution
    # with ThreadpoolController().limit(limits=1, user_api='blas'):
    #     sVdm = np.linalg.inv(Vdm)
    # D[1, :, :] = (Vr @ sVdm).T
    # D[0, :, :] = (Vs @ sVdm).T
    # D[2, :, :] = (Vt @ sVdm).T

    # PERF: Solve with LU factorization instead of explicitly inverting, disabling input matrix checks
    VT = Vdm.T
    lu, piv    = sp.linalg.lu_factor(VT,             check_finite=False)
    D[1, :, :] = sp.linalg.lu_solve((lu, piv), Vr.T, check_finite=False)
    D[0, :, :] = sp.linalg.lu_solve((lu, piv), Vs.T, check_finite=False)
    D[2, :, :] = sp.linalg.lu_solve((lu, piv), Vt.T, check_finite=False)

    return D


def polynomial_derivative_matrix_pyram(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the polynomial derivative matrix for a pyramid
    """
    a  = np.where((1 - xGP[2, :]) <= 1.e-12, -1.0, 2 * (1 + xGP[0, :]) / (1. - xGP[2, :] + 1.e-20) - 1.)
    b  = np.where((1 - xGP[2, :]) <= 1.e-12, -1.0, 2 * (1 + xGP[1, :]) / (1. - xGP[2, :] + 1.e-20) - 1.)
    c  = xGP[2, :]

    nDOFs = NDOFS_ELEM(105, order-1)
    Vdm = np.zeros((   nDOFs, nDOFs))
    Vr  = np.zeros((   nDOFs, nDOFs))
    Vs  = np.zeros((   nDOFs, nDOFs))
    Vt  = np.zeros((   nDOFs, nDOFs))
    D   = np.zeros((3, nDOFs, nDOFs))

    iX = 0
    for iZETA in range(order):
        for iETA in range(order - iZETA):
            for iXI in range(order - iZETA):
                # fXI  = P_{iXI }^{(0,0)}(a), dfXI  = 0.5*(iXI +1)*P_{iXI -1}^{(1,1)}(a)
                fXI    = sp.special.jacobi(iXI  , 0, 0)(a)
                dfXI   = sp.special.jacobi(iXI  , 0, 0).deriv()(a)

                # fETA = P_{iETA}^{(0,0)}(b), dfETA = 0.5*(iETA+1)*P_{iETA-1}^{(1,1)}(b)
                fETA   = sp.special.jacobi(iETA , 0, 0)(b)
                dfETA  = sp.special.jacobi(iETA , 0, 0).deriv()(b)

                # fZETA is the dependent direction: alpha = 2*(iXI+iETA)+2, beta = 0
                fZETA  = sp.special.jacobi(iZETA, 2*(iXI+iETA)+2, 0)(c)
                dfZETA = sp.special.jacobi(iZETA, 2*(iXI+iETA)+2, 0).deriv()(c)

                # Fill Vdm columns
                Vdm[:, iX] = 2 * fXI * fETA * fZETA * (1-c)**(iXI+iETA)

                # Vr term
                Vr[:, iX] = dfXI*fETA*fZETA
                if iXI+iETA > 0:
                    Vr[:, iX] *= 2*(1-c)**(iXI+iETA-1)

                # Vs term
                Vs[:, iX] = fXI*dfETA*fZETA
                if iXI+iETA > 0:
                    Vs[:, iX] *= 2*(1-c)**(iXI+iETA-1)

                # Vt term
                tmp = (1-c)**(iXI+iETA)*dfZETA
                if iXI+iETA > 0:
                    tmp -= (iETA+iXI)*(1-c)**(iETA+iXI-1)*fZETA
                Vt[:, iX] = 2*tmp*fXI*fETA
                Vt[:, iX] += (1+a)*Vr[:, iX] + (1+b)*Vs[:, iX]

                # Final scaling
                Vr[:, iX] *= 2
                Vs[:, iX] *= 2

                iX += 1

    # PERF: Need to use threadpoolctl here as BLAS has issues with multiprocessing
    #       > Workaround by restricting to one thread during concurrent execution
    # with ThreadpoolController().limit(limits=1, user_api='blas'):
    #     sVdm = np.linalg.inv(Vdm)
    # D[0, :, :] = (Vr @ sVdm).T
    # D[1, :, :] = (Vs @ sVdm).T
    # D[2, :, :] = (Vt @ sVdm).T

    # PERF: Solve with LU factorization instead of explicitly inverting, disabling input matrix checks
    VT = Vdm.T
    lu, piv    = sp.linalg.lu_factor(VT,             check_finite=False)
    D[0, :, :] = sp.linalg.lu_solve((lu, piv), Vr.T, check_finite=False)
    D[1, :, :] = sp.linalg.lu_solve((lu, piv), Vs.T, check_finite=False)
    D[2, :, :] = sp.linalg.lu_solve((lu, piv), Vt.T, check_finite=False)

    return D


def polynomial_derivative_matrix_tetra(order: int, xGP: np.ndarray) -> np.ndarray:
    """ Compute the polynomial derivative matrix for a tetra
    """
    a  = np.where(np.abs(xGP[1, :] + xGP[2, :]) <= 1.e-12, -1.0, 2 * (1 + xGP[0, :]) / (- xGP[1, :] - xGP[2, :] + 1.e-20) - 1.)
    b  = np.where((1 - xGP[2, :]) <= 1.e-12, -1.0, 2 * (1 + xGP[1, :]) / (1. - xGP[2, :] + 1.e-20) - 1.)
    c  = xGP[2, :]

    nDOFs = NDOFS_ELEM(104, order-1)
    Vdm = np.zeros((   nDOFs, nDOFs))
    Vr  = np.zeros((   nDOFs, nDOFs))
    Vs  = np.zeros((   nDOFs, nDOFs))
    Vt  = np.zeros((   nDOFs, nDOFs))
    D   = np.zeros((3, nDOFs, nDOFs))

    # Precompute constants
    sqrt2    = np.sqrt(2.)

    iX = 0
    for iZETA in range(order):
        for iETA in range(order - iZETA):
            for iXI in range(order - iETA - iZETA):
                # fXI(a), dfXI(a)
                fXI    = sp.special.jacobi(iXI  , 0             , 0)(a)
                dfXI   = sp.special.jacobi(iXI  , 0             , 0).deriv()(a)

                # fETA(b; alpha=2*iXI+1), dfETA(b)
                fETA   = sp.special.jacobi(iETA , 2*iXI+1       , 0)(b)
                dfETA  = sp.special.jacobi(iETA , 2*iXI+1       , 0).deriv()(b)

                # fZETA(c; alpha=2*(iXI+iETA)+2), dfZETA(c)
                fZETA  = sp.special.jacobi(iZETA, 2*(iXI+iETA)+2, 0)(c)
                dfZETA = sp.special.jacobi(iZETA, 2*(iXI+iETA)+2, 0).deriv()(c)

                # Fill Vdm columns
                Vdm[:, iX] = 2 * sqrt2 * fXI * fETA * fZETA * (1-b)**iXI * (1-c)**(iXI+iETA)

                # Vr term
                Vr[:, iX] = 2 * sqrt2 * dfXI * fETA * fZETA
                if iXI > 0:
                    Vr[:, iX] *= ((0.5*(1-b))**(iXI-1))
                if iXI+iETA > 0:
                    Vr[:, iX] *= ((0.5*(1-c))**(iXI+iETA-1))

                # Vs term
                tmp = (1-b)**iXI*dfETA
                if iXI > 0:
                    tmp -= iXI*(1-b)**(iXI-1)*fETA
                if iXI+iETA > 0:
                    tmp *= (1-c)**(iXI+iETA-1)
                tmp *= 2.*sqrt2*fXI*fZETA
                Vs[:, iX] = tmp + Vr[:, iX] * (1+a)

                # Vt term
                Vt[:, iX] = 2*(1+a)*Vr[:, iX] + (1+b)*tmp
                tmp = dfZETA*(1-c)**(iETA+iXI)
                if iXI+iETA > 0:
                    tmp -= (iXI+iETA)*(1-c)**(iETA+iXI-1)*fZETA
                tmp *= 2*sqrt2*fXI*fETA*(1-b)**iXI
                Vt[:, iX] += tmp

                Vr[:, iX] *= 4
                Vs[:, iX] *= 2
                iX += 1

    # PERF: Need to use threadpoolctl here as BLAS has issues with multiprocessing
    #       > Workaround by restricting to one thread during concurrent execution
    # with ThreadpoolController().limit(limits=1, user_api='blas'):
    #     sVdm = np.linalg.inv(Vdm)
    # D[0, :, :] = (Vr @ sVdm).T
    # D[1, :, :] = (Vs @ sVdm).T
    # D[2, :, :] = (Vt @ sVdm).T

    # PERF: Solve with LU factorization instead of explicitly inverting, disabling input matrix checks
    VT = Vdm.T
    lu, piv    = sp.linalg.lu_factor(VT,             check_finite=False)
    D[0, :, :] = sp.linalg.lu_solve((lu, piv), Vr.T, check_finite=False)
    D[1, :, :] = sp.linalg.lu_solve((lu, piv), Vs.T, check_finite=False)
    D[2, :, :] = sp.linalg.lu_solve((lu, piv), Vt.T, check_finite=False)

    return D


if not NUMBA_AVAILABLE:
    def change_basis_3D(Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
        """ Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
            to another 3D tensor product node positions (number of nodes N_out+1)
            defined by (N_out+1) interpolation point  positions xi_Out(0:N_Out)
            xi is defined in the 1D reference element xi=[-1,1]
        """
        # INFO: numpy-version, not compatible with numba
        # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x3D_In)
        # x3D_Buf1 = np.tensordot(Vdm, x3D_In , axes=(1, 1))
        # x3D_Buf1 = np.moveaxis(x3D_Buf1, 0, 1)  # Correct the shape to (dim1, n_Out, n_In, n_In)
        n_In     = Vdm.shape[1]
        X1       = x3D_In.reshape(x3D_In.shape[0], n_In, n_In * n_In)
        x3D_Buf1 = (Vdm @ X1).reshape(x3D_In.shape[0], Vdm.shape[0], n_In, n_In)  # (dim1, n_Out, n_In, n_In)

        # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x3D_Buf1)
        x3D_Buf2 = np.tensordot(Vdm, x3D_Buf1, axes=(1, 2))
        x3D_Buf2 = np.moveaxis(x3D_Buf2, 0, 2)  # Correct the shape to  (dim1, n_Out, n_Out, n_In)

        # Third contraction along the kN_In axis (axis 1 of Vdm, axis 3 of x3D_Buf2)
        # x3D_Out  = np.tensordot(Vdm, x3D_Buf2, axes=(1, 3))
        # x3D_Out  = np.moveaxis(x3D_Out , 0, 3)  # Correct the shape to (dim1, n_Out, n_Out, n_Out)
        # x3D_Out = x3D_Buf2 @ Vdm.T
        # PERF: This is actually slower than the individual contractions
        # x3D_Out  = np.einsum('pi,qj,rk,dijk->dpqr', Vdm, Vdm, Vdm, x3D_In, optimize=True)

        return x3D_Buf2 @ Vdm.T

else:
    @jit((types.float64[:, :, :, ::1])(types.float64[:, ::1], types.float64[:, :, :, :]), nopython=True, cache=True, nogil=True)
    def change_basis_3D(Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
        """ Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
            to another 3D tensor product node positions (number of nodes N_out+1)
            defined by (N_out+1) interpolation point  positions xi_Out(0:N_Out)
            xi is defined in the 1D reference element xi=[-1,1]
        """
        dim1, n_In, _, _ = x3D_In.shape
        n_Out = Vdm.shape[0]

        # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x3D_In)
        buf1 = np.zeros((dim1, n_Out, n_In, n_In), dtype=x3D_In.dtype)
        for d in range(dim1):
            for k in range(n_In):
                buf1[d, :, :, k] = Vdm @ np.ascontiguousarray(x3D_In[d, :, :, k])

        # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x3D_Buf1)
        buf2 = np.zeros((dim1, n_Out, n_Out, n_In), dtype=x3D_In.dtype)
        for d in range(dim1):
            for k in range(n_Out):
                buf2[d, k, :, :] = Vdm @ np.ascontiguousarray(buf1[d, k, :, :])

        # Third contraction along the kN_In axis (axis 1 of Vdm, axis 3 of x3D_Buf2)
        buf3_Flat = buf2.reshape(-1, n_In) @ Vdm.T

        return buf3_Flat.reshape(dim1, n_Out, n_Out, n_Out)


if not NUMBA_AVAILABLE:
    def change_basis_2D(Vdm: np.ndarray, x2D_In: np.ndarray) -> np.ndarray:
        """ Interpolate a 2D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
            to another 2D tensor product node positions (number of nodes N_out+1)
            defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
            xi is defined in the 1D reference element xi=[-1,1]
        """
        # INFO: numpy-version, not compatible with numba
        # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x2D_In)
        # x2D_Buf1 = np.tensordot(Vdm, x2D_In, axes=(1, 1))
        # x2D_Buf1 = np.moveaxis(x2D_Buf1, 0, 1)  # Correct the shape to (dim1, n_Out, n_In, n_In)
        x2D_Buf1 = Vdm @ x2D_In

        # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of x2D_Buf1)
        # x2D_Out = np.tensordot(Vdm, x2D_Buf1, axes=(1, 2))
        # x2D_Out = np.moveaxis(x2D_Out, 0, 2)  # Correct the shape to  (dim1, n_Out, n_Out, n_In)
        # x2D_Out = x2D_Buf1 @ Vdm.T
        # PERF: This is actually slower than the individual contractions
        # x2D_Out = np.einsum('pi,qj,dij->dpq', Vdm, Vdm, x2D_In, optimize=True)

        return x2D_Buf1 @ Vdm.T

else:
    @jit((types.float64[:, :, ::1])(types.float64[:, ::1], types.float64[:, :, :]), nopython=True, cache=True, nogil=True)
    def change_basis_2D(Vdm: np.ndarray, x2D_In: np.ndarray) -> np.ndarray:
        """ Interpolate a 2D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
            to another 2D tensor product node positions (number of nodes N_out+1)
            defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
            xi is defined in the 1D reference element xi=[-1,1]
        """
        dim1, n_In, _ = x2D_In.shape
        n_Out = Vdm.shape[0]

        # First contraction along the iN_In axis (axis 1 of Vdm, axis 1 of x2D_In)
        x2D_Flat = np.ascontiguousarray(x2D_In)
        buf1 = np.zeros((dim1, n_Out, n_In), dtype=x2D_In.dtype)
        for d in range(dim1):
            buf1[d] = Vdm @ x2D_Flat[d]

        # Second contraction along the jN_In axis (axis 1 of Vdm, axis 2 of buf1)
        # Result shape: (dim1, n_Out, n_Out)
        # Reshaping to 2D for a single GEMM call: (dim1 * n_Out, n_In) @ (n_In, n_Out)
        x2D_Out = buf1.reshape(-1, n_In) @ Vdm.T

        return x2D_Out.reshape(dim1, n_Out, n_Out)


def evaluate_jacobian(xGeo_In: np.ndarray, VdmGLtoAP: np.ndarray, D_EqToGL: np.ndarray) -> np.ndarray:
    """ Calculate the Jacobian of the mapping for a given element
    """
    dim1  = xGeo_In.shape[0]
    n_In  = xGeo_In.shape[1]
    n_Out = D_EqToGL.shape[0]

    # Perform tensor contraction for the first derivative (Xi direction)
    # dXdXiGL   = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 1))
    # dXdXiGL   = np.moveaxis(dXdXiGL  , 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)
    _X1     = xGeo_In.reshape(dim1, n_In, n_In * n_In)
    dXdXiGL = (D_EqToGL @ _X1).reshape(dim1, n_Out, n_In, n_In)

    # Perform tensor contraction for the second derivative (Eta direction)
    dXdEtaGL  = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 2))
    dXdEtaGL  = np.moveaxis(dXdEtaGL , 1, 0)  # Correct the shape to (3, n_Out, n_In, n_In)
    # PERF: This is actually slower than the individual contractions
    # dXdEtaGL  = np.einsum('qj,dijk->dqik', D_EqToGL, xGeo_In, optimize=True)

    # Perform tensor contraction for the third derivative (Zeta direction)
    # dXdZetaGL = np.tensordot(D_EqToGL, xGeo_In, axes=(1, 3))
    # dXdZetaGL = np.moveaxis(dXdZetaGL, 1, 0)  # Correct the shape to (3, nGeoRef, nGeoRef, nGeoRef)
    _X3       = xGeo_In.reshape(dim1, n_In * n_In, n_In)
    dXdZetaGL = (_X3 @ D_EqToGL.T).reshape(dim1, n_In, n_In, n_Out)
    dXdZetaGL = np.transpose(dXdZetaGL, (0, 3, 1, 2))
    # PERF: This is actually slower than the individual contractions
    # dXdZetaGL = np.einsum('rk,dijk->drij', D_EqToGL, xGeo_In, optimize=True)

    # Change basis for each direction
    dXdXiAP   = change_basis_3D(VdmGLtoAP, dXdXiGL  )
    dXdEtaAP  = change_basis_3D(VdmGLtoAP, dXdEtaGL )
    dXdZetaAP = change_basis_3D(VdmGLtoAP, dXdZetaGL)

    # Precompute cross products between dXdEtaAP and dXdZetaAP for all points
    # cross_eta_zeta = np.cross(dXdEtaAP, dXdZetaAP, axis=0)  # Shape: (3, nGeoRef, nGeoRef, nGeoRef)
    # > Manually compute cross product
    cross_eta_zeta = np.empty_like(dXdEtaAP)
    cross_eta_zeta[0] = dXdEtaAP[1] * dXdZetaAP[2] - dXdEtaAP[2] * dXdZetaAP[1]
    cross_eta_zeta[1] = dXdEtaAP[2] * dXdZetaAP[0] - dXdEtaAP[0] * dXdZetaAP[2]
    cross_eta_zeta[2] = dXdEtaAP[0] * dXdZetaAP[1] - dXdEtaAP[1] * dXdZetaAP[0]

    # Fill output Jacobian array
    # jacOut = np.einsum('ijkl,ijkl->jkl', dXdXiAP, cross_eta_zeta)
    # PERF: This is actually slower than the individual contractions
    # jacOut = np.sum(dXdXiAP * cross_eta_zeta, axis=0)

    return np.einsum('ijkl,ijkl->jkl', dXdXiAP, cross_eta_zeta)


def evaluate_jacobian_simplex(xGeo_In: np.ndarray, _: np.ndarray, D_EqToGL: np.ndarray) -> np.ndarray:
    # Perform tensor contraction for each derivative
    # Change basis for each direction
    dXdXiAP   = xGeo_In[:, :] @ D_EqToGL[0, :, :]
    dXdEtaAP  = xGeo_In[:, :] @ D_EqToGL[1, :, :]
    dXdZetaAP = xGeo_In[:, :] @ D_EqToGL[2, :, :]

    # Manually compute cross product
    cross_eta_zeta = np.empty_like(dXdEtaAP)
    cross_eta_zeta[0] = dXdEtaAP[1] * dXdZetaAP[2] - dXdEtaAP[2] * dXdZetaAP[1]
    cross_eta_zeta[1] = dXdEtaAP[2] * dXdZetaAP[0] - dXdEtaAP[0] * dXdZetaAP[2]
    cross_eta_zeta[2] = dXdEtaAP[0] * dXdZetaAP[1] - dXdEtaAP[1] * dXdZetaAP[0]

    # Fill output Jacobian array
    # jacOut = np.sum(dXdXiAP * cross_eta_zeta, axis=0)

    return np.sum(dXdXiAP * cross_eta_zeta, axis=0)

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
