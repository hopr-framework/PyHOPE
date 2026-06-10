#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
# Copyright (c) 2022 Gabriel Altay (Original Version)
#
# Hilbert(Z) curve adapted from
# hilbert.c - Computes Hilbert space-filling curve coordinates, without recursion, from integer index, and vice versa, and other
# Hilbert-related calculations.  Also known as Pi-order or Peano scan.
#
# Author:      Doug Moore
#              Dept. of Computational and Applied Math
#              Rice University
#              http://www.caam.rice.edu/~dougm
# Date:        Sun Feb 20 2000
# Copyright (c) 1998-2000, Rice University
#
# Acknowledgement:
# This implementation is based on the work of A. R. Butz ("Alternative Algorithm for Hilbert's Space-Filling Curve", IEEE Trans.
# Comp., April, 1971, pp 424-426) and its interpretation by Spencer W. Thomas, University of Michigan
# (http://www-personal.umich.edu/~spencer/Home.html) in his widely available C software.  While the implementation here differs
# considerably from his, the first two interfaces and the style of some comments are very much derived from his work.
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
from collections.abc import Iterable
from typing import Literal
from typing import overload
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def HilbertCurveNumpy() -> None:
    """ Monkey-patch for hilbertcurve.HilbertCurve:
        - Adds a NumPy-vectorized _distances_from_points_numpy(self, points)
        - Wraps distances_from_points to use the vectorized path for ndarray inputs

        Assumptions:
        - points is (M, n) with integer coordinates in [0, 2**p - 1]
        - p * n <= 63 so distances fit into uint64

        The patch is idempotent: calling apply_hilbert_numpy_patch() multiple times is harmless
    """
    # If the package isn't available, silently skip patching
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except Exception:
        return None

    # Avoid re-patching
    if getattr(HilbertCurve, '_numpy_patch_applied', False):
        return None

    # Typing helpers
    @overload
    def _distances_from_points_numpy(self, points: npt.NDArray, match_type: Literal[False] = False) -> list: ...         # noqa: ANN001
    @overload
    def _distances_from_points_numpy(self, points: list       , match_type: Literal[True])          -> list: ...         # noqa: ANN001
    @overload
    def _distances_from_points_numpy(self, points: npt.NDArray, match_type: Literal[True])          -> npt.NDArray: ...  # noqa: ANN001
    # Function
    def _distances_from_points_numpy(self,
                                     points    : list | npt.NDArray,
                                     match_type: bool = False) -> npt.NDArray | list:
        """ Batch implementation for distances_from_points in numpy
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != self.n:
            raise ValueError(f'points must be (M, {self.n})')

        # Work in uint64 for per-coordinate logic (coordinates fit in 64-bit)
        # > Copy to not mutate caller memory
        upts = pts.astype(np.uint64, copy=True)

        # Range checks
        max_x = np.uint64(self.max_x)
        if (upts > max_x).any():
            raise ValueError('point coordinates out of range [0, 2**p - 1]')

        M     = upts.shape[0]
        n     = self.n
        pbits = self.p

        # Inverse undo excess work
        q = np.uint64(1) << np.uint64(pbits - 1)  # m
        while q > 1:
            pmask = q - 1

            for i in range(n):
                mask = (upts[:, i] & q) != 0

                if mask.any():
                    upts[mask, 0] ^= pmask

                if (~mask).any():
                    t = (upts[~mask, 0] ^ upts[~mask, i]) & pmask
                    upts[~mask, 0] ^= t
                    upts[~mask, i] ^= t
            q >>= 1

        # Gray encode
        for i in range(1, n):
            upts[:, i] ^= upts[:, i - 1]

        tmask = np.zeros(M, dtype=np.uint64)
        q     = np.uint64(1) << np.uint64(pbits - 1)
        while q > 1:
            mask = (upts[:, n - 1] & q) != 0
            if mask.any():
                tmask[mask] ^= (q - 1)
            q >>= 1

        upts ^= tmask[:, None]

        # Interleave bit-planes into a big (potentially >64-bit) integer per row
        # > Build as Python ints to avoid overflow; keep it efficient by
        # >   - computing the per-plane "combine" (0..(2^n-1)) vectorized,
        # >   - updating the big integer with one vectorized object op per plane
        # > Using dtype=object leverages Python big-int shifts/ors elementwise
        h = np.zeros(M, dtype=object)

        # Precompute weights across dims for each plane: (1 << (n-1-i))
        dim_weights = np.array([1 << (n - 1 - i) for i in range(n)], dtype=np.uint64)

        for bit in range(pbits - 1, -1, -1):
            # bits_plane: (M, n) of 0/1 for current bit
            bits_plane = (upts >> np.uint64(bit)) & np.uint64(1)
            # Combine per row into [0 .. 2^n-1] by weighted sum across dims
            combine = (bits_plane * dim_weights).sum(axis=1).astype(np.uint64)
            # Update big-int index: shift by n, then OR the small combine
            # > Cast combine to object just for the OR
            h = (h << n) | combine.astype(object)

        # Result type parity
        if match_type and isinstance(points, np.ndarray):
            return np.array(h, dtype=points.dtype, copy=False)

        return list(map(int, h))

    # Attach the vectorized helper
    HilbertCurve._distances_from_points_numpy = _distances_from_points_numpy

    # Wrap the public API to prefer the NumPy path
    _orig_dfp = HilbertCurve.distances_from_points

    def _dfp_patched(self, points: Iterable[Iterable[int]], match_type: bool = False) -> Iterable[int]:  # noqa: ANN001
        """ Wrapper for the monkey-patched numby implementation
        """
        # Prefer NumPy path when possible
        if isinstance(points, np.ndarray):
            try:
                distances = self._distances_from_points_numpy(points, match_type=match_type)
            except Exception as e:
                raise RuntimeError('HilbertCurve.distances_from_points_numpy encountered an unexpected error') from e
                # Fallback to original behavior on any unexpected issue
                # distances = _orig_dfp(self, points, match_type=match_type)
        else:
            distances = _orig_dfp(self, points, match_type=match_type)

        return distances

    HilbertCurve.distances_from_points = _dfp_patched
    HilbertCurve._numpy_patch_applied  = True


def _bit_transpose(n_dims: int, n_bits: int, coords: npt.NDArray) -> npt.NDArray:
    """ Transpose the bit-planes of a packed coordinate word
    """
    in_b          = n_bits
    in_field_ends = np.uint64(1)
    in_mask       = np.uint64((1 << in_b) - 1)
    result        = np.zeros_like(coords)

    while (ut_b := in_b >> 1):
        shift_amt     = np.uint64((n_dims - 1) * ut_b)
        ut_field_ends = in_field_ends | (in_field_ends << (shift_amt + ut_b))
        ut_mask       = (ut_field_ends << np.uint64(ut_b)) - ut_field_ends
        ut_coords     = np.zeros_like(coords)

        if in_b & 1:
            # Odd number of bits: peel off the top bit of each field separately
            in_field_starts = in_field_ends << np.uint64(in_b - 1)
            odd_shift       = np.uint64(2 * shift_amt)
            for d in range(n_dims):
                chunk       = coords & in_mask
                coords    >>= np.uint64(in_b)
                result     |= (chunk & in_field_starts) << odd_shift
                odd_shift  += np.uint64(1)
                chunk      &= ~in_field_starts
                chunk       = (chunk | (chunk << shift_amt)) & ut_mask
                ut_coords  |= chunk << np.uint64(d * ut_b)
        else:
            for d in range(n_dims):
                chunk      = coords & in_mask
                coords   >>= np.uint64(in_b)
                chunk      = (chunk | (chunk << shift_amt)) & ut_mask
                ut_coords |= chunk << np.uint64(d * ut_b)

        coords        = ut_coords
        in_b          = ut_b
        in_field_ends = ut_field_ends
        in_mask       = ut_mask

    return result | coords


def _adjust_rotation(n_dims  : int        , bits    : npt.NDArray,
                     rotation: npt.NDArray, nd1_ones: int) -> npt.NDArray:
    """ rotation = (rotation + 1 + ffs(bits)) % nDims
    """
    nd1 = np.uint64(nd1_ones)
    b   = bits & (-bits.astype(np.int64)).astype(np.uint64) & nd1
    # Count trailing zeros of b per element: each '>>1' step increments rotation
    while np.any(b):
        active    = b != 0
        rotation  = np.where(active, rotation + np.uint64(1), rotation)
        b         = np.where(active, b >> np.uint64(1), b)
    rotation += np.uint64(1)
    return np.where(rotation >= np.uint64(n_dims), rotation - np.uint64(n_dims), rotation)


# ----------------------------------------------------------------------------------------------------------------------------------
# Hilbert (Z-order) curve
# ----------------------------------------------------------------------------------------------------------------------------------
def hilbert(n_dims: int, n_bits: int,
            coords: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """ Convert integer coordinates to Hilbert curve indices
    """
    M  = coords.shape[0]
    u  = coords.astype(np.uint64)

    if n_dims == 1:  # pragma: no cover
        return u[:, 0].astype(np.int64)

    n_dims_bits = n_dims * n_bits
    nd_ones     = np.uint64(( 1 << n_dims     ) - 1)
    nth_bits    = np.uint64(((1 << n_dims_bits) - 1) // int(nd_ones))
    nd1_ones    = int(nd_ones >> np.uint64(1))

    # Pack coordinates into a single word: packed[m] = u[m,0] | u[m,1]<<n_bits | ...
    packed = np.zeros(M, dtype=np.uint64)
    for d in range(n_dims):
        packed |= u[:, d] << np.uint64(d * n_bits)

    if n_bits > 1:
        # Bit-transpose then Gray-decode
        packed  = _bit_transpose(n_dims, n_bits, packed.copy())
        packed ^= packed >> np.uint64(n_dims)

        rotation = np.zeros(M, dtype=np.uint64)
        flip_bit = np.zeros(M, dtype=np.uint64)
        result   = np.zeros(M, dtype=np.uint64)

        b = n_dims_bits
        while b > 0:
            b    -= n_dims
            bits  = (packed >> np.uint64(b)) & nd_ones
            bits  = (flip_bit ^ bits)

            # rotateRight(bits, rotation, n_dims)
            rot_r = rotation % np.uint64(n_dims)
            bits  = ((bits >> rot_r) | (bits << (np.uint64(n_dims) - rot_r))) & nd_ones

            result  <<= np.uint64(n_dims)
            result   |= bits
            flip_bit  = np.uint64(1) << rotation

            rotation = _adjust_rotation( n_dims, bits.copy(), rotation, nd1_ones)

        result ^= nth_bits >> np.uint64(1)

    else:
        # n_bits == 1: trivial Gray decode
        result = packed

    # Final Gray decode of the index itself
    d = 1
    while d < n_dims_bits:
        result ^= result >> np.uint64(d)
        d      *= 2

    return result.astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------------------
# Morton (Z-order) curve
# ----------------------------------------------------------------------------------------------------------------------------------
def morton(n_dims: int, n_bits: int,
           coords: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """ Convert integer coordinates to Morton (Z-order) indices
    """
    M      = coords.shape[0]
    result = np.zeros(M, dtype=np.int64)

    for d in range(n_dims):
        col = coords[:, d].astype(np.int64)
        for i in range(n_bits):
            # Bit i of dimension d maps to position n_dims*i + (n_dims-1-d)
            bit_pos = n_dims * i + (n_dims - 1 - d)
            result |= ((col >> np.int64(i)) & np.int64(1)) << np.int64(bit_pos)

    return result
