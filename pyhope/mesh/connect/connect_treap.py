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
import bisect
import random
from typing import Optional, List, Tuple
from functools import cache
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
from pyhope.mesh.mesh_vars import SIDE
# ==================================================================================================================================


class LinkOffsetManager:
    """
    Batch Update Manager for Connection Offsets

    Instead of updating each node’s stored link on every insertion, this manager maintains breakpoints for cumulative index shifts
    """
    __slots__ = ('breakpoints', '_offset_cache')

    def __init__(self) -> None:
        # For all indices, initial offset is 0
        self.breakpoints: List[Tuple[int, int]] = [(0, 0)]

    def get_stored_index(self, effective_index: int) -> int:
        """
        Given an effective index (i.e. the logical index in the list), compute the corresponding stored index at which the
        breakpoint update should occur. The effective index E is related to a stored index S by E = S + offset(S), where
        offset(S) is constant between breakpoints.
        """
        # Precompute effective boundaries for each breakpoint.
        effective_boundaries = [s + off for s, off in self.breakpoints]
        pos = bisect.bisect_right(effective_boundaries, effective_index) - 1
        return effective_index - self.breakpoints[pos][1]

    def update(self, insert_index: int, delta: int) -> None:
        """
        Record that all stored links with value >= insert_index should be increased by delta
        """
        pos = bisect.bisect_left(self.breakpoints, (insert_index, -float('inf')))
        if pos < len(self.breakpoints) and self.breakpoints[pos][0] == insert_index:
            index, current_offset = self.breakpoints[pos]
            self.breakpoints[pos] = (index, current_offset + delta)
            pos += 1
        else:
            prev_offset = self.breakpoints[pos - 1][1] if pos > 0 else 0
            self.breakpoints.insert(pos, (insert_index, prev_offset + delta))
            pos += 1
        # Update all subsequent breakpoints.
        for i in range(pos, len(self.breakpoints)):
            index, current_offset = self.breakpoints[i]
            self.breakpoints[i]   = (index, current_offset + delta)
        # Clear cached offsets as breakpoints have changed
        self._get_offset.cache_clear()

    # @lru_cache(maxsize=None)
    @cache
    def _get_offset(self, index: int) -> int:
        """
        Return the cumulative offset for a given stored link (cached)
        """
        pos = bisect.bisect_right(self.breakpoints, (index, float('inf'))) - 1
        return self.breakpoints[pos][1]

    def get_offset(self, index: int) -> int:
        return self._get_offset(index)


class SideNode:
    __slots__ = ('value', 'link')

    def __init__(self,
                 value: SIDE,
                 link: Optional[int]) -> None:
        """
        value: a SIDE object
        link : the stored connection (an int) from the SIDE (side.connection)
        """
        self.value = value
        self.link  = link   # This is the base (stored) connection value

    def effective_link(self, offset_manager: LinkOffsetManager) -> Optional[int]:
        """
        Compute the effective connection (link) by adding the current offset
        """
        # If link is None, return None; otherwise, add the cumulative offset
        if self.link is None:
            return None
        return self.link + offset_manager.get_offset(self.link)


# ----- Treap Helpers ---------------------------------------------------------------------------
class _TreapNode:
    __slots__ = ('data', 'priority', 'left', 'right', 'size')

    def __init__(self, data: SideNode) -> None:
        self.data    : SideNode = data
        self.size    : int      = 1
        self.priority: float    = random.random()  # random priority for balancing
        self.left    : Optional[_TreapNode] = None
        self.right   : Optional[_TreapNode] = None


def _update_size(node: Optional[_TreapNode]) -> None:
    if node is not None:
        node.size = 1
        if node.left is not None:
            node.size += node.left.size
        if node.right is not None:
            node.size += node.right.size


def _split(root: Optional[_TreapNode], index: int) -> Tuple[Optional[_TreapNode], Optional[_TreapNode]]:
    """
    Splits the treap into two treaps:
      - left : first 'index' elements
      - right: the remaining elements
    """
    if root is None:
        return (None, None)
    left_size = root.left.size if root.left else 0
    if index <= left_size:
        left, new_left = _split(root.left, index)
        root.left = new_left
        _update_size(root)
        return (left, root)
    else:
        new_index  = index - left_size - 1
        new_right, right = _split(root.right, new_index)
        root.right = new_right
        _update_size(root)
        return (root, right)


def _merge(left: Optional[_TreapNode], right: Optional[_TreapNode]) -> Optional[_TreapNode]:
    """
    Merges two treaps where all keys in left come before keys in right.
    """
    if left is None or right is None:
        return left or right
    if left.priority < right.priority:
        left.right = _merge(left.right, right)
        _update_size(left)
        return left
    else:
        right.left = _merge(left, right.left)
        _update_size(right)
        return right


class Treap:
    """
    This class provides randomized balanced BST known as a Treap to reduce the O(n) cost of arbitrary insertions and random access
    inherent in a doubly linked list.
    """
    __slots__ = ('_root', '_size', 'offset_manager', '_node_at')

    def __init__(self, offset_manager: LinkOffsetManager) -> None:
        self._root: Optional[_TreapNode] = None
        self._size: int = 0
        # Shared manager for batch updates
        self.offset_manager = offset_manager
        # Cache for node_at lookups using native lru_cache
        self._node_at = self._node_at_impl

    def __len__(self) -> int:
        return self._size

    # @lru_cache(maxsize=None)
    @cache
    def _node_at_impl(self, index: int) -> SideNode:
        """
        Retrieve the ListNode at the given index using treap search. This method is cached; the cache is cleared on structural
        modifications.
        """
        if not 0 <= index < self._size:
            raise IndexError('Index out of range')

        # Traverse the treap to find the node at the given index
        node = self._root
        while node is not None:
            left_size = node.left.size if node.left else 0
            if index < left_size:
                node = node.left
            elif index == left_size:
                return node.data
            else:
                index -= left_size + 1
                node = node.right
        raise IndexError("Index not found")

    def node_at(self, index: int) -> SideNode:
        """
        Retrieve the node at the given index (via cached lookup)
        """
        return self._node_at(index)

    def __getitem__(self, index: int) -> SideNode:
        return self.node_at(index)

    def insert(self, effective_index: int, new_node: SideNode, update_offset: bool = True) -> None:
        """
        Insert new_node at the logical position corresponding to the effective_index
        """
        if not 0 <= effective_index <= self._size:
            raise IndexError('Index out of range')

        # Invalidate the node_at cache since the list structure is about to change
        self._node_at.cache_clear()

        # Insert the new node into the treap at logical index
        new_treap   = _TreapNode(new_node)
        left, right = _split(self._root, effective_index)
        merged      = _merge(left, new_treap)
        self._root  = _merge(merged, right)
        self._size += 1

        if update_offset:
            # Update the offset manager using the computed stored index corresponding
            # to the effective index.
            stored_index = self.offset_manager.get_stored_index(effective_index)
            self.offset_manager.update(stored_index, 1)

    def update(self, index: int, new_value: SIDE) -> None:
        """
        Update the value of the node at the given index with the new SIDE object
        """
        # For update, structure remains unchanged so cache need not be cleared
        node = self.node_at(index)
        node.value = new_value

    def inorder(self, t: Optional[_TreapNode], result: List[SideNode]) -> None:
        """
        Recursively traverse the treap in order and append node data to the result list
        """
        if t is None:
            return

        self.inorder( t.left, result)
        result.append(t.data)
        self.inorder( t.right, result)

    def to_list(self) -> List[SideNode]:
        """
        Return a Python list of the nodes (in order) via an in-order traversal of the treap
        """
        result: List[SideNode] = []
        self.inorder(self._root, result)
        return result

    def __iter__(self):
        """
        Iterate over the nodes in order
        """
        return iter(self.to_list())


def list_to_treap(sides: List[SIDE], offset_manager: LinkOffsetManager) -> Treap:
    """
    Convert a list of SIDE objects into a Treap (balanced BST) for efficient insertion and random access

    When building from an existing list, we do not update offsets because the
    stored connection (side.connection) is already valid
    """
    dll = Treap(offset_manager)
    for side in sides:
        node = SideNode(value=side, link=side.connection)
        # Do not update the offset manager during this bulk conversion
        dll.insert(len(dll), node, update_offset=False)
    return dll


def treap_to_list(dll: Treap) -> List[SIDE]:
    """
    Convert the Treap (balanced BST) back into a list of SIDE objects

    For each node, update its SIDE object's connection field using the effective link,
    and update sideID to reflect the node's new position in the Treap.
    """
    nodes = dll.to_list()
    for idx, node in enumerate(nodes):
        # Get the new (effective) connection
        if node.value.connection is not None and node.value.connection >= 0:
            node.value.connection = node.effective_link(dll.offset_manager) if node.link is not None else None
        # Update sideID with the node's new index
        node.value.sideID = idx
    return [node.value for node in nodes]
