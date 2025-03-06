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
from dataclasses import dataclass
from typing import Optional, List, Tuple, cast
from typing import final
from functools import cache
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
from pyhope.mesh.mesh_vars import SIDE
# ----------------------------------------------------------------------------------------------------------------------------------
RED   = True
BLACK = False
LEFT  = True
RIGHT = False
# ==================================================================================================================================


@final
class LinkOffsetManager:
    """ Batch Update Manager for Connection Offsets

        Instead of updating each node's stored link on every insertion, this manager maintains breakpoints for cumulative index
        shifts
    """

    __slots__ = ('breakpoints',)

    def __init__(self) -> None:
        # Single breakpoint at stored_index=0 => offset=0
        self.breakpoints: List[Tuple[int, int]] = [(0, 0)]

    def update(self,
               effective_index: int,
               delta          : int) -> None:
        """ Record that all stored links with value >= insert_index should be increased by delta

            Perform a binary search on breakpoints, then update or insert the breakpoint and adjust all subsequent offsets
        """
        # S = E - offset(S)
        insert_index = effective_index - self.get_offset(effective_index)
        pos          = bisect.bisect_left(self.breakpoints, (insert_index, -float('inf')))

        # If there is a breakpoint exactly at insert_index, update its offset
        if pos < len(self.breakpoints) and self.breakpoints[pos][0] == insert_index:
            index, current_offset = self.breakpoints[pos]
            self.breakpoints[pos] = (index, current_offset + delta)
            pos += 1
        # Otherwise, insert a new breakpoint
        else:
            prev_offset = self.breakpoints[pos - 1][1] if pos > 0 else 0
            self.breakpoints.insert(pos, (insert_index, prev_offset + delta))
            pos += 1

        # INFO: This is the original implementation
        # > Directly update the breakpoints list
        # # Increase offset of all subsequent breakpoints
        # for i in range(pos, len(self.breakpoints)):
        #     index, current_offset = self.breakpoints[i]
        #     # Calculate the new offset by adding delta to the current offset
        #     self.breakpoints[i] = (index, current_offset + delta)

        # NOTE: This is the alternative implementation
        # > Convert the slice of breakpoints to a NumPy array
        breakpoints_array = np.array(self.breakpoints[pos:], dtype=[('index', int), ('offset', int)]).reshape(-1, 2)
        # Add delta to the 'offset' column
        breakpoints_array['offset'] += delta
        # Convert back to a list of tuples
        self.breakpoints[ pos:    ]  = cast(list[tuple[int, int]], breakpoints_array.tolist())

        # Clear any cached offset computations
        self.get_offset.cache_clear()

    @cache
    def get_offset(self,
                   index: int) -> int:
        """ Cached offset lookup for stored index.
        """
        # Binary search for the region containing 'index'
        pos = bisect.bisect_right(self.breakpoints, (index, float('inf'))) - 1
        if pos < 0:
            # If index < the first breakpoint, offset=0
            return 0
        return self.breakpoints[pos][1]


@final
@dataclass(init=False, repr=False, eq=False, slots=False)
class SideNode:
    # PERF: Convert class to dataclass for better performance
    # """
    # value: a SIDE object
    # link : the stored connection (an int) from the SIDE (side.connection)
    # """
    # value = SIDE
    # link  = Optional[int]  # This is the base (stored) connection value
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

    def effective_link(self,
                       offset_manager: LinkOffsetManager) -> Optional[int]:
        """
        Compute the effective connection (link) by adding the current offset
        """
        if self.link is None:
            return None
        return cast(int, self.link) + offset_manager.get_offset(self.link)


# ----- Red-Black Tree Helpers ---------------------------------------------------------------------------
@final
@dataclass(init=True, repr=False, eq=False, slots=True)
class _RBTreeNode:
    # PERF: Convert class to dataclass for better performance
    # __slots__ = ('data', 'color', 'left', 'right', 'parent', 'size')
    #
    # def __init__(self,
    #              data: SideNode) -> None:
    #     self.data  : SideNode = data
    #     self.color : bool     = RED  # New nodes are initially red
    #     self.left  : Optional[_RBTreeNode] = None
    #     self.right : Optional[_RBTreeNode] = None
    #     self.parent: Optional[_RBTreeNode] = None
    #     self.size  : int = 1
    data  : SideNode
    color : bool     = RED  # New nodes are initially red
    left  : Optional[_RBTreeNode] = None
    right : Optional[_RBTreeNode] = None
    parent: Optional[_RBTreeNode] = None
    size  : int = 1


@final
class RedBlackTree:
    """
    This class provides a balanced binary search tree implemented as a Red-Black Tree,
    augmented with subtree sizes to support efficient arbitrary insertions and random access.
    """
    __slots__ = ('_root', '_size', 'offset_manager', '_node_at')

    def __init__(self,
                 offset_manager: LinkOffsetManager) -> None:
        self._root: Optional[_RBTreeNode] = None
        self._size: int = 0
        # Shared manager for batch updates
        self.offset_manager = offset_manager
        # Cache for node_at lookups using native cache
        self._node_at = self._node_at_impl

    def __len__(self) -> int:
        return self._size

    def _left_rotate(self,
                     x: _RBTreeNode) -> None:
        """ Rotate the subtree rooted at x to the left
        """
        y = x.right

        # Cannot left-rotate without right child
        if y is None:
            return

        x.right  = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent

        if x.parent is None:
            self._root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left   = x
        x.parent = y
        # Update subtree sizes
        x.size   = 1 + (x.left.size if x.left else 0) + (x.right.size if x.right else 0)
        y.size   = 1 + (y.left.size if y.left else 0) + (y.right.size if y.right else 0)

    def _right_rotate(self,
                      x: _RBTreeNode) -> None:
        """ Rotate the subtree rooted at x to the right
        """
        y = x.left

        # Cannot right-rotate without left child
        if y is None:
            return

        x.left   = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent

        if x.parent is None:
            self._root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y

        y.right  = x
        x.parent = y
        # Update subtree sizes
        x.size   = 1 + (x.left.size if x.left else 0) + (x.right.size if x.right else 0)
        y.size   = 1 + (y.left.size if y.left else 0) + (y.right.size if y.right else 0)

    def _insert_fixup(self, z: _RBTreeNode) -> None:
        """ Fixup the red-black tree after insertion
        """
        while z.parent is not None and z.parent.color == RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right

                if y is not None and y.color == RED:
                    z.parent.color = BLACK
                    y.color        = BLACK
                    z.parent.parent.color = RED
                    z = cast(_RBTreeNode, z.parent.parent)
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._right_rotate(cast(_RBTreeNode, z.parent.parent))

            else:
                y = z.parent.parent.left
                if y is not None and y.color == RED:
                    z.parent.color = BLACK
                    y.color        = BLACK
                    z.parent.parent.color = RED
                    z = cast(_RBTreeNode, z.parent.parent)
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(cast(_RBTreeNode, z.parent.parent))
        self._root.color = BLACK

    @cache
    def _node_at_impl(self, index: int) -> SideNode:
        if not 0 <= index < self._size:
            raise IndexError('Index out of range')

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
        Insert new_node at the logical position corresponding to the effective_index in the red-black tree.
        """
        if not 0 <= effective_index <= self._size:
            raise IndexError('Index out of range')

        # Invalidate the node_at cache since the tree structure is about to change
        if update_offset:
            self._node_at.cache_clear()

        # Create a new red-black tree node for new_node
        z: _RBTreeNode           = _RBTreeNode(new_node)
        y: Optional[_RBTreeNode] = None
        x: Optional[_RBTreeNode] = self._root
        current_index: int       = effective_index
        branch: Optional[bool]   = None

        # Find the insertion point using order statistics
        while x is not None:
            y         = x
            x.size   += 1  # Increment subtree size for nodes along the insertion path
            left_size = x.left.size if x.left is not None else 0

            if current_index <= left_size:
                branch = LEFT
                x = x.left
            else:
                current_index -= left_size + 1
                branch = RIGHT
                x = x.right

        z.parent = y
        if y is None:
            self._root = z
        else:
            if branch == LEFT:
                y.left  = z
            else:
                y.right = z

        # Fix-up the red-black tree properties
        self._insert_fixup(z)
        self._size += 1

        if update_offset:
            self.offset_manager.update(effective_index, 1)

    def update(self, index: int, new_value: SIDE) -> None:
        """
        Update the value of the node at the given index with the new SIDE object
        """
        node = self.node_at(index)
        node.value = new_value

    def inorder(self, t: Optional[_RBTreeNode], result: List[SideNode]) -> None:
        """
        Recursively traverse the red-black tree in order and append node data to the result list
        """
        if t is None:
            return
        self.inorder( t.left, result)
        result.append(t.data)
        self.inorder( t.right, result)

    def _to_list(self) -> List[SideNode]:
        """
        Return a Python list of the nodes (in order) via an in-order traversal of the red-black tree
        """
        result: List[SideNode] = []
        self.inorder(self._root, result)
        return result

    def __iter__(self):
        """
        Iterate over the nodes in order
        """
        return iter(self.to_list())

    @staticmethod
    def _build_tree(nodes: tuple[_RBTreeNode, ...], start: int, end: int) -> Optional[_RBTreeNode]:
        """Recursively build a balanced tree from the tuple of nodes

        This method assumes that the nodes are in the desired in-order sequence
        """
        # Return if subtree is empty
        if start > end:
            return None

        mid  = (start + end) // 2
        root = nodes[mid]

        # Recursively build left and right subtrees
        root.left  = RedBlackTree._build_tree(nodes, start, mid - 1)
        if root.left is not None:
            root.left.parent  = root

        root.right = RedBlackTree._build_tree(nodes, mid + 1, end)
        if root.right is not None:
            root.right.parent = root

        # Compute subtree size in one pass
        left_size  = root.left.size  if root.left  is not None else 0  # noqa: E272
        right_size = root.right.size if root.right is not None else 0  # noqa: E272
        root.size  = 1 + left_size + right_size

        return root

    @classmethod
    def from_list(cls, sides: List[SIDE], offset_manager: LinkOffsetManager) -> RedBlackTree:
        """Optimized bulk conversion from a sorted list to a red–black tree

        Assumes that the provided list is already in the desired in-order sequence.
        All nodes are initialized as black to maintain red–black properties.
        """
        rbt   = cls(offset_manager)

        # Create a list of nodes with the SIDE data
        nodes = tuple(_RBTreeNode(SideNode(value=side, link=side.connection), color=BLACK) for side in sides)

        rbt._root = cls._build_tree(nodes, 0, len(nodes) - 1)
        rbt._size = len(nodes)

        return rbt

    # PERF: This is an improved implementation but still not optimal
    # def from_list(cls, sides: List[SIDE], offset_manager: LinkOffsetManager) -> RedBlackTree:
    #     """ Convert a list of SIDE objects into a red-black tree (balanced BST) for efficient insertion and random access
    #
    #         When building from an existing list, we do not update offsets because the offset manager because the stored
    #         connection (side.connection) is already valid
    #     """
    #     rbt = cls(offset_manager)
    #     for side in sides:
    #         node = SideNode(value=side, link=side.connection)
    #         # Do not update the offset manager during this bulk conversion
    #         rbt.insert(len(rbt), node, update_offset=False)
    #     return rbt

    # INFO: This is the original implementation
    # def to_list(self) -> List[SIDE]:
    #     """ Convert the red-black tree (balanced BST) back into a list of SIDE objects
    #
    #         For each node, update its SIDE object's connection field using the effective link, and update sideID to reflect
    #         the node's new position in the red-black tree.
    #     """
    #     nodes = self._to_list()
    #     for idx, node in enumerate(nodes):
    #         if node.value.connection is not None and node.value.connection >= 0:
    #             node.value.connection = node.effective_link(self.offset_manager) if node.link is not None else None
    #         node.value.sideID = idx
    #     return [node.value for node in nodes]

    def to_list(self) -> List[SIDE]:
        """ Convert the red-black tree back into a list of SIDE objects

        Instead of computing each node's effective link individually (using a binary search for each call), we batch-process updates
        by precomputing the breakpoints and then sweeping through the nodes that require an update.
        """
        nodes = self._to_list()

        # Extract the current breakpoints from the offset manager.
        # Assumed to be sorted by the stored index.
        breakpoints = self.offset_manager.breakpoints

        # Create a list of tuples containing (stored_link, SideNode)
        update_nodes: list[tuple[int, SideNode]] = []

        # Only consider nodes with a valid stored connection (>= 0)
        for node in nodes:
            # We use the stored link (node.link) for the update computation
            if node.link is not None and node.link >= 0:
                update_nodes.append((node.link, node))

        # Sort the update_nodes by the stored connection value
        update_nodes.sort(key=lambda tup: tup[0])

        bp_index = 0
        bp_count = len(breakpoints)
        # Iterate over the nodes (sorted by connection) and assign the effective link
        for stored_link, node in update_nodes:
            # Advance the breakpoint pointer while the next breakpoint index is <= stored_link
            while bp_index < bp_count - 1 and breakpoints[bp_index + 1][0] <= stored_link:
                bp_index += 1
            offset = breakpoints[bp_index][1]
            node.value.connection = stored_link + offset

        # Update the sideID for all nodes based on in-order position
        for idx, node in enumerate(nodes):
            node.value.sideID = idx

        return [node.value for node in nodes]
