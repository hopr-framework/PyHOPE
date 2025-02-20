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
from typing import Optional, List, Tuple
from functools import lru_cache
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import bisect
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

    Instead of updating each node’s stored link on every insertion, this
    manager maintains breakpoints for cumulative index shifts
    """
    __slots__ = ('breakpoints', '_offset_cache')

    def __init__(self) -> None:
        self.breakpoints: List[Tuple[int, int]] = [(0, 0)]  # For all indices, initial offset is 0

    def update(self, insert_index: int, delta: int) -> None:
        """
        Record that all stored links with value >= insert_index should be increased
        by delta
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
            self.breakpoints[i] = (index, current_offset + delta)

    def get_offset(self, index: int) -> int:
        """
        Return the cumulative offset for a given stored link
        """
        pos = bisect.bisect_right(self.breakpoints, (index, float('inf'))) - 1
        return self.breakpoints[pos][1]


class ListNode:
    __slots__ = ('value', 'link', 'prev', 'next')

    def __init__(self,
                 value: SIDE,
                 link: Optional[int],
                 next: Optional[ListNode] = None,
                 prev: Optional[ListNode] = None) -> None:
        """
        value: a SIDE object
        link : the stored connection (an int) from the SIDE (side.connection)
        """
        self.value = value
        self.link  = link   # This is the base (stored) connection value
        self.prev  = prev
        self.next  = next

    def effective_link(self, offset_manager: LinkOffsetManager) -> Optional[int]:
        """
        Compute the effective connection (link) by adding the current offset
        """
        # If link is None, return None; otherwise, add the cumulative offset
        if self.link is None:
            return None
        return self.link + offset_manager.get_offset(self.link)


class DoublyLinkedList:
    __slots__ = ('head', 'tail', '_size', 'offset_manager', '_node_at')

    def __init__(self, offset_manager: LinkOffsetManager) -> None:
        self.head : Optional[ListNode] = None
        self.tail : Optional[ListNode] = None
        self._size: int                = 0
        # Shared manager for batch updates
        self.offset_manager = offset_manager
        # Cache for node_at lookups using native lru_cache
        self._node_at = lru_cache(maxsize=None)(self._node_at_impl)

    def __len__(self) -> int:
        return self._size

    def _node_at_impl(self, index: int) -> ListNode:
        """
        Internal implementation for retrieving the node at the given index (by traversing from head or tail)
        > This method is cached via lru_cache
        """
        if index < 0 or index >= self._size:
            raise IndexError('Index out of range')

        # Decide traverse direction: from head or tail.
        if index < self._size // 2:
            current = self.head
            cur_index = 0
            while cur_index < index:
                current    = current.next
                cur_index += 1
        else:
            current = self.tail
            cur_index = self._size - 1
            while cur_index > index:
                current    = current.prev
                cur_index -= 1

        if current is None:
            raise IndexError("Node not found")

        return current

    def node_at(self, index: int) -> ListNode:
        """
        Retrieve the node at the given index (by using a cached lookup)
        """
        return self._node_at(index)

    def __getitem__(self, index: int) -> ListNode:
        return self.node_at(index)

    def insert(self, effective_index: int, new_node: ListNode, update_offset: bool = True) -> None:
        """
        Insert new_node at the logical position that corresponds to the given
        effective_index (already offset-adjusted)
        """
        logical_index = effective_index

        if logical_index < 0 or logical_index > self._size:
            raise IndexError("Index out of range")

        # Invalidate the node_at cache since the list structure is about to change
        self._node_at.cache_clear()

        # Standard insertion logic in the doubly linked list at logical_index
        if self._size == 0:
            self.head = self.tail = new_node
        elif logical_index == 0:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        elif logical_index == self._size:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        else:
            current = self.node_at(logical_index)
            prev_node = current.prev
            if prev_node:
                prev_node.next = new_node
                new_node.prev = prev_node
            new_node.next = current
            current.prev = new_node

        self._size += 1
        if update_offset:
            # Update the offset manager using the computed logical index
            self.offset_manager.update(logical_index - self.offset_manager.get_offset(effective_index), 1)

    def update(self, index: int, new_value: SIDE) -> None:
        """
        Update the value of the node at the given index with the new SIDE object
        """
        # For update, structure remains unchanged so cache need not be cleared
        node = self.node_at(index)
        node.value = new_value

    def to_list(self) -> List[ListNode]:
        """ Return a Python list of the nodes (in order)
        """
        result: List[ListNode] = []
        current = self.head
        while current:
            result.append(current)
            current = current.next
        return result

    def __iter__(self):
        """
        Iterate over the nodes in the list
        """
        current = self.head
        while current:
            yield current
            current = current.next


def list_to_dllist(sides: List[SIDE], offset_manager: LinkOffsetManager) -> DoublyLinkedList:
    """
    Convert a list of SIDE objects into a DoublyLinkedList

    When building from an existing list, we do not update offsets because the
    stored connection (side.connection) is already valid
    """
    dll = DoublyLinkedList(offset_manager)
    for side in sides:
        node = ListNode(value=side, link=side.connection)
        # Do not update the offset manager during this bulk conversion.
        dll.insert(len(dll), node, update_offset=False)
    return dll


def dllist_to_list(dll: DoublyLinkedList) -> List[SIDE]:
    """
    Convert the DoublyLinkedList back to a list of SIDE objects

    For each node, update its SIDE object's connection field using the effective link,
    and update sideID to reflect the node's new position in the list
    """
    nodes = dll.to_list()
    for idx, node in enumerate(nodes):
        if node.value.connection is not None and node.value.connection >= 0:
            node.value.connection = node.effective_link(dll.offset_manager) if node.link is not None else None
        # Update sideID with the node's new index.
        node.value.sideID = idx
    return [node.value for node in nodes]
