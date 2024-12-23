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
import threading
from multiprocessing import Pool, Queue
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from alive_progress import alive_bar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def distribute_work(elems, chunk_size: int) -> list[object]:
    """Distribute elements into chunks of a given size
    """
    return [elems[i:i + chunk_size] for i in range(0, len(elems), chunk_size)]


def update_progress(progress_queue, total_elements) -> None:
    """ Function to update the progress bar from the queue
    """
    with alive_bar(total_elements, title='│             Processing Elements', length=33) as bar:
        for _ in range(total_elements):
            # Block until we receive a progress update from the queue
            progress_queue.get()
            bar()


def run_in_parallel(process_chunk, elems, chunk_size=10):
    """Run the element processing in parallel using a specified number of processes
    """
    # Local imports ----------------------------------------
    from pyhope.common.common_vars import np_mtp
    # ------------------------------------------------------

    chunks = distribute_work(elems, chunk_size)
    total_elements = len(elems)
    progress_queue = Queue()

    # Use a separate thread for the progress bar
    progress_thread = threading.Thread(target=update_progress, args=(progress_queue, total_elements))
    progress_thread.start()

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=np_mtp) as pool:
        # Map work across processes in chunks
        results = []
        for chunk_result in pool.imap_unordered(process_chunk, chunks):
            results.extend(chunk_result)
            # Update progress for each processed element in the chunk
            for _ in chunk_result:
                progress_queue.put(1)

    # Wait for the progress bar thread to finish
    progress_thread.join()
    return results
