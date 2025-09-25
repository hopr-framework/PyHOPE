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
import sys
import traceback
from multiprocessing import Pool, Queue, Process
from typing import Callable
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from alive_progress import alive_bar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def distribute_work(elems: tuple, chunk_size: int) -> tuple:
    """Distribute elements into chunks of a given size
    """
    return tuple(elems[i:i + chunk_size] for i in range(0, len(elems), chunk_size))


def update_progress(progress_queue: Queue, total_elements: int) -> None:
    """ Function to update the progress bar from the queue
    """
    with alive_bar(total_elements, title='│             Processing Elements', length=33) as bar:
        processed_count = 0
        while processed_count < total_elements:
            # Block until we receive a progress update from the queue
            chunk_size = progress_queue.get()
            if chunk_size is None:  # Sentinel value to stop
                break
            bar(chunk_size)
            processed_count += chunk_size


def run_in_parallel(process_chunk: Callable, elems: tuple, chunk_size: int = 10) -> list:
    """Run the element processing in parallel using a specified number of processes
    """
    # Local imports ----------------------------------------
    from pyhope.common.common import IsInteractive
    from pyhope.common.common_vars import np_mtp
    # ------------------------------------------------------

    chunks = distribute_work(elems, chunk_size)
    total_elements = len(elems)

    # Return early if there's no work to be done
    if total_elements == 0:
        return []

    progress_queue = Queue()

    # Create a progress bar target
    target = update_progress if IsInteractive() else None

    # Use a separate thread for the progress bar
    progress_thread = Process(target=target, args=(progress_queue, total_elements))
    progress_thread.start()

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=np_mtp) as pool:
        # Map work across processes in chunks
        results = []
        try:
            # Using imap_unordered to get results as they complete
            for chunk_result in pool.imap_unordered(process_chunk, chunks):
                results.extend(chunk_result)
                # Update progress for each processed chunk
                progress_queue.put(len(chunk_result))
        except Exception:
            # Terminate processes and print traceback (exception only contains the error message)
            pool.terminate()
            progress_thread.terminate()
            print(traceback.format_exc())
            sys.exit(1)

    # Wait for the process and progress threads to finish and synchronize
    pool.join()
    pool.close()
    progress_thread.join()
    progress_thread.close()
    return results
