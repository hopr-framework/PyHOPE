# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 - 2026 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
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
# Continuous Integration/Continuous Deployment
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
groot: Final[Path]             = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
files: Final[tuple[Path, ...]] = (Path(groot / '.github/workflows/ci.yml'),
                                  Path(groot / '.gitlab-ci.yml'))
# ==================================================================================================================================


def update_python() -> None:
    versions = {}
    url      = ('https://hub.docker.com/v2/repositories/library/python/tags?name=slim-trixie&page_size=100&ordering=last_updated')

    while url:
        req = urllib.request.Request(url, headers={'User-Agent': 'pyhope-ci-script'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        for result in data.get('results', []):
            tag   = result.get('name', '')
            match = re.match(r'^3\.(\d+)\.(\d+)-slim-trixie$', tag)
            if match:
                minor, patch = int(match.group(1)), int(match.group(2))
                key = f'3.{minor}'
                if key not in versions or patch > versions[key]['patch']:
                    versions[key] = {'patch': patch, 'full': tag}

        url = data.get('next')

    print(f'Detected latest Python versions: { {k: v["full"] for k, v in versions.items()} }')

    for path in files:
        if not path.exists():
            print(f'Skipping {path}: file not found.')
            continue

        content = path.read_text(encoding='utf-8')

        for key, info in versions.items():
            minor   = key.split('.')[1]          # '3.11' -> '11'
            patch   = info['patch']
            content = re.sub(rf'(3\.{minor}\.)\d+(-slim-trixie)', rf'\g<1>{patch}\g<2>', content)
            content = re.sub(rf'(PYTHON3{minor}_VERSION:\s*3\.{minor}\.)\d+(-trixie)', rf'\g<1>{patch}\g<2>', content)

        path.write_text(content, encoding='utf-8')
        print(f'Successfully updated {path}')


if __name__ == '__main__':
    update_python()
