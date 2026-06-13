# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-04-16
### Added
- Add the original Hilbert(Z)/Morton(Z) space-filling curve from HOPR [c14fe07]
- Add support for high-order extrusion [9718c3d][212b316]
- Add support for volume zones during extrusion [a96ca55]
- Add ruff linter, vulture and ty as pre-commit githook [877cdb7][131be03]

### Changed
- Abort if postDeform transformation is illegally combined with meshScale [20961bc]
- Output the BC index when running into the error of finding multiple BCs on the same internal face [1f34550][67027bf]
- Try/except import gmsh errors [fe19d71]
- Update rules for ty v0.30 [ab5238a]
- Update ty, reduce Git LFS bandwidth pressure during CI/CD [c0af215]
- Enforce annotations using ruff linter [19d340c]

### Fixed
- Fix FEM edge ordering and orientation [d56c052]
- Fix flipped elements during splitToPrism [cc8ce94][1386a37]
- Fix static code analysis issues [a4734fe][62d2ba9]
- Fix NaN in Jacobi polynomials [2fb6846]

### Dependencies
- Dependencies: Enable Dependabot [597ef91]
- Bump actions/configure-pages from 5 to 6 [166c11e]
- Bump actions/deploy-pages from 4 to 5 [87c6be7]
- Bump actions/upload-pages-artifact from 3 to 4 [4f34bf4]
- Bump actions/download-artifact from 4 to 8 [037279b]
- Bump actions/cache from 4 to 5 [6aae174]
- Bump actions/checkout from 4 to 6 [b860a24]
- Bump actions/upload-artifact from 4 to 7 [938ed62]

## [0.10.0] - 2026-03-09
### Added
- Add extrusion of 2D meshes [3b26e70][6993cb0][9a7cf94]
- Add rebuild step for deformed mortar geometry [6b680a5][0bb967f]
- Add SplitToHex for hexahedra [90b0e35]
- Add mesh read-in support for `.geo` file type [6bc3b32]
- Add mesh deformation sine template [1bd05db]
- Add PyHOPE version to generated HDF5 file [4050207]
- Add GitHub actions CI/CD [dd8e0a9]
- Add PyPI trusted publishing [fe73b52]
- Add unit tests [f939259]
- Add OpenSSF Baseline and Best Practices badges [9c57b17]

### Changed
- Lift library metadata, extend `staticmethods` [d595c7d]
- Improve CGNS reader compatibility [d9de6b6]
- Improve code support for numba dependency checking [7c9cf34]
- Improve code performance of FEM calculations [db75884]
- Improve code performance of mortar calculations [88b8dd8]
- Improve code performance of health-check [fbfb6b9]
- Improve code compatibility with Gmsh, ty [8fe1f09]
- Improve code check compatibility with PICLas [e90f13c]
- Update bundled Gmsh to v4.15.1 [09b6898]
- Update rules for ruff v0.15.2 [cd4f9a3]
- Update CI/CD container to Fedora 43 [dd34e1e]
- Transition to namespace packages [84a2de8]

### Fixed
- Apply stretching directions in reference space [18bc20f]
- Apply edge correction (coons mapping) during post deformation [e54beca]
- Calculation of Jacobians for simplex elements [fb23202]
- Explicitly specify Gmsh parameters [4c4149c][5cdca76]
- Permit regular package install [cf98482]
- Sort GlobalSideIDs along the SFC [6962de1]

## [0.9.0] - 2026-01-14
### Added
- Add numba to lower to machine-code [500b1d4]
- Add Jacobians for mixed element meshes [631bdd1]
- Add Gmsh output for meshes [bc6d8de]
- Add node ordering for VTK Lagrange Hexahedrons [b10a246]
- Add JOSS paper to documentation [4f86c2d]

### Changed
- Improve code performance of Jacobian calculation [0843b0d]
- Reduce multiprocessing memory pressure [b407a77]
- Update Gmsh to v4.15.0 [204570a]
- Update ty linting for numba [9be42fa]
- Update CI/CD consistency checks to current versions of ty, ruff, [based]pyright, and vulture [9daefde]
- Switch CI/CD to Python 3.14 and reduce docker image size [3468340]

### Fixed
- Restrict number of threads during concurrent BLAS execution [f866ac2]

## [0.8.0] - 2025-10-21
This release brings PyHOPE to functional parity with the HOPR features currently in active use. Any additional used but missing features will be treated as bugs.
### Added
- Add `nElems_IJK`/`Elem_IJK` arrays for meshes with structured dimensions [92f7168]
- Add debugMesh output in XDMF format [907df1f]
- Add element and zone ID to the debugMesh [9bb3338]
- Add two additional mesh sorting algorithms (snake/lex) [df517c8]
- Add consistency checks to various sections [f68bdb6]
- Add type checking with [ty](https://github.com/astral-sh/ty) [df517c8]

### Changed
- Improve error message for invalid MeshMode [a8d78ef]
- Improve code performance in various sections [5403b5b]

### Fixed
- Missing integer conversion in DebugMesh [a5c3248]
- Consider multiple periodicity during FEMconnect [43414bd]

## [0.1.1] - 2025-09-19
### Added
- Add support for multiple mesh zones [40dc87c]
- Add support for XDMF output [07319b6]
- Add documentation on GitHub pages [23c90ae]
- Add community guidelines [73ecc84]
- Add consistency check after initial Gmsh mesh creation [e32730a]
- Add health checks [21e771e][93a094c]
- Add support for Gitlab code coverage [531fec3][f5fe93a]

### Changed
- Pre-install Gmsh during CI/CD [ff11bb8]
- Updated bundled Gmsh to v4.14.1 [0b01a2e][396a71a]
- General improvements to code performance [1361cfb]
- Explicitly request OpenMP multithreading in Gmsh [c074aad]

### Fixed
- Fix inner boundary conditions being counted twice [01deffb]
- Fix DEFVAR parsing for floats [69ffe11]
- Remove dead code with vulture [c2d1ec8]
- Remove unnecessary intermediate meshio.Mesh [744d362]
- Several bugfixes for GitHub pages [61af0c5][d0addd1][6f2f19c]

## [0.1.0] - 2025-06-25

### Added
- Add PyHOPE context manager [97a5194a]

### Changed
- Major rework of GAMBIT mesh reader [57067512]

## [0.0.9] - 2025-06-12

### Added
- FEMConnect: Edge and vertex connectivity [5b72caa4]
- Reader: Support for Gambit mesh format [d1e8eec5]
- Support for ElemCounter, nUniqueSides, nUniqueNodes with HOPR format [9304460a]

### Changed
- Improve performance during topology changes [e6245f4f]
- Take the element volume into account when calculating the watertightness tolerance [27620b59]
- Reword the error message for malformed comma-separated arrays [c7751b88]
- Reword the readme to clarify installation steps using venv [8e818d6a]
- Only open GMSH GUI when running with Display attached [8718ec3e]

### Fixed
- Fix checking PyHOPE directory instead of current directory for git commit [a56ba0b8]
- Fix several issues in the readintools (default, DEFVAR) [b9e5d431]

## [0.0.8] - 2025-03-07

### Added
- Ability to split elements zonewise [63414dbc]
- Allow for directly processing mesh files [3cfd8cfd]
- Advanced parameter Options to ensure compatibility with HOPR [604776c5]
- Add contributors according to `git shortlog -s -n` [8cb8da50]

### Changed
- Improve mortar support (HOPR reader, matching performance, ...) [481c2eae]
- Linting and performance fixes [4181bb87] [e9a601ec]
- Analytic Gmsh -> Meshio mapping [9c10f165]
- Require Python 3.10+ for PyRight and Ruff [fdb7d8dd]
- Permit creating builtin-tetras and split them to hex [78b4cefb]
- Set new ruff parameter "-target-version" [c017b0ce]

### Fixed
- Fix extra offset for simplex/splitToHex [2191fa60]
- Several bugfixes improving overall stability [63354530]

## [0.0.7-1] - 2025-02-13

### Added & Changed
- Add CI/CD pipeline for checking several Python versions [4757fb65]

### Fixed
- Fix incompatibility with Python 3.10 and 3.11 [6dc59ec8]

## [0.0.7] - 2025-02-11

### Added & Changed
For this release there is a major feature merge [902b2b49] to the main branch which contains:
- Add support for simplex elements
- Add support for mixed meshes
- Add support for serendipity element processing from CGNS
- Further flexibility in element stretching
- Implement mesh transformations using templates
- Search for meshes and templates in multiple dirs including CWD
- Implement periodic mortar sides

Other changes:
- Implement stretching and scaling [c07fa435]

### Fixed
- Fix compatibility for Python 3.10 to 3.13 using the typing-extensions [3aaeacf4]
- Fix offset calculation during Gmsh to meshio conversion [5270006]

## [0.0.5] - 2025-01-08

### Added
- Add mortar connections with hanging nodes [36da08f]
- Add output of unique GlobalNodeIDs [206b172]
- Add reader for HOPR HDF5 format [0b8acaf]
- Add generator and stub directory for Pyright [2102804]
- Add uv as package manager [ba9601a]

### Changed
- Improve performance of mesh generation [1404bd2]
- Improve class decorators and result caching [917143e]
- Refactor CI/CD and beautify the output [f2f1836]
- Refactor CI/CD to use NRG docker containers [5f67862]
- Refactor CI/CD to use uv as package manager [739a55c]

## [0.0.4] - 2024-12-04

### Added
- Add libraries and compatibility for Linux on ARM (aarch64) and macOS on ARM (arm64) [a2e2eebc]
- Test convergence of generated meshes using FLEXI in the CI/CD pipeline [711973d0]
- Added first scaffolding of non-hexahedral elements [ab941452]

### Changed
- Rework singleton logic [4a30d04e]

### Fixed
- Fix version and commit logic [ea625cb2]
- Fix wrong type in OutputFormat declaration [c2b2d6b6]

## [0.0.3] - 2024-11-20

### Added
- Support for high-order CGNS meshes via agglomeration [2162820d]
- Added CHANGELOG.md

### Changed
- Eliminate duplicate points before creating element objects [1f060c2]
- Version number is uniquely defined in `pyproject.toml`

## [0.0.2] - 2024-11-12

### Fixed
- Improve detection of Gmsh origin and fix installation issues [b1bd5111]

## [0.0.1] - 2024-11-11
Initial release
