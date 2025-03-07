# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
