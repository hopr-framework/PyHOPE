# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.5] - 2025-xx-xx

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
