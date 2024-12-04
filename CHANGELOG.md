# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
