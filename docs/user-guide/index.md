# Introduction

Scope and features of PyHOPE

---

High-order numerical methods surch as Discontinuous Galerkin, Spectral Element Methods, or pFEM, require unstructured high-order meshes in order to retain their accuracy if the computational domain includes curved boundaries. The HOPR HDF5 curved mesh format is specifically designed to specifically designed for parallel read-in of unstructured three-dimensional meshes of arbitrary order, including tetrahedra, pyramids, prisms, and hexahedra. Information stored in HOPR format facilitates non-overlapping input output (I/O) through collocation of the required mesh information, including the vertex and side information together with element connectivity, in per-element packages. Each package is assigned a unique identifier via ordering along structured dimensions or a space-filling curve.

PyHOPE is a Python library for reading, writing, and manipulating HOPR HDF5 curved mesh files. 

- [Installation](installation.md): Installation and verification of the local installation of PyHOPE

At the current state, it features two modes for mesh generation:

- [Internal mesh generator](mesh-generators/internal.md): A simple mesh generator for generating curved meshes of basic block-structured geometries 
- [External mesh generator](mesh-generators/external.md): Read and convert meshes generated with external mesh generators such as ANSA, Gmsh, or Cubit

PyHOPE is controlled via a parameter file in INI format. The parameter file specifies the mesh generation mode, the geometry and mesh parameters, and the output file name.

- [Parameter file format](parameter-file.md): Description of the parameter file format and available options

PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. 

- [Mesh Format](../mesh-format.md): Brief description of the HOPR HDF5 mesh format

For more information and tutorials, please visit the [HOPR documentation](https://hopr.readthedocs.io). Furthermore, PyHOPE utilizes [Gmsh](https://gmsh.info) for the initial mesh generation and conversion before switching to its internal representation. The internal representation is loosely based on [meshio](https://github.com/nschloe/meshio) but augmented with additional information required for high-order meshes.
