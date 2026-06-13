# PyHOPE

Mesh Generator for High-Order Meshes in Python

---

PyHOPE (Python High-Order Preprocessing Environment) is an open-source Python framework for the generation of three-dimensional unstructured high-order meshes. These meshes are needed by high-order numerical methods like Discontinuous Galerkin, Spectral Element Methods, or pFEM, in order to retain their accuracy if the computational domain includes curved boundaries.

<div class="text-center" style="margin-bottom: 2em;">
<a href="getting-started/" class="btn btn-primary btn-spacing" role="button">Getting Started</a>
<a href="user-guide/"      class="btn btn-primary btn-spacing" role="button">User Guide</a>
<a href="developer-guide/" class="btn btn-primary btn-spacing" role="button">Developer Guide</a>
<a href="mesh-format/"     class="btn btn-primary btn-spacing" role="button">Mesh Format</a>
</div>

PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. HOPR is written in modern Fortran and is maintained as legacy code and as a framework for reference. For more information and tutorials, please refer to the HOPR source code and documentation.

<div class="text-center" style="margin-bottom: 2em;">
<a href="https://github.com/hopr-framework/hopr" class="btn btn-primary btn-spacing btn-hopr"> <img src="assets/hopr-logo.png" alt="HOPR Logo" style="height:1.5em; vertical-align:middle; margin-right:0.5em;">HOPR Code</a>
<a href="https://hopr.readthedocs.io"            class="btn btn-primary btn-spacing btn-hopr"> <img src="assets/hopr-logo.png" alt="HOPR Logo" style="height:1.5em; vertical-align:middle; margin-right:0.5em;">HOPR Documentation</a>
</div>

PyHOPE has been developed by the Numerics Research Group (NRG) led by Prof. Andrea Beck at the Institute of Aerodynamics and Gas Dynamics at the University of Stuttgart, Germany.  

This is a scientific project. If you use PyHOPE for publications or presentations in science, please support the project by citing our publications given at [numericsresearchgroup.org](https://numericsresearchgroup.org/publications.html).
