---
title: 'PyHOPE: A Python Toolkit for Three-Dimensional Unstructured High-Order Meshes'
tags:
  - Python
  - Mesh Generation
  - Spectral Element Methods
  - High Order
  - Unstructured Meshes
authors:
  - name: Patrick Kopper
    orcid: 0000-0002-7613-0739
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Marcel P. Blind
    orcid: 0000-0002-4737-9359
    equal-contrib: true
    affiliation: 1
  - name: Anna Schwarz
    orcid: 0000-0002-3181-8230
    affiliation: 1
  - name: Marius Kurz
    orcid: 0009-0005-0844-0769
    affiliation: 2
  - name: Felix Rodach
    orcid: 0009-0003-1800-5600
    affiliation: 1
  - name: Stephen M. Copplestone
    orcid: 0000-0001-5557-1044
    affiliation: 3
  - name: Andrea D. Beck
    orcid: 0000-0003-3634-7447
    affiliation: 1
affiliations:
  - name: Institute of Aerodynamics and Gas Dynamics, University of Stuttgart, Stuttgart, Germany
    index: 1
  - name: Centrum Wiskunde & Informatica, Amsterdam, Netherlands
    index: 2
  - name: boltzplatz - numerical plasma dynamics GmbH, Stuttgart, Germany
    index: 3
date: 1 April 2025
bibliography: paper.bib
---

# Summary
PyHOPE (Python High-Order Preprocessing Environment) is a framework for generating and converting high-order meshes composed of standard 3D element types, designed for massively parallel spectral element solvers on high-performance computing (HPC) systems.
PyHOPE builds on and extends Gmsh [@Geuzaine2009] which is used for the initial mesh generation and/or mesh read-in before conversion of the mesh to its internal representation and application of boundary conditions.
Parallel read-in is crucial on HPC clusters which typically use parallel distributed file systems to enable and scale storage access by striping data across multiple servers.
Primary output format of PyHOPE is the HOPR [@Hindenlang2014] HDF5 curved mesh format which is specifically designed for parallel read-in of unstructured three-dimensional meshes of arbitrary order, including tetrahedra, pyramids, prisms, and hexahedra.
Information stored in HOPR format facilitates non-overlapping input/output (I/O) through collocation of the required mesh information, including the vertex and side information together with element connectivity, in per-element packages.
Each package is assigned a unique identifier via ordering along structured dimensions or a space-filling curve.

# Statement of Need
The Discontinuous Galerkin Spectral Element Method (DGSEM) is a powerful numerical approach for solving partial differential equations, particularly in high-performance computing applications.
Prominent examples of DGSEM codes originating and being actively developed at the University of Stuttgart include the FLEXI family[^1] geared towards solving the compressible (multiphase) Navier-Stokes equations along with PICLas[^2] which focuses on plasma simulation with a Particle-in-Cell/Direct Simulation Monte Carlo approach.
A crucial aspect of DGSEM is the mapping of the equations to be solved from reference space to physical space, which relies on the computation of the Jacobian determinant to ensure accurate transformations for curved high-order elements.
At the same time, DGSEM features a highly local stencil since grid elements are connected solely via the numerical flux through adjacent element faces.
Providing mesh and solution data in an on-disc data format which facilitates non-overlapping I/O is key to efficient parallel data access, thereby minimizing execution time.
While HOPR has traditionally served as a reference implementation of a mesh generator for this process, PyHOPE is a modern alternative that enhances readability and extensibility.
Designed with excellent scaling and parallel computing capabilities in mind, PyHOPE offers a more user-friendly and adaptable solution for researchers and engineers working on high-order mesh generation and transformation.

[^1]: [https://numericsresearchgroup.org/codes.html](https://numericsresearchgroup.org/codes.html)
[^2]: [https://github.com/piclas-framework/piclas](https://github.com/piclas-framework/piclas)

# Features
PyHOPE facilitates high-order mesh generation and transformation through a streamlined and accessible interface.
PyHOPE is distributed as open-source software on GitHub[^3] and as package on the Python Package Index (PyPI)[^4].
It reads user input from a single plain-text configuration file in INI format, rendering setup and execution straightforward.
The software supports the generation of block‑structured meshes using canonical volumetric element types (tetrahedral, pyramidal, prismatic, and hexahedral), allowing for mesh stretching and post‑deformation at arbitrary polynomial orders.
PyHOPE can automatically create rectilinear boundary layer meshes based on a desired wall resolution or stretching factor.
It enables the read-in and merging of both internally and externally created curved meshes, incorporates boundary conditions, and automatically adapts sub-meshes to the desired polynomial order.
PyHOPE also offers conversion of simplex elements into fully hexagonal cells through geometric subdivision, as well as mesh sorting along structured dimensions or space-filling Hilbert curves.
Additionally, it calculates mesh connectivity information, supporting periodic and non-conforming interfaces with hanging nodes, while accommodating potential anisotropy.
To ensure robustness, PyHOPE performs comprehensive sanity checks, verifying mesh watertightness, correct surface orientation, and valid Jacobian mappings.
PyHOPE detects available simultaneous multithreading (SMT) capabilities and automatically enables process-based parallelism using the Python multiprocessing module.
Through these capabilities, PyHOPE provides a modern and efficient tool for high-order numerical simulations, with a strong emphasis on scalability and parallel processing.

[^3]: [https://github.com/hopr-framework/PyHOPE](https://github.com/hopr-framework/PyHOPE)
[^4]: [https://pypi.org/project/PyHOPE](https://pypi.org/project/PyHOPE)

# Examples
PyHOPE is used to generate unstructured high-order meshes for a variety of engineering applications, ranging from canonical cases such as channel flows or the Taylor-Green vortex to complex geometries such as airfoils and complete airplanes.
High-order meshes in HOPR format also find application in electromagnetics and plasma simulation, such as optical lenses and gyrotrons.
PyHOPE comes with several tutorials that are included in the [GitHub repository](https://github.com/hopr-framework/PyHOPE/tree/main/tutorials), together with the external mesh files where appropriate.
These tutorials cover both the creation of block-structured grids using PyHOPE's inbuilt functionality as well as the read-in of externally created meshes.
The available post-deformation options and topology conversion features are outlined as well.
All tutorial cases are also used for regression checking during Continuous Integration/Continuous Deployment (CI/CD).

One notable example from the field of aerospace engineering is the application of PyHOPE to the Common Research Model (CRM), a widely used aerodynamic benchmark.
For this example, the CAD file is initially meshed using second-order simplex elements in the external generator ANSA v24 and exported in high-order curved CGNS 4.2.0 format based on HDF5.
While the specific mesh used here does not include boundary layers, ANSA supports layered and mixed-element meshing, which has been successfully tested in other configurations.
The same methodology is also applicable to quasi-2D meshes in ANSA.
In all cases, a fully-curved volume mesh is generated using the internal second-order volume mesher.
PyHOPE reads the CGNS file, reconstructs boundary conditions, and creates a fully-curved HOPR mesh including connectivity information for vertices, edges, and sides.
The final mesh contains 147763 second-order elements and can be processed by PyHOPE on any standard machine using approximately 520 MB RAM (peak RSS).
Notably, this workflow enables full utilization of CFD meshes generated by commercial tools such as, but not limited to, ANSA or Pointwise, which are widely used in industry.
As a result, meshing complex geometries of industrial relevance -- such as automobiles or advanced aerospace configurations -- is no longer a limitation.
\autoref{fig:CRM} shows the second-order unstructured surface grid and the instantaneous distribution of surface pressure on the NASA CRM, respectively.
The latter was calculated using FLEXI, an open-source framework for the solution of the compressible Navier-Stokes equations using DGSEM.

![NASA Common Research Model (CRM). Left: Second-order unstructured surface grid. Right: Instantaneous distribution of surface pressure.\label{fig:CRM}](figures/CRM/CRM.jpg){width=100%}

# Related Software
PyHOPE shares similarities with other high-order mesh generation tools such as HOPR and can generate meshes for use in DGSEM frameworks.
However, with enhanced adaptability, improved usability, and support for advanced meshing techniques, PyHOPE goes beyond existing tools in enabling more complex mesh generation, rendering it a powerful alternative for researchers and engineers.
The following spectral element solvers have (optional) support for meshes generated in PyHOPE.

+--------------+-------------+-------------------------+--------------------------------------------+
| Framework    | Language    | Equation System         | Reference                                  |
+:=============+:============+:========================+:===========================================+
| FLEXI        | Fortran     | NSE                     | [@Krais2021]                               |
+--------------+-------------+-------------------------+--------------------------------------------+
| ƎLEXI        | Fortran     | NSE/MRG                 | [@Kopper2023]                              |
+--------------+-------------+-------------------------+--------------------------------------------+
| GALÆXI       | Fortran/C   | NSE                     | [@Kurz2025]                                |
+--------------+-------------+-------------------------+--------------------------------------------+
| FLUXO        | Fortran     | NSE/MHD/Maxwell         | [@RuedaRamirez2017]                        |
+--------------+-------------+-------------------------+--------------------------------------------+
| HORSES3D     | Fortran     | NSE/Cahn-Hilliard       | [@Ferrer2023]                              |
+--------------+-------------+-------------------------+--------------------------------------------+
| PICLas       | Fortran     | Maxwell/Poisson         | [@Fasoulas2019]                            |
+==============+=============+=========================+============================================+
| ~*Equation Systems: NSE - Navier-Stokes, MRG - Maxey-Riley-Gatignol, MHD - Magnetohydrodynamics*~ |
+==============+=============+===========================+==========================================+

# Acknowledgements
This work was funded by the European Union.
This work has received funding from the European High Performance Computing Joint Undertaking (JU) and Sweden, Germany, Spain, Greece, and Denmark under grant agreement No 101093393.
The research presented in this paper was funded in parts by the state of Baden-Württemberg under the project Aerospace 2050 MWK32-7531-49/13/7 "FLUTTER".

# References
