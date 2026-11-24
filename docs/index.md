# PyHOPE {: .home-page }

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

<div class="text-center" style="margin-bottom: 2em;" markdown="1">
[![HOPR Logo](assets/hopr-logo.png){ .hopr-logo } HOPR Code](https://github.com/hopr-framework/hopr){ .btn .btn-primary .btn-spacing .btn-hopr }
[![HOPR Logo](assets/hopr-logo.png){ .hopr-logo } HOPR Documentation](https://hopr.readthedocs.io){ .btn .btn-primary .btn-spacing .btn-hopr }
</div>

PyHOPE has been developed by the Numerics Research Group (NRG) led by Prof. Andrea Beck at the Institute of Aerodynamics and Gas Dynamics at the University of Stuttgart, Germany.

## Citation

This is a scientific project. If you use PyHOPE for publications or presentations in science, please support the project by citing the following article:

> Kopper, P., Blind, M. P., Schwarz, A., Kurz, M., Rodach, F., Copplestone, S. M., & Beck, A. D. (2025). PyHOPE: A Python Toolkit for Three-Dimensional Unstructured High-Order Meshes. *Journal of Open Source Software*, 10(115), 8769. [https://doi.org/10.21105/joss.08769](https://doi.org/10.21105/joss.08769)

??? quote "BibTeX"

    ```bibtex
    @article{kopper2025pyhope:joss,
      title        = {{PyHOPE}: A Python Toolkit for Three-Dimensional Unstructured High-Order Meshes},
      author       = {Kopper, Patrick and Blind, Marcel P. and Schwarz, Anna and Kurz, Marius and Rodach, Felix and Copplestone, Stephen M. and Beck, Andrea D.},
      journal      = {Journal of Open Source Software},
      year         = {2025},
      volume       = {10},
      number       = {115},
      pages        = {8769},
      publisher    = {The Open Journal},
      doi          = {10.21105/joss.08769}
    }
    ```

The repository can also be cited using its Zenodo DOI at [10.5281/zenodo.17414843](https://doi.org/10.5281/zenodo.17414843).

??? quote "BibTeX (Zenodo)"

    ```bibtex
    @misc{kopper2025pyhope:repo,
      title        = {{PyHOPE}: A Python Toolkit for Three-Dimensional Unstructured High-Order Meshes},
      author       = {Kopper, Patrick and Blind, Marcel P. and Schwarz, Anna and Kurz, Marius and Rodach, Felix and Copplestone, Stephen M. and Beck, Andrea D.},
      year         = {2025},
      howpublished = {\url{https://github.com/hopr-framework/PyHOPE}},
      doi          = {10.5281/zenodo.17414843}
    }
    ```
