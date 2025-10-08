# Library Usage

Using PyHOPE as a Python Library

---

PyHOPE can be included in other Python libraries. PyHOPE exposes its functionally via runtime contexts defined by [Context Managers](https://docs.python.org/3/library/stdtypes.html#typecontextmanager). The following Python code loads a HOPR HDF5 mesh and derived quantities.
```python
from pyhope import Basis, Mesh
with Mesh('1-01-cartbox_mesh.h5') as m:
    elems = m.elems
    lobatto_nodes = Basis.legendre_gauss_lobatto_nodes(order=m.nGeo)
```

## Currently implemented functions

Library functionally of PyHOPE is geared towards usage in post-processing tools which require mesh information to reconstruct meaningful quantities, for example, solution gradients for Schlieren visualization. Functions currently exposed through the Python interface are the following. For a complete interface definition of currently implemented functions, see the [source code](https://github.com/hopr-framework/PyHOPE/blob/main/pyhope/__init__.py).

- Basis
    - legendre_gauss_nodes
    - legendre_gauss_lobatto_nodes
    - barycentric_weights
    - polynomial_derivative_matrix
    - lagrange_interpolation_polys
    - calc_vandermonde
    - change_basis_3D
    - change_basis_2D
    - evaluate_jacobian
- Mesh
    - *Context manager to generate a mesh from a given file*
