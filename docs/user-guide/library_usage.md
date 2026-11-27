---
icon: lucide/book-marked
---
# Library Usage

Using PyHOPE as a Python Library

---

PyHOPE can be included in other Python libraries. PyHOPE exposes its functionality via runtime contexts defined by [Context Managers](https://docs.python.org/3/library/stdtypes.html#typecontextmanager). The following Python code loads a HOPR HDF5 mesh and derived quantities.
```python
from pyhope import Basis, Mesh
with Mesh('1-01-cartbox_mesh.h5') as m:
    elems = m.elems
    lobatto_nodes = Basis.legendre_gauss_lobatto_nodes(order=m.nGeo)
```

## Currently implemented functions

Library functionality of PyHOPE is geared towards usage in post-processing tools which require mesh information to reconstruct meaningful quantities, for example, solution gradients for Schlieren visualization. Functions currently exposed through the Python interface are the following.
<!-- For a complete interface definition of currently implemented functions, see the [source code](https://github.com/hopr-framework/PyHOPE/blob/main/pyhope/__init__.py). -->

### Basis
#### legendre_gauss_nodes {: .api }
```python
legendre_gauss_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
""" Return Legendre-Gauss nodes and weights for a given order in 1D
"""
```
#### legendre_gauss_lobatto_nodes {: .api }
```python
legendre_gauss_lobatto_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
""" Return Legendre-Gauss-Lobatto nodes and weights for a given order in 1D
"""
```
#### equi_nodes_prism {: .api }
```python
equi_nodes_prism(order: int) -> np.ndarray:
""" Return equidistant nodes on a wedge/prism
"""
```
#### equi_nodes_pyram {: .api }
```python
equi_nodes_pyram(order: int) -> np.ndarray:
""" Return equidistant nodes on a pyramid
"""
```
#### equi_nodes_tetra {: .api }
```python
equi_nodes_tetra(order: int) -> np.ndarray:
""" Return equidistant nodes on a tetrahedron
"""
```
#### barycentric_weights {: .api }
```python
barycentric_weights(_: int, xGP: np.ndarray) -> np.ndarray:
""" Compute the barycentric weights for a given node set
    > Algorithm 30, Kopriva (2009), Implementing Spectral Methods for Partial Differential Equations
"""
```
#### polynomial_derivative_matrix {: .api }
```python
polynomial_derivative_matrix(order: int, xGP: np.ndarray) -> np.ndarray:
""" Compute the polynomial derivative matrix for a given node set
    > Algorithm 37, Kopriva (2009), Implementing Spectral Methods for Partial Differential Equations
"""
```
#### polynomial_derivative_matrix_prism {: .api }
```python
polynomial_derivative_matrix_prism(order: int, xGP: np.ndarray) -> np.ndarray:
""" Compute the polynomial derivative matrix for a prism
"""
```
#### polynomial_derivative_matrix_pyram {: .api }
```python
polynomial_derivative_matrix_pyram(order: int, xGP: np.ndarray) -> np.ndarray:
""" Compute the polynomial derivative matrix for a pyramid
"""
```
#### polynomial_derivative_matrix_tetra {: .api }
```python
polynomial_derivative_matrix_tetra(order: int, xGP: np.ndarray) -> np.ndarray:
""" Compute the polynomial derivative matrix for a tetrahedron
"""
```
#### lagrange_interpolation_polys {: .api }
```python
lagrange_interpolation_polys(x: Union[float, np.ndarray], order: int, xGP: np.ndarray, wBary: np.ndarray) -> np.ndarray:
""" Computes all Lagrange functions evaluated at position x in [-1;1]
    > Algorithm 34, Kopriva
"""
```
#### calc_vandermonde {: .api }
```python
calc_vandermonde(n_In: int, n_Out: int, wBary_In: np.ndarray, xi_In: np.ndarray, xi_Out: np.ndarray) -> np.ndarray:
""" Build a 1D Vandermonde matrix using the Lagrange basis functions of degree N_In,
    evaluated at the interpolation points xi_Out
"""
```
#### change_basis_3D {: .api }
```python
change_basis_3D(Vdm: np.ndarray, x3D_In: np.ndarray) -> np.ndarray:
""" Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
    to another 3D tensor product node positions (number of nodes N_out+1)
    defined by (N_out+1) interpolation point  positions xi_Out(0:N_Out)
    xi is defined in the 1D reference element xi=[-1,1]
"""
```
#### change_basis_2D {: .api }
```python
change_basis_2D(Vdm: np.ndarray, x2D_In: np.ndarray) -> np.ndarray:
""" Interpolate a 2D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
    to another 2D tensor product node positions (number of nodes N_out+1)
    defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
    xi is defined in the 1D reference element xi=[-1,1]
"""
```
#### change_basis_1D {: .api }
```python
change_basis_1D(Vdm: np.ndarray, x1D_In: np.ndarray) -> np.ndarray:
""" Interpolate a 1D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
    to another 1D tensor product node positions (number of nodes N_out+1)
    defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
    xi is defined in the 1D reference element xi=[-1,1]
"""
```
#### evaluate_jacobian {: .api }
```python
evaluate_jacobian(xGeo_In: np.ndarray, VdmGLtoAP: np.ndarray, D_EqToGL: np.ndarray) -> np.ndarray:
""" Calculate the Jacobian of the mapping for a given element
"""
```
#### evaluate_jacobian_simplex {: .api }
```python
evaluate_jacobian_simplex(xGeo_In: np.ndarray, _: Any, D_EqToGL: np.ndarray) -> np.ndarray:
""" Calculate the Jacobian of the mapping for a given simplex element
"""
```
### Mesh
Context manager to generate a mesh from a given file.

#### Mesh {: .api }
```python
Mesh(*args: str, stdout: bool = False, stderr: bool = True):
""" Mesh context manager to generate a mesh from a given file

    Args:
        *args: The mesh file path(s) to be processed
        stdout (bool): If False, standard output is suppressed
        stderr (bool): If False, standard error  is suppressed

    Yields:
        Mesh: An object containing the generated mesh and its properties
"""
```
