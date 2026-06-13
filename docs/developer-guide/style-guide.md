# Style Guide

Rules and recommendations for code style

---

A consistent style across the codebase reduces friction when reading and reviewing code written by others. Where a rule is marked _must_, it is mandatory; where it is marked _should_, it is a strong recommendation that may be deviated from only with good reason.

**Why do we need a style guide?**

  - It creates a unified appearance and coding structure
  - It makes the code more understandable and therefore important information is understood more easily
  - It forces the developers to think more actively about their work

**General rules**

  - Coding language: English
  - A maximum of 132 characters are allowed per line (incl. comments)
  - Indentation: 4 spaces (no tabs!)
  - Line breaks in comments _must_ be followed by the next line indented appropriately

## Nested Functions
Code _should_ avoid nested functions. Nesting a function inside another function creates a new scope that is difficult to test in isolation, obscures the control flow, and can introduce subtle bugs through closure over mutable variables. Instead, helper logic _should_ be extracted into a separate, named function at module or class level.

**Avoid:**
```python
def GenerateMesh() -> None:
    """ Generate the mesh 
    """
    def _build_cartesian():
        # ... builds the mesh ...
        pass

    mesh = _build_cartesian()
```

**Prefer:**
```python
def MeshCartesian() -> meshio.Mesh:
    """ Build and return a Cartesian mesh 
    """
    # ... builds the mesh ...

def GenerateMesh() -> None:
    """ Generate the mesh
        Mode 1 - Use internal mesh generator
        Mode 2 - Readin external mesh through GMSH
    """
    mesh = MeshCartesian()
```

## Object-Oriented Programming
Code _should_ follow object-oriented programming principles. Related data and the functions that operate on it _should_ be grouped into classes rather than kept as loose collections of module-level variables and free functions. Use `@dataclass` for lightweight data containers and reserve full class definitions for objects that carry behavior.

**Example:** Data container using `@dataclass`
```python
from dataclasses import dataclass
from typing import Optional
import numpy.typing as npt

@dataclass(init=True, repr=False, eq=False, slots=True)
class ELEM:
    type   : Optional[int]         = None
    zone   : Optional[int]         = None
    elemID : Optional[int]         = None
    sides  : Optional[list]        = None
    nodes  : Optional[npt.NDArray] = None

    # Comparison operator for bisect
    def __lt__(self, other) -> bool:
        return self.elemID < other.elemID
```

The `slots=True` argument instructs Python to use `__slots__` instead of the default per-instance `__dict__`. Normally, each instance stores its attributes in a dictionary, which is flexible but carries memory and access overhead. With `__slots__`, Python instead allocates a fixed, compact block of memory whose layout is fixed at class definition time.

**Example:** Class with properties and methods
```python
@singleton
class Common():
    def __init__(self: Self) -> None:
        self._program: Final[str] = self.__program__
        self._version: Final      = self.__version__

    @property
    @cache
    def __version__(self) -> Version:
        """ Retrieve the package version from installed metadata 
        """
        ...
```

## Global Variables
Code _must_ avoid global variables if possible. Global mutable state makes the execution order matter in non-obvious ways, hinders testing, and creates hidden dependencies between modules. Where a value must be shared across modules, it _should_ be encapsulated in a dedicated variables module (e.g. `mesh_vars.py`) with a clear type annotation, so that the origin and type of each value is always explicit.

**Avoid:**
```python
# At the top of a file (hard to trace, easy to accidentally overwrite)
nGeo = 1
mesh = None
```

**Prefer**: Explicit module-level declarations with type annotations
```python
nGeo : int          # Order of spline-reconstruction for curved surfaces
mesh : meshio.Mesh  # MeshIO object holding the mesh
bcs  : list[Optional['BC']]  # Boundary conditions
```

Consumers then import the module by name, making the provenance of each value clear at every call site. Import _must_ use the full namespace path.
```python
import pyhope.mesh.mesh_vars as mesh_vars

mesh_vars.nGeo = GetInt('NGeo')
```

## Function Signatures
Functions _should_ always contain type annotations for the arguments and the return values. Functions _should_ contain a brief description in the header.
```python
def change_basis_3D(Vdm: npt.NDArray[np.float64], x3D_In: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Interpolate a 3D tensor product Lagrange basis defined by (N_in+1) 1D interpolation point positions xi_In(0:N_In)
        to another 3D tensor product node positions (number of nodes N_out+1)
        defined by (N_out+1) interpolation point positions xi_Out(0:N_Out)
        xi is defined in the 1D reference element xi=[-1,1]
    """
```
## Functions and Control Structures
User-defined functions and subroutines _should_ carry meaning in their name. If their name is composed of multiple words, they are to be joined together without underscores (`_`) and the first letter of each word _should_ be capitalised (camelCase). Furthermore, the words `Get`, `get`, `Is`, `is`, `Do`, `do` at the start of a function name signal the function's intent.

**Examples:**
```python
# "Get" — retrieves a typed parameter value from the configuration
def GetInt(name: str, default: Optional[str] = None, number: Optional[int] = None) -> int:
    ...

# "Is" — queries an environmental condition, returns bool
def IsInteractive() -> bool:
    """ Check if the program is running in an interactive terminal
    """
    ...
```

Functions that neither retrieve a value nor test a condition _should_ use a descriptive verb that names the action performed, such as `Build`, `Write`, `Connect`, `Generate`, or `Check`.

## Workflow Description
In addition to the header description, a short workflow table of contents at the beginning of the file or function _should_ be included for longer subroutines in which multiple tasks are completed. The table of contents lists each logical step as a numbered comment, so a reader can grasp the overall structure before reading the body. Furthermore, the steps _must_ be found at the appropriate position within the code as a matching numbered comment. The step number _must not_ just appear once in the header and then disappear.
