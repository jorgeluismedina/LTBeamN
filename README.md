# PyLTB — Lateral-Torsional Buckling FEM Solver

A finite element solver for **elastic lateral-torsional buckling (LTB)** of singly and doubly symmetric I-beams — uniform and linearly tapered — with arbitrary boundary conditions, intermediate restraints, and general load configurations.

The formulation follows the references below, implemented as a two-stage pipeline: a **static pre-analysis** (in-plane bending) feeds internal forces into a **linearized buckling eigenvalue problem** (out-of-plane stability).

---

## References

| Reference                                                       | Role in the code                                                    |
| --------------------------------------------------------------- | ------------------------------------------------------------------- |
| Andrade et al. (2005) — *LTB of singly symmetric tapered beams* | Tapered inertia terms $I_\psi$, $I_{w\psi}$, $I_{y\psi}$            |
| Beyer et al. (2015) — *Elastic stability with arbitrary BCs*    | Numerical assembly of $\mathbf{K}_g$ (Eq. 17 / 20), spring supports |

---

## Project Structure

```
src/
├── elements/
│   ├── base_beam.py       # Abstract base class Beam
│   ├── ltbeam.py          # LTBeam  — uniform element (closed-form integrals)
│   └── ltbeamtap.py       # LTBeamTap — tapered element (Gauss quadrature)
│
├── sections/
│   ├── section_bs.py      # ISection_BS — doubly symmetric I-section
│   ├── section_ms.py      # ISection_MS — monosymmetric I-section
│   └── section_utils.py   # Linear interpolation between sections
│
├── constructors.py         # ElementFactory — uniform and tapered factories
├── model.py                # StabilityModel — mesh, loads, BCs
├── material.py             # Material (E, ν, G)
│
├── static.py               # StaticSolver  — in-plane bending (Step 1)
├── stability.py            # StabilitySolver — LTB eigenvalue problem (Step 2)
│
├── shape_funcs.py          # Hermite cubic shape functions N, dN, d²N
├── gauss_quad.py           # 1D / nD Gauss–Legendre quadrature on [0,1]
└── plotting.py             # Diagrams: N, V, M, deformed shape, buckling modes
```

---

## Theoretical Background

### Degrees of Freedom

Each node carries two independent sets of DOFs:

**In-plane** (static problem, 3 DOF/node):

$$\mathbf{d}^\text{vrx} = \{u,\; w,\; w_{,x}\}$$

**Out-of-plane** (stability problem, 4 DOF/node):

$$\mathbf{d}^\text{ltr} = \{v,\; v_{,x},\; \theta,\; \theta_{,x}\}$$

### Stiffness Matrix

The elastic stiffness matrix for the lateral-torsional problem has the block structure:

$$\mathbf{K}_0^\text{ltr} = \begin{bmatrix} EI_z \mathbf{K}_{vv} & \mathbf{0} \\ \mathbf{0} & EI_w \mathbf{K}_{\theta\theta}^w + GI_t \mathbf{K}_{\theta\theta}^t \end{bmatrix}$$

For tapered elements, taper-induced coupling terms enter through $I_\psi$, $I_{w\psi}$, $I_{y\psi}$ via the constitutive matrix:

$$\mathbf{D}^\text{ltr} = \begin{bmatrix} EI_z & 0 & EI_{y\psi} \\ 0 & EI_w & EI_{w\psi} \\ EI_{y\psi} & EI_{w\psi} & GI_t + EI_\psi \end{bmatrix}$$

### Geometric Stiffness Matrix

The geometric stiffness assembles three contributions (Beyer 2015, Eq. 17):

$$\mathbf{K}_g = \underbrace{\mathbf{K}_g^N}_{\text{axial}} + \underbrace{\mathbf{K}_g^{MV}}_{\text{bending + shear}} + \underbrace{\mathbf{K}_g^Q}_{\text{load height}}$$

For a cross-section slice at $\xi$:

$$d\mathbf{K}_g = \left[ N \left( \mathbf{v}' \otimes \mathbf{v}' + i_0^2\, \mathbf{\theta}' \otimes \mathbf{\theta}' + z_S\, \text{sym}(\mathbf{v}' \otimes \mathbf{\theta}') \right) + M_y\, \text{sym}(\mathbf{v}' \otimes \mathbf{\theta}') - 2\beta_z M_y\, \mathbf{\theta}' \otimes \mathbf{\theta}' - V_z\, \text{sym}(\mathbf{v}' \otimes \mathbf{\theta}) + q_z e_z\, \mathbf{\theta} \otimes \mathbf{\theta} \right] dx$$

### Buckling Eigenvalue Problem

After applying boundary conditions, the critical load multiplier $\mu_{cr}$ satisfies:

$$(\mathbf{K}_0 - \mu_{cr}\, \mathbf{K}_g)\, \boldsymbol{\phi} = \mathbf{0}$$

Solved via `scipy.linalg.eigh` on the reduced free-DOF system.

---

## Element Types

### `LTBeam` — Uniform element

Uses **closed-form integrated** matrices: $\int_0^L N_i'' N_j''\,dx$, $\int_0^L N_i' N_j'\,dx$, etc. All section properties are constant. Wagner coefficient $\beta_z$, shear center $z_S$, and polar radius $i_0$ enter analytically.

### `LTBeamTap` — Tapered element

Uses **4-point Gauss quadrature** on $\xi \in [0,1]$. At each integration point:

1. Section geometry is linearly interpolated between end sections.
2. Taper inertias are computed by numerical differentiation of $I_w(\xi)$:

$$I_\psi = \frac{d^2 I_w}{dx^2}, \quad I_{w\psi} = \frac{d I_w}{dx}, \quad I_{y\psi} \approx 2\left(\dot{a_T}_1 I_{z,fa} - \dot{a_B}_2 I_{z,fb}\right)$$

---

## Cross-Section Types

### `ISection_BS` — Doubly symmetric I-section

Parameters: `h, bf, tw, tf, r`

Computed properties: $A$, $I_y$, $I_z$, $I_t$ (Saint-Venant + fillet correction), $I_w$, $i_0$. By symmetry: $z_S = 0$, $\beta_z = 0$.

### `ISection_MS` — Monosymmetric I-section

Parameters: `h, bf1, bf2, tw, tf1, tf2, r1, r2`

Computed properties: all of the above plus shear center $z_S$ (from $I_z$-weighted centroid), warping constant $I_w = \frac{I_{z,f1} I_{z,f2}}{I_{z,f1}+I_{z,f2}} h_s^2$, and Wagner coefficient:

$$\beta_z = \frac{1}{2I_y} \int_A z(y^2 + z^2)\,dA - z_S$$

---

## Workflow

```python
import numpy as np
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.model import StabilityModel
from src.static import StaticSolver
from src.stability import StabilitySolver

# 1. Material & sections
steel = Material(E=210e6, nu=0.3, dens=7850)
sec   = ISection_MS(h=0.5, bf1=0.2, bf2=0.15, tw=0.01, tf1=0.015, tf2=0.012, r1=0, r2=0)

# 2. Build model
model = StabilityModel()
model.add_materials([steel])
model.add_sections([sec] * (n_nodes))
model.add_nodes(coordinates)           # (n_nodes, 1) array of x-coordinates
model.add_uniform_elements(elem_data)  # [[etype, mat_id, nodei, nodej], ...]

# 3. Boundary conditions & loads
model.add_verax_restraints(vrx_data)   # in-plane restraints
model.add_lator_restraints(ltr_data)   # out-of-plane restraints
model.add_elem_loads(load_data)        # distributed loads

# 4. Static pre-analysis (computes N, V, M used in Kg)
static = StaticSolver(model)
static.solve()

# 5. Buckling analysis
stab = StabilitySolver(model)
stab.solve()

print("Critical load multipliers:", stab.mu_crs[:3])
```

---

## Load & Restraint Encoding

### Element loads (`add_elem_loads`)

```
[id_elem, qxpos, qzpos, qxez, qzez, qxi, qzi, qxj, qzj]
```

| Field            | Meaning                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `qxpos`, `qzpos` | Height code for axial / transverse load: `0`→G, `1`→SC, `2`→bottom flange, `3`→top flange |
| `qxez`, `qzez`   | Additional eccentricity relative to the height code (signed)                              |
| `qxi`, `qzi`     | Intensities at node $i$ (axial, transverse)                                               |
| `qxj`, `qzj`     | Intensities at node $j$                                                                   |

### Nodal loads (`add_nodal_loads`)

```
[node, fxpos, fzpos, fxez, fzez, Fx, Fz, Mx]
```

### Lateral springs (`add_lateral_springs`)

```
[node, pos, kv, kt]
```

`kv` = lateral translational stiffness, `kt` = torsional stiffness, applied at height `pos`.

---

## Dependencies

```
numpy
scipy
matplotlib
```

---

## Key Design Decisions

- **Two separate DOF spaces**: the in-plane (vrx) and out-of-plane (ltr) problems are fully decoupled in assembly; internal forces flow from static → stability as load parameters.
- **Tapered elements via Gauss quadrature**: closed-form integrals are replaced by numerical integration, with section properties recomputed at each Gauss point including taper corrections.
- **No global coordinate transformation**: elements are assumed collinear (1D beam axis), so local = global in-plane coordinates.
- **Shear center as stability reference**: $\mathbf{K}_g$ is built in the shear-center frame; load height effects enter as $q_z e_z \,\theta^2$ contributions.
