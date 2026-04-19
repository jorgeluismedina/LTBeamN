
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections.section_ms import ISection_MS
from src.sections.utils import interpolate_multiple_sections
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed

# Materiales
material1 = Material(E=2.10e11, nu=0.3, dens=1.0)
materials = [material1]

# Secciones
section1 = ISection_MS(h=0.6127, bf1=0.15, bf2=0.15, 
                       tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]

section2 = ISection_MS(h=0.6127*0.2, bf1=0.15, bf2=0.15, 
                       tw=0.0095, tf1=0.0127, tf2=0.0127, r1=0.00, r2=0.00) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
L = 4 #[m]
nelems = 20

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Generacion de secciones
node_sections = interpolate_multiple_sections(section1, section2, norm_coords)


# Informacion de elementos
elements_data = []
for e in range(nelems):
    # formato: [etype, mat_id, nodei, nodej]
    elements_data.append([1, 0, e, e+1]) 

elements_data = np.array(elements_data)


# ----- RESTRICCIONES --------
# Empotramiento
verax_restraints = np.array([
    [0,       1, 1, 1],
])
# Empotramiento
lator_restraints = np.array([
    [0,       1, 1, 1, 1],
])


# ----- CARGAS NODALES --------
# Carga puntual en la punta sobre la mesa superior
idx = 2
ratios = [0, 1, 2, 4]
r = ratios[idx]
nodal_loads = np.array([
    [nelems, 3,   r*-50000.0, -50000.0, 0.0]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)



# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
solver1 = StaticSolver(model)
solver1.solve()
maxN, maxV, maxM, maxw = solver1.max_vals() 

# Resolcion del problema de estabilidad
solver2 = StabilitySolver(model)
solver2.solve()
mu_cr = solver2.mu_crs[0]

# Resultados y comparacion
mu_cr_ref     = [1.979, 1.742, 1.475, 1.006]
mu_cr_ltbeamn = [1.943, 1.722, 1.469, 1.010]


print("\n" + "="*55)
print(" ANALYSIS RESULTS ".center(55))
print("="*55)

print("\n MESH DATA")
print(f"  Number of nodes:                 {model.nnods:>20}")
print(f"  Number of elements:              {model.nelems:>20}")

print("\n STATIC ANALYSIS")
print(f"  Axial max.        Nmax:          {maxN/1e3:>16.4f} kN")
print(f"  Shear max.        Vmax:          {maxV/1e3:>16.4f} kN")
print(f"  Moment max.       Mmax:          {maxM/1e3:>16.4f} kNm")
print(f"  Displacement max. w_max:         {maxw*1e3:>16.4f} mm")

print("\n STABILITY ANALYSIS")
print(f"  Ratio r=N/Q:                            {r:>12}")
print(f"  Critical load factor μ_cr (PyLTB):      {mu_cr:>12.4f}")
print(f"  Critical load factor μ_cr (Reference):  {mu_cr_ref[idx]:>12.4f}")
print(f"  Critical load factor μ_cr (LTBeamN):    {mu_cr_ltbeamn[idx]:>12.4f}")
print(f"  Result diff. with Reference:            {abs(mu_cr - mu_cr_ref[idx])/mu_cr_ref[idx]*100:>11.2f} %")
print(f"  Result diff. with LTBeamN:              {abs(mu_cr - mu_cr_ltbeamn[idx])/mu_cr_ltbeamn[idx]*100:>11.2f} %")
print("\n" + "="*55 + "\n")



"""
# ----- PLOTEO DE RESULTADOS --------
# Problema estatico
all_diagrams = solver1.prepare_diagrams()

plot_diagram(model, all_diagrams[0], "Axial Force Diagram")
plot_diagram(model, all_diagrams[1], "Shear Force Diagram")
plot_diagram(model, all_diagrams[2], "Bending Moment Diagram")
plot_deformed(model, all_diagrams[3])

# Problema de estabilidad
plot_buckling_modes(model, solver2.mu_crs, solver2.modes) 
plt.show()
"""