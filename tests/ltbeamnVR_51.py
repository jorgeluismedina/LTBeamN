
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections.section_bs import ISection_BS
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=0.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.2, tw=0.010, tf=0.015, r=0.0) #[m]



# ----- CONSTRUCCION DE LA MALLA --------
L = 19.5 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 200
# Con 160 elementos mu_cr = 9.4700, error con Ansys delta = 0.63%
# Con 200 elementos mu_cr = 9.4787, error con Ansys delta = 0.73%
# Con 500 elememtos mu_cr = 9.5000, error con Ansys delta = 0.95%


# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)

# Generacion de secciones
node_sections = [sect1] * coordinates.shape[0]

# Informacion de elementos
elements_data = []
for e in range(nelems):
    elements_data.append([0, 0, e, e+1])
elements_data = np.array(elements_data)



# ----- RESTRICCIONES --------
# Restricciones problema estatico
verax_restraints = np.array([
    [0,         1, 1, 0],
    [nelems/2,  1, 1, 0],
    [nelems,    1, 1, 0]
])

# restricciones problema de estabilidad
lator_restraints = np.array([
    [0,         1, 0, 1, 0],
    [nelems/2,  1, 0, 1, 0],
    [nelems,    1, 0, 1, 0]
])



# ----- CARGAS DE ELEMENTO --------
# Cargas distribuida uniforme
elem_loads = []
for e in range(nelems//4):
    elem_loads.append([e, 0,   0.0, -1000.0, 0.0, -1000.0])


for e in range(nelems//4+1, nelems):
    elem_loads.append([e, 0,   0.0, -3000.0, 0.0, -3000.0])

elem_loads = np.array(elem_loads)




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(node_sections)
model.add_nodes(coordinates)
model.add_uniform_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)


# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
solver1 = StaticSolver(model)
verax_disps, verax_react = solver1.solve()

# Resolcion del problema de estabilidad
solver2 = StabilitySolver(model)
mu_crs, modes = solver2.solve()

# Resultados y comparacion
mu_cr = mu_crs[0]
mu_cr_ansys = 9.4102
mu_cr_ltbeamn = 9.4267

print(f"Factor de carga critico μ_cr (PyLTB):   {mu_cr:.4f}")
print(f"Factor de carga critico μ_cr (Ansys):   {mu_cr_ansys:.4f}")
print(f"Factor de carga critico μ_cr (LTBeamN): {mu_cr_ltbeamn:.4f}")
print(f"Diff de resultados con Ansys:   {abs(mu_cr - mu_cr_ansys)/mu_cr_ansys * 100:.2f} %")
print(f"Diff de resultados con LTBeamN: {abs(mu_cr - mu_cr_ltbeamn)/mu_cr_ltbeamn * 100:.2f} %")


 

# ----- PLOTEO DE RESULTADOS --------
# Problema estatico
all_fields = solver1.generate_fields()
all_diagrams = solver1.prepare_diagrams(all_fields)

plot_diagram(model, all_diagrams[0], "Axial Force Diagram")
plot_diagram(model, all_diagrams[1], "Shear Force Diagram")
plot_diagram(model, all_diagrams[2], "Bending Moment Diagram")
plot_deformed(model, all_diagrams[3])

# Problema de estabilidad
plot_buckling_modes(model, mu_crs, modes) 
plt.show()