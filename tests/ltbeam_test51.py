
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections import ISection_BS, ISection_MS
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=0.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.2, tw=0.010, tf=0.015, r=0.0) #[m]

sections = [sect1]
sect1.summary()



# ----- CONSTRUCCION DE LA MALLA --------
L = 19.5 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 200
# Con 160 elementos mu_cr = 9.4700, error con Ansys delta = 0.63%
# Con 200 elementos mu_cr = 9.4787, error con Ansys delta = 0.73%
# Con 500 elememtos mu_cr = 9.5000, error con Ansys delta = 0.95%


# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
elements_data = []

# Informacion de elementos
for e in range(nelems):
    elements_data.append([1, 0, 0, e, e+1]) # etype, mat_id, sec_id, nodei, nodej
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
    elem_loads.append([e,   0, -1000, 0, -1000]) # id_elem, q1i, q2i, q1j, q2j


for e in range(nelems//4+1, nelems):
    elem_loads.append([e,   0, -3000, 0, -3000]) # id_elem, q1i, q2i, q1j, q2j

elem_loads = np.array(elem_loads)




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(coordinates)
model.add_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)


# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
solver1 = StaticSolver(model)
verax_disps, verax_react = solver1.solve()
#print(verax_react)
#print(verax_disps.reshape(model.nnods, model.nvrx_dofn))

# Resolcion del problema de estabilidad
solver2 = StabilitySolver(model)
mu_crs, modes = solver2.solve()
print(f"factor de carga critico μ_cr: {mu_crs[0]:.4f}")

 

# ----- PLOTEO DE RESULTADOS --------
# Problema estatico
all_fields = solver1.generate_fields()
all_diagrams = solver1.prepare_diagrams(all_fields)

#print(all_fields[3][-1])

plot_diagram(model, all_diagrams[0], "Axial Force Diagram")
plot_diagram(model, all_diagrams[1], "Shear Force Diagram")
plot_diagram(model, all_diagrams[2], "Bending Moment Diagram")
plot_deformed(model, all_diagrams[3])

# Problema de estabilidad
plot_buckling_modes(model, mu_crs, modes) 
plt.show()