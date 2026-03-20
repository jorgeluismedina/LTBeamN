
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
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_MS(h=0.3, bf1=0.2, bf2=0.12, 
                    tw=0.01, tf1=0.01, tf2=0.01, 
                    r1=0.01, r2=0.01) #[m]
sections = [sect1]




# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 50 #Con 25 elementos ya se alcanza el valor teorico de momento critico

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)

# Informacion de elementos
elements_data = []

for e in range(nelems):
    elements_data.append([1, 0, 0, e, e+1]) # etype, mat_id, sec_id, nodei, nodej
elements_data = np.array(elements_data)



# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  1, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])



# ----- CARGAS DE ELEMENTO --------
# Carga distribuida uniforme unitaria
elem_loads = []
for e in range(nelems):
    elem_loads.append([e,   0, -1, 0, -1]) # id_elem, q1i, q2i, q1j, q2j
elem_loads = np.array(elem_loads)




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
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

print(f"Momento Crítico Calculado: {mu_crs[0]/1000:.4f} kNm")
print(mu_crs[0]/1000, mu_crs[1]/1000, mu_crs[2]/1000, mu_crs[3]/1000)
 

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