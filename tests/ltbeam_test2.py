
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
material1 = Material(E=2.1e11, nu=0.2, dens=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.15, tw=0.015, tf=0.015, r=0.01) #[m]

sections = [sect1]
sect1.summary()
#sect2.summary()



# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 26 #Con 25 elementos ya se alcanza el valor teorico de momento critico

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
elements_data = []

# Informacion de elementos
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
model.add_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_elem_loads(elem_loads)


# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
solver1 = StaticSolver(model)
verax_disps, verax_react = solver1.solve()
#print(model.elems[0].K0_ltr)
#print(verax_disps.reshape(mod.nnods, mod.nvrx_dofn))

# Resolcion del problema de estabilidad
solver2 = StabilitySolver(model)
mu_crs, modes = solver2.solve()


# verificacion para flexion pura
EIz = material1.E * sect1.Iz
GIt = material1.G * sect1.It
EIw = material1.E * sect1.Iw
M_critico_ana = np.pi / L * np.sqrt(EIz*GIt * (1 + (np.pi**2*EIw)/(L**2*GIt)))
M_critico_num = mu_crs[0]
print(f"Momento Crítico Calculado: {M_critico_num/1000:.4f} kNm")

 
print(model.elems[-1].forces)

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

