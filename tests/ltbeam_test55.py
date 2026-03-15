
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from src.model import StabilityModel
from src.material import Material
from src.sections import ISection_MS
from src.solvers.static import StaticSolver
from src.solvers.stability import StabilitySolver
from src.plotting import plot_buckling_modes, plot_diagram, plot_deformed

# Materiales
material1 = Material(E=2.1e11, nu=0.3, dens=0.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_MS(h=0.3, bf1=0.15, bf2=0.20, 
                    tw=0.01, tf1=0.012, tf2=0.015, r1=0.0, r2=0.0) #[m]

sections = [sect1]
sect1.summary()



# ----- CONSTRUCCION DE LA MALLA --------
L = 19.5 #[m]
# numero de elementos pares para que exista un nodo en el centro
nelems = 150 
# Con 150 elementos mu_cr = 8.0760, error con Ansys delta = 0.14%
# Con 250 elementos mu_cr = 8.0911, error con Ansys delta = 0.32%
# Con 350 elementos mu_cr = 8.0978, error con Ansys delta = 0.41%

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



# ----- CARGAS NODALES --------
nodal_loads = np.array([
    [nelems/4,    0, -10000, 0]
])




# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(coordinates)
model.add_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)


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


plot_diagram(model, all_diagrams[0], "Axial Force Diagram")
plot_diagram(model, all_diagrams[1], "Shear Force Diagram")
plot_diagram(model, all_diagrams[2], "Bending Moment Diagram")
plot_deformed(model, all_diagrams[3])

# Problema de estabilidad
plot_buckling_modes(model, mu_crs, modes) 
plt.show()