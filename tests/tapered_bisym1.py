
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
material1 = Material(E=2.1e11, nu=0.3, dens=1.0) #[N/m2] # cambio a nu=0.3 por que LTBeamN no me deja cambiar a 0.2
materials = [material1]

# Secciones
section1 = ISection_MS(h=0.3, bf1=0.20, bf2=0.20, 
                       tw=0.01, tf1=0.015, tf2=0.015, r1=0.0, r2=0.0) #[m]

section2 = ISection_MS(h=0.2, bf1=0.15, bf2=0.15, 
                       tw=0.01, tf1=0.015, tf2=0.015, r1=0.0, r2=0.0) #[m]




# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 200 

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)
norm_coords = coordinates / L

# Generacion de secciones
sections = interpolate_multiple_sections(section1, section2, norm_coords)




# Informacion de elementos
elements_data = []
for e in range(nelems):
    # etype, mat_id, sec_id1, sec_id2, nodei, nodej
    elements_data.append([2, 0, e, e+1, e, e+1]) 

elements_data = np.array(elements_data)


# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0,       1, 1, 0],
    [nelems,  0, 1, 0]
])

lator_restraints = np.array([
    [0,       1, 0, 1, 0],
    [nelems,  1, 0, 1, 0]
])


# ----- CARGAS NODALES --------
# Carga de flexion pura unitaria
nodal_loads = np.array([
    [0,       0, 0, -1000],
    [nelems,  0, 0,  1000]
])


# ----- CREACION Y SETEO DEL MODELO -------- 
model = StabilityModel()
model.add_materials(materials)
model.add_sections(sections)
model.add_nodes(coordinates)
model.add_tapered_elements(elements_data)
model.add_verax_restraints(verax_restraints)
model.add_lator_restraints(lator_restraints)
model.add_nodal_loads(nodal_loads)


#print(model.elems[0].dh1, model.elems[0].dh2)
#print(model.elems[1].dh1, model.elems[1].dh2)

# ----- RESOLUCION DEL MODELO --------
# Resolucion del problema estatico
solver1 = StaticSolver(model)
verax_disps, verax_react = solver1.solve()

# Resolucion del problema de estabilidad
solver2 = StabilitySolver(model)
mu_crs, modes = solver2.solve()

# Resultados y comparacion
mu_cr = mu_crs[0] 
mu_cr_ltbeamn = 241.83
print(f"Factor de carga critico μ_cr (PyLTB):   {mu_cr:.4f}")
print(f"Factor de carga critico μ_cr (LTBeamN): {mu_cr_ltbeamn:.4f}")
print(f"Diferencia de resultados: {abs(mu_cr - mu_cr_ltbeamn)/mu_cr_ltbeamn*100:.2f} %")



"""
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
"""
