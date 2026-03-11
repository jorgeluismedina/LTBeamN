
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
from src.plotting import plot_buckling_modes, plot_1d_diagram

# Materiales
material1 = Material(E=2.1e11, nu=0.2, dens=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.15, tw=0.015, tf=0.015, r=0.01) #[m]

sect2 = ISection_MS(h=0.3, bf1=0.2, bf2=0.12, 
                    tw=0.015, tf1=0.015, tf2=0.015, 
                    r1=0.01, r2=0.01) #[m]

sections = [sect1, sect2]
sect1.summary()
#sect2.summary()



# ----- CONSTRUCCION DE LA MALLA --------
L = 5 #[m]
nelems = 25 #Con 25 elementos ya se alcanza el valor teorico de momento critico

# Coordenadas de nodos
coordinates = np.linspace(0, L, nelems+1)[:,None] 
elements_data = []

# Informacion de elementos
for e in range(nelems):
    elements_data.append([1, 0, 0, e, e+1]) # etype, mat_id, sec_id, nodei, nodej
elements_data = np.array(elements_data)



# ----- RESTRICCIONES --------
verax_restraints = np.array([
    [0, 1, 1, 0],
    [nelems, 1, 1, 0]
])

lator_restraints = np.array([
    [0, 1, 0, 1, 0],
    [nelems, 1, 0, 1, 0]
])


# ----- CARGAS NODALES --------
# Carga de flexion pura unitaria
nodal_loads = np.array([
    [0, 0, 0, -1],
    [nelems, 0, 0, 1]
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
#print(model.elems[0].K0_ltr)
#print(verax_disps.reshape(mod.nnods, mod.nvrx_dofn))

# Resolcion del problema de estabilidad
solver2 = StabilitySolver(model)
mu_cr, modes = solver2.solve()
modes = solver2.reconstruct_full_modes(modes, nmodes=5)


# verificacion para flexion pura
EIz = material1.E * sect1.Iz
GIt = material1.G * sect1.It
EIw = material1.E * sect1.Iw
M_critico_ana = np.pi / L * np.sqrt(EIz*GIt * (1 + (np.pi**2*EIw)/(L**2*GIt)))
M_critico_num = mu_cr[0]
print(f"Momento Crítico Calculado: {M_critico_num/1000:.4f} kNm")
print(f"Momento Crítico Teorico: {M_critico_ana/1000:.4f} kNm")
 


# ----- PLOTEO DE RESULTADOS --------
# Problema estatico
all_fields = solver1.generate_fields()
plot_1d_diagram(model.elems, all_fields[0],  all_fields[1])
plot_1d_diagram(model.elems, all_fields[0],  all_fields[2])
plot_1d_diagram(model.elems, all_fields[0],  all_fields[3])


# Problema de estabilidad
plot_buckling_modes(mu_cr, modes, model) 
plt.show()
