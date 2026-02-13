
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from ltbeamn.femclass import StabilityModel, Model
from ltbeamn.material import Material
from ltbeamn.sections import ISection_BS, ISection_MS
from ltbeamn.solvers import solve_linear_static2, solve_stability, buckling_modes, check_symmetric
from ltbeamn.plotting import plot_buckling_modes, plot_1d_diagram

# Materiales
material1 = Material(elast=2.1e11, poiss=0.2, dense=1.0) #[N/m2]
materials = [material1]

# Secciones
sect1 = ISection_BS(h=0.3, bf=0.15, tw=0.015, tf=0.015, r=0.01) #[m]

sect2 = ISection_MS(h=0.3, bf1=0.2, bf2=0.12, 
                    tw=0.015, tf1=0.015, tf2=0.015, 
                    r1=0.01, r2=0.01) #[m]

sections = [sect1, sect2]

# Beam Length
L = 5 #[m]
nelems = 10

coordinates = np.linspace(0, L, nelems+1)[:,None] 
sect1.summary()
#sect2.summary()

mod = StabilityModel(ndofn1=3, ndofn2=4)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

for j in range(nelems):
    mod.add_element('LTBeam', material1, sect1, [j, j+1])


mod.add_node_restraint1(0, [1, 1, 0]) # 1 significa grado de libertad restricto y 0 libre 
mod.add_node_restraint1(nelems, [1, 1, 0])

mod.add_node_restraint2(0, [1, 0, 1, 0]) # 1 significa grado de libertad restricto y 0 libre 
mod.add_node_restraint2(nelems, [1, 0, 1, 0])


mod.set_restraints1()
mod.set_restraints2()

# ----- CARGAS --------
#mod.add_node_load(0, [0, 0, -1]) # [N]
#mod.add_node_load(nelems, [0, 0, 1]) # [N]

for e in range(nelems):
    mod.add_elem_load(e, [0,-1,0,-1]) #[N]


# SOLUCION
glob_disps, reactions = solve_linear_static2(mod)
#print(glob_disps.reshape((mod.nnods, mod.ndofn1)), '\n')

mod.calculate_forces(glob_disps)
for elem in mod.elems:
    print(elem.forces) 


all_fields = mod.generate_fields()
ax1 = plot_1d_diagram(mod.elems, all_fields[0],  all_fields[1])
ax1 = plot_1d_diagram(mod.elems, all_fields[0],  all_fields[2])
ax1 = plot_1d_diagram(mod.elems, all_fields[0],  all_fields[3])

print(2/3)
print(float(2.0)/float(3.0))

# ESTABILIDAD
vals, vecs = solve_stability(mod) 

M_critico_num = vals[0]

EIz = material1.elast * sect1.Iz
GIt = material1.shear * sect1.It
EIw = material1.elast * sect1.Iw
M_critico_ana = np.pi / L * np.sqrt(EIz*GIt * (1 + (np.pi**2*EIw)/(L**2*GIt)))
print(f"Momento Crítico Calculado: {M_critico_num/1000:.4f} kNm")
#print(f"Momento Crítico Teorico: {M_critico_ana/1000:.4f} kNm")


fig1 = plot_buckling_modes(vals, vecs, mod) 
plt.show()


'''
Estoy programando un codigo de elementos finitos que resuelve vigas de seccion I monosimetricas en compresion y flexion y que a su vez hace un analisis de estabilidad elastico para encontrar cargas criticas y modos de pandeo. Mi objetivo es replicar el programa que realizo el autor de una articulo que no te paso ahora. El autor realizo el programa para calcular vigas uniformes y no-uniformes (secciones que varia linealmente). Mi plan inicial es programar primero para secciones no uniformes. Entonces tengo escrito el siguiente codigo que quiero que lo hagas mas elegante y eficiente y mas practico.
'''

#GIt = material1.shear * 7.32500e-7
#M_critico_ana = np.pi / L * np.sqrt(EIz*GIt * (1 + (np.pi**2*EIw)/(L**2*GIt)))
#print(f"Momento Crítico Teorico: {M_critico_ana/1000:.4f} kNm")