
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from ltbeamn.femclass import Model
from ltbeamn.material import Material
from ltbeamn.sections import RectangularSection
from ltbeamn.solvers import solve_linear_static
from ltbeamn.plotting import plot_2dmodel, prepare_diagrams, plot_2d_diagram, plot_2d_deformed_shape

material1 = Material(elast=2.1e10, poiss=0.3, dense=1.0) #[N/m2]
materials = [material1]

V20x30 = RectangularSection(base=0.2, height=0.3, orientation=0) #[m2]
sections = [V20x30]


coordinates = np.array([[0.0, 0.0],
                        [0.0, 3.0],
                        [5.0, 2.0],
                        [5.0, 0.0]])

mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

mod.add_element('Frame2D', material1, V20x30, [0, 1])
mod.add_element('Frame2D', material1, V20x30, [1, 2])
mod.add_element('Frame2D', material1, V20x30, [2, 3])


mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(3, [1, 1, 1])

mod.add_node_load(1, [50000, 0, 0]) # [N]

mod.add_elem_load(1, [1000, -10000, 5000, -15000]) #[KN/m2]



# Solucion
glob_disps, reactions = solve_linear_static(mod)
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')


mod.calculate_forces(glob_disps)



all_fields = mod.generate_fields()
elements = mod.elems
all_diagrams = prepare_diagrams(elements, all_fields)


ax1 = plot_2d_diagram(elements, all_diagrams[0])
ax2 = plot_2d_diagram(elements, all_diagrams[1])
ax3 = plot_2d_diagram(elements, all_diagrams[2])
ax4 = plot_2d_deformed_shape(elements, all_diagrams[3])
plt.show()


