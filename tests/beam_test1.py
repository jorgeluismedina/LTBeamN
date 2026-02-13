

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

material1 = Material(elast=2.1e10, poiss=0.3, dense=1.0) #[N/m2]
materials = [material1]

V20x30 = RectangularSection(base=0.2, height=0.3, orientation=0) #[m2]
sections = [V20x30]


coordinates = np.array([[0.0],
                        [8.0],
                        [13.0]])

mod = Model(ndofn=2)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

mod.add_element('Beam', material1, V20x30, [0, 1])
mod.add_element('Beam', material1, V20x30, [1, 2])

mod.add_node_restraint(0, [1, 1])
mod.add_node_restraint(1, [1, 0])
mod.add_node_restraint(2, [1, 1])

mod.add_elem_load(0, [-10, -10])
mod.add_elem_load(1, [-10, -10])



# Solucion
glob_disps, reactions = solve_linear_static(mod)
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')

