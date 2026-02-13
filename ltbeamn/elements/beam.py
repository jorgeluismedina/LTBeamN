
import numpy as np
import scipy as sp
from .base_elem import FrameElement


class Beam(FrameElement):
    def __init__(self, mater, section, coord, conec, dof): #elast, xarea, i_mom)
        super().__init__(mater, section, coord, conec, dof)
        self.init_element()

    
    def compute_stiff_mat(self):
        EI = self.mater.elast * self.section.Iy
        twoEI = 2 * EI / self.length
        fourEI = 4 * EI / self.length
        twelveEI = 12 * EI / self.length**3
        sixEI = 6 * EI / self.length**2

        K = np.array([
            [  twelveEI,   sixEI, -twelveEI,   sixEI],
            [   sixEI,     fourEI, -sixEI,     twoEI],
            [-twelveEI,   -sixEI,  twelveEI,  -sixEI],
            [   sixEI,      twoEI, -sixEI,     fourEI]
        ])

        return K


    
    def init_element(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.stiff = self.compute_stiff_mat()
        self.loads = np.zeros(4)
        self.forces = np.zeros(4)

    def add_loads(self, qi, qj):
        # Añadir en coordenadas locales=globales
        # qi = intensidad en el nodo i en direccion perpendicular de la barra
        # qj = intensidad en el nodo j en direccion perpendicular de la barra
        qi_qj = qi - qj

        self.loads[0] = -(3/20 * (qi_qj) - qi/2) * self.length
        self.loads[1] = -(1/30 * (qi_qj) - qi/12) * self.length**2

        self.loads[2] = -(7/20 * (qi_qj) - qi/2) * self.length
        self.loads[3] =  (1/20 * (qi_qj) - qi/12) * self.length**2



    def calculate_forces(self, glob_disps):
        # A Coordenadas locales=globales
        self.forces = self.stiff @ glob_disps
        self.forces = self.forces - self.loads


        
