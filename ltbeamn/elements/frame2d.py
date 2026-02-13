

import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Frame2D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        self.init_element()

    def compute_trans_mat(self):
        c, s = self.dirvec
        rmatx = np.array([[c, s],[-s, c]])
        T = np.eye(6)
        T[0:2, 0:2] = rmatx
        T[3:5, 3:5] = rmatx

        return T

    def compute_stiff_mat(self):
        EA = self.mater.elast * self.section.A
        EI = self.mater.elast * self.section.Iy
        L  = self.length
        oneEA = EA /L
        twoEI = 2 * EI / L
        fourEI = 4 * EI / L
        sixEI = 6 * EI / L**2
        twelveEI = 12 * EI / L**3
        

        K = np.array([
            [ oneEA,          0,        0,   -oneEA,          0,      0.0],
            [     0,   twelveEI,    sixEI,        0,  -twelveEI,    sixEI],
            [     0,      sixEI,   fourEI,        0,     -sixEI,    twoEI],
            [-oneEA,          0,        0,    oneEA,          0,      0.0],
            [     0,  -twelveEI,   -sixEI,        0,   twelveEI,   -sixEI],
            [     0,      sixEI,    twoEI,        0,     -sixEI,   fourEI]
        ], dtype=float)

        return K
    

    def compute_mass_mat(self):
        pA = self.mater.dense * self.section.A
        L  = self.length

        M = np.array([
            [140,        0,         0,    70,         0,         0],
            [  0,      156,      22*L,     0,        54,     -13*L],
            [  0,     22*L,    4*L**2,     0,      13*L,   -3*L**2],
            [ 70,        0,         0,   140,         0,         0],
            [  0,       54,      13*L,     0,       156,     -22*L],
            [  0,    -13*L,   -3*L**2,     0,     -22*L,    4*L**2]
        ], dtype=float)

        return (pA * L / 420) * M


    
    def init_element(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        T = self.compute_trans_mat()
        K = self.compute_stiff_mat()
        M = self.compute_mass_mat()
        self.trans = T
        self.stiff = T.T @ K @ T
        self.mass = T.T @ M @ T
        self.load_intensities = np.zeros(4)
        self.loads = np.zeros(6)
        self.forces = np.zeros(6)
        self.disps = np.zeros(6)

    def add_loads(self, q1i, q2i, q1j, q2j):
        # Añadir en coordenadas locales
        # q1i = intensidad en el nodo i en direccion de la barra
        # q1j = intensidad en el nodo j en direccion de la barra
        # q2i = intensidad en el nodo i en direccion perpendicular de la barra
        # q2j = intensidad en el nodo j en direccion perpendicular de la barra
        self.load_intensities = [q1i, q2i, q1j, q2j]
        L = self.length

        self.loads[0] =  (q1i/3 + q1j/6) * L
        self.loads[1] =  (7*q2i + 3*q2j) * L / 20
        self.loads[2] =  (3*q2i + 2*q2j) * L**2 / 60

        self.loads[3] =  (q1j/3 + q1i/6) * L
        self.loads[4] =  (3*q2i + 7*q2j) * L / 20
        self.loads[5] = -(2*q2i + 3*q2j) * L**2 / 60

        # a coordenadas globales
        self.loads = self.trans.T @ self.loads
    
    def calculate_forces(self, glob_disps):
        # A Coordenadas locales
        self.disps = self.trans @ glob_disps # locales
        glob_forces = self.stiff @ glob_disps # globales
        self.forces = self.trans @ (glob_forces - self.loads) # locales


    def get_element_fields(self, esc1=1e-5, esc2=1e-5, esc3=100):
        EA = self.mater.elast * self.section.A
        EI = self.mater.elast * self.section.Iy
        L  = self.length

        x = np.linspace(0,L,50)
        x2 = x**2
        x3 = x2*x
        x4 = x3*x
        x5 = x4*x

        q1i, q2i, q1j, q2j = self.load_intensities
        sl1 = (q1j - q1i) / L
        sl2 = (q2j - q2i) / L

        # self.forces son fuerzas del nodo
        # ahora Ni debe cambiar de signo para pasar a la fuerza de elemento
        Ni = -self.forces[0] 
        Vi =  self.forces[1]
        Mi = -self.forces[2] # por convencion

        ui  = self.disps[0]
        wi  = self.disps[1]
        thi = self.disps[2]

        N = -sl1/2*x2 - q1i*x + Ni
        u = (-sl1/6*x3 - q1i*x2 + Ni*x) / EA + ui

        V =  sl2/2*x2 + q2i*x + Vi
        M =  (sl2/6*x3 + q2i/2*x2 + Vi*x + Mi) # poner menos?
        w =  (sl2/120*x5 + q2i/24*x4 + Vi*x3/6 + Mi*x2/2) / EI + thi*x + wi

        return x, N, V, M, u, w

    






