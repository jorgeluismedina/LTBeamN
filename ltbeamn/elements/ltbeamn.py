
import numpy as np
import scipy as sp
from .base_elem import FrameElement
from ..shape_funcs import shape_beam, deriv1beam, deriv2beam

class LTBeamN(FrameElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        self.init_element()

    '''
    def get_trans_mat(self):
        c, s = self.dirvec
        rmatx = np.array([[c, s],[-s, c]])
        T = np.eye(6)
        T[0:2, 0:2] = rmatx
        T[3:5, 3:5] = rmatx

        return T
    '''
    #def get_const_mat(self, r):


    def get_shape_mat(self, r):
        shape = shape_beam(r)
        L = self.length
        N = np.zeros((2,8))
        N[0, 0::4] = shape[0::2]
        N[0, 1::4] = shape[1::2] / L
        N[1, 2::4] = shape[0::2]
        N[1, 3::4] = shape[1::2] / L
        return N
    
    def get_strain_mat(self, r):
        deriv1 = deriv1beam(r)
        deriv2 = deriv2beam(r)
        L = self.length
        L2 = L*L
        B = np.zeros((3,8))
        B[0, 0::2] = deriv2[0::2] / L2
        B[0, 1::2] = deriv2[1::2] / L
        B[1, 2::2] = deriv2[0::2] / L2
        B[1, 3::2] = deriv2[1::2] / L
        B[2, 2::2] = deriv1[0::2] / L
        B[2, 3::2] = deriv1[1::2]
        return B # en funcion de r (coordenadas naturales)
    




    def get_stiff_mat(self):
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
    

    def get_mass_mat(self):
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
        T = self.get_trans_mat()
        K = self.get_stiff_mat()
        M = self.get_mass_mat()
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
        #qi_qj = q2i - q2j
        L = self.length

        self.loads[0] =  (q1i/3 + q1j/6) * L
        self.loads[1] =  (7*q2i + 3*q2j) * L / 20#-(3/20 * (qi_qj) - q2i/2) * L
        self.loads[2] =  (3*q2i + 2*q2j) * L**2 / 60#-(1/30 * (qi_qj) - q2i/12) * L**2

        self.loads[3] =  (q1j/3 + q1i/6) * L
        self.loads[4] =  (3*q2i + 7*q2j) * L / 20#-(7/20 * (qi_qj) - q2i/2) * L
        self.loads[5] = -(2*q2i + 3*q2j) * L**2 / 60#(1/20 * (qi_qj) - q2i/12) * L**2

        # a coordenadas globales
        self.loads = self.trans.T @ self.loads
    
    def calculate_forces(self, glob_disps):
        # A Coordenadas locales
        self.disps = self.trans @ glob_disps # locales
        glob_forces = self.stiff @ glob_disps # globales
        self.forces = self.trans @ (glob_forces - self.loads) # locales