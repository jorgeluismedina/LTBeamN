
import numpy as np
import scipy as sp
from .base_beam import Beam

class LTBeam(Beam):
    def __init__(self, mater, section, coord, conec, verax_dof, lator_dof):
        super().__init__(mater, section, coord, conec, verax_dof, lator_dof)

        self.init_geometry()
        self.set_dof_indices()

        # Inicializar matrices de rigidez y geometricas
        self.K0_vrx = self.compute_verax_K0()
        self.K0_ltr = self.compute_lator_K0()
        self.Kg_ltr = np.zeros((8, 8))

        # Solo problema estatico
        self.loads  = np.zeros(6)
        self.forces = np.zeros(6)
        self.disps  = np.zeros(6)
        self.load_intensities = np.zeros(4)

        
  
    def init_geometry(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)

    
    def set_dof_indices(self):
        # Indices para u (0, 3) y w (1, 2, 4, 5)
        idx_u = [0, 3]
        idx_w = [1, 2, 4, 5]

        # Indices para v (0, 1, 4, 5) y theta (2, 3, 6, 7)
        idx_v = [0, 1, 4, 5]
        idx_t = [2, 3, 6, 7]

        # Indices de submatrices para rigidez y acoplamiento
        self.idx_uu = np.ix_(idx_u, idx_u)
        self.idx_ww = np.ix_(idx_w, idx_w)  
        self.idx_vv = np.ix_(idx_v, idx_v)
        self.idx_tt = np.ix_(idx_t, idx_t)
        self.idx_vt = np.ix_(idx_v, idx_t)
        self.idx_tv = np.ix_(idx_t, idx_v)


    def ddNiddNj_matrix(self):
        """ Integral (Ni'' * Nj'') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 12,   6*L,   -12,   6*L],
            [ 6*L,  4*L**2, -6*L, 2*L**2],
            [-12,  -6*L,    12,   -6*L],
            [ 6*L,  2*L**2, -6*L, 4*L**2]
        ]) / L**3

        return matrix
    
    def dNidNj_matrix(self):
        """ Integral (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 36,   3*L,   -36,   3*L],
            [ 3*L,  4*L**2, -3*L, -L**2],
            [-36,  -3*L,    36,  -3*L],
            [ 3*L, -L**2,  -3*L,  4*L**2]
        ]) / (30*L)

        return matrix
    
    def dNidNj_1_xi_matrix(self):
        """ Integral (1-xi) * (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 18,    0,      -18,    3*L ],
            [ 0,    3*L**2,   0,   -0.5*L**2],
            [-18,    0,       18,   -3*L ],
            [ 3*L, -0.5*L**2,  -3*L,    L**2]
        ]) / (30*L)

        return matrix
    
    def dNidNj_xi_matrix(self):
        """ Integral xi * (Ni' * Nj') función de forma cúbica Hermite """
        L = self.length
        matrix = np.array([
            [ 18,    3*L,    -18,   0   ],
            [ 3*L,   L**2,   -3*L,   -0.5*L**2],
            [-18,   -3*L,     18,   0   ],
            [ 0,    -0.5*L**2,   0,  3*L**2]
        ]) / (30*L)

        return matrix
    
    def dNiNj_matrix(self):
        """ Integral (Ni' * Nj) función de forma cúbica Hermite, asimetrica """
        L = self.length
        matrix = np.array([
            [ -0.5,    -L/10,     -0.5,    L/10],
            [  L/10,    0,        -L/10,   L**2/60],
            [  0.5,     L/10,      0.5,   -L/10],
            [ -L/10,   -L**2/60,   L/10,     0]
        ]) 

        return matrix

    
    def compute_verax_K0(self):
        """ Matriz de rigidez flexion vertical (w, w,x) y desp. axial (u) (6x6) """
        EA  = self.mater.E * self.section.A
        EIy = self.mater.E * self.section.Iy

        # Matrices base
        axial_base = np.array([[1, -1],[-1, 1]]) / self.length
        bending_base = self.ddNiddNj_matrix()

        # Ensamblaje de matriz de rigidez viga convencional
        K0_vrx = np.zeros((6, 6))
        K0_vrx[self.idx_uu] = EA * axial_base
        K0_vrx[self.idx_ww] = EIy * bending_base

        return K0_vrx
    
    
    def compute_lator_K0(self):
        """ Matriz de rigidez flexion lateral (v, v,x) y torsion (theta, theta,x) (8x8) """
        EIz = self.mater.E * self.section.Iz
        GIt = self.mater.G * self.section.It
        EIw = self.mater.E * self.section.Iw

        # Matrices base
        bending_base = self.ddNiddNj_matrix()
        torsion_base = self.dNidNj_matrix()

        # Ensamblaje de matriz de rigidez flexión lateral y torsión con acoplamiento
        K0_ltr = np.zeros((8, 8))
        K0_ltr[self.idx_vv] = EIz * bending_base                          # Bloque v-v (Flexión lateral)
        K0_ltr[self.idx_tt] = (EIw * bending_base) + (GIt * torsion_base) # Bloque t-t (Torsión = Warping + St.Venant)
        
        return K0_ltr
        
    
    def compute_lator_KgN(self): 
        """ Matriz geometrica por carga axial (8x8) """
        N = self.forces[0]
        zs = self.section.zS
        r02 = self.section.i0**2    

        # Matriz base
        base = self.dNidNj_matrix()

        # Ensamblaje de matriz geometrica por carga axial
        KgN = np.zeros((8, 8))
        
        # Bloques directos
        block_vv = N * base
        KgN[self.idx_vv] = block_vv        # Bloque v-v (Flexión lateral)
        KgN[self.idx_tt] = r02 * block_vv  # Bloque t-t (Torsión)

        # Bloques de acoplamiento (zc)
        block_vt = N * zs * base
        KgN[self.idx_vt] += block_vt # Bloque vt (Acoplamiento)
        KgN[self.idx_tv] += block_vt # Bloque tv = vt (Acoplamiento)

        return KgN
    

    def compute_lator_KgMV(self):
        """ Matriz geometrica por momento y cortante (KgM + KgV) (8x8) """
        beta_z = self.section.beta_z
        L = self.length

        M1 = -self.forces[2] # Momento en nodo i (signo invertido por carga nodal)
        M2 =  self.forces[5] # Momento en nodo j
        Vz = (M1 - M2) / L
  
        My_base = (M1 * self.dNidNj_1_xi_matrix() 
                   + M2 * self.dNidNj_xi_matrix()) # Integral de My * Ni' * Nj'
        
        Vz_base  = Vz * self.dNiNj_matrix() # Integral de Vz * Ni' * Nj (asimetrica)

        # Ensamblaje de matriz geometrica por momento y cortante
        KgMV = np.zeros((8, 8))
        
        # Bloques directos diagonal (t-t), (v-v) = 0
        # Termino: 2 * beta_z * My * t' * t'
        block_tt = - 2 * beta_z * My_base
        KgMV[self.idx_tt] += block_tt

        # Bloques de acoplamiento (v-t y t-v) 
        # Termino: My * v' * t' - Vz * v' * t
        block_vt = My_base - Vz_base 
        KgMV[self.idx_vt] += block_vt
        KgMV[self.idx_tv] += block_vt.T 

        return KgMV
    

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


    def calculate_forces(self, glob_disps):
        # A Coordenadas locales=globales
        self.disps = glob_disps # ya son locales
        self.forces = self.K0_vrx @ glob_disps - self.loads
        self.disps[np.abs(self.disps) < 1e-12] = 0
        self.forces[np.abs(self.forces) < 1e-9] = 0

    def update_lator_Kg(self):
        """ Actualiza la matriz geometrica con fuerzas internas """
        KgN = self.compute_lator_KgN()
        KgMV = self.compute_lator_KgMV()
        self.Kg_ltr = KgN + KgMV


    def get_fields(self):
        EA = self.mater.E * self.section.A
        EI = self.mater.E * self.section.Iy
        L  = self.length

        x = np.linspace(0,L,2)
        x2 = x**2
        x3 = x2*x
        x4 = x3*x
        x5 = x4*x

        q1i, q2i, q1j, q2j = self.load_intensities
        sl1 = (q1j - q1i) / L
        sl2 = (q2j - q2i) / L

        # self.forces son fuerzas del nodo
        # deben cambiar de signo para pasar a la fuerza de elemento
        Ni = -self.forces[0] 
        Vi =  self.forces[1] # para que salga como en Ftool no cambia
        Mi = -self.forces[2]

        ui  = self.disps[0]
        wi  = self.disps[1]
        thi = self.disps[2]

        N = -sl1/2*x2 - q1i*x + Ni
        u = (-sl1/6*x3 - q1i*x2 + Ni*x) / EA + ui

        V =  sl2/2*x2 + q2i*x + Vi
        M =  (sl2/6*x3 + q2i/2*x2 + Vi*x + Mi)
        w =  (sl2/120*x5 + q2i/24*x4 + Vi*x3/6 + Mi*x2/2) / EI + thi*x + wi

        # Limpieza de valores muy pequeños
        N[np.abs(N) < 1e-9] = 0
        V[np.abs(V) < 1e-9] = 0
        M[np.abs(M) < 1e-9] = 0

        u[np.abs(u) < 1e-12] = 0
        w[np.abs(w) < 1e-12] = 0

        return x, N, V, M, u, w