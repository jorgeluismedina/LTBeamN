
import numpy as np
import scipy as sp
from .base_elem import FrameElement

class LTBeam(FrameElement):
    def __init__(self, mater, section, coord, conec, dof1, dof2): #elast, xarea, i_mom)
        super().__init__(mater, section, coord, conec, dof1)
        self.init_element()
        self.dof1 = dof1
        self.dof2 = dof2

    
    # vertical deflection
    def compute_bend_stiff_mat(self):
        EA = self.mater.elast * self.section.A
        EI = self.mater.elast * self.section.Iy
        L  = self.length
        oneEA  = EA / L
        twoEI  = 2 * EI / L
        fourEI = 4 * EI / L
        sixEI  = 6 * EI / L**2
        twelveEI = 12 * EI / L**3
        

        Kb = np.array([
            [ oneEA,          0,        0,   -oneEA,          0,      0.0],
            [     0,   twelveEI,    sixEI,        0,  -twelveEI,    sixEI],
            [     0,      sixEI,   fourEI,        0,     -sixEI,    twoEI],
            [-oneEA,          0,        0,    oneEA,          0,      0.0],
            [     0,  -twelveEI,   -sixEI,        0,   twelveEI,   -sixEI],
            [     0,      sixEI,    twoEI,        0,     -sixEI,   fourEI]
        ], dtype=float)

        return Kb
    
    def compute_lator_stiff_mat(self):
        EIz = self.mater.elast * self.section.Iz
        GIt = self.mater.shear * self.section.It
        EIw = self.mater.elast * self.section.Iw
        L   = self.length

        # --- Sub-matrices base ---
        
        # 1. Matriz de Flexión (Hermite standard) -> Para EIz y EIw
        # Int(N'' * N'')
        k_bending_base = np.array([
            [ 12,   6*L,   -12,   6*L],
            [ 6*L,  4*L**2, -6*L, 2*L**2],
            [-12,  -6*L,    12,  -6*L],
            [ 6*L,  2*L**2, -6*L, 4*L**2]
        ]) / L**3

        # 2. Matriz de "Cuerda" (Geometric-like) -> Para GIt (St. Venant)
        # Int(N' * N') - Consistente con interpolación cúbica
        k_stvenant_base = np.array([
            [ 36,   3*L,   -36,   3*L],
            [ 3*L,  4*L**2, -3*L, -L**2],
            [-36,  -3*L,    36,  -3*L],
            [ 3*L, -L**2,  -3*L,  4*L**2]
        ]) / (30*L)

        # --- Ensamblaje ---
        K = np.zeros((8, 8))
        
        # Indices para v (0, 1, 4, 5) y theta (2, 3, 6, 7)
        idx_v = [0, 1, 4, 5]
        idx_t = [2, 3, 6, 7]
        
        # Bloque v-v (Flexión lateral)
        K[np.ix_(idx_v, idx_v)] = EIz * k_bending_base
        # Bloque t-t (Torsión = Alabeo + St.Venant)
        K[np.ix_(idx_t, idx_t)] = (EIw * k_bending_base) + (GIt * k_stvenant_base)
        
        return K
        
    
    # 1. Matriz Geométrica por Carga Axial (KgN)
    def compute_lator_geom_mat_N(self): 
        N = self.forces[0]
        L = self.length
        zs = self.section.zS
        r02 = self.section.i0**2    

        # Matriz base (Integral N' * N') / 30L
        base = np.array([
            [ 36,   3*L,   -36,   3*L],
            [ 3*L,  4*L**2, -3*L, -L**2],
            [-36,  -3*L,    36,  -3*L],
            [ 3*L, -L**2,  -3*L,  4*L**2]
        ]) / (30 * L)

        KgN = np.zeros((8, 8))
        idx_v = [0, 1, 4, 5] 
        idx_t = [2, 3, 6, 7]

        # Bloques directos
        KgN[np.ix_(idx_v, idx_v)] = N * base           # Flexión lateral
        KgN[np.ix_(idx_t, idx_t)] = N * r02 * base     # Torsión (Wagner)

        # Bloques de acoplamiento (zc)
        # Se suman a ambos lados para garantizar simetría numérica
        k_cross = N * zs * base
        KgN[np.ix_(idx_v, idx_t)] += k_cross
        KgN[np.ix_(idx_t, idx_v)] += k_cross # Simetría: k_cross.T = k_cross

        return KgN
    
    # 2. Matriz Geométrica por Momento y Cortante (KgM + KgV)
    def compute_lator_geom_mat_MV(self):
        L = self.length
        beta_z = self.section.beta_z
        # Ajuste de signos por que son fuerzas nodales
        M1 = -self.forces[2]
        M2 =  self.forces[5]
        Vz = (M1 - M2) / L # OJO CON EL SIGNO: Depende de tu convención. 
                           # Usualmente V = (M1 + M2)/L si giran en mismo sentido horario,
                           # o (M2 - M1)/L. Verifica tu convención de signos de FEM.
                           # Aquí asumo: M1 y M2 son momentos nodales en sentido antihorario.
                           # Equilibrio: M1 + M2 + V*L = 0 -> V = -(M1+M2)/L. 
                           # Ajusta según tu formulación estática.
        
        # Matrices de integración para interpolación lineal (1-x/L) y (x/L)
        m_base1 = np.array([
            [ 18,    0,      -18,    3*L ],
            [ 0,    3*L**2,   0,   -0.5*L**2],
            [-18,    0,       18,   -3*L ],
            [ 3*L, -0.5*L**2,  -3*L,    L**2]
        ]) / (30*L)
        
        m_base2 = np.array([
            [ 18,    3*L,    -18,   0   ],
            [ 3*L,   L**2,   -3*L,   -0.5*L**2],
            [-18,   -3*L,     18,   0   ],
            [ 0,    -0.5*L**2,   0,  3*L**2]
        ]) / (30*L)

        # Matriz ponderada por los momentos nodales
        # Esto representa la integral de M(x) * phi_i' * phi_j'
        k_My_base = M1 * m_base1 + M2 * m_base2

        # Matriz de acoplamiento por Cortante (Integral V * v' * theta)
        # Ojo: Esta es la integral asimétrica de base.
        k_Vz_base = Vz * np.array([
            [ -0.5,    -L/10,     -0.5,    L/10],
            [  L/10,    0,        -L/10,   L**2/60],
            [  0.5,     L/10,      0.5,   -L/10],
            [ -L/10,   -L**2/60,   L/10,     0]
        ])

        KgMV = np.zeros((8, 8))
        idx_v = [0, 1, 4, 5]
        idx_t = [2, 3, 6, 7]

        # A. Término Wagner debido al momento (Bloque theta-theta)
        # Contribución: -0.5 * Int( M * beta_z * theta,x^2 ) -> El 0.5 se va con la variación
        # El signo depende del funcional. Usualmente es -M * beta_z
        KgMV[np.ix_(idx_t, idx_t)] -= 2 * beta_z * k_My_base

        # B. Términos de Acoplamiento v-theta
        # Parte 1: Momento (Simétrica) -> M * v,x * theta,x
        # Parte 2: Cortante (Asimétrica base) -> V * v,x * theta
        
        # Combinamos para el bloque fuera de la diagonal
        # K_vt = Int(M v' t') - Int(V v' t)
        block_vt = k_My_base - k_Vz_base 
        
        # Aqui aplicamos la "simetrización forzada" que es en realidad
        # la suma de los términos de variación cruzada
        KgMV[np.ix_(idx_v, idx_t)] += block_vt
        KgMV[np.ix_(idx_t, idx_v)] += block_vt.T 

        return KgMV
    
    
    def init_element(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.bend_stiff = self.compute_bend_stiff_mat()
        self.lator_stiff = self.compute_lator_stiff_mat()

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


    def calculate_forces(self, glob_disps):
        # A Coordenadas locales=globales
        self.disps = glob_disps # ya son locales
        self.forces = self.bend_stiff @ glob_disps
        self.forces = self.forces - self.loads

    def compute_lator_geom_mat(self):
        KgN  = self.compute_lator_geom_mat_N()
        KgMV = self.compute_lator_geom_mat_MV()
        self.lator_geom = KgN + KgMV


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
        vi  = self.disps[1]
        thi = self.disps[2]

        N = -sl1/2*x2 - q1i*x + Ni
        u = (-sl1/6*x3 - q1i*x2 + Ni*x) / EA + ui

        V =  sl2/2*x2 + q2i*x + Vi
        M =  (sl2/6*x3 + q2i/2*x2 + Vi*x + Mi)
        v =  (sl2/120*x5 + q2i/24*x4 + Vi*x3/6 + Mi*x2/2) / EI + thi*x + vi

        return x, N, V, M, u, v