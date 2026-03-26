
import numpy as np
import scipy as sp
from src.elements.base_beam import Beam
from src.sections.utils import interpolate_section
from src.shape_funcs import N_hermite, dN_hermite, ddN_hermite
from src.gauss_quad import gauss_1d


class LTBeamTap(Beam):
    def __init__(self, mater, section_i, section_j, coord, conec, verax_dof, lator_dof):
        super().__init__(mater, coord, conec, verax_dof, lator_dof)

        self.section_i = section_i
        self.section_j = section_j

        self.init_geometry()

        self.gpoints, self.gweights = gauss_1d(4)

        # Inicializar matrices de rigidez y geometricas
        self.K0_vrx, self.K0_ltr = self.compute_K0_matrices()
        self.Kg_ltr = np.zeros((8, 8))

        # Solo problema estatico
        self.loads  = np.zeros(6)
        self.forces = np.zeros(6)
        self.disps  = np.zeros(6)
        self.load_intensities = np.zeros(4)

        

    def init_geometry(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)

        # Derivadas de las alturas de las mesas respecto a zS para calcular I_ypsi       
        h1_i = abs(self.section_i.zf1 - self.section_i.zS)
        h2_i = abs(self.section_i.zf2 - self.section_i.zS)
        
        h1_j = abs(self.section_j.zf1 - self.section_j.zS)
        h2_j = abs(self.section_j.zf2 - self.section_j.zS)
        
        self.dh1 = (h1_j - h1_i) / self.length
        self.dh2 = (h2_j - h2_i) / self.length
        
    

    def interpolate_at_gauss(self, xi):
        """
        Interpola sección en punto de Gauss y añade inercias del taper.
        """
        delta = 1e-6
        L = self.length
        
        # Interpolar sección principal
        gsec = interpolate_section(self.section_i, self.section_j, xi)
        
        # Secciones para diferenciación numérica
        sec_plus = interpolate_section(self.section_i, self.section_j, xi + delta)
        sec_minus = interpolate_section(self.section_i, self.section_j, xi - delta)
        
        #h1 = gsec.zf1 - gsec.zS
        #h2 = gsec.zf2 - gsec.zS
        
        # Calcular inercias del taper
        #I_psi = 4 * (self.dh1**2 * gsec.Izf1 + self.dh2**2 * gsec.Izf2)
        #I_wpsi = 2 * (h1 * self.dh1 * gsec.Izf1 + h2 * self.dh2 * gsec.Izf2)

        I_psi = 2 * (sec_plus.Iw - 2*gsec.Iw + sec_minus.Iw) / (delta * L)**2
        I_wpsi = (sec_plus.Iw - sec_minus.Iw) / (2 * delta * L)
        I_ypsi = 2 * (self.dh1 * gsec.Izf1 - self.dh2 * gsec.Izf2) # Aproximacion
        
        # Actualizar sección con inercias del taper
        gsec.update_tapered_inertias(I_psi, I_wpsi, I_ypsi)
        
        return gsec
    

    def compute_interpolation_vectors(self, xi):
        """ Vectores para ensamblar term-wise la parte de Kg_ltr"""
        L = self.length
        N = N_hermite(xi)
        dN = dN_hermite(xi)

       # Vector v' (derivada de la flexión lateral)
        vec_dv = np.zeros(8)
        vec_dv[0::4] = dN[0::2] / L  
        vec_dv[1::4] = dN[1::2]

        # Vector theta' (derivada del giro torsional)
        vec_dt = np.zeros(8)
        vec_dt[2::4] = dN[0::2] / L  
        vec_dt[3::4] = dN[1::2]     
            
        # Vector theta (giro torsional)
        vec_t = np.zeros(8)
        vec_t[2::4] = N[0::2]        
        vec_t[3::4] = N[1::2] * L          

        return vec_dv, vec_t, vec_dt
    
    
    def compute_verax_B(self, xi):
        """ Matriz deformacion-desplazamiento axial flexion vertical (2x6)"""
        L = self.length
        ddN = ddN_hermite(xi)
        
        B = np.zeros((2,6))
        # Deformación axial: ε = du/dx
        B[0, 0] = -1/L; 
        B[0, 3] =  1/L  
        # Curvatura: κ = d²w/dx²
        B[1, 1] = ddN[0] / L**2; 
        B[1, 2] = ddN[1] / L; 
        B[1, 4] = ddN[2] / L**2; 
        B[1, 5] = ddN[3] / L 

        return B

    
    def compute_lator_B(self, xi):
        """ Matriz deformacion-desplazamiento torsion flexion lateral (3x8)"""
        L = self.length
        dN = dN_hermite(xi)
        ddN = ddN_hermite(xi)
         # corregir indices
        B = np.zeros((3,8))
        # Curvatura lateral: κ_v = d²v/dx²
        B[0, 0::4] = ddN[0::2] / L**2
        B[0, 1::4] = ddN[1::2] / L
        # Curvatura de warping: κ_w = d²θ/dx²
        B[1, 2::4] = ddN[0::2] / L**2
        B[1, 3::4] = ddN[1::2] / L
        # Torsión: γ = dθ/dx
        B[2, 2::4] =  dN[0::2] / L
        B[2, 3::4] =  dN[1::2]

        return B 
    
    def compute_verax_D(self, gauss_section):
        """ Matriz constitutiva axial flexion vertical (2x2)"""
        EA  = self.mater.E * gauss_section.A
        EIy = self.mater.E * gauss_section.Iz
        return np.diag([EA, EIy])
    
    
    def compute_lator_D(self, gauss_section):
        """ Matriz constitutiva torsion flexion lateral (3x3)"""
        EIz = self.mater.E * gauss_section.Iz
        EIw = self.mater.E * gauss_section.Iw
        GIt = self.mater.G * gauss_section.It

        EI_psi  = self.mater.E * gauss_section.I_psi
        EI_wpsi = self.mater.E * gauss_section.I_wpsi
        EI_ypsi = self.mater.E * gauss_section.I_ypsi

        return np.array([
            [EIz,      0,        EI_ypsi],
            [0,        EIw,      EI_wpsi],
            [EI_ypsi,  EI_wpsi,  GIt + EI_psi]
        ])
    


    def compute_K0_matrices(self):
        """ Matriz de rigidez Axial-Flexion vertical (6x6)"""
        """ Matriz de rigidez Torsion-Flexion lateral (8x8)"""
        K0_vrx = np.zeros((6, 6))
        K0_ltr = np.zeros((8, 8))
        L = self.length

        for xi, w in zip(self.gpoints, self.gweights):
            # Interpolar sección en punto de Gauss (UNA sola vez)
            section = self.interpolate_at_gauss(xi)

            # Matrices constitutivas
            D_vrx = self.compute_verax_D(section)
            D_ltr = self.compute_lator_D(section)

            # Matrices de deformación-desplazamiento
            B_vrx = self.compute_verax_B(xi)
            B_ltr = self.compute_lator_B(xi)

            # Acumular contribuciones
            K0_vrx += (B_vrx.T @ D_vrx @ B_vrx) * w * L
            K0_ltr += (B_ltr.T @ D_ltr @ B_ltr) * w * L
        
        return K0_vrx, K0_ltr


    

    def update_lator_Kg(self):
        """ Matriz geometrica Torsion-Flexion lateral (8x8)"""
        Kg_ltr = np.zeros((8, 8))
        L = self.length
        
        N1 = -self.forces[0] # Axial izquierda
        N2 =  self.forces[3] # Axial derecha
        M1 = -self.forces[2] # Momento izquierda
        M2 =  self.forces[5]  # Momento derecha
        V_z = (M1 - M2) / L  # Cortante

        
        for xi, w in zip(self.gpoints, self.gweights):
            section = self.interpolate_at_gauss(xi)

            # Interpolar fuerzas internas
            M_xi = M1 * (1 - xi) + M2 * xi
            N_xi = N1 * (1 - xi) + N2 * xi 

            # Propiedades geométricas en la rebanada actual
            zS = section.zS
            i02 = section.i0**2
            beta_z = section.beta_z

            # Vectores de interpolación para ensamblar término a término
            vec_dv, vec_t, vec_dt = self.compute_interpolation_vectors(xi)

            # Ensamblaje numérico de la Ecuación 17 (Beyer et al.)
            # Términos de Fuerza Axial N
            term_N = N_xi * (
                np.outer(vec_dv, vec_dv) + 
                i02 * np.outer(vec_dt, vec_dt) + 
                zS * (np.outer(vec_dv, vec_dt) + np.outer(vec_dt, vec_dv))
            )
            
            # Términos de Momento My
            term_M = M_xi * (
                np.outer(vec_dv, vec_dt) + np.outer(vec_dt, vec_dv) - 
                2 * beta_z * np.outer(vec_dt, vec_dt)
            )
            
            # Término de Cortante Vz
            term_V = -V_z * (np.outer(vec_dv, vec_t) + np.outer(vec_t, vec_dv))
            
            Kg_ltr += (term_N + term_M + term_V) * w * L
            
        self.Kg_ltr = Kg_ltr

    

    def add_loads(self, qxi, qzi, qxj, qzj):
        # Añadir en coordenadas locales
        # qxi = intensidad en el nodo i en direccion de la barra
        # qxj = intensidad en el nodo j en direccion de la barra
        # qzi = intensidad en el nodo i en direccion perpendicular de la barra
        # qzj = intensidad en el nodo j en direccion perpendicular de la barra
        self.load_intensities = [qxi, qzi, qxj, qzj]
        L = self.length

        self.loads[0] =  (qxi/3 + qxj/6) * L
        self.loads[1] =  (7*qzi + 3*qzj) * L / 20
        self.loads[2] =  (3*qzi + 2*qzj) * L**2 / 60

        self.loads[3] =  (qxj/3 + qxi/6) * L
        self.loads[4] =  (3*qzi + 7*qzj) * L / 20
        self.loads[5] = -(2*qzi + 3*qzj) * L**2 / 60


    def calculate_forces(self, glob_disps):
        # A Coordenadas locales=globales
        self.disps = glob_disps # ya son locales
        self.forces = self.K0_vrx @ glob_disps - self.loads
        self.disps[np.abs(self.disps) < 1e-12] = 0
        self.forces[np.abs(self.forces) < 1e-9] = 0


    #"""
    def get_fields(self):
        L  = self.length
        x = np.linspace(0,L,3)
        xi = x/L

        qxi, qzi, qxj, qzj = self.load_intensities
        slx = (qxj - qxi) / L
        slz = (qzj - qzi) / L

        # self.forces son fuerzas del nodo
        # deben cambiar de signo para pasar a la fuerza de elemento
        Ni = -self.forces[0] 
        Vi =  self.forces[1] # para que salga como en Ftool no cambia
        Mi = -self.forces[2]

        N_diag = -slx/2*x**2 - qxi*x + Ni
        V_diag =  slz/2*x**2 + qzi*x + Vi
        M_diag =  slz/6*x**3 + qzi/2*x**2 + Vi*x + Mi

        # Desplazamiento Axial: Interpolacion lineal
        u = (1 - xi) * self.disps[0] + xi * self.disps[3]

        # Desplazamiento Vertical: Interpolacion cubica
        Nh = N_hermite(xi)
        w =  (Nh[0]*self.disps[1] + Nh[1]*L*self.disps[2] + 
              Nh[2]*self.disps[4] + Nh[3]*L*self.disps[5])

        # Limpieza de valores muy pequeños
        N_diag[np.abs(N_diag) < 1e-9] = 0
        V_diag[np.abs(V_diag) < 1e-9] = 0
        M_diag[np.abs(M_diag) < 1e-9] = 0

        u[np.abs(u) < 1e-12] = 0
        w[np.abs(w) < 1e-12] = 0

        return x, N_diag, V_diag, M_diag, u, w
        #"""