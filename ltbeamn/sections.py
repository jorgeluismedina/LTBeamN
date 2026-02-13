
import numpy as np


class FrameSection2D:
    def __init__(self, A, Iy):
        self.A  = A  # cross sectional area
        self.Iy = Iy # second moment of area respect to 3



# de momenento solo secciones bisimetricas
class FrameSection3D:
    def __init__(self, A, Iy, Iz, J, orientation):
        self.A  = A  # cross sectional area
        self.Iy = Iy # second moment of area respect to y
        self.Iz = Iz # second moment of area respect to z
        self.J  = J  # torsional constant
        self.orientation = orientation
        
        #self.warping_const



class RectangularSection: # Porticos en 3D
    def __init__(self, base, height, orientation):
        self.b = base
        self.h = height
        self.orientation = orientation #0 a 90 Este angulo que sea variable
        self.set_section_props()

    def set_section_props(self):
        self.A  = self.b * self.h
        self.Iy = self.b * self.h**3 / 12
        self.Iz = self.h * self.b**3 / 12
        self.get_torsional_const()
        #beta = 1/3 - 0.21 * b/h * (1 - 1/12 * (b/h)**4)

    def get_torsional_const(self):
        # calculo de J (constante torsional)
        if self.b == self.h:
            beta = 0.141
            self.J = beta * (self.b**3) * self.h
        else:
            b_min = min(self.b, self.h)
            h_max = max(self.b, self.h)
            ratio = b_min / h_max
            # Fórmula de aproximación polinómica
            beta = (1/3) * (1 - 0.63*ratio + 0.052*ratio**5) 
            self.J = beta * (b_min**3) * h_max





# Seccion I bi-simetrica
class ISection_BS:
    def __init__(self, h, bf, tw, tf, r):
        self.h  = h     # total height
        self.bf = bf    # flanges width
        self.tw = tw    # web thick
        self.tf = tf    # flanges thick
        self.r  = r     # radius of fillets

        self.hw = h - 2*tf # altura del alma entre flanges
        self.zG = h / 2
        self.zS = 0.0
        self.beta_z = 0.0
        self.compute_area()
        self.compute_bending_inertias()
        self.compute_torsional_inertia()
        self.compute_warping_inertia()
        self.compute_polar_radius()


    def compute_area(self):
        A_web = self.tw * self.hw
        A_flanges = 2 * self.bf * self.tf
        A_corners = self.r**2 * (4 - np.pi) # area de los fillets
        self.A = A_web + A_flanges + A_corners
    
    def compute_bending_inertias(self):
        four_pi = 4 - np.pi
        term = 4*self.r**4 * (1/3 - np.pi/16 - 1/(9*four_pi))

        term1y = 1/12 * (self.bf * self.h**3 - (self.bf - self.tw) * self.hw**3)
        term2y = four_pi*self.r**2 * (self.zG - self.tf - self.r + 2*self.r/(3*four_pi))**2
        self.Iy = term1y + term + term2y

        term1z = 1/6*self.tf*self.bf**3 + 1/12*self.tw**3*self.hw
        term2z = four_pi*self.r**2 * (self.tw/2 + self.r - 2*self.r/(3*four_pi))**2
        self.Iz = term1z + term + term2z

    def compute_torsional_inertia(self): # constante torsional
        # Saint-Venant base
        self.It = (2 * self.bf * self.tf**3 +
                   self.hw  * self.tw**3) / 3
        #print(self.It)
        if self.r > 0.0:
            D = ((self.tf + self.r)**2 + 
                  self.tw * (self.r + self.tw / 4)) / (2*self.r + self.tf) 
                
            alpha = (0.2204 * (self.tw / self.tf) +
                     0.1355 * (self.r / self.tf) -
                     0.0865 * (self.tw * self.r / self.tf**2) -
                     0.0725 * (self.tw / self.tf)**2)

            #corr_high = 0.0175 * (self.tf**8) / (self.bf**4)
            I2tr = alpha * D**4 - 0.21*self.tf**4 #- corr_high
            self.It += 2*I2tr
        
    def compute_warping_inertia(self):
        Izf = self.bf * self.tf**3 / 12
        self.Iw = 0.25 * self.Iz * (self.h - self.tf)**2

    def compute_polar_radius(self): #respecto al centro de corte
        self.i0 = np.sqrt((self.Iy + self.Iz) / self.A)

    def summary(self):
        print("\n" + "="*50)
        print(" I-SECTION (BISYMMETRIC) – GEOMETRY & PROPERTIES")
        print("="*50)

        # --- Geometry ---
        print("\n[ Geometry ]")
        print(f"  h   = {self.h:.4f}")
        print(f"  bf  = {self.bf:.4f}")
        print(f"  tf  = {self.tf:.4f}")
        print(f"  tw  = {self.tw:.4f}")
        print(f"  hw  = {self.hw:.4f}")
        print(f"  r   = {self.r:.4f}")

        # --- Properties ---
        print("\n[ Properties ]")
        print(f"  A  = {self.A:.4e}")
        print(f"  Iy = {self.Iy:.4e}")
        print(f"  Iz = {self.Iz:.4e}")
        print(f"  It = {self.It:.4e}")
        print(f"  Iw = {self.Iw:.4e}")
        print(f"  βz = {self.beta_z:.6f}")
        print(f"  zG = {self.zG:.6f}  (from bottom fiber)")
        print(f"  zS = {self.zS:.6f}  (relative to centroid)")
        print(f"  i0 = {self.i0:.6f}  (respect to shear center)")
        print("\n" + "="*50 + "\n")








class ISection_MS:
    def __init__(self, h, bf1, bf2, tw, tf1, tf2, r1, r2):
        self.h   = h     # total height
        self.bf1 = bf1   # top flange width
        self.bf2 = bf2   # bottom flange width
        self.tw  = tw    # web thick
        self.tf1 = tf1   # top flange thick
        self.tf2 = tf2   # botttom flange thick
        self.r1  = r1    # top fillets radious
        self.r2  = r2    # bottom fillets radious
        self.compute_basic()
        self.compute_area()
        self.compute_gravity_center()
        self.compute_bending_inertias()
        self.compute_shear_center()
        self.compute_torsional_inertia()
        self.compute_warping_inertia()
        self.compute_polar_radius()
        self.compute_wagner_coeff()

    def compute_basic(self):
        # propiedades ala superior
        self.Af1  = self.bf1 * self.tf1
        self.Iyf1 = self.bf1 * self.tf1**3 / 12
        self.Izf1 = self.tf1 * self.bf1**3 / 12
        self.zGf1 = self.h - self.tf1 / 2
        # propiedades ala inferior
        self.Af2  = self.bf2 * self.tf2
        self.Iyf2 = self.bf2 * self.tf2**3 / 12
        self.Izf2 = self.tf2 * self.bf2**3 / 12
        self.zGf2 = self.tf2 / 2
        # propiedades alma
        self.hw = self.h - self.tf1 - self.tf2
        self.Aw = self.hw * self.tw
        self.Iyw = self.tw * self.hw**3 / 12
        self.Izw = self.hw * self.tw**3 / 12
        self.zGw = self.tf2 + self.hw/2
        # terminos repetidos
        one_qpi = 1 - np.pi/4
        four_pi = 4 * one_qpi
        # propiedades fillets superiores
        self.Ar1 = one_qpi * self.r1**2
        self.Ir1 = (1/3 - np.pi/16 - 1/(9*four_pi)) * self.r1**4
        self.vr1 = (1 - 2/(3*four_pi)) * self.r1
        self.zGr1 = self.h - self.tf1 - self.vr1
        # propiedades fillets inferiores
        self.Ar2 = one_qpi * self.r2**2
        self.Ir2 = (1/3 - np.pi/16 - 1/(9*four_pi)) * self.r2**4
        self.vr2 = (1 - 2/(3*four_pi)) * self.r2
        self.zGr2 = self.tf2 + self.vr2

    def compute_area(self):
        self.A = self.Af1 + self.Af2 + self.Aw + 2*self.Ar1 + 2*self.Ar2

    def compute_gravity_center(self):
        self.zG = (self.Af1 * self.zGf1 + 
                   self.Af2 * self.zGf2 +
                   self.Aw * self.zGw +
                   2 * self.Ar1 * self.zGr1 +
                   2 * self.Ar2 * self.zGr2) / self.A
        # centroides de las partes respecto al centroide total
        self.zf1 = self.zGf1 - self.zG
        self.zf2 = self.zGf2 - self.zG
        self.zw = self.zGw - self.zG
        self.zr1 = self.zGr1 - self.zG
        self.zr2 = self.zGr2 - self.zG

    def compute_bending_inertias(self):
        Iyf1 = self.Iyf1 + self.Af1 * self.zf1**2
        Iyf2 = self.Iyf2 + self.Af2 * self.zf2**2
        Iyw  = self.Iyw + self.Aw * self.zw**2
        Iyr1 = self.Ir1 + self.Ar1 * self.zr1**2
        Iyr2 = self.Ir2 + self.Ar2 * self.zr2**2
        Izr1 = self.Ir1 + self.Ar1 * (self.tw/2 + self.vr1)**2
        Izr2 = self.Ir2 + self.Ar2 * (self.tw/2 + self.vr2)**2

        self.Iy = Iyf1 + Iyf2 + Iyw + 2*Iyr1 + 2*Iyr2
        self.Iz = self.Izf1 + self.Izf2 + self.Izw + 2*Izr1 + 2*Izr2

    def compute_shear_center(self): # respecto del centroide
        self.zS = (self.Izf1 * self.zf1 + 
                   self.Izf2 * self.zf2 +
                   self.Izw * self.zw +
                   2 * self.Ir1 * self.zr1 +
                   2 * self.Ir2 * self.zr2) / self.Iz

    def compute_torsional_inertia(self):
        # Saint-Venant base
        self.It = (self.bf1 * self.tf1**3 +
                   self.bf2 * self.tf2**3 +
                   self.hw  * self.tw**3) / 3
        #print(self.It)
        # correction functions
        def alpha(tf, r):
            return (0.2204 * (self.tw / tf) + 
                    0.1355 * (r / tf) -
                    0.0865 * (self.tw * r / tf**2) -
                    0.0725 * (self.tw / tf)**2)
        
        def D(tf, r):
            return ((tf+r)**2 + self.tw*(r+self.tw/4)) / (2*r+tf)
        
        if self.r1 > 0.0:
            alpha1 = alpha(self.tf1, self.r1)
            D1 = D(self.tf1, self.r1)
            self.It += alpha1 * D1**4 - 0.21*self.tf1**4

        if self.r2 > 0.0:
            alpha2 = alpha(self.tf2, self.r2)
            D2 = D(self.tf2, self.r2)
            self.It += alpha2 * D2**4 - 0.21*self.tf2**4
        

    def compute_warping_inertia(self):
        I_ratio = self.Izf1 * self.Izf2 / (self.Izf1 + self.Izf2)
        hs = (self.zGf1 - self.zGf2)
        self.Iw = I_ratio * hs**2

    def compute_polar_radius(self):
        self.i0 = np.sqrt((self.Iy + self.Iz) / self.A + self.zS**2)

    def compute_wagner_coeff(self):
        sf1 = self.zf1 * (self.Izf1 + self.Af1*self.zf1**2 + 3*self.Iyf1)
        sf2 = self.zf2 * (self.Izf2 + self.Af2*self.zf2**2 + 3*self.Iyf2)
        sw = self.zw * (self.Izw + self.Aw*self.zw**2 + 3*self.Iyw)
        sr1 = 2 * self.zr1 * (4*self.Ir1 + self.Ar1*self.zr1**2)
        sr2 = 2 * self.zr2 * (4*self.Ir2 + self.Ar2*self.zr2**2)

        self.beta_z = 1 / (2*self.Iy) * (sf1 + sf2 + sw + sr1 + sr2) - self.zS

    def summary(self):
        print("\n" + "="*55)
        print(" I-SECTION (MONOSYMMETRIC) – GEOMETRY & PROPERTIES")
        print("="*55)

        # --- Geometry ---
        print("\n[ Geometry ]")
        print(f"  h   = {self.h:.4f}")
        print(f"  bf1 = {self.bf1:.4f}")
        print(f"  bf2 = {self.bf2:.4f}")
        print(f"  tf1 = {self.tf1:.4f}")
        print(f"  tf2 = {self.tf2:.4f}")
        print(f"  tw  = {self.tw:.4f}")
        print(f"  hw  = {self.hw:.4f}")
        print(f"  r1  = {self.r1:.4f}")
        print(f"  r2  = {self.r2:.4f}")

        # --- Properties ---
        print("\n[ Properties ]")
        print(f"  A  = {self.A:.4e}")
        print(f"  Iy = {self.Iy:.4e}")
        print(f"  Iz = {self.Iz:.4e}")
        print(f"  It = {self.It:.4e}")
        print(f"  Iw = {self.Iw:.4e}")
        print(f"  βz = {self.beta_z:.6f}")
        #print()
        print(f"  zG = {self.zG:.6f}  (from bottom fiber)")
        print(f"  zS = {self.zS:.6f}  (relative to centroid)")
        print(f"  i0 = {self.i0:.6f}  (respect to shear center)")
        print("\n" + "="*55 + "\n")






        




        





'''
class ISection_MS:
    def __init__(self, h, bf1, bf2, tw, tf1,tf2):
        self.h   = h     # total height
        self.bf1 = bf1   # top flange width
        self.bf2 = bf2   # bottom flange width
        self.tw  = tw    # web thick
        self.tf1 = tf1   # top flange thick
        self.tf2 = tf2   # botttom flange thick
        self.compute_basic()
        self.compute_inertias_centroidal()
        self.compute_shear_center()
        self.compute_torsional_const()
        self.compute_warping_const()
        self.compute_polar_radius()
        
    def compute_basic(self):
        # Areas parciales
        A_top = self.btf * self.tf
        A_web = self.tw * (self.h - 2.0*self.tf)
        A_bot = self.bbf * self.tf

        # Centroides parciales (medidos desde la base)
        z_top = self.h - self.tf/2.0
        z_web = self.tf + (self.h - 2.0*self.tf)/2.0
        z_bot = self.tf/2.0

        self.Ai = np.array([A_top, A_web, A_bot])
        self.zi = np.array([z_top, z_web, z_bot])
        self.A = A_top + A_web + A_bot

        # Centroide medido desde la base
        self.zG = np.dot(self.Ai, self.zi) / self.A
    
    
    def compute_inertias_centroidal(self): # respecto al centroide
        A_top, A_web, A_bot = self.Ai
        z_top, z_web, z_bot = self.zi
        zG = self.zG

        # Calculo de Iy, inercias locales
        h_web = self.h - 2.0*self.tf
        Iy_top_local = self.btf * (self.tf**3) / 12
        Iy_web_local = self.tw * (h_web**3) / 12.0
        Iy_bot_local = self.bbf * (self.tf**3) / 12.0

        # teorema de ejes paralelos
        Iy_top = Iy_top_local + A_top * (z_top - zG)**2
        Iy_web = Iy_web_local + A_web * (z_web - zG)**2
        Iy_bot = Iy_bot_local + A_bot * (z_bot - zG)**2
        self.Iy = Iy_top + Iy_web + Iy_bot

        # Calculo de Iz, inercias de cada area. ya son globales por que xc = 0
        Iz_top = self.tf * (self.btf**3) / 12
        Iz_web = h_web * (self.tw**3) / 12
        Iz_bot = self.tf * (self.bbf**3) / 12
        self.Iz = Iz_top + Iz_web + Iz_bot

    def compute_torsional_const(self): # como si fuera thin-walled
        # longitudes parciales de linea neutra
        m_top = self.btf
        m_web = self.h - self.tf
        m_bot = self.bbf

        self.It = 1/3 * (m_top * self.tf**3+
                         m_web * self.tw**3+
                         m_bot * self.tf**3)
    


    def compute_shear_center(self):
        # inercias locales de las mesas
        Iz_top = self.tf * (self.btf**3) / 12
        Iz_bot = self.tf * (self.bbf**3) / 12
        # measure of monossimetry
        rho = Iz_top / self.Iz
        # distance between flange shear centers
        h_s = self.zi[0] - self.zi[2]
        # Centro de corte medido desde la base
        self.zC = self.zi[0] - (1 - rho) * h_s

    def compute_warping_const(self):
        # inercias locales de las mesas
        Iz_top = self.tf * (self.btf**3) / 12
        Iz_bot = self.tf * (self.bbf**3) / 12
        # distancias de los centroides locales al centro de corte
        a = self.zi[0] - self.zC
        b = self.zi[2] - self.zC # sale negativo
        # Constante de warping
        self.Iw = a**2 * Iz_top + b**2 * Iz_bot



    def compute_polar_radius(self):
        # radio de giro polar al cuadrado respecto del centro de corte
        self.ryz2 = (self.Iy + self.Iz) / self.A + self.zC**2



    

        
    def get_I_psi(self):
        return
    
    def get_I_omega_psi(self):
        return
    
    def get_I_y_psi(self):
        return

'''


