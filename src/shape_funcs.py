
import numpy as np

# r = xi
# s = eta
# t = zeta

# FUNCIONES DE FORMA EN SENTIDO ANTIHORARIO
# COLOCAR CONECTIVIDADES EN SENTIDO ANTIHORARIO
#def shape_bar(r):



def shape_beam(r):
    r2 = r*r
    r3 = r2*r

    N1 = 1 - 3*r2 + 2*r3
    N2 = r - 2*r2 + r3
    N3 = 3*r2 - 2*r3
    N4 = -r2 + r3

    return np.array([N1, N2, N3, N4])


def deriv1beam(r):
    r2 = r*r

    d1N1 = -6*r + 6*r2
    d1N2 = 1 - 4*r + 3*r2
    d1N3 = 6*r - 6*r2
    d1N4 = -2*r + 3*r2
    
    return np.array([d1N1, d1N2, d1N3, d1N4])


def deriv2beam(r):
    d2N1 = -6 + 12*r
    d2N2 = -4 + 6*r
    d2N3 = 6 - 12*r
    d2N4 = -2 + 6*r

    return np.array([d2N1, d2N2, d2N3, d2N4])

    
      



