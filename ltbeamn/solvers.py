
import numpy as np
import scipy as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import spsolve


def check_symmetric(a, rtol=1e-6, atol=1e-5):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_conver(resid, force, tol):
    resid = sp.linalg.norm(resid)
    force = sp.linalg.norm(force)
    ratio = resid/force
    return ratio <= tol




def solve_linear_static(model): # dense matrix

    model.set_restraints()
    fixd_dof = model.fixd_dof
    free_dof = model.free_dof

    glob_stiff = model.assemb_global_stiff()
    glob_loads = model.assemb_global_loads()
    glob_disps = model.assemb_global_disps()

    # Reduccion del sistema
    stiff_ff = glob_stiff[np.ix_(free_dof, free_dof)]
    stiff_sf = glob_stiff[np.ix_(fixd_dof, free_dof)]
    glob_loads -= glob_stiff[:,fixd_dof] @ glob_disps[fixd_dof] # desplazamientos impuestos -> fuerzas

    # Resolucion del Sistema
    # Cholesky solo va a funcionar cuando la matriz sea Sym-Pos-definite
    # los nodos tienen que estar en sentido antihorario para que sea SPD
    free_disps = cho_solve(cho_factor(stiff_ff), glob_loads[free_dof])
    glob_disps[free_dof] = free_disps
    glob_react = stiff_sf @ free_disps - glob_loads[fixd_dof]

    return glob_disps, glob_react



# Para la clase Stability Model
def solve_linear_static2(model): # dense matrix

    model.set_restraints1()
    fixd_dof = model.fixd_dof1
    free_dof = model.free_dof1

    glob_stiff = model.assemb_global_stiff1()
    glob_loads = model.assemb_global_loads()
    glob_disps = model.assemb_global_disps()

    # Reduccion del sistema
    stiff_ff = glob_stiff[np.ix_(free_dof, free_dof)]
    stiff_sf = glob_stiff[np.ix_(fixd_dof, free_dof)]
    glob_loads -= glob_stiff[:,fixd_dof] @ glob_disps[fixd_dof] # desplazamientos impuestos -> fuerzas

    # Resolucion del Sistema
    # Cholesky solo va a funcionar cuando la matriz sea Sym-Pos-definite
    # los nodos tienen que estar en sentido antihorario para que sea SPD
    free_disps = cho_solve(cho_factor(stiff_ff), glob_loads[free_dof])
    glob_disps[free_dof] = free_disps
    glob_react = stiff_sf @ free_disps - glob_loads[fixd_dof]

    return glob_disps, glob_react




def buckling_modes(Kff, Kgff):
    # Calculo de los autovectores y autovalores
    vals, vecs = sp.linalg.eig(Kff, Kgff)
    vals = np.real(vals)
    # solo psoitivos
    pos_indices = np.where(vals > 0)[0]
    vals = vals[pos_indices]
    vecs = vecs[:, pos_indices]
    # Ordenamiento de autovalores de manera creciente
    idx = vals.argsort()
    vals = vals[idx]
    vecs = vecs[:, idx] 

    return vals, vecs



def solve_stability(model):
    model.set_restraints2()
    free_dof = model.free_dof2

    glob_stiff = model.assemb_global_stiff2()
    glob_geom  = model.assemb_global_geom()

    # Reduccion del sistema
    stiff_ff = glob_stiff[np.ix_(free_dof, free_dof)]
    geom_ff  = glob_geom[np.ix_(free_dof, free_dof)]
    
    # Autovlores y Autovectores
    vals, vecs = buckling_modes(stiff_ff, -geom_ff)

    return vals, vecs




def detailed_check_SPD(Kg): # matriz semidefinida positiva
    eigs = np.linalg.eigvalsh(Kg)
    
    print(f"Autovalor mínimo: {np.min(eigs):.2e}")
    print(f"Autovalor máximo: {np.max(eigs):.2e}")
    
    # Contar autovalores según su magnitud
    zero_eigs = np.sum(np.abs(eigs) < 1e-10)  # Prácticamente cero
    positive_eigs = np.sum(eigs > 1e-10)      # Claramente positivos
    negative_eigs = np.sum(eigs < -1e-10)     # Claramente negativos
    
    print(f"Autovalores ~0 (modos cuerpo rígido): {zero_eigs}")
    print(f"Autovalores >0 (modos elásticos): {positive_eigs}") 
    print(f"Autovalores <0 (problemas!): {negative_eigs}")
    
    # Para un elemento shell de 8 nodos libre, esperamos:
    # - 6 autovalores cero (modos cuerpo rígido)
    # - 42 autovalores positivos (modos elásticos)
    # - 0 autovalores negativos
    
    expected_rigid_body_modes = 6
    if zero_eigs == expected_rigid_body_modes and negative_eigs == 0:
        print("✓ ¡PERFECTO! El elemento tiene el comportamiento esperado")
    else:
        print("✗ Comportamiento inesperado - revisar implementación")
