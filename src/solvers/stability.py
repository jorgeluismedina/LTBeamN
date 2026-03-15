
import numpy as np
import scipy as sp


class StabilitySolver():
    def __init__(self, model):
        self.model = model

    def assemble_lator_K0(self):
        """Ensambla matriz de rigidez global."""
        nltr_dofs = self.model.nltr_dofs
        K0_ltr = np.zeros((nltr_dofs, nltr_dofs))
        for elem in self.model.elems:
            K0_ltr[np.ix_(elem.ltr_dof, elem.ltr_dof)] += elem.K0_ltr
        
        return K0_ltr

    def assemble_lator_Kg(self):
        """Ensambla matriz geometrica global."""
        nltr_dofs = self.model.nltr_dofs
        Kg_ltr = np.zeros((nltr_dofs, nltr_dofs))
        for elem in self.model.elems:
            elem.update_lator_Kg()
            Kg_ltr[np.ix_(elem.ltr_dof, elem.ltr_dof)] += elem.Kg_ltr
        
        return Kg_ltr
    
    def process_lator_restraints(self):
        """Separa DOFs fijos y libres."""
        dofs, vals = self.model.assemble_global_vec(
            self.model.altr_dof,
            self.model.sltr_nodes, 
            self.model.ltr_restraints
        )
        adof = self.model.altr_dof.ravel()
        sdof = dofs[vals.astype(bool)] # DOFs fijos
        fdof = np.setdiff1d(adof, sdof) # DOFs libres
        
        return fdof, sdof
    
    def solve(self):
        """ Resuelve el problema de estabilidad y retorna resultados."""
        # Ensambla
        self.K0 = self.assemble_lator_K0()
        self.Kg = self.assemble_lator_Kg()
        free, supp = self.process_lator_restraints()
        
        # Reduce a DOFs libres    
        K0_ff = self.K0[np.ix_(free, free)]
        Kg_ff = self.Kg[np.ix_(free, free)]
        
        # Resuelve autovectores y autovalores 
        # invirtiendo el problema (-Kg * phi = lam_cr * K0 * phi)
        # lam_cr = 1 / mu_cr, donde mu_cr es la carga crítica de pandeo
        # los autovectores modes son columnas, ca columna es un modo de pandeo
        lam_crs, modes = sp.linalg.eigh(-Kg_ff, K0_ff)

        # solo autovalores reales positivos
        pos_indices = np.where(lam_crs > 1e-12)[0]
        lam_crs = lam_crs[pos_indices]
        modes  = modes[:, pos_indices]

        # Calculo de mu_critico 
        mu_crs = 1 / lam_crs

        # Ordenamiento de autovalores de manera creciente
        idx = mu_crs.argsort()
        mu_crs = mu_crs[idx]
        modes = modes[:, idx]

        # Reconstruccion de modos completos con apoyos incluidos
        full_modes = np.zeros((self.model.nltr_dofs, mu_crs.size))
        full_modes[free, :] = modes
        return mu_crs, full_modes
    
    def reconstruct_full_modes(self, mu_crs, modes, nmodes = 5):
        """ Reconstruye modos completos con apoyos incluidos."""
        full_modes = np.zeros((self.model.nltr_dofs, nmodes))
        free, supp = self.process_lator_restraints()
        for i in range(nmodes):
            full_modes[free, i] = modes[:, i]
            full_modes[supp, i] = 0.0 # Apoyos sin desplazamiento
        
        return mu_crs[:nmodes], full_modes