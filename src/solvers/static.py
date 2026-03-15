
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import spsolve



class StaticSolver():
    def __init__(self, model):
        self.model = model


    def assemble_verax_K0(self):
        """Ensambla matriz de rigidez global."""
        nvrx_dofs = self.model.nvrx_dofs
        K0_vrx = np.zeros((nvrx_dofs, nvrx_dofs))
        for elem in self.model.elems:
            K0_vrx[np.ix_(elem.vrx_dof, elem.vrx_dof)] += elem.K0_vrx
        
        return K0_vrx
    

    def assemble_verax_F(self):
        """Ensambla vector de cargas."""
        F = np.zeros(self.model.nvrx_dofs)
        
        if self.model.loaded_nodes:
            dofs, vals = self.model.assemble_global_vec(
                self.model.avrx_dof, 
                self.model.loaded_nodes, 
                self.model.nodal_loads
            )
            F[dofs] = vals

        if self.model.loaded_elems:
            for id_elem in self.model.loaded_elems:
                F[self.model.elems[id_elem].vrx_dof] += self.model.elems[id_elem].loads

        return F

    
    def process_verax_restraints(self):
        """Separa DOFs fijos y libres."""
        dofs, vals = self.model.assemble_global_vec(
            self.model.avrx_dof,
            self.model.svrx_nodes, 
            self.model.vrx_restraints
        )
        adof = self.model.avrx_dof.ravel()
        sdof = dofs[vals.astype(bool)] # DOFs fijos
        fdof = np.setdiff1d(adof, sdof) # DOFs libres
        
        return fdof, sdof

    
    def solve(self):
        """ Resuelve el problema y retorna resultados."""
        # Ensambla
        self.K0 = self.assemble_verax_K0()
        self.F = self.assemble_verax_F()
        free, supp = self.process_verax_restraints()
        
        # Reduce a DOFs libres    
        K0_ff = self.K0[np.ix_(free, free)]
        K0_sf = self.K0[np.ix_(supp, free)]
        F_f = self.F[free]
        F_s = self.F[supp]
        
        # Resuelve
        disps_f = cho_solve(cho_factor(K0_ff), F_f)
        
        disps = np.zeros(self.model.nvrx_dofs)
        disps[free] = disps_f
        react = K0_sf @ disps_f - F_s
        
        self.compute_internal_forces(disps)
        
        return disps, react
    
    def compute_internal_forces(self, glob_disps):
        """Calcula fuerzas internas verticales en elementos."""
        for elem in self.model.elems:
            elem.calculate_forces(glob_disps[elem.vrx_dof])


    def generate_fields(self):
        """Genera los diagramas locales"""
        x_fields = []
        N_fields = []
        V_fields = []
        M_fields = []
        u_fields = []
        w_fields = []

        for elem in self.model.elems:
            x, N, V, M, u, w = elem.get_fields()
            x_fields.append(x)
            N_fields.append(N)
            V_fields.append(V)
            M_fields.append(M)
            u_fields.append(u)
            w_fields.append(w)

        all_fields = [x_fields,
                      N_fields,
                      V_fields,
                      M_fields,
                      u_fields,
                      w_fields]
        
        return all_fields
    
    
    def prepare_diagrams(self, all_fields, esc1=0.6, esc2=0.8, esc3=0.6):
        all_x, all_N, all_V, all_M, all_u, all_w = all_fields

        max_N = np.max(np.abs(np.asarray(all_N))) or 1
        max_V = np.max(np.abs(np.asarray(all_V))) or 1
        max_M = np.max(np.abs(np.asarray(all_M))) or 1
        max_u = np.max(np.abs(np.asarray(all_u)))
        max_w = np.max(np.abs(np.asarray(all_w)))
        max_def = max(max_u, max_w) or 1

        N_globals = []
        V_globals = []
        M_globals = []
        def_shapes = []

        for e, elem in enumerate(self.model.elems):
            x = all_x[e]
            X0 = elem.coord[0]
            
            # Coordenadas globales
            X = X0 + x
            
            # Diagramas (perpendiculares a X, en Y)
            N_diag_Y = all_N[e] / max_N * esc1
            V_diag_Y = all_V[e] / max_V * esc1
            M_diag_Y = all_M[e] / max_M * esc2
            
            # Deformada
            X_def = X + all_u[e] / max_def * esc3
            Y_def = all_w[e] / max_def * esc3
            
            # Almacena
            N_globals.append(np.vstack([X, N_diag_Y, all_N[e]]))
            V_globals.append(np.vstack([X, V_diag_Y, all_V[e]]))
            M_globals.append(np.vstack([X, M_diag_Y, all_M[e]]))
            def_shapes.append(np.vstack([X_def, Y_def]))

        return N_globals, V_globals, M_globals, def_shapes
    

    
    """
    def prepare_diagrams(self, fields, esc1=0.5, esc2=0.7, esc3=0.5):
        all_x, all_N, all_V, all_M, all_u, all_w = fields
        max_N = np.max(np.abs(np.asarray(all_N)))
        max_V = np.max(np.abs(np.asarray(all_V)))
        max_M = np.max(np.abs(np.asarray(all_M)))
        max_u = np.max(np.abs(np.asarray(all_u)))
        max_w = np.max(np.abs(np.asarray(all_w)))
        max_def = max(max_u, max_w)

        N_globals = []
        V_globals = []
        M_globals = []
        def_shapes = []

        for e, elem in enumerate(self.model.elems):
            x = all_x[e]
            N = all_N[e] / max_N # axial normalizada
            V = all_V[e] / max_V # corte  normalizada
            M = all_M[e] / max_M # momento normalizada
            u = all_u[e] / max_def # def axial normalizada
            w = all_w[e] / max_def # def vertical normalizada

            c, s = elem.dir_vec 
            # Coordenadas globales de puntos a lo largo del elemento
            X0, Y0 = elem.coord[0]
            X = X0 + c*x
            Y = Y0 + s*x

            # Desplazamientos globales (vectorizado)
            u_global = c*u - s*w    # Componente X global
            w_global = s*u + c*w    # Componente Y global

            # Diagramas rotados (perpendiculares al elemento)
            N_diag_X = X - s*N*esc1
            N_diag_Y = Y + c*N*esc1

            V_diag_X = X - s*V*esc1
            V_diag_Y = Y + c*V*esc1

            M_diag_X = X + s*M*esc2  # Escala más pequeña para momento
            M_diag_Y = Y - c*M*esc2  # Momentos ploteados alrevez por convencion

            X_def = X + u_global*esc3
            Y_def = Y + w_global*esc3

            N_globals.append(np.vstack([N_diag_X, N_diag_Y, all_N[e]]))
            V_globals.append(np.vstack([V_diag_X, V_diag_Y, all_V[e]]))
            M_globals.append(np.vstack([M_diag_X, M_diag_Y, all_M[e]]))
            def_shapes.append(np.vstack([X_def, Y_def]))

        return N_globals, V_globals, M_globals, def_shapes

    """





