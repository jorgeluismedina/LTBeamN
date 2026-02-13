
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse
from .constructors import constructor1, constructor2


'''
GLOSARIO DE VARIABLES
ndofn = Numero de grados de libertad por nodo
nnods = Numero de nodos en la estructura
ndofs = Numero de grados de libertad de la estructura
elems = Lista de todos los elementos finitos
'''
class Model():
    def __init__(self, ndofn):
        self.ndofn = ndofn 
        self.elems = []


        self.fixd_nodes = [] #tags
        self.restraints = [] #values

        self.loaded_nodes = [] #tags
        self.nodal_loads = [] #values

        self.imposed_disp_nodes = [] #tags
        self.imposed_disps = [] #values

        self.loaded_elems = [] #tags

    def add_materials(self, materials): # Son necesarios ahora?
        self.mater = materials

    def add_sections(self, sections): # Son necesarios ahora?
        self.sections = sections

    def assemb_global_vec(self, node_tag, values):
        dof = self.all_nodof[node_tag].flatten()
        vals = np.array(values).flatten()
        return dof, vals

    def add_nodes(self, coordinates):
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        self.ndofs = self.ndofn * self.nnods
        self.all_dof = np.arange(self.ndofs, dtype=int)
        self.all_nodof = np.reshape(self.all_dof, (self.nnods, self.ndofn))

    def add_node_restraint(self, tag, restraints):
        self.fixd_nodes.append(tag)
        self.restraints.append(restraints)

    def set_restraints(self):
        dofs, vals = self.assemb_global_vec(self.fixd_nodes, self.restraints)
        self.fixd_dof = list(dofs[vals.astype(bool)])
        self.free_dof = list(np.setdiff1d(self.all_dof, self.fixd_dof))

    def add_node_load(self, tag, loads): 
        self.loaded_nodes.append(tag)
        self.nodal_loads.append(loads)


    def add_element(self, etype, material, section, conec):
        coord = self.coord[conec]
        dof   = self.all_nodof[conec].ravel()
        self.elems.append(constructor1(etype, material, section, coord, conec, dof))
        

    def add_elem_load(self, elem_tag, loads):
        self.loaded_elems.append(elem_tag)
        self.elems[elem_tag].add_loads(*loads)
          
    
    def assemb_global_loads(self):
        glob_loads = np.zeros(self.ndofs)

        if self.loaded_nodes:
            dofs, vals = self.assemb_global_vec(self.loaded_nodes, self.nodal_loads)
            glob_loads[dofs] = vals

        if self.loaded_elems:
            for id_elem in self.loaded_elems:
                glob_loads[self.elems[id_elem].dof] += self.elems[id_elem].loads
            
        return glob_loads
    
    
    def add_node_disp(self, tag, imposed_disps): 
        self.imposed_disp_nodes.append(tag)
        self.imposed_disps.append(imposed_disps)

    def assemb_global_disps(self):
        glob_disps = np.zeros(self.ndofs)

        if self.imposed_disp_nodes:
            dofs, vals = self.assemb_global_vec(self.imposed_disp_nodes, self.imposed_disps)
            glob_disps[dofs] = vals

        return glob_disps

    def assemb_global_stiff(self):
        glob_stiff = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            glob_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff 

        return glob_stiff

    
    def assemb_global_stiff_sparse(self):
        rows = []
        cols = []
        vals = []

        for elem in self.elems:
            dof = elem.dof
            n = dof.size
            stiff = elem.stiff
            rows.append(np.repeat(dof, n))
            cols.append(np.tile(dof, n))
            vals.append(stiff.ravel())
        
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        glob_stiff = coo_matrix((vals, (rows, cols)), shape=(self.ndofs, self.ndofs)).tocsr()
        return glob_stiff
    

    def calculate_forces(self, glob_disps):
        for elem in self.elems:
            edisp = glob_disps[elem.dof]
            elem.calculate_forces(edisp)



    def generate_fields(self):
        x_fields = []
        N_fields = []
        V_fields = []
        M_fields = []
        u_fields = []
        v_fields = []

        for elem in self.elems:
            x, N, V, M, u, v = elem.get_element_fields()
            x_fields.append(x)
            N_fields.append(N)
            V_fields.append(V)
            M_fields.append(M)
            u_fields.append(u)
            v_fields.append(v)

        all_fields = [x_fields,
                      N_fields,
                      V_fields,
                      M_fields,
                      u_fields,
                      v_fields]
        return all_fields





class StabilityModel():
    def __init__(self, ndofn1, ndofn2):
        self.ndofn1 = ndofn1
        self.ndofn2 = ndofn2
        self.elems = []

        # estatico
        self.fixd_nodes1 = [] #tags
        self.restraints1 = [] #values
        # estabilidad
        self.fixd_nodes2 = [] #tags
        self.restraints2 = [] #values

        self.loaded_nodes = [] #tags
        self.nodal_loads = [] #values

        self.imposed_disp_nodes = [] #tags
        self.imposed_disps = [] #values

        self.loaded_elems = [] #tags

    def add_materials(self, materials): 
        self.mater = materials

    def add_sections(self, sections):
        self.sections = sections

    def assemb_global_vec(self, all_nodof, node_tag, values):
        dof = all_nodof[node_tag].flatten()
        vals = np.array(values).flatten()
        return dof, vals

    def add_nodes(self, coordinates):
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        # Grados de libertad problema estatico (u, w, w,x)
        self.ndofs1 = self.ndofn1 * self.nnods
        self.all_dof1 = np.arange(self.ndofs1, dtype=int)
        self.all_nodof1 = np.reshape(self.all_dof1, (self.nnods, self.ndofn1))
        # Grados de libertad problema de estabilidad (v, v,x, th, th,x)
        self.ndofs2 = self.ndofn2 * self.nnods
        self.all_dof2 = np.arange(self.ndofs2, dtype=int)
        self.all_nodof2 = np.reshape(self.all_dof2, (self.nnods, self.ndofn2))

    def add_node_restraint1(self, tag, restraints):
        # nodos restrictos estatico
        self.fixd_nodes1.append(tag)
        self.restraints1.append(restraints)

    def add_node_restraint2(self, tag, restraints):
        # nodos restrictos estatico
        self.fixd_nodes2.append(tag)
        self.restraints2.append(restraints)

    def set_restraints1(self):
        # estatico
        dofs1, vals1 = self.assemb_global_vec(self.all_nodof1, self.fixd_nodes1, self.restraints1)
        self.fixd_dof1 = list(dofs1[vals1.astype(bool)])
        self.free_dof1 = list(np.setdiff1d(self.all_dof1, self.fixd_dof1))

    def set_restraints2(self):
        # estabilidad
        dofs2, vals2 = self.assemb_global_vec(self.all_nodof2, self.fixd_nodes2, self.restraints2)
        self.fixd_dof2 = list(dofs2[vals2.astype(bool)])
        self.free_dof2 = list(np.setdiff1d(self.all_dof2, self.fixd_dof2))




    def add_node_load(self, tag, loads): 
        self.loaded_nodes.append(tag)
        self.nodal_loads.append(loads)

    def add_element(self, etype, material, section, conec):
        coord = self.coord[conec]
        dof1  = self.all_nodof1[conec].ravel()
        dof2 =  self.all_nodof2[conec].ravel()
        self.elems.append(constructor2(etype, material, section, coord, conec, dof1, dof2))
        self.nelem = len(self.elems)
        
    def add_elem_load(self, elem_tag, loads):
        self.loaded_elems.append(elem_tag)
        self.elems[elem_tag].add_loads(*loads)
          


    def assemb_global_loads(self):
        glob_loads = np.zeros(self.ndofs1)

        if self.loaded_nodes:
            dofs, vals = self.assemb_global_vec(self.all_nodof1, self.loaded_nodes, self.nodal_loads)
            glob_loads[dofs] = vals

        if self.loaded_elems:
            for id_elem in self.loaded_elems:
                glob_loads[self.elems[id_elem].dof] += self.elems[id_elem].loads
            
        return glob_loads
    
    

    def add_node_disp(self, tag, imposed_disps): 
        self.imposed_disp_nodes.append(tag)
        self.imposed_disps.append(imposed_disps)

    def assemb_global_disps(self):
        glob_disps = np.zeros(self.ndofs1)

        if self.imposed_disp_nodes:
            dofs, vals = self.assemb_global_vec(self.all_nodof1, self.imposed_disp_nodes, self.imposed_disps)
            glob_disps[dofs] = vals

        return glob_disps

    
    # para el elemento ltbeam
    def assemb_global_stiff1(self):
        glob_bend_stiff = np.zeros((self.ndofs1, self.ndofs1))
        for elem in self.elems:
            glob_bend_stiff[np.ix_(elem.dof1, elem.dof1)] += elem.bend_stiff 

        return glob_bend_stiff
    
    # para el elemento ltbeam
    def assemb_global_stiff2(self):
        glob_lator_stiff = np.zeros((self.ndofs2, self.ndofs2))
        for elem in self.elems:
            glob_lator_stiff[np.ix_(elem.dof2, elem.dof2)] += elem.lator_stiff 

        return glob_lator_stiff

    # para el elemento ltbeam
    def assemb_global_geom(self):
        glob_lator_geom = np.zeros((self.ndofs2, self.ndofs2))
        for elem in self.elems:
            elem.compute_lator_geom_mat()
            glob_lator_geom[np.ix_(elem.dof2, elem.dof2)] += elem.lator_geom

        return glob_lator_geom
    
    
    def assemb_global_stiff_sparse(self):
        rows = []
        cols = []
        vals = []

        for elem in self.elems:
            dof = elem.dof
            n = dof.size
            stiff = elem.stiff
            rows.append(np.repeat(dof, n))
            cols.append(np.tile(dof, n))
            vals.append(stiff.ravel())
        
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        glob_stiff = coo_matrix((vals, (rows, cols)), shape=(self.ndofs1, self.ndofs1)).tocsr()
        return glob_stiff


    
    
    def calculate_forces(self, glob_disps):
        for elem in self.elems:
            edisp = glob_disps[elem.dof1]
            elem.calculate_forces(edisp)



    def generate_fields(self):
        x_fields = []
        N_fields = []
        V_fields = []
        M_fields = []
        u_fields = []
        w_fields = []

        for elem in self.elems:
            x, N, V, M, u, v = elem.get_element_fields()
            x_fields.append(x)
            N_fields.append(N)
            V_fields.append(V)
            M_fields.append(M)
            u_fields.append(u)
            w_fields.append(v)

        all_fields = [x_fields,
                      N_fields,
                      V_fields,
                      M_fields,
                      u_fields,
                      w_fields]
        return all_fields




        


