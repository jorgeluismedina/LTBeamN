
import numpy as np
from .constructors import ElementFactory


class StabilityModel():
    def __init__(self):
        self.nvrx_dofn = 3 # numero de DOF por nodo (u, w, w,x) 
        self.nltr_dofn = 4 # numero de DOF por nodo (v, v,x, th, th,x)
        self.elems = []

        # estatico
        self.svrx_nodes = [] # tags de nodos con DOF de flex. vertical y desp. axial restringidos
        self.vrx_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # estabilidad
        self.sltr_nodes = [] # tags de nodos con DOF de flex. lateral y torsion restringidos
        self.ltr_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # estatico
        self.loaded_nodes = [] # tags de nodos cargados
        self.nodal_loads = [] # cargas nodales

        self.imposed_disp_nodes = [] #tags
        self.imposed_disps = [] #values

        self.loaded_elems = [] #tags

    def add_materials(self, materials): 
        self.materials = materials

    def add_sections(self, sections):
        self.sections = sections

    def assemble_global_vec(self, all_dof, node_tag, values):
        dof = all_dof[node_tag].flatten()
        vals = np.array(values).flatten()
        return dof, vals

    def add_nodes(self, coordinates):
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        # DOF problema estatico (u, w, w,x)
        self.nvrx_dofs = self.nvrx_dofn * self.nnods # numero total de DOF problema estatico
        self.avrx_dof = np.arange(self.nvrx_dofs).reshape((self.nnods, self.nvrx_dofn)) # DOF ordenados por nodo
        # DOF problema de estabilidad (v, v,x, th, th,x)
        self.nltr_dofs = self.nltr_dofn * self.nnods # numero total de DOF problema estabilidad
        self.altr_dof = np.arange(self.nltr_dofs).reshape((self.nnods, self.nltr_dofn)) # DOF ordenados por nodo

    def add_elements(self, elements_data):
        """ Funcion solo para añadir elementos barra"""
        for elem_data in elements_data:
            etype, mat_id, sec_id, nodei, nodej = elem_data

            mat = self.materials[int(mat_id)]
            sec = self.sections[int(sec_id)]
            conec = [int(nodei), int(nodej)]
            
            coord = self.coord[conec]
            vrx_dof = self.avrx_dof[conec].flatten()
            ltr_dof = self.altr_dof[conec].flatten()

            elem = ElementFactory.create(etype, mat, sec, coord, conec, vrx_dof, ltr_dof)
            self.elems.append(elem)


    def add_verax_restraints(self, verax_restraints_data):
        self.svrx_nodes = list(verax_restraints_data[:,0].astype(int))
        self.vrx_restraints = verax_restraints_data[:,1:].astype(int)

    def add_lator_restraints(self, lator_restraints_data):
        self.sltr_nodes = list(lator_restraints_data[:,0].astype(int))
        self.ltr_restraints = lator_restraints_data[:,1:].astype(int)


    def add_nodal_loads(self, nodal_loads_data):
        """ Añade cargas verticales y axiales en coordenadas locales"""
        self.loaded_nodes = list(nodal_loads_data[:,0].astype(int))
        self.nodal_loads = nodal_loads_data[:,1:].astype(float)

    def add_elem_loads(self, elem_loads_data):
        """ Añade cargas verticales de elemento en coordenads locales"""
        self.loaded_elems = list(elem_loads_data[:,0].astype(int))
        self.elem_loads = elem_loads_data[:,1:].astype(float)

        for load_data in elem_loads_data:
            id_elem = int(load_data[0])
            loads = load_data[1:].astype(float)
            self.elems[id_elem].add_loads(*loads)




        


