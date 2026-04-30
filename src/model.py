
import numpy as np
from .constructors import ElementFactory


class StabilityModel():
    def __init__(self):
        self.nvrx_dofn = 3 # numero de DOF por nodo (u, w, w,x) 
        self.nltr_dofn = 4 # numero de DOF por nodo (v, v,x, th, th,x)
        self.elems = [] # cambiar el atributo a elements?
        #self.conectivities = []

        # estatico
        self.svrx_nodes = [] # tags de nodos con DOF de flex. vertical y desp. axial restringidos
        self.vrx_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # estabilidad
        self.sltr_nodes = [] # tags de nodos con DOF de flex. lateral y torsion restringidos
        self.ltr_restraints = [] # estados de restriccion (1 o 0) para cada DOF restringido

        # apoyos elasticos laterales
        self.spring_nodes = []
        self.spring_pos = []
        self.spring_kv = [] # rigidez traslacional lateral (v)
        self.spring_kt = [] # rigidez torsional (theta)

        # cargas estatico
        self.loaded_nodes = [] # tags de nodos cargados
        self.nodal_loads = [] # cargas nodales
        self.fx_load_pos = [] # posicion de carga axial
        self.fz_load_pos = [] # posicion de carga vertical

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
        self.coord = coordinates # cambiar atributo a coordinates?
        self.nnods = coordinates.shape[0]

        # DOF problema estatico (u, w, w,x)
        self.nvrx_dofs = self.nvrx_dofn * self.nnods # numero total de DOF problema estatico
        self.avrx_dof = np.arange(self.nvrx_dofs).reshape((self.nnods, self.nvrx_dofn)) # DOF ordenados por nodo

        # DOF problema de estabilidad (v, v,x, th, th,x)
        self.nltr_dofs = self.nltr_dofn * self.nnods # numero total de DOF problema estabilidad
        self.altr_dof = np.arange(self.nltr_dofs).reshape((self.nnods, self.nltr_dofn)) # DOF ordenados por nodo


    def build_node_alignments(self):
        """ Lista de alineacion de cada nodo, segun los elementos conectados. """
        self.node_align = ['centroid'] * self.nnods
        visited = np.zeros(self.nnods, dtype=bool)
        for elem in self.elems:
            for node in elem.conec:
                if not visited[node]:
                    self.node_align[node] = elem.align   # indexado por node id
                    visited[node] = True


    def add_uniform_elements(self, elements_data):
        """ Funcion solo para añadir elementos barra """
        for elem_data in elements_data:
            etype, mat_id, nodei, nodej = elem_data

            mat   = self.materials[int(mat_id)]
            sec   = self.sections[int(nodei)]
            conec = [int(nodei), int(nodej)]
            
            coord   = self.coord[conec]
            vrx_dof = self.avrx_dof[conec].flatten()
            ltr_dof = self.altr_dof[conec].flatten()

            elem = ElementFactory.create_uniform(etype, mat, sec, 
                                                 coord, conec, 
                                                 vrx_dof, ltr_dof)
            self.elems.append(elem)
        
        self.nelems = len(self.elems)
        self.build_node_alignments()
            

    def add_tapered_elements(self, elements_data, align=0):
        """
        Añade elementos de seccion variable.
 
        Parametros
        ----------
        elements_data : array-like — cada fila: [etype, mat_id, nodei, nodej]
        align : string
            Alineacion del eje de referencia local:
                0 → centroide G(x)    — sin acoplamiento axial-flexión (default)
                3 → fibra superior    — taper hacia abajo
                2 → fibra inferior    — taper hacia arriba
        """
        for elem_data in elements_data:
            etype, mat_id, nodei, nodej = elem_data

            mat   = self.materials[int(mat_id)]
            seci  = self.sections[int(nodei)]
            secj  = self.sections[int(nodej)]
            conec = [int(nodei), int(nodej)]
            
            coord   = self.coord[conec]
            vrx_dof = self.avrx_dof[conec].flatten()
            ltr_dof = self.altr_dof[conec].flatten()

            elem = ElementFactory.create_tapered(etype, mat, 
                                                 seci, secj,
                                                 coord, conec, 
                                                 vrx_dof, ltr_dof,
                                                 align=align)
            self.elems.append(elem)
        
        self.nelems = len(self.elems)
        self.build_node_alignments()


    def add_verax_restraints(self, verax_restraints_data):
        self.svrx_nodes     = list(verax_restraints_data[:,0].astype(int))
        self.vrx_restraints = verax_restraints_data[:,1:].astype(int)

    def add_lator_restraints(self, lator_restraints_data):
        self.sltr_nodes     = list(lator_restraints_data[:,0].astype(int))
        self.ltr_restraints = lator_restraints_data[:,1:].astype(int)

    def add_lateral_springs(self, springs_data):
        """
        Apoyos elasticos nodales en el problema lateral-torsional.
        Formato: [node, pos, kv, kt]
            kv  : rigidez traslacional lateral [F/L]  (v-DOF)
            kt  : rigidez torsional            [F·L]  (θ-DOF)
            pos : pos. vertical, 0→G, 1→SC, 2→ala inf, 3→ala sup

        """
        self.spring_nodes = list(springs_data[:, 0].astype(int))
        self.spring_pos   = springs_data[:, 1].astype(int)
        self.spring_kv    = springs_data[:, 2].astype(float)
        self.spring_kt    = springs_data[:, 3].astype(float)


    def add_nodal_loads(self, nodal_loads_data):
        """
        Cargas puntuales nodales en coordenadas locales.
 
        Formato: [node, fxpos, fzpos, Fx, Fz, Mx]
            fzpos : altura de Fz — 0→G, 1→SC, 2→ala inf, 3→ala sup
                    (usado en el problema de estabilidad, StabilitySolver)
            fxpos : altura de Fx — mismos códigos
                    (la corrección ΔM = Fx·ez la aplica StaticSolver
                    en assemble_verax_F, con la geometría del elemento conectado)
            Fx, Fz, Mx : carga axial, vertical y momento nodal
        """
        self.loaded_nodes = list(nodal_loads_data[:,0].astype(int))
        self.fx_loads_pos = nodal_loads_data[:,1].astype(int)
        self.fz_loads_pos = nodal_loads_data[:,2].astype(int)
        self.nodal_loads  = nodal_loads_data[:,3:].astype(float) # [Fx, Fz, Mx]
        

    def add_elem_loads(self, elem_loads_data):
        """
        Cargas distribuidas de elemento en coordenadas locales.
 
        Formato: [id_elem, qzpos, qxpos, qxi, qzi, qxj, qzj]
            qzpos : altura de qz — 0→G, 1→SC, 2→ala inf, 3→ala sup
            qxpos : altura de qx — mismos códigos
            qxi, qzi : intensidades en nodo i (axial, transversal)
            qxj, qzj : intensidades en nodo j (axial, transversal)
        """
        self.loaded_elems = list(elem_loads_data[:,0].astype(int))
        self.qx_load_pos  = elem_loads_data[:,1].astype(int)
        self.qz_load_pos  = elem_loads_data[:,2].astype(int)
        self.elem_loads   = elem_loads_data[:,3:].astype(float)
        

        for load_data in elem_loads_data:
            id_elem = int(load_data[0])
            qxpos   = load_data[1].astype(int)
            qzpos   = load_data[2].astype(int)
            loads   = load_data[3:].astype(float)

            self.elems[id_elem].add_loads(qxpos, qzpos, *loads)




        


