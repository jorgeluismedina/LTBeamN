
import numpy as np
from .constructors import ElementFactory


class StabilityModel():
    def __init__(self):
        self.nvrx_dofn = 3 # numero de DOF por nodo (u, w, w,x) 
        self.nltr_dofn = 4 # numero de DOF por nodo (v, v,x, th, th,x)
        self.elems = [] # cambiar el atributo a elements?

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

        # estatico
        self.loaded_nodes = [] # tags de nodos cargados
        self.nodal_loads = [] # cargas nodales
        self.nodal_load_qzpos = [] # alturas de carga vertical

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

    def add_uniform_elements(self, elements_data):
        """ Funcion solo para añadir elementos barra"""
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
            

    def add_tapered_elements(self, elements_data, align=0):
        """
        Añade elementos barra de seccion variable (tapered).

        Parametros
        ----------
        elements_data : array-like
            Cada fila: [etype, mat_id, nodei, nodej]
        align : int
            Tipo de alineacion de secciones a lo largo del eje de la barra.
            Determina sobre que punto de la seccion transversal pasa el eje
            de referencia x del elemento, lo que afecta al offset del centroide
            y por ende al acoplamiento axial-flexion en K0_vrx, y a la
            distribucion de N y M que alimenta la matriz geometrica Kg.

            0 → eje x pasa por el centroide G(x) en cada seccion.
                Sin acoplamiento axial-flexion. Caso mas simple; N = 0 bajo
                cargas verticales sobre barra horizontal (equivalente al
                comportamiento previo del codigo).
            1 → eje x alineado con la fibra superior (ala superior horizontal).
                Geometria tipica de correas y cabrios con taper hacia abajo.
            2 → eje x alineado con la fibra inferior (ala inferior horizontal).
                Geometria tipica de vigas con taper hacia arriba.

            Para barras horizontales con cargas verticales, la diferencia
            entre align=0, 1 y 2 es practicamente nula (N ~ 0 en todos los
            casos). El efecto es significativo solo en barras inclinadas o
            bajo carga axial combinada con flexion.
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
            pos : posicion vertical en la seccion
                0 → centro de corte
                1 → centroide
                2 → ala inferior
                3 → ala superior
            
        Pasar 0 para omitir un tipo de muelle en un nodo.
        """
        self.spring_nodes = list(springs_data[:, 0].astype(int))
        self.spring_pos   = springs_data[:, 1].astype(int)
        self.spring_kv    = springs_data[:, 2].astype(float)
        self.spring_kt    = springs_data[:, 3].astype(float)


    def add_nodal_loads(self, nodal_loads_data):
        """ 
        Añade cargas verticales, axiales y de momento en coordenadas locales
        Formato: [node, qx, qz, Mx, qzpos=0]
            qzpos:
                0 → centro de corte
                1 → centroide
                2 → ala inferior
                3 → ala superior
        """
        self.loaded_nodes   = list(nodal_loads_data[:,0].astype(int))
        self.nodal_load_pos = nodal_loads_data[:,1].astype(int)
        self.nodal_loads    = nodal_loads_data[:,2:].astype(float)
        

    def add_elem_loads(self, elem_loads_data):
        """ 
        Añade cargas verticales de elemento en coordenads locales
        Formato: [id_elem, qxi, qzi, qxj, qzj, qzpos=0]
             qzpos:
                0 → centro de corte
                1 → centroide
                2 → ala inferior
                3 → ala superior
        """
        self.loaded_elems   = list(elem_loads_data[:,0].astype(int))
        self.elem_loads_pos = elem_loads_data[:,1].astype(int)
        self.elem_loads     = elem_loads_data[:,2:].astype(float)
        

        for load_data in elem_loads_data:
            id_elem = int(load_data[0])
            pos     = load_data[1].astype(int)
            loads   = load_data[2:].astype(float)
            self.elems[id_elem].add_loads(pos, *loads)




        


