from .elements.ltbeam import LTBeam
from .elements.ltbeamtap import LTBeamTap




class ElementFactory:
    """Factory para crear elementos"""
    
    registry = {}
    
    @classmethod
    def register(cls, name, element_class):
        """Registra nuevo tipo de elemento."""
        cls.registry[name] = element_class
        

    
    @classmethod
    def create_uniform(cls, etype, material, section, 
                       coord, conec, 
                       verax_dof, lator_dof):
        """Crea instancia del elemento."""
        if etype not in cls.registry:
            raise ValueError(f"Tipo de elemento no registrado: {etype}")
        
        return cls.registry[etype](material, section, 
                                   coord, conec, 
                                   verax_dof, lator_dof)
    

    @classmethod
    def create_tapered(cls, etype, material,
                       section_i, section_j,
                       coord, conec,
                       verax_dof, lator_dof):
        """Crea instancia del elemento."""
        if etype not in cls.registry:
            raise ValueError(f"Tipo de elemento no registrado: {etype}")
        
        return cls.registry[etype](material, 
                                   section_i, section_j, 
                                   coord, conec, 
                                   verax_dof, lator_dof)
        


# Registrar elemento disponible
ElementFactory.register(1, LTBeam)
ElementFactory.register(2, LTBeamTap)



