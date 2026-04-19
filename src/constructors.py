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
                       coord, conec, verax_dof, lator_dof):
        """Crea instancia del elemento uniforme."""
        if etype not in cls.registry:
            raise ValueError(f"Element type not supported: {etype}")
        
        return cls.registry[etype](material, section, 
                                   coord, conec, 
                                   verax_dof, lator_dof)
    

    @classmethod
    def create_tapered(cls, etype, material,
                       section_i, section_j,
                       coord, conec,
                       verax_dof, lator_dof):
        
        """Crea instancia del elemento tapered."""
        if etype not in cls.registry:
            raise ValueError(f"Element type not supported: {etype}")
        
        return cls.registry[etype](material, 
                                   section_i, section_j, 
                                   coord, conec, 
                                   verax_dof, lator_dof)
        


# Registrar elemento disponible
ElementFactory.register(0, LTBeam)
ElementFactory.register(1, LTBeamTap)



