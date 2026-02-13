
import numpy as np
from .elements.truss2d import Truss2D
from .elements.beam import Beam
from .elements.ltbeam import LTBeam
from .elements.frame2d import Frame2D




def constructor1(etype, mater, section, coord, conec, dof):
    if etype == 'Truss2D':
        return Truss2D(mater, section, coord, conec, dof)
    
    elif etype == 'Beam':
        return Beam(mater, section, coord, conec, dof)
    
    elif etype == 'Frame2D':
        return Frame2D(mater, section, coord, conec, dof)
    
    else:
        raise ValueError(f"Not supported element type: {etype}")
    
    
    
def constructor2(etype, mater, section, coord, conec, dof1, dof2):
    if etype == 'LTBeam':
        return LTBeam(mater, section, coord, conec, dof1, dof2)
    
    #elif etype == 'LTBeamN':
        #return LTBeamN(mater, section, coord, conec, dof1, dof2)

    else:
        raise ValueError(f"Not supported element type: {etype}")




