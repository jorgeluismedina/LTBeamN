
import numpy as np
from abc import ABC, abstractmethod

class Beam():
    def __init__(self, mater, section, coord, conec, verax_dof, lator_dof):
        self.mater = mater
        self.coord = coord
        self.conec = conec
        self.section = section
        self.vrx_dof = verax_dof # vertical deflection and axial displacement
        self.ltr_dof = lator_dof # lateral deflection and torsion
    


