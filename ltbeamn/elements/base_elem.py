
import numpy as np
from abc import ABC, abstractmethod

class Element():
    def __init__(self, mater, coord, conec, dof):
        self.mater = mater
        self.coord = coord
        self.conec = conec
        self.dof = dof
        self.loads = None
        

class FrameElement(Element):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, coord, conec, dof)
        self.section = section


