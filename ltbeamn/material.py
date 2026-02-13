
import numpy as np

class Material:
    def __init__(self, elast, poiss, dense):
        self.elast = elast
        self.poiss = poiss
        self.dense = dense
        self.shear = elast / (2 * (1 + poiss))