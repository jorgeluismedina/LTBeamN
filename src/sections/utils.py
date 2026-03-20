

import numpy as np
from src.sections.section_bs import ISection_BS
from src.sections.section_ms import ISection_MS


def interpolate_section(section1, section2, xi):
    """
    Interpola linealmente las dimensiones entre dos secciones.
    
    Crea una nueva sección con propiedades interpoladas en el punto xi.
    Interpolación lineal: prop(xi) = prop1 * (1 - xi) + prop2 * xi
    
    Parámetros:
    -----------
    section1 : ISection_BS o ISection_MS
        Sección en el nodo inicial (xi = 0)
    section2 : ISection_BS o ISection_MS
        Sección en el nodo final (xi = 1)
    xi : float
        Coordenada normalizada [0, 1] donde interpolar
        - xi = 0.0 → retorna sección idéntica a section1
        - xi = 0.5 → retorna sección en el centro
        - xi = 1.0 → retorna sección idéntica a section2
    
    Retorna:
    --------
    section_interpolada : ISection_BS o ISection_MS
        Nueva sección con propiedades interpoladas
    
    Raises:
    -------
    TypeError
        Si las secciones no son del mismo tipo
    ValueError
        Si xi está fuera del rango [0, 1]
    
    """

    # ===== INTERPOLACIÓN PARA SECCIÓN MONO-SIMÉTRICA =====

        
        # Interpolar cada dimensión geométrica
    h = section1.h + (section2.h - section1.h) * xi
    bf1 = section1.bf1 + (section2.bf1 - section1.bf1) * xi
    bf2 = section1.bf2 + (section2.bf2 - section1.bf2) * xi
    tw = section1.tw + (section2.tw - section1.tw) * xi
    tf1 = section1.tf1 + (section2.tf1 - section1.tf1) * xi
    tf2 = section1.tf2 + (section2.tf2 - section1.tf2) * xi
    r1 = section1.r1 + (section2.r1 - section1.r1) * xi
    r2 = section1.r2 + (section2.r2 - section1.r2) * xi
        
    # Crear nueva sección con dimensiones interpoladas
    return ISection_MS(h=h, bf1=bf1, bf2=bf2, tw=tw, 
                        tf1=tf1, tf2=tf2, r1=r1, r2=r2)


def interpolate_multiple_sections(section1, section2, points):

    sections = []
    for xi in points:
        sections.append(interpolate_section(section1, section2, xi))

    return sections
