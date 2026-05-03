
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
 
 
def _get_flange_widths(sec):
    """Compatibilidad ISection_BS (bf) e ISection_MS (bf1, bf2)."""
    if hasattr(sec, 'bf1'):
        return sec.bf1, sec.bf2
    return sec.bf, sec.bf
 
 
def _section_outline(sec):
    """
    Puntos del perfil I en coordenadas locales (y, z) referenciadas al centro de corte.
    Retorna lista de segmentos: cada segmento es un array (n, 2).
    """
    bf1, bf2 = _get_flange_widths(sec)
    zG, h, zS = sec.zG, sec.h, sec.zS
 
    z_top = h - zG - zS   # fibra superior rel. al centro de corte
    z_bot =    -zG - zS   # fibra inferior rel. al centro de corte
 
    flange_top = np.array([[-bf1/2, z_top], [bf1/2, z_top]])
    flange_bot = np.array([[-bf2/2, z_bot], [bf2/2, z_bot]])
    web        = np.array([[0,      z_bot], [0,     z_top]])
 
    return [flange_top, flange_bot, web]
 
 
def _deform_segment(seg, vi, thi, zS):
    """
    Rotación thi alrededor del centro de corte + traslación lateral vi.
    Entrada: seg (n, 2) en (y_local, z_local) rel. al CS.
    Salida: (y_global, z_global).
    """
    c, s = np.cos(thi), np.sin(thi)
    y = seg[:, 0] * c - seg[:, 1] * s + vi
    z = seg[:, 0] * s + seg[:, 1] * c + zS
    return y, z
 
 
def plot_buckling_mode_3d(model, mu_crs, modes, imode=0, scale=1.0):
    """
    Plotea en 3D el modo de pandeo lateral-torsional, estilo LTBeamN.
 
    Parámetros
    ----------
    model  : StabilityModel
    mu_crs : array de factores de carga crítica
    modes  : (nltr_dofs, nmodes)  — columnas = autovectores
    imode  : índice del modo a plotear
    scale  : amplificación de los desplazamientos (visual)
    """
    mode = modes[:, imode]
 
    # DOFs: [v, v', theta, theta'] por nodo
    v  = mode[0::4]   # desplazamiento lateral
    th = mode[2::4]   # ángulo de torsión
 
    # Normalizar por el pico de v
    v_peak = np.max(np.abs(v)) or 1.0
    v  = v  / v_peak * scale
    th = th / v_peak * scale
 
    x = model.coord   # posiciones nodales (1D)
 
    fig = plt.figure(figsize=(14, 6))
    ax  = fig.add_subplot(111, projection='3d')
 
    # ── Rebanadas transversales ────────────────────────────────────────────
    for i, xi in enumerate(x):
        sec = model.sections[i]
        for seg in _section_outline(sec):
            y_def, z_def = _deform_segment(seg, v[i], th[i], sec.zS)
            ax.plot([xi, xi], y_def, z_def, color='blue', lw=1, alpha=0.6)
 
    # ── Líneas longitudinales por los bordes de las alas ──────────────────
    # Cuatro esquinas: top-left, top-right, bottom-left, bottom-right
    tips = {'tl': [], 'tr': [], 'bl': [], 'br': []}
 
    for i, xi in enumerate(x):
        sec = model.sections[i]
        bf1, bf2 = _get_flange_widths(sec)
        zG, h, zS = sec.zG, sec.h, sec.zS
        z_top = h - zG - zS
        z_bot =    -zG - zS
 
        corners = {
            'tl': np.array([[-bf1/2, z_top]]),
            'tr': np.array([[ bf1/2, z_top]]),
            'bl': np.array([[-bf2/2, z_bot]]),
            'br': np.array([[ bf2/2, z_bot]]),
        }
        for key, pt in corners.items():
            y_def, z_def = _deform_segment(pt, v[i], th[i], zS)
            tips[key].append([xi, y_def[0], z_def[0]])
 
    for key, pts in tips.items():
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='blue', lw=1)
 
    # ── Formato ────────────────────────────────────────────────────────────
    ax.set_xlabel('x', labelpad=6)
    ax.set_ylabel('y  (lateral)', labelpad=6)
    #ax.set_zlabel('z  (vertical)', labelpad=6)
    ax.set_title(
        rf'Mode {imode + 1}  —  $\mu_{{cr}} = {mu_crs[imode]:.3f}$',
        fontsize=11, pad=10
    )
    #ax.set_box_aspect([4, 1, 1])   # relación de aspecto razonable para vigas
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)
 
    return fig, ax