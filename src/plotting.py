
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import cast
 
from src.shape_funcs import N_hermite
from src.sections.section_utils import interpolate_section
 
 
# ── Estilo global ──────────────────────────────────────────────────────────
 
_BEAM_COLOR     = '#888888'   # eje / viga no deformada 2D
_DIAGRAM_COLOR  = '#0000ff'   # diagramas de esfuerzos
_DEFORMED_COLOR = '#0000ff'   # deformada estática
_CRIT_COLOR     = '#f00000'   # etiquetas de valores críticos
_MODE_COLORS    = ['#f00000', '#0000ff', "#01B701", '#000000']  # v, v', θ, θ'
_BEAM3D_COLOR   = '#0000ff'   # secciones deformadas 3D
_UNDEF_COLOR    = '#888888'   # viga no deformada 3D
 
_FIG_W, _FIG_H = 13, 5
 
 
# ── Helpers internos ───────────────────────────────────────────────────────
 
def critical_indices(values):
    """Índices del máximo y mínimo global (ignora valores ~0)."""
    if np.allclose(values, values[0], atol=1e-12):
        return [len(values) // 2]
    indices = set()
    for fn in (np.argmax, np.argmin):
        idx = fn(values)
        if np.abs(values[idx]) > 1e-12:
            indices.add(idx)
    return sorted(indices)
 
 
def label_offset(y_range, y_val):
    """Offset vertical proporcional al rango para etiquetas críticas."""
    offset = y_range * 0.04
    return offset if y_val >= 0 else -offset
 
 
def get_flange_widths(sec):
    if hasattr(sec, 'bf1'):
        return sec.bf1, sec.bf2
    return sec.bf, sec.bf
 
 
def section_outline(sec):
    """Segmentos del perfil I en coords locales (y, z) rel. al centro de corte."""
    bf1, bf2 = get_flange_widths(sec)
    zG, h, zS = sec.zG, sec.h, sec.zS
    z_top = h - zG - zS
    z_bot =    -zG - zS
    return [
        np.array([[-bf1/2, z_top], [bf1/2, z_top]]),  # ala superior
        np.array([[-bf2/2, z_bot], [bf2/2, z_bot]]),  # ala inferior
        np.array([[0,      z_bot], [0,     z_top]]),   # alma
    ]
 
 
def deform_segment(seg, v, theta, zS):
    """Rotación theta alrededor del CS + traslación lateral v."""
    c, s = np.cos(theta), np.sin(theta)
    y = seg[:, 0] * c - seg[:, 1] * s + v
    z = seg[:, 0] * s + seg[:, 1] * c + zS
    return y, z
 
 
def section_at(elem, xi):
    """Sección interpolada en xi ∈ [0,1] (uniforme o tapered)."""
    if hasattr(elem, 'section_i'):
        return interpolate_section(elem.section_i, elem.section_j, xi)
    return elem.section
 
 
def interp_mode(elem_ltr_dof, mode, L, xis):
    """
    Interpola v(xi) y theta(xi) con funciones de forma de Hermite.
    Las rotaciones se escalan por L antes de aplicar N_hermite.
    """
    d    = mode[elem_ltr_dof]
    d_v  = np.array([d[0], d[1]*L, d[4], d[5]*L])
    d_th = np.array([d[2], d[3]*L, d[6], d[7]*L])
    N    = N_hermite(xis)   # (4, n_pts)
    return N.T @ d_v, N.T @ d_th
 
 
# ── Funciones de ploteo ────────────────────────────────────────────────────
 
def plot_diagram(model, diagrams, title=""):
    """
    Diagrama de esfuerzos (N, V o M) con etiquetas en puntos críticos.
 
    Parámetros
    ----------
    model    : StabilityModel
    diagrams : salida de StaticSolver.prepare_diagrams — lista de (x, y_norm, vals)
    title    : título del gráfico
    """
    all_x    = np.concatenate([d[0] for d in diagrams])
    all_y    = np.concatenate([d[1] for d in diagrams])
    all_vals = np.concatenate([d[2] for d in diagrams])
    y_range  = all_y.max() - all_y.min() or 1.0
 
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
 
    for e, elem in enumerate(model.elements):
        diag_x, diag_y, _ = diagrams[e]
        ax.plot(elem.coords, np.zeros(2), color=_BEAM_COLOR, lw=1, alpha=0.8)
        ax.plot(diag_x, diag_y, color=_DIAGRAM_COLOR, lw=1)

        ax.plot([diag_x[0],  elem.coords[0]], [diag_y[0],  0], '--',
                color=_DIAGRAM_COLOR, lw=0.5, alpha=0.5)
        
        ax.plot([diag_x[-1], elem.coords[1]], [diag_y[-1], 0], '--',
                color=_DIAGRAM_COLOR, lw=0.5, alpha=0.5)
 
    for idx in critical_indices(all_y):
        offset = label_offset(y_range, all_y[idx])
        ax.text(all_x[idx], all_y[idx] + offset, f'{all_vals[idx]:.3e}',
                color=_CRIT_COLOR, fontsize=8, ha='center', va='center')
 
    #ax.axhline(0, color=_BEAM_COLOR, lw=0.5, alpha=0.5)
    #ax.set_xlabel('x', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig, ax
 

def plot_deformed(model, def_shapes, title="Deformed shape"):
    """
    Viga original (gris) y deformada estática (azul).
 
    Parámetros
    ----------
    model      : StabilityModel
    def_shapes : salida de StaticSolver.prepare_diagrams — lista de (X_def, Y_def)
    title      : título del gráfico
    """
    all_x    = np.concatenate([d[0] for d in def_shapes])
    all_y    = np.concatenate([d[1] for d in def_shapes])
    #all_vals = np.concatenate([d[2] for d in def_shapes])
    y_range  = all_y.max() - all_y.min() or 1.0

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
 
    for e, elem in enumerate(model.elements):
        X_def, Y_def = def_shapes[e]
        ax.plot(elem.coords, np.zeros(2), color=_BEAM_COLOR,    lw=1, alpha=0.8)
        ax.plot(X_def, Y_def,             color=_DEFORMED_COLOR, lw=1)

    for idx in critical_indices(all_y):
        offset = label_offset(y_range, all_y[idx])
        ax.text(all_x[idx], all_y[idx] + offset, f'{all_y[idx]:.3e}',
                color=_CRIT_COLOR, fontsize=8, ha='center', va='center')
 
    #ax.set_xlabel('x', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig, ax



def plot_buckling_modes(model, mu_crs, modes, nmodes=2):
    """
    Diagramas 2D de los modos de pandeo lateral-torsional (v, v', θ, θ').
    Cada componente se normaliza por su propio pico.
 
    Parámetros
    ----------
    model  : StabilityModel
    mu_crs : array de factores de carga crítica
    modes  : (nltr_dofs, nmodes)
    nmodes : número de modos a plotear
    """
    labels = [r'$v$', r"$v'$", r'$\theta$', r"$\theta'$"]
    x      = model.coords
 
    fig, axes = plt.subplots(nmodes, 1, figsize=(_FIG_W, 2.8*nmodes), sharex=True)
    if nmodes == 1:
        axes = [axes]
 
    for i, ax in enumerate(axes):
        mode = modes[:, i]
        dofs = [mode[k::4] for k in range(4)]
        dofs = [d / (np.max(np.abs(d)) or 1.0) for d in dofs]
 
        for d, color, label in zip(dofs, _MODE_COLORS, labels):
            ax.plot(x, d, color=color, lw=1, label=label)
 
        ax.axhline(0, color=_BEAM_COLOR, lw=0.6, alpha=0.5)
        ax.set_ylabel(
            f'Mode {i+1}\n$\\mu_{{cr}}={mu_crs[i]:.2f}$',
            fontsize=9, rotation=0, labelpad=48, va='center'
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(x[0], x[-1])
        ax.grid(True, alpha=0.3, lw=0.5, ls='--')
 
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=4, frameon=False, fontsize=11)
 
    #axes[-1].set_xlabel('x', fontsize=10)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    return fig, axes 





def draw_axis_arrows(ax, x0, y0, z0, length):
    """Trípode de ejes al estilo CAD en la posición (x0, y0, z0)."""
    axes_def = [
        ([1, 0, 0], '#e74c3c', 'X'),
        ([0, 1, 0], '#27ae60', 'Y'),
        ([0, 0, 1], '#2980b9', 'Z'),
    ]
    for direction, color, label in axes_def:
        dx, dy, dz = [d * length for d in direction]

        ax.quiver(x0, y0, z0, dx, dy, dz,
                  color=color, linewidth=1.2,
                  arrow_length_ratio=0.2, normalize=False)
        
        ax.text(x0 + dx*1.15, y0 + dy*1.15, z0 + dz*1.15,
                label, color=color, fontsize=9*length*7, fontweight='bold',
                ha='center', va='center')




def plot_buckling_mode_3d(model, mu_crs, modes, imode=0, scale=1.0, n_sec=5):
    mode   = modes[:, imode]
    v_peak = np.max(np.abs(mode[0::4])) or 1.0
    mode_n = mode / v_peak * scale
 
    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
 
    ax = cast(Axes3D, fig.add_subplot(111, projection='3d'))
 
    # Sin paredes, sin ejes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): # type: ignore[union-attr]
        pane.fill = False
        pane.set_edgecolor('none')
    ax.set_axis_off()
 
    for elem in model.elements:
        L   = elem.length
        x0  = elem.coords[0]
        xis = np.linspace(0, 1, n_sec)
 
        v_arr, th_arr = interp_mode(elem.ltr_dofs, mode_n, L, xis)
 
        tips_undef = {k: [] for k in ('tl', 'tr', 'bl', 'br')}
        tips_def   = {k: [] for k in ('tl', 'tr', 'bl', 'br')}
 
        for k, xi in enumerate(xis):
            sec = section_at(elem, xi)
            bf1, bf2 = get_flange_widths(sec)
            zG, h, zS = sec.zG, sec.h, sec.zS
            z_top, z_bot = h - zG - zS, -zG - zS
            x_k = x0 + xi * L
            z_align = sec.z_from_ref(elem.align, 0)
 
            for seg in section_outline(sec):
                y0, z0 = seg[:, 0], seg[:, 1] + zS + z_align
                ax.plot([x_k, x_k], y0, z0,
                        color=_UNDEF_COLOR, lw=0.5, alpha=1)
                y_def, z_def = deform_segment(seg, v_arr[k], th_arr[k], zS)
                ax.plot([x_k, x_k], y_def, z_def + z_align,
                        color=_BEAM3D_COLOR, lw=0.5, alpha=1)
 
            corners = [('tl', [-bf1/2, z_top]),
                       ('tr', [ bf1/2, z_top]),
                       ('bl', [-bf2/2, z_bot]),
                       ('br', [ bf2/2, z_bot])]
 
            for key, pt in corners:
                pt_arr = np.array([pt])
                tips_undef[key].append([x_k, pt_arr[0, 0], pt_arr[0, 1] + zS + z_align])
                y_def, z_def = deform_segment(pt_arr, v_arr[k], th_arr[k], zS)
                tips_def[key].append([x_k, y_def[0], z_def[0] + z_align])
 
        for pts in tips_undef.values():
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=_UNDEF_COLOR, lw=0.5, alpha=1)
 
        for pts in tips_def.values():
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=_BEAM3D_COLOR, lw=0.5)
 
    # Trípode de ejes
    arrow_len = (model.coords[-1] - model.coords[0]) * 0.02
    ax.set_aspect('equal')  # type: ignore[arg-type]
    draw_axis_arrows(ax, 0.0, 0.0, 0.0, arrow_len)
    
    #ax.margins(x=0.1, y=0.3, z=0.3)
    ax.set_title(
        rf'Mode {imode+1}  —  $\mu_{{cr}} = {mu_crs[imode]:.3f}$',
        fontsize=11, pad=2, y=0.97
    )
    return fig, ax








