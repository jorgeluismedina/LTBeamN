
import numpy as np
import matplotlib.pyplot as plt



def _get_global_critical_points(diag_y):
    critical_indices = set()
    
    # Constante → centro
    if np.allclose(diag_y, diag_y[0], atol=1e-12):
        idx_center = len(diag_y) // 2
        critical_indices.add(idx_center)
    
    
    # Variable
    else:
        # Máximo y mínimo globales
        idx_max = np.argmax(diag_y)
        idx_min = np.argmin(diag_y)
        
        if np.abs(diag_y[idx_max]) > 1e-12:
            critical_indices.add(idx_max)

        if np.abs(diag_y[idx_min]) > 1e-12:
            critical_indices.add(idx_min)
    
    return sorted(list(critical_indices))



def plot_diagram(model, diagrams, title=""):
    """
    Plotea diagrama con etiquetas globales en puntos críticos.
    
    Args:
        model: Modelo con elementos
        diagrams: Salida de prepare_diagrams (diag_x, diag_y, vals)
        title: Título del gráfico
    """
    # ========== ANÁLISIS GLOBAL ==========
    # Extrae TODOS los valores para análisis global
    all_diag_x = []
    all_diag_y = []
    all_vals = []
    
    for (diag_x, diag_y, vals) in diagrams:
        all_diag_x.extend(diag_x)
        all_diag_y.extend(diag_y)
        all_vals.extend(vals)
    
    all_diag_x = np.array(all_diag_x)
    all_diag_y = np.array(all_diag_y)
    all_vals = np.array(all_vals)
    
    # Encuentra puntos críticos GLOBALES
    critical_indices = _get_global_critical_points(all_diag_y)
    
    # ========== PLOTEO ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for e, elem in enumerate(model.elems):
        x_elem = elem.coord
        y_elem = np.zeros_like(x_elem)
        diag_x, diag_y, vals = diagrams[e]
        
        # Plotea geometría y diagrama
        ax.plot(x_elem, y_elem, color='gray', lw=1, markersize=5, alpha=0.7)
        ax.plot(diag_x, diag_y, 'b', lw=1)
        ax.plot([diag_x[0], x_elem[0]], [diag_y[0], 0], 'b--', lw=0.5, alpha=0.5)
        ax.plot([diag_x[-1], x_elem[1]], [diag_y[-1], 0], 'b--', lw=0.5, alpha=0.5)
    
    # Plotea SOLO etiquetas en puntos críticos globales
    for idx in critical_indices:
        x_crit = all_diag_x[idx]
        y_crit = all_diag_y[idx]
        val_crit = all_vals[idx]
        
        offset = 0.1 if y_crit>=0 else -0.1
        ax.text(x_crit, y_crit + offset, f"{val_crit:.3e}",
                color='red', fontsize=8, ha='center', va='center')
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    
    return fig, ax







"""
def plot_diagram(model, diagrams, title=""):
    # Plotea N, V, M
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for e, elem in enumerate(model.elems):
        x_elem = elem.coord
        y_elem = np.zeros_like(x_elem)
        diag_x, diag_y, vals = diagrams[e]
        
        ax.plot(x_elem, y_elem, color='gray', lw=1)
        ax.plot(diag_x, diag_y, 'b', lw=1)
        ax.plot([diag_x[0], x_elem[0]], [diag_y[0], 0], 'b--', lw=0.5, alpha=0.5)
        ax.plot([diag_x[-1], x_elem[1]], [diag_y[-1], 0], 'b--', lw=0.5, alpha=0.5)
        
        ax.text(diag_x[0], diag_y[0]*1.01, f"{vals[0]:.3e}", 
                color='red', fontsize=8, ha='center', va='top')
        ax.text(diag_x[-1], diag_y[-1]*1.01, f"{vals[-1]:.3e}", 
                color='red', fontsize=8, ha='center', va='top')
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return fig, ax

"""



def plot_deformed(model, def_shapes, title="Deformed Shape"):
    """Plotea original + deformada"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for e, elem in enumerate(model.elems):
        x_elem = elem.coord
        y_elem = np.zeros_like(x_elem)
        X_def, Y_def = def_shapes[e]
        
        ax.plot(x_elem, y_elem, color='gray', lw=1, alpha=0.7)
        ax.plot(X_def, Y_def, color='b', lw=1)
    
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)
    return fig, ax








def plot_buckling_modes(model, mu_crs, modes, nmodes=2):

    # Creación de la figura
    fig, axes = plt.subplots(nmodes, 1, figsize=(12, 3*nmodes), sharex=True)
    if nmodes == 1: axes = [axes]

    coord = model.coord

    for i in range(nmodes):
        # Extraer el autovalor y el autovector
        mu_cr = mu_crs[i]
        mode = modes[:, i]

        # Extracción de desplazamientos
        v   = mode[0::4]
        dv  = mode[1::4]
        th  = mode[2::4]
        dth = mode[3::4]

        # Normalización
        v   = v / np.max(np.abs(v))
        dv  = dv / np.max(np.abs(dv))
        th  = th / np.max(np.abs(th))
        dth = dth / np.max(np.abs(dth))

        # Gráficos
        axes[i].plot(coord, v,   'r-', lw=1, label=r'$v$')
        axes[i].plot(coord, dv,  'b-', lw=1, label=r'$v_{,x}$')
        axes[i].plot(coord, th,  'k-', lw=1, label=r'$\theta$')
        axes[i].plot(coord, dth, 'g-', lw=1, label=r'$\theta_{,x}$')
        
        # Etiquetas y estilo
        axes[i].set_ylabel(rf'Mode {i+1}' + '\n' + rf'$\mu_{{cr}}={mu_cr:.1f}$', 
                          fontsize=9, rotation=0, labelpad=20, va='center')
        axes[i].grid(True, alpha=0.3, ls='--')
        axes[i].set_ylim(-1.05, 1.05)  # Sin espacio en Y
        axes[i].set_xlim(coord[0], coord[-1])
        axes[i].set_xlabel('x', fontsize=10)
        
        # Leyenda solo en el primer subplot
        if i == 0:
            axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), 
                           ncol=4, frameon=False, fontsize=12)
            
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    return fig, axes










