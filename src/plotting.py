
import numpy as np
import matplotlib.pyplot as plt



def plot_matrix(Mat, part=1):
    figure = plt.figure()
    plt.spy(Mat, origin='upper', extent=(0, Mat.shape[1], Mat.shape[0], 0))
    num_rows, num_cols = Mat.shape
    plt.xticks(np.arange(0, num_cols + 1, part))  # Ticks enteros para las columnas
    plt.yticks(np.arange(0, num_rows + 1, part))  # Ticks enteros para las filas
    plt.grid(True)
    return figure

def plot_imshow(Mat, n, part=1):
    figure = plt.figure(n)
    plt.imshow(Mat, cmap='viridis', origin='upper', 
           interpolation='none',  # Evitar interpolación
           extent=(0, Mat.shape[1], Mat.shape[0], 0))
    plt.colorbar()
    num_rows, num_cols = Mat.shape
    plt.xticks(np.arange(0, num_cols + 1, part))  # Ticks enteros para las columnas
    plt.yticks(np.arange(0, num_rows + 1, part))  # Ticks enteros para las filas
    plt.grid(True)
    return figure




def plot_1d_diagram(elements, xfields, diagrams):

    fig, ax = plt.subplots()
    elev = np.array([0.0, 0.0])

    for e, elem in enumerate(elements):
        x = xfields[e]
        vals = diagrams[e]

        # node i diagram info
        xi = elem.coord[0][0]
        val_i = vals[0]

        # node j diagram info
        xj = elem.coord[1][0]
        val_j = vals[-1]

        ax.plot(elem.coord, elev, color='gray', lw=0.8)
        ax.plot(x + xi, vals, 'b', lw=0.8)
        ax.plot([xi, xi], [0, val_i], 'b', lw=0.8)
        ax.plot([xj, xj], [0, val_j], 'b', lw=0.8)

        ax.text(xi, val_i*1.05, f"{val_i:.3e}",
                color='red', fontsize=7, ha='center', va='center')
        
        ax.text(xj, val_j*1.05, f"{val_j:.3e}",
                color='red', fontsize=7, ha='center', va='center')

    ax.axis('equal')

    return fig, ax





def plot_buckling_modes(vals, modes, model):
    n_modes = min(5, len(vals))

    # Creacion de la figura
    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 1.5*n_modes), sharex=True)
    if n_modes == 1: axes = [axes] # Ajuste si solo hay un modo

    nmodes = modes.shape[-1]
    coord = model.coord

    for i in range(nmodes):
        # Extraer el autovalor y el autovector
        val_crit = vals[i]
        fullmode = modes[:, i]

        # Extraccion de desplazamiento lateral y rotacion 
        v_disp  = fullmode[0::4]
        the_rot = fullmode[2::4]

        # Normalizacion
        v_disp  = v_disp / np.max(np.abs(v_disp))
        the_rot = the_rot / np.max(np.abs(the_rot))

        # --- Gráfico de Desplazamiento Lateral (v) y Torsión (theta) ---
        axes[i].plot(coord, v_disp, 'b-o', markersize=3)
        axes[i].plot(coord, the_rot, 'r-s', markersize=3)
        axes[i].set_ylabel(r'Mode {}, $\lambda$={:.1f}'.format(i+1, val_crit))
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].set_title('Lateral displacement (v) \n, Torsional twist ($\\theta$)')

    plt.tight_layout()

    return fig, axes










