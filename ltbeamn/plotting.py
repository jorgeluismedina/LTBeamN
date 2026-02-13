
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




def plot_2dmodel(mod):

    plt.figure()
    ax = plt.gca()

    for elem in mod.elems:
        x = elem.coord[:,0]
        y = elem.coord[:,1]
        ax.plot(x, y, 'b-o', lw=1, markersize=2)

    for n, coor in enumerate(mod.coord):
        ax.text(coor[0]+0.05, coor[1]+0.05, str(n), 
                fontsize=7, ha='left', va='bottom')
        
    ax.axis('equal')

    return ax



def prepare_diagrams(elements, fields, esc1=0.5, esc2=0.7, esc3=0.5):
    all_x, all_N, all_V, all_M, all_u, all_w = fields
    max_N = np.max(np.abs(np.asarray(all_N)))
    max_V = np.max(np.abs(np.asarray(all_V)))
    max_M = np.max(np.abs(np.asarray(all_M)))
    max_u = np.max(np.abs(np.asarray(all_u)))
    max_w = np.max(np.abs(np.asarray(all_w)))
    max_def = max(max_u, max_w)

    N_globals = []
    V_globals = []
    M_globals = []
    def_shapes = []

    for e, elem in enumerate(elements):
        x = all_x[e]
        N = all_N[e] / max_N # axial normalizada
        V = all_V[e] / max_V # corte  normalizada
        M = all_M[e] / max_M # momento normalizada
        u = all_u[e] / max_def # def axial normalizada
        w = all_w[e] / max_def # def vertical normalizada

        c, s = elem.dirvec 
        # Coordenadas globales de puntos a lo largo del elemento
        X0, Y0 = elem.coord[0]
        X = X0 + c*x
        Y = Y0 + s*x

        # Desplazamientos globales (vectorizado)
        u_global = c*u - s*w   # Componente X global
        w_global = s*u + c*w    # Componente Y global

        # Diagramas rotados (perpendiculares al elemento)
        N_diag_X = X - s*N*esc1
        N_diag_Y = Y + c*N*esc1

        V_diag_X = X - s*V*esc1
        V_diag_Y = Y + c*V*esc1

        M_diag_X = X + s*M*esc2  # Escala más pequeña para momento
        M_diag_Y = Y - c*M*esc2  # Momentos ploteados alrevez por convencion

        X_def = X + u_global*esc3
        Y_def = Y + w_global*esc3

        N_globals.append(np.vstack([N_diag_X, N_diag_Y, all_N[e]]))
        V_globals.append(np.vstack([V_diag_X, V_diag_Y, all_V[e]]))
        M_globals.append(np.vstack([M_diag_X, M_diag_Y, all_M[e]]))
        def_shapes.append(np.vstack([X_def, Y_def]))

    return N_globals, V_globals, M_globals, def_shapes




def get_normal_vector(coordi, coordj):
    temp = coordj - coordi
    length = np.linalg.norm(temp)
    vector = temp / length
    return vector



def plot_2d_diagram(elements, diagrams):

    plt.figure()
    ax = plt.gca()

    for e, elem in enumerate(elements):
        # Geometria original
        x_elem = elem.coord[:,0]
        y_elem = elem.coord[:,1]
        dirvec = elem.dirvec
        diag_x, diag_y, vals = diagrams[e]

        # node i diagram info
        coordi = elem.coord[0]
        diag_i = np.array([diag_x[0], diag_y[0]])
        dvec_i = get_normal_vector(diag_i, coordi)
        diag_i_val = vals[0]

        # node j diagram info
        coordj = elem.coord[1]
        diag_j = np.array([diag_x[-1], diag_y[-1]])
        dvec_j = get_normal_vector(diag_j, coordj)
        diag_j_val = vals[-1]


        textang = -np.arccos(dirvec[0])/np.pi*180
        texti_xy = diag_i + ( 0.2*dirvec - 0.12*dvec_i)
        textj_xy = diag_j + (-0.2*dirvec - 0.12*dvec_j)

        ax.plot(x_elem, y_elem, color='gray', lw=0.8)
        ax.plot(diag_x, diag_y, 'b', lw=0.8)
        ax.plot([diag_i[0], coordi[0]],[diag_i[1], coordi[1]], 'b', lw=0.8)
        ax.plot([diag_j[0], coordj[0]],[diag_j[1], coordj[1]], 'b', lw=0.8)

        
        ax.text(texti_xy[0], texti_xy[1], f"{diag_i_val:.3e}", rotation = textang,
                color='red', fontsize=7, ha='center', va='center')
        
        ax.text(textj_xy[0], textj_xy[1], f"{diag_j_val:.3e}", rotation =textang,
                color='red', fontsize=7, ha='center', va='center')


    ax.axis('equal')

    return ax




def plot_2d_deformed_shape(elements, deformed_shapes):
    plt.figure()
    ax = plt.gca()

    for e, elem in enumerate(elements):
        # Geometria original
        x_elem = elem.coord[:,0]
        y_elem = elem.coord[:,1]
        def_x, def_y = deformed_shapes[e]

        ax.plot(x_elem, y_elem, color='gray', lw=0.8)
        ax.plot(def_x, def_y, 'b', lw=0.8)

    ax.axis('equal')

    return ax




def plot_1d_diagram(elements, xfields, diagrams):

    plt.figure()
    ax = plt.gca()
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

    return ax





def plot_buckling_modes(vals, vecs, mod):
    n_modes = min(5, len(vals))

    # Creacion de la figura
    fig, axes = plt.subplots(n_modes, 2, figsize=(10, 1.5*n_modes), sharex=True)
    if n_modes == 1: axes = [axes] # Ajuste si solo hay un modo

    coord = mod.coord
    ndofs = mod.ndofs2

    for i in range(n_modes):
        # Extraer el autovalor y el autovector
        val_crit = vals[i]
        vec_modo = vecs[:, i]

        # Reconstruir el vector completo (con apoyos)
        full_vec = np.zeros(ndofs)
        full_vec[mod.free_dof2] = vec_modo

        # Extraccion de desplazamiento lateral y rotacion 
        v_disp  = full_vec[0::4]
        the_rot = full_vec[2::4]

        # Normalizacion
        v_disp  = v_disp / np.max(np.abs(v_disp))
        the_rot = the_rot / np.max(np.abs(the_rot))

        # --- Gráfico de Desplazamiento Lateral (v) ---
        axes[i][0].plot(coord, v_disp, 'b-o', markersize=3)
        axes[i][0].set_ylabel(r'Mode {}, $\lambda$={:.1f}'.format(i+1, val_crit))
        axes[i][0].grid(True, alpha=0.3)
        if i == 0: axes[i][0].set_title('Lateral displacement (v)')

        # --- Gráfico de Torsión (theta) ---
        axes[i][1].plot(coord, the_rot, 'r-s', markersize=3)
        axes[i][1].grid(True, alpha=0.3)
        if i == 0: axes[i][1].set_title('Torsional twist ($\\theta$)')

    plt.tight_layout()

    return fig










