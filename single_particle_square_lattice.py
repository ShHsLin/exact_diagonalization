import scipy.linalg
import numpy as np

'''
lattice
'''

def generate_lattice_idx(Lx, Ly):
    lattice_idx = np.zeros([Lx, Ly], dtype=np.int)
    idx = 0
    for i in range(Lx):
        for j in range(Ly):
            lattice_idx[i, j] = idx
            idx += 1

    return lattice_idx

def lattice_to_idx(lattice, lattice_idx):
    return np.einsum('ij,ij->', lattice, lattice_idx)

def idx_to_lattice(idx, Lx, Ly):
    lattice = np.zeros([Lx, Ly], dtype=np.int)
    tmp = 0
    for i in range(Lx):
        for j in range(Ly):
            if tmp == idx:
                lattice[i, j] = 1
                return lattice, i, j
            else:
                pass
            tmp += 1

    raise

def near_neigh(x, y, Lx, Ly, PBC=False):
    if PBC:
        nn_list = []
        new_x = (x + Lx + 0) % Lx
        new_y = (y + Ly + 1) % Ly
        nn_list.append((new_x, new_y))
        new_x = (x + Lx + 1) % Lx
        new_y = (y + Ly + 0) % Ly
        nn_list.append((new_x, new_y))
        new_x = (x + Lx + 0) % Lx
        new_y = (y + Ly - 1) % Ly
        nn_list.append((new_x, new_y))
        new_x = (x + Lx - 1) % Lx
        new_y = (y + Ly + 0) % Ly
        nn_list.append((new_x, new_y))
        return nn_list
    else:
        nn_list = []
        new_x = (x + Lx + 0) % Lx
        new_y = (y + Ly + 1) % Ly
        if new_y > y:
            nn_list.append((new_x, new_y))

        new_x = (x + Lx + 1) % Lx
        new_y = (y + Ly + 0) % Ly
        if new_x > x:
            nn_list.append((new_x, new_y))

        new_x = (x + Lx + 0) % Lx
        new_y = (y + Ly - 1) % Ly
        if new_y < y:
            nn_list.append((new_x, new_y))

        new_x = (x + Lx - 1) % Lx
        new_y = (y + Ly + 0) % Ly
        if new_x < x:
            nn_list.append((new_x, new_y))

        return nn_list

def get_Hamiltonian(lattice_idx, t=1, PBC=False):
    Lx, Ly = lattice_idx.shape
    N = Lx * Ly
    Hamiltonian = np.zeros([N, N])
    for idx in range(N):
        lattice, x, y = idx_to_lattice(idx, Lx, Ly)
        nn_list = near_neigh(x, y, Lx, Ly, PBC=PBC)
        for nn in nn_list:
            lattice_copy = lattice.copy()
            lattice_copy[x, y] = 0
            lattice_copy[nn[0], nn[1]] = 1
            new_idx = lattice_to_idx(lattice_copy, lattice_idx)
            Hamiltonian[new_idx, idx] = -t

    return Hamiltonian


def occupation_number(psi):
    # return np.einsum('i,i->i', psi, psi.conjugate())
    return psi

def double_ft(A, w, dt=0.05, decay=0.1):
    '''
    Goal perform double fourier transform on the time-dependent connected correlation function.

    Input: A
    '''
    num_steps, sys_size = A.shape
    L = int(sys_size ** 0.5)
    assert L**2 == sys_size
    # eta = - np.log(decay) / (num_steps * dt)
    # exp_jwt = np.exp( 1j * (w + 1j * eta) * np.arange(num_steps) * dt )
    eta = - np.log(decay) / ((num_steps * dt)**2)
    exp_jwt = np.exp( 1j * w * np.arange(num_steps) * dt ) * np.exp(
        - eta * ((np.arange(num_steps) * dt)**2))
    S_jw = exp_jwt.dot(A) * 2 * np.pi / num_steps
    S_jw = S_jw.reshape((L, L))

    S_qw = np.zeros((L, L), dtype=np.complex)
    for idx_x in range(L):
        qx = 2*np.pi * idx_x / L
        for idx_y in range(L):
            qy = 2*np.pi * idx_y / L
            exp_jqx = np.exp( -1j * qx * ( np.arange(L) - L//2))
            exp_jqy = np.exp( -1j * qy * ( np.arange(L) - L//2))
            S_qw[idx_x, idx_y] = exp_jqx.dot(S_jw).dot(exp_jqy)


    S_qw = (1. / sys_size) * S_qw
    return S_qw

def get_S_qw(A, omega=(-5,10), dw=0.1, dt=0.05):
    S_array = []
    w_array = []
    num_omega = int((omega[1] - omega[0]) // dw) + 1
    for idx_w in range(num_omega):
        w = omega[0] + dw * idx_w
        S_array.append(double_ft(A, w, dt))
        w_array.append(w)


    return np.array(S_array), np.array(w_array)





'''
lattice_idx = idx = state_idx
'''
if __name__ == '__main__':
    Lx = Ly = 31
    N = Lx * Ly
    lattice_idx = generate_lattice_idx(Lx, Ly)
    H = get_Hamiltonian(lattice_idx, t=1.)
    # print(H)
    # import matplotlib.pyplot as plt
    # plt.imshow(H)
    # plt.show()
    # Random State 
    psi = np.zeros(N)
    psi[N//2] = 1
    # Ground State

    # psi[Ly*(Lx//2)+Ly//2] = 1

    dt = 0.05
    T = 5.
    steps = int(T / dt)
    occup_array = np.zeros([steps, N], dtype=np.complex128)
    import scipy.sparse
    import scipy.sparse.linalg
    H = scipy.sparse.csc_matrix(H)
    U = scipy.sparse.linalg.expm(-1j * dt * H)
    for i in range(steps):
        occup_array[i, :] = occupation_number(psi)
        psi = U.dot(psi)

    # np.save('single_particle.npy', occup_array[:, N//2])
    import matplotlib.pyplot as plt
    plt.imshow(occup_array.real)
    plt.colorbar()
    plt.title("ED")
    plt.xlabel(r"Site")
    plt.ylabel(r"Steps ($dt=%f$)" % dt)
    plt.show()


    L = Lx
    S_array, w_array = get_S_qw(occup_array, omega=(-5,10), dw=0.05, dt=dt)

    S = S_array
    final_S = np.concatenate([S[:, :L//2, 0], S[:, L//2, :L//2]] + [S[:, idx, idx:idx+1] for idx in range(L//2, -1, -1)], axis=1)
    final_S_ = final_S.copy().real
    # final_S_[final_S_<0] = 0

    ###############################
    ### Plotting exact solution ###
    ###############################
    x1 = np.arange(L//2)
    x2 = np.arange(L//2)
    x3 = np.arange(L//2, -1, -1)
    scale_x1 = 2 * np.pi / L * x1
    scale_x2 = 2 * np.pi / L * x2
    scale_x3 = 2 * np.pi / L * x3
    x2 = x2 + L//2
    x3 = 2*(L//2) + np.arange(L//2+1)

    scale_y1 = -2 * np.cos(scale_x1) -2
    scale_y2 = -2 * np.cos(scale_x2) + 2
    scale_y3 = -4 * np.cos(scale_x3)

    y1 = (-w_array[0] + scale_y1) / (w_array[1] - w_array[0])
    y2 = (-w_array[0] + scale_y2) / (w_array[1] - w_array[0])
    y3 = (-w_array[0] + scale_y3) / (w_array[1] - w_array[0])
    # plt.gca().plot(np.concatenate([x1,x2,x3]), np.concatenate([y1,y2,y3]), 'r')
    plt.gca().plot(x1, y1, 'r')
    plt.gca().plot(x2, y2, 'b')
    plt.gca().plot(x3, y3, 'k')
    ################################


    plt.imshow(final_S_, origin='lower', aspect=0.25); plt.colorbar()
    plt.xticks([0, L//2, 2 * (L//2), 3 * (L//2)] , [r'$(0,0)$',r'$(0,\pi)$',r'$(\pi,\pi)$',r'$(0,0)$',])
    plt.yticks([0, 50, 100, 120, 160, 200] , [r'$%.1f$'% w_array[0],
                                              r'$%.1f$'% w_array[50],
                                              r'$%.1f$'% w_array[100],
                                              r'$%.1f$'% w_array[120],
                                              r'$%.1f$'% w_array[160],
                                              r'$%.1f$'% w_array[200]])
    plt.title(u'$S^{zz}(k, \omega)$')
    fig = plt.gcf()
    fig.set_size_inches(4, 9)
    plt.show()

    # final_S_ = S_array.copy().real.reshape([301,-1])
    # final_S_[final_S_<0] = 0
    # plt.imshow(final_S_, origin='lower')
    # plt.show()


