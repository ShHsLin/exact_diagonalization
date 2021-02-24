import scipy.linalg
import numpy as np

'''
We represent the honeycomb lattice by the square lattice.
At each site on square lattice, we have one unit cell, which
contain two physical sites.
'''

def generate_lattice_idx(Lx, Ly):
    '''
    Since one unit cell have two sites, each site corresponds
    to two indices.
    '''
    lattice_idx = np.zeros([Lx, Ly, 2], dtype=np.int)
    idx = 0
    for i in range(Lx):
        for j in range(Ly):
            lattice_idx[i, j, 0] = idx
            lattice_idx[i, j, 1] = idx + 1
            idx += 2

    return lattice_idx

def lattice_to_idx(lattice, lattice_idx):
    return np.einsum('ijk,ijk->', lattice, lattice_idx)

def idx_to_lattice(idx, Lx, Ly):
    lattice = np.zeros([Lx, Ly, 2], dtype=np.int)
    tmp = 0
    for i in range(Lx):
        for j in range(Ly):
            if tmp == idx:
                lattice[i, j, 0] = 1
                return lattice, i, j, 0
            elif tmp + 1 == idx:
                lattice[i, j, 1] = 1
                return lattice, i, j, 1
            else:
                pass
            tmp += 2

    raise

def near_neigh(x, y, z, Lx, Ly, PBC=False):
    if PBC:
        nn_list = []
        if z == 0:
            new_x = x
            new_y = y
            new_z = 1
            nn_list.append((new_x, new_y, new_z))
            new_x = (x + Lx - 1) % Lx
            new_y = y
            new_z = 1
            nn_list.append((new_x, new_y, new_z))
            new_x = x
            new_y = (y + Ly - 1) % Ly
            new_z = 1
            nn_list.append((new_x, new_y, new_z))
        elif z == 1:
            new_x = x
            new_y = y
            new_z = 0
            nn_list.append((new_x, new_y, new_z))
            new_x = (x + Lx + 1) % Lx
            new_y = y
            new_z = 0
            nn_list.append((new_x, new_y, new_z))
            new_x = x
            new_y = (y + Ly + 1) % Ly
            new_z = 0
            nn_list.append((new_x, new_y, new_z))
        else:
            raise
    else:
        nn_list = []
        if z == 0:
            new_x = x
            new_y = y
            new_z = 1
            nn_list.append((new_x, new_y, new_z))
            new_x = (x + Lx - 1) % Lx
            new_y = y
            new_z = 1
            if new_x < x:
                nn_list.append((new_x, new_y, new_z))

            new_x = x
            new_y = (y + Ly - 1) % Ly
            new_z = 1
            if new_y < y:
                nn_list.append((new_x, new_y, new_z))

        elif z == 1:
            new_x = x
            new_y = y
            new_z = 0
            nn_list.append((new_x, new_y, new_z))
            new_x = (x + Lx + 1) % Lx
            new_y = y
            new_z = 0
            if new_x > x:
                nn_list.append((new_x, new_y, new_z))

            new_x = x
            new_y = (y + Ly + 1) % Ly
            new_z = 0
            if new_y > y:
                nn_list.append((new_x, new_y, new_z))

        else:
            raise

    return nn_list


def get_Hamiltonian(lattice_idx, t=1, PBC=False):
    Lx, Ly, _ = lattice_idx.shape
    N = Lx * Ly * 2
    Hamiltonian = np.zeros([N, N])
    for idx in range(N):
        lattice, x, y, z = idx_to_lattice(idx, Lx, Ly)
        nn_list = near_neigh(x, y, z, Lx, Ly, PBC=PBC)
        for nn in nn_list:
            lattice_copy = lattice.copy()
            lattice_copy[x, y, :] = 0
            lattice_copy[nn[0], nn[1], nn[2]] = 1
            new_idx = lattice_to_idx(lattice_copy, lattice_idx)
            Hamiltonian[new_idx, idx] = -t

    return Hamiltonian


def occupation_number(psi):
    # return np.einsum('i,i->i', psi, psi.conjugate())
    return psi

def double_ft(A, w, dt=0.05, sub_idx=0):
    '''
    Goal perform double fourier transform on the time-dependent connected correlation function.

    Input:
        A is the occupation matrix (num_steps, sys_size).
    '''
    num_steps, sys_size = A.shape
    L = int((sys_size/2) ** 0.5)
    assert 2 * L**2 == sys_size
    # eta = - np.log(0.5) / (num_steps * dt)
    # exp_jwt = np.exp( 1j * (w + 1j * eta) * np.arange(num_steps) * dt )
    eta = - np.log(0.1) / ((num_steps * dt)**2)
    exp_jwt = np.exp( 1j * w * np.arange(num_steps) * dt ) * np.exp(
        - eta * ((np.arange(num_steps) * dt)**2))
    S_jw = exp_jwt.dot(A) * 2 * np.pi / num_steps
    # S_jw = S_jw.real
    # S_jw = exp_jwt.dot(A.real) * 2 * np.pi / num_steps
    S_jw = S_jw.reshape((L, L, 2))

    S_qw = np.zeros((L, L), dtype=np.complex)
    for idx_x in range(L):
        qx = 2*np.pi * idx_x / L
        for idx_y in range(L):
            qy = 2*np.pi * idx_y / L
            ## R2 is now the x direction
            ## R1 is now the y direction
            exp_jqx = np.exp( -1j * (3/2.*qx + np.sqrt(3)/2.*qy) * ( np.arange(L) - L//2))
            exp_jqy = np.exp( -1j * (3/2.*qx - np.sqrt(3)/2.*qy) * ( np.arange(L) - L//2))
            if sub_idx == 0:
                exp_jqz = np.exp( -1j * qx * np.array([0., 1.]))
            elif sub_idx == 1:
                exp_jqz = np.exp( -1j * qx * np.array([-1, 0.]))

            # S_qw[idx_x, idx_y] = np.einsum('ijk,i,j,k->', S_jw, exp_jqx, exp_jqy, exp_jqz)
            S_qw[idx_x, idx_y] = np.einsum('ij,i,j->', np.real(np.einsum('ijk,k->ij', S_jw, exp_jqz)),
                                           exp_jqx, exp_jqy)


    S_qw = (1. / 2.) * S_qw
    return S_qw

def get_S_qw(A, num_omega=301, dt=0.05, sub_idx=0):
    S_array = []
    w_array = []
    for idx_w in range(num_omega):
        w = -5 + dt * idx_w
        S_array.append(double_ft(A, w, dt, sub_idx=sub_idx))
        w_array.append(w)


    return np.array(S_array), np.array(w_array)





'''
lattice_idx = idx = state_idx
'''
if __name__ == '__main__':
    Lx = Ly = 5
    N = Lx * Ly * 2
    lattice_idx = generate_lattice_idx(Lx, Ly)
    H = get_Hamiltonian(lattice_idx, t=1., PBC=False)
    # print(H)
    # import matplotlib.pyplot as plt
    # plt.imshow(H)
    # plt.show()
    # Random State
    occup_array_list = []
    S_list = []
    for sublattice_site in [0, 1]:
        psi = np.zeros(N)
        psi[N//2 - 1 +sublattice_site] = 1
        # Ground State

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
        occup_array_list.append(occup_array)

        L = Lx
        S_array, w_array = get_S_qw(occup_array, num_omega=301, dt=dt,
                                    sub_idx=sublattice_site)
        S_list.append(S_array)
        try:
            PEPS_A = np.load('/space/ga63zuh/mosesmove_11_11/DSF/free_fermion/S_xt_L%d_D2_Dc4_B.npy' % Lx)
            # PEPS_A = np.load('/space/ga63zuh/mosesmove_11_11/DMRG2/TE_MZ_XY_hc/S_xt_L5_D2_Dc4_XY.npy')
            # PEPS_A[1::2] = PEPS_A[1::2,::-1,::-1,:]
            PEPS_A = PEPS_A.reshape([100,50])
            # PEPS_B = np.load('/space/ga63zuh/mosesmove_11_11/A_B_L%d.npy' % Lx)
            # plt.plot(np.arange(100)*np.sqrt(2),PEPS_A[:,25].real,'r-')
            plt.plot(np.arange(PEPS_A.shape[0]),PEPS_A[:,Lx**2 - 1].real,'r-')
            plt.plot(np.arange(PEPS_A.shape[0]),occup_array_list[0][:,Lx**2 -1].real,'b-')
            plt.show()
            plt.imshow((occup_array_list[0]-PEPS_A).real)
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(e)
            pass


    import pdb;pdb.set_trace()

    S = S_array = S_list[1] + S_list[1]
    plt.imshow(S_array[301//2,:,:].real, origin='lower'); plt.colorbar()
    plt.show()

    # ###############################
    # ### Plotting exact solution ###
    # ###############################
    # x1 = np.arange(L//3)
    # x2 = np.arange(L//3)
    # x3 = np.arange(L//3, -1, -1)
    # scale_x1 = 2 * np.pi / L * x1
    # scale_x2 = 2 * np.pi / L * x2
    # scale_x3 = 2 * np.pi / L * x3
    # x2 = x2 + L//3
    # x3 = 2*(L//3) + np.arange(L//3+1)

    # scale_y1 = np.sqrt(1 + 4 * np.cos(np.sqrt(3)/2.*scale_x1) +
    #                    4 * np.cos(np.sqrt(3)/2.*scale_x1)**2)
    # scale_y2 = np.sqrt(1 - 4 * np.cos(np.sqrt(3)/2.*scale_x2) +
    #                    4 * (np.cos(np.sqrt(3)/2.*scale_x2)**2))
    # scale_y3 = np.sqrt(1 + 4 * np.cos(np.sqrt(3)/2.*scale_x3) * np.cos(3/2.*scale_x3) +
    #                    4 * np.cos(np.sqrt(3)/2.*scale_x3)**2)

    # y1 = (-w_array[0] + scale_y1) / (w_array[1] - w_array[0])
    # y2 = (-w_array[0] + scale_y2) / (w_array[1] - w_array[0])
    # y3 = (-w_array[0] + scale_y3) / (w_array[1] - w_array[0])
    # # plt.gca().plot(np.concatenate([x1,x2,x3]), np.concatenate([y1,y2,y3]), 'r')
    # plt.gca().plot(x1, y1, 'r')
    # plt.gca().plot(x2, y2, 'b')
    # plt.gca().plot(x3, y3, 'k')
    # ################################

    # final_S = np.concatenate([S[:, :2*L//3, 0], S[:, 2*L//3, :2*L//(3*np.sqrt(3)) +1]] + [S[:, idx, idx:idx+1] for idx in range(2*L//(3*np.sqrt(3)) +1, -1, -1)], axis=1)
    final_S = np.concatenate([S[:, :L//3, 0], S[:, L//3, :L//3]] + [S[:, idx, idx:idx+1] for idx in range(L//3, -1, -1)], axis=1)
    final_S_ = final_S.copy().real
    # final_S_[final_S_<0] = 0



    plt.imshow(final_S_, origin='lower', aspect=0.25); plt.colorbar()
    plt.xticks([0, L//3, 2*(L//3), 3 * (L//3)] , [r'$(0,0)$',r'$(0,\frac{2}{3}\pi)$',r'$(\frac{2}{3}\pi,\frac{2}{3}\pi)$',r'$(0,0)$',])
    plt.yticks([0, 40, 80, 100, 120, 160, 200],[r'$%.1f$'% w_array[0],
                                                r'$%.1f$'% w_array[40],
                                                r'$%.1f$'% w_array[80],
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


