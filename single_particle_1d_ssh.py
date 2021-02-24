import scipy.linalg
import numpy as np

'''
lattice
'''

def get_Hamiltonian(L, t=1, PBC=True):
    Hamiltonian = np.zeros([2*L, 2*L])
    ta = tb = t
    for idx in range(2*L):
        if idx % 2 == 0:
            Hamiltonian[idx, (idx+1)%L] = -ta
            Hamiltonian[idx, (idx+L-1)%L] = -tb
        else:
            Hamiltonian[idx, (idx+1)%L] = -tb
            Hamiltonian[idx, (idx+L-1)%L] = -ta

    return Hamiltonian


def occupation_number(psi):
    # return np.einsum('i,i->i', psi, psi.conjugate())
    return psi

def double_ft(A, w, dt=0.05):
    '''
    Goal perform double fourier transform on the time-dependent connected correlation function.

    Input: A
    '''
    num_steps, sys_size = A.shape
    L = sys_size
    # eta = - np.log(0.5) / (num_steps * dt)
    # exp_jwt = np.exp( 1j * (w + 1j * eta) * np.arange(num_steps) * dt )
    eta = - np.log(0.1) / ((num_steps * dt)**2)
    # exp_jwt = np.cos(w * np.arange(num_steps) * dt )
    exp_jwt = np.exp( 1j * w * np.arange(num_steps) * dt ) * np.exp(
        - eta * ((np.arange(num_steps) * dt)**2))
    S_jw = exp_jwt.dot(A) * 2 * np.pi / num_steps
    S_jw = S_jw.reshape([L])

    S_qw = np.zeros([L], dtype=np.complex)
    for idx_x in range(L):
        qx = 2*np.pi * idx_x / L
        exp_jqx = np.exp( -1j * qx * ( np.arange(L) - L//2))
        S_qw[idx_x] = exp_jqx.dot(S_jw)

    S_qw = (1. / sys_size) * S_qw
    return S_qw

def get_S_qw(A, num_omega=301, dt=0.05):
    S_array = []
    w_array = []
    for idx_w in range(num_omega):
        w = -5 + 0.05 * idx_w
        S_array.append(double_ft(A, w, dt))
        w_array.append(w)


    return np.array(S_array), np.array(w_array)





'''
lattice_idx = idx = state_idx
'''
if __name__ == '__main__':
    L = 150
    N = 2 * L
    H = get_Hamiltonian(L, t=1.)
    # print(H)
    # import matplotlib.pyplot as plt
    # plt.imshow(H)
    # plt.show()
    # Random State 
    psi = np.zeros(N)
    psi[N//2] = 1
    # Ground State

    # psi[Ly*(Lx//2)+Ly//2] = 1

    dt = 0.02
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


    S_array, w_array = get_S_qw(occup_array, num_omega=301, dt=dt)


    S = S_array
    # final_S = np.concatenate([S[:, :L//2, 0], S[:, L//2, :L//2]] + [S[:, idx, idx:idx+1] for idx in range(L//2, -1, -1)], axis=1)
    final_S_ = S.copy().real
    # final_S_[final_S_<0] = 0

    x = np.arange(L)
    scale_x = 2 * np.pi / L * x
    scale_y = -2 * np.cos(scale_x)
    y = (-w_array[0] + scale_y) / (w_array[1] - w_array[0])
    plt.gca().plot(x, y, 'r')
    plt.imshow(final_S_, origin='lower', aspect=0.25); plt.colorbar()
    # plt.xticks([0, L//2, 2 * (L//2), 3 * (L//2)] , [r'$(0,0)$',r'$(0,\pi)$',r'$(\pi,\pi)$',r'$(0,0)$',])
    plt.xticks([0, L//2, L-1] , [r'$0$',r'$\pi$',r'$2\pi$'])
    plt.yticks([0, 60, 100, 140, 160, 200] , [r'$%.1f$'% w_array[0],
                                              r'$%.1f$'% w_array[60],
                                              r'$%.1f$'% w_array[100],
                                              r'$%.1f$'% w_array[140],
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


