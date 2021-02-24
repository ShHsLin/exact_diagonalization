import numpy as np
# from contract_peps import *
# from peps import PEPS

import pickle
# import setup


def double_ft(A, w, dt=0.05):
    '''
    Goal perform double fourier transform on the time-dependent connected correlation function.

    Input: A
    '''
    num_steps, sys_size = A.shape
    L = int(sys_size ** 0.5)
    assert L**2 == sys_size
    # eta = - np.log(0.5) / (num_steps * dt)
    # exp_jwt = np.exp( 1j * (w + 1j * eta) * np.arange(num_steps) * dt )
    eta = - np.log(0.99) / ((num_steps * dt)**2)
    exp_jwt = np.exp( 1j * w * np.arange(num_steps) * dt ) * np.exp(
        - eta * ((np.arange(num_steps) * dt)**2))
    S_jw = exp_jwt.dot(A) * 2 * np.pi / num_steps
    S_jw = S_jw.reshape((L, L))

    S_qw = np.zeros((L, L), dtype=np.complex)
    for idx_x in range(L):
        qx = 2*np.pi * idx_x / L
        for idx_y in range(L):
            qy = 2*np.pi * idx_y / L
            exp_jqx = np.exp( -1j * qx * ( np.arange(L) - L/2.))
            exp_jqy = np.exp( -1j * qy * ( np.arange(L) - L/2.))
            S_qw[idx_x, idx_y] = exp_jqx.dot(S_jw).dot(exp_jqy)


    S_qw = (1. / sys_size) * S_qw
    return S_qw

def get_S_qw(A, num_omega=301, dt=0.05):
    S_array = []
    w_array = []
    for idx_w in range(num_omega):
        w = 0.05 * idx_w
        S_array.append(double_ft(A, w))
        w_array.append(w)


    return np.array(S_array), np.array(w_array)


Corr_list = []
Occu_list = []
L=5
D = 2
chi = 2
num_steps = 100
for idx in range(num_steps):
    print(idx)
    filename_GS = '../../data/time_evolution_Ising_2d/L%d_MZ0_D%d_chi%d_GS/step%d.pkl' % (L, D, chi, idx)
    wf_GS = pickle.load(open(filename_GS, 'rb'))
    dtype = wf_GS.T_array[0][0].dtype

    Sz = np.array([[1, 0], [0., -1.]], dtype=dtype)
    Splus = np.array([[0, 1], [0., 0.]], dtype=dtype)

    # filename_Sz = '../../data/time_evolution_Ising_2d/L%d_MZ1_D%d_chi%d_Sz/step%d.pkl' % (L, D, chi, idx)
    # wf_Sz = pickle.load(open(filename_Sz, 'rb'))
    # Corr = compute_td_corr(wf_GS, wf_Sz, Sz, D_max= 2 * D**2, Verbose=True)
    filename_Splus = '../../data/time_evolution_Ising_2d/L%d_MZ0_D%d_chi%d_Splus/step%d.pkl' % (L, D, chi, idx)
    wf_Splus = pickle.load(open(filename_Splus, 'rb'))

    Occu = compute_single_site_Op(wf_Splus, Sz, D_max=2 * D**2, Verbose=True)
    Occu_list.append(Occu.real.flatten())

    Corr = compute_td_corr(wf_GS, wf_Splus, Splus, D_max=2 * D**2, Verbose=True)
    Corr_list.append(Corr.flatten())

import matplotlib.pyplot as plt
import pdb;pdb.set_trace()
plt.imshow(np.array(Occu_list)); plt.colorbar(); plt.show()
# 
# L = 11 
# D = 6
# chi = 12
# fn_Corr = '../../data/time_evolution_Ising_2d/L%d_MZ1_D%d_chi%d_Splus/Corr_D%d.npy' % (L, D, chi, 2*D**2)
# # fn_Corr = '../../data/time_evolution_Ising_2d/L%d_MZ1_D%d_chi%d_Sz/Corr_D%d.npy' % (L, D, chi, 2*D**2)
# num_steps=30
# Corr_list = np.load(fn_Corr)[:num_steps,:]
# print(Corr_list.shape)
# 
# ###### # np.save(fn_Corr, np.array(Corr_list))

A = np.array(Corr_list).copy()
AA = A[:,(L*L)//2]
# AA = AA-AA[0]
AA = np.real(AA)
d_list=[]
dt=0.05
eta = - np.log(0.99) / ((num_steps * dt)**2)
for ww in np.arange(300)/10.:
    exp_jwt = np.exp( 1j * ww * np.arange(num_steps) * dt ) * np.exp( - eta * ((np.arange(num_steps) * dt)**2))
    d_list.append(AA.dot(exp_jwt))

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3,1,1)
plt.plot(np.arange(num_steps)*dt, AA)
plt.subplot(3,1,2)
plt.plot(np.arange(num_steps)*dt, AA * np.exp(- eta * ((np.arange(num_steps) * dt)**2)))
plt.subplot(3,1,3)
plt.plot(np.arange(300)*0.1, d_list)
# plt.show()
# import pdb;pdb.set_trace()



A = np.array(Corr_list)
A = A.real
# A = A - A[0,:]
S, W = get_S_qw(A, 301)

import matplotlib.pyplot as plt
plt.figure()
raw_S = S.real.reshape([301,L*L]).T
raw_S += np.concatenate([raw_S[:, 1:], raw_S[:, -1: ]], axis=1)
raw_S += np.concatenate([raw_S[:, :1], raw_S[:, :-1]], axis=1)
import pdb;pdb.set_trace()
min_idx_raw_S = np.argmax(raw_S, axis=1) * 0.05
import pdb;pdb.set_trace()
plt.imshow(min_idx_raw_S.reshape([L, L]), origin='lower')
plt.colorbar()
# plt.show()


S = S.reshape((301, L, L))
# final_S = np.concatenate([S[:, L//2:, L//2], S[:, -1, L//2:]] + [S[:, idx, idx:idx+1] for idx in range(L-1, L//2-1, -1)], axis=1)
final_S = np.concatenate([S[:, :L//2, 0], S[:, L//2, :L//2]] + [S[:, idx, idx:idx+1] for idx in range(L//2, -1, -1)], axis=1)
final_S_ = final_S.copy().real
final_S_[final_S_<0] = 0


import matplotlib.pyplot as plt
plt.figure()
plt.imshow(A, origin='lower')
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.xlabel(u'Site')
plt.ylabel(u'$Time\ t[1/\Delta], \Delta=20$')
plt.title(u'$S^{zz}(j, t_n)$')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)
fig = plt.gcf()
fig.set_size_inches(9, 4)
# plt.savefig('dt%d_Szz.pdf' % num_steps)
# plt.show()

plt.figure()
plt.imshow(final_S_, origin='lower', aspect=0.25); plt.colorbar()
plt.xlabel(u'$(k_x, k_y)$'); plt.ylabel(u'$\omega$');
plt.xticks([0, L//2, 2 * (L//2), 3 * (L//2)] , [r'$(0,0)$',r'$(0,\pi)$',r'$(\pi,\pi)$',r'$(0,0)$',])
plt.yticks([0, 50, 100, 120, 160, 200] , [r'$0$',r'$2.5$', r'$5.0$',r'$6.0$',r'$8.$',r'$10.$'])
plt.title(u'$S^{zz}(k, \omega)$')
fig = plt.gcf()
fig.set_size_inches(4, 9)
# plt.savefig('dt%d_Sqw.pdf' % num_steps)
# plt.show()

plt.figure()
plt.imshow(final_S.real, origin='lower', aspect=0.25); plt.colorbar()
plt.xlabel(u'$(k_x, k_y)$'); plt.ylabel(u'$\omega$');
plt.xticks([0, L//2, 2 * (L//2), 3 * (L//2)] , [r'$(0,0)$',r'$(0,\pi)$',r'$(\pi,\pi)$',r'$(0,0)$',])
plt.yticks([0, 50, 100, 120, 160, 200] , [r'$0$',r'$2.5$', r'$5.0$',r'$6.0$',r'$8.$',r'$10.$'])
fig = plt.gcf()
fig.set_size_inches(4, 9)
# plt.savefig('dt%d_Sqw_neg.png' % num_steps)
plt.show()



