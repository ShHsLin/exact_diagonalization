import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


# some fake data
data = np.random.randn(1000)
# evaluate the histogram
filename = sys.argv[1]
if '.npy' in filename:
    wf = np.load(filename)
else:
    wf = np.genfromtxt(filename)
    L=16
    X_computation_basis = np.genfromtxt('basis_L%d.csv' % L, delimiter=',')
    wf = wf[np.sum(X_computation_basis, axis=-1)==8]


wf = wf.astype(np.complex128)
log_wf = np.log(wf)

fig, axes = plt.subplots(1, 2, sharey=True)


weights_list = [np.abs(wf)**2, np.ones_like(wf)/wf.size]
fmt_list = ['-', '--']
for idx, weights in enumerate(weights_list):
    fmt = fmt_list[idx]
    real_values, real_base = np.histogram(log_wf.real, bins=40, weights=weights)
    real_cumulative = np.cumsum(real_values)
    imag_values, imag_base = np.histogram(log_wf.imag, bins=40, weights=weights)
    imag_cumulative = np.cumsum(imag_values)
    axes[0].plot(real_base[:-1], real_cumulative, fmt, c='blue',)
    axes[1].plot(imag_base[:-1], imag_cumulative, fmt, c='blue',)


hist_kws={'weights':np.abs(wf)**2}
sns.distplot(log_wf.real, color="r", ax=axes[0], hist_kws=hist_kws)
sns.distplot(log_wf.imag, color="r", ax=axes[1], hist_kws=hist_kws)

## plot the survival function
# plt.plot(base[:-1], len(data)-cumulative, c='green')

plt.show()

