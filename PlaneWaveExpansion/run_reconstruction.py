import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('./PlaneWaveExpansion')

from InvProblemsAuxFunctions import (plane_wave_sensing_matrix, get_measurement_vectors,normalise,Ridge_regression,
                                     run_mcmc,hierarchical_model, sparse_hierarchical_model, bayesian_lasso,
                                     nmse, plot_sf)

# %%
filename = './Data/SoundFieldControlPlanarDataset.h5'
pref, fs, grid, pm, grid_measured, noise, f_vec, c = get_measurement_vectors(filename=filename, frequency_domain=False)

taps = pref.shape[-1]
noise = noise[:, :taps]

# plt.magnitude_spectrum(noise.mean(0), Fs=8000, scale='dB')
# plt.magnitude_spectrum(pref.mean(0), Fs=8000, scale='dB')
# plt.xlim([0, 5000])
# plt.show()

f_vec = np.fft.rfftfreq(pref.shape[-1], d=1 / fs)
Pref = np.fft.rfft(pref)
Pm = np.fft.rfft(pm)
Noise = np.fft.rfft(noise, n=pm.shape[-1])
# %%
f = 500.
f_ind = np.argmin(f_vec <= f)
f = f_vec[f_ind]

H, k = plane_wave_sensing_matrix(f, n_pw=1200,
                                 sensor_grid=grid_measured,
                                 c=c, normalise=False)

Href, _ = plane_wave_sensing_matrix(f,
                                    k_samp=k,
                                    n_pw=1200,
                                    sensor_grid=grid,
                                    c=c, normalise=False)

# %%

# Pm_ff, mu_P, scale_P = standardise(Pm[:, f_ind])
# Pref_ff, mu_Pref, scale_Pref = scale_maxabs(Pm[:, f_ind])
Pm_ff, norm = normalise(Pm[:, f_ind])

data_dict = {
    "f":f,
    "H" : H,
    "pm" : Pm_ff

}

coeffs_lstsq  = np.linalg.lstsq(H, Pm_ff, rcond= True)[0]
# %%
start_ridge = time.time()

coeffs_ridge, alpha_ridge = Ridge_regression(H, Pm_ff)
#  = lasso_cvx_cmplx(H, Pm_ff, Noise_var= abs(Noise[:, f_ind]/norm).mean())
end_ridge = time.time()

elapsed_time_ridge = time.strftime("%Mm %Ss", time.gmtime(end_ridge - start_ridge ))

print("Took ", elapsed_time_ridge)
# %%

start_bayes = time.time()
coeffs, posterior = run_mcmc(hierarchical_model, data_dict= data_dict
                             , num_chains=2,  num_posterior_samples = 1500)
end_bayes = time.time()
elapsed_time_normal = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes ))
print("MCMC took ", elapsed_time_normal)

#%%
start_bayes = time.time()
coeffs2, posterior2 = run_mcmc(bayesian_lasso, data_dict= data_dict
                             , num_chains=2, num_posterior_samples = 1000)
end_bayes = time.time()
elapsed_time_lasso = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes ))
print("MCMC took ", elapsed_time_lasso)

#%%
start_bayes = time.time()
coeffs2, posterior2 = run_mcmc(sparse_hierarchical_model, data_dict= data_dict
                             , num_chains=2, num_posterior_samples = 1000)
end_bayes = time.time()
elapsed_time_sparse = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes ))

print("MCMC took ", elapsed_time_sparse)

#%%
# p_post = Href.dot(coeffs.T)
p_post2 = Href.dot(coeffs2.T)
p_ridge = Href.dot(coeffs_ridge)*norm
# p_ridge = rescale(p_ridge, mu_P, scale_P)
# p_post2 = rescale(p_post2, mu_P, scale_P)
# p_post = rescale(p_post, mu_P, scale_P)
#%%
# print("Posterior samples: {}, time: {:.4f}, error: {:.5f}".format(coeffs.shape[0], -(start_bayes - end_bayes),
#                                                                   nmse(Pref[:, f_ind], p_post.mean(axis =-1))))


print("Sparse Posterior samples: {}, time: {:.4f}, error: {:.5f}".format(coeffs2.shape[0], -(start_bayes - end_bayes),
                                                                  nmse(Pref[:, f_ind], p_post2.mean(axis =-1))))
print("Ridge samples: {}, time: {}, error: {:.5f}".format(p_ridge.shape[0], elapsed_time_ridge,
                                                                  nmse(Pref[:, f_ind], p_ridge)))
# %%

plims = [Pref[:, f_ind].real.min(), Pref[:, f_ind].real.max()]

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax, _ = plot_sf(Pref[:, f_ind].imag, grid[0], grid[1], ax=ax, clim = plims, normalise= False)
ax.set_title('truth')
ax = fig.add_subplot(1, 3, 2)
ax, _ = plot_sf(p_ridge.imag, grid[0], grid[1], ax=ax, clim = plims, normalise= False)
ax.set_title('MAP - Normal Prior')
ax = fig.add_subplot(1, 3, 3)
ax, _ = plot_sf(p_post2.mean(axis =-1).imag, grid[0], grid[1], ax=ax, clim = plims, normalise= False)
ax.set_title('MAP - Laplace Prior')
fig.tight_layout()
fig.show()
# %%
stdlims = [p_post2.std(axis =-1).min(), p_post2.std(axis =-1).max()]
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax, im = plot_sf(p_post2.std(axis =-1), grid[0], grid[1], ax=ax, cmap = 'bone', normalise = False, clim= stdlims)
ax.set_title('std - Normal Prior')
ax = fig.add_subplot(1, 2, 2)
ax, _ = plot_sf(p_post2.std(axis =-1), grid[0], grid[1], ax=ax, cmap = 'bone', normalise = False, clim= stdlims)
ax.set_title('std - Laplace Prior')
fig.colorbar(im)
fig.tight_layout()
fig.show()
