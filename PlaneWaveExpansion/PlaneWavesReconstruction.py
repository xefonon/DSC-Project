import numpy as np
from sklearn import linear_model
import h5py
from sklearn.neighbors import NearestNeighbors
import time
from utils_soundfields import plot_sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from celer import LassoCV, Lasso
import click
#%%
def reference_grid(steps, xmin=-.7, xmax=.7, z = 0.):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    X, Y = np.meshgrid(x, y)
    Z = z*np.ones_like(X)

    gridnew = np.vstack((X.reshape(1, -1), Y.reshape(1, -1), Z.reshape(1, -1)))
    return gridnew

def fib_sphere(num_points, radius=1):
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)

    x_batch = np.tensordot(radius, x, 0)
    y_batch = np.tensordot(radius, y, 0)
    z_batch = np.tensordot(radius, z, 0)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 20)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()
    return [x_batch, y_batch, z_batch]

def wavenumber(f, n_PW, c=343):
    k = 2 * np.pi * f / c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

def get_sensing_mat(f, n_pw, X, Y, Z, k_samp=None, c=343, mesh = True):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c)

    kx, ky, kz = k_samp
    if np.ndim(kx) < 2:
        kx = np.expand_dims(kx, 0)
        ky = np.expand_dims(ky, 0)
        kz = np.expand_dims(kz, 0)
    elif np.ndim(kx) > 2:
        kx = np.squeeze(kx, 1)
        ky = np.squeeze(ky, 1)
        kz = np.squeeze(kz, 1)

    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=mesh)
    column_norm = np.linalg.norm(H, axis = 1, keepdims = True)
    H = H/column_norm
    return H, k_out

def build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=False):
    if mesh:
        H = np.exp(-1j*(np.einsum('ij,k -> ijk', kx, X.flatten()) + np.einsum('ij,k -> ijk', ky, Y.flatten()) +
                   np.einsum('ij,k -> ijk', kz, Z.flatten())))
    else:
        H = np.exp(-1j*(np.einsum('ij,k -> ijk', kx, X) + np.einsum('ij,k -> ijk', ky, Y) +
                   np.einsum('ij,k -> ijk', kz, Z)))
    return H.squeeze(0).T
def load_measurement_data(filename):

    with h5py.File(filename, "r") as f:
        data_keys = f.keys()
        meta_data_keys = f.attrs.keys()
        data_dict = {}
        for key in data_keys:
            data_dict[key] = f[key][:]
        for key in meta_data_keys:
            data_dict[key] = f.attrs[key]
        f.close()
    return data_dict

def subsample_gridpoints(grid, subsample = 5):
    r0 = grid.mean(axis=-1)
    tempgrid = grid - r0[:, None]
    xmin, xmax = round(tempgrid[0].min(), 3), round(tempgrid[0].max(), 3)
    # ymin, ymax = round(tempgrid[1].min(), 3), round(tempgrid[1].max(), 3)
    newgrid = reference_grid(subsample, xmin, xmax)
    newgrid += r0[:,None]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(grid.T)
    distances, indices = nbrs.kneighbors(newgrid.T)
    return grid[:, indices.squeeze(-1)], indices.squeeze(-1)

def get_measurement_vectors(filename, subsample_points = 10, frequency_domain = True):
    data_dict = load_measurement_data(filename)
    refdata = data_dict['RIRs_bottom']
    fs = data_dict['fs']
    grid = data_dict['grid_bottom']
    measureddata = data_dict['RIRs_bottom']
    # grid_measured = data_dict['grid_bottom']
    grid -= grid.mean(axis = -1)[:, None]
    grid_measured, indcs = subsample_gridpoints(grid, subsample= subsample_points)
    measureddata = measureddata[indcs]
    if frequency_domain:
        f_vec = np.fft.rfftfreq(refdata.shape[-1], d= 1/fs)
        refdata = np.fft.rfft(refdata)
        measureddata = np.fft.rfft(measureddata)
    else:
        f_vec = None
    return refdata, fs, grid, measureddata, grid_measured, f_vec

def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack

def LASSOLARS_regression(H, p, n_plwav=None, cv=True):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    if cv:
        reg_las = linear_model.LassoLarsCV(cv=5, fit_intercept=True, normalize=True)
    else:
        alpha_lass = 2.62e-6
        reg_las = linear_model.LassoLars(alpha=alpha_lass, fit_intercept=True, normalize=True)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        pass

    return q_las, alpha_lass


def OrthoMatchPursuit_regression(H, p, n_plwav=None):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg_las = linear_model.OrthogonalMatchingPursuitCV(cv=5, max_iter=1e4)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    return q_las

def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-2, 1e-7, 50),
                                   fit_intercept=True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=True)

    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk
def LASSO_regression(H, p, n_plwav=None, cv=True):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    if cv:
        reg_las = linear_model.LassoCV(cv=5, alphas=np.geomspace(1, 1e-12, 50),
                                       fit_intercept=True, normalize=False, tol = 1e-4, n_jobs= 8, max_iter= 10000)
    else:
        alpha_lass = 2.62e-6
        reg_las = linear_model.Lasso(alpha=alpha_lass,
                                     fit_intercept=True, normalize=False)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        pass
    return q_las, alpha_lass

def LASSO_regression_celer(H, p, n_plwav=None):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg_las = LassoCV( cv=3, alphas=np.geomspace(1e-2, 1e-8, 20),
                                       fit_intercept=True, tol = 1e-4, n_jobs= 12)
    # reg_las = Lasso( alpha=1e-2,fit_intercept=True, tol = 1e-6,max_iter=1000)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        alpha_lass = None
        pass
    return q_las, alpha_lass


# %%
# @click.command()
# @click.option(
#     "--data_dir", default='../Data', type=str, help="Directory of training data"
# )
# @click.option(
#     "--save_dir",
#     default="./Reconstructions",
#     type=str,
#     help="Directory for saving PW model results"
# )
# @click.option(
#     "--train_epochs",
#     default=1e8,
#     type=int,
#     help="Number of epochs for which to train PINN (in total)"
# )
# @click.option(
#     "--siren",
#     default=True,
#     type=bool,
#     help="Use sinusoidal activations"
# )
# @click.option(
#     "--real_data",
#     default=True,
#     type=bool,
#     help="Use measurement data"
# )
# @click.option(
#     "--standardize_data",
#     default=False,
#     type=bool,
#     help="Standardize measurement data with global mean and std"
# )

filename = './Data/SoundFieldControlPlanarDataset.h5'
pref, fs, grid, pm, grid_measured, f_vec = get_measurement_vectors(filename= filename)


f = 725.
f_ind = np.argmin(f_vec <= f)
f = f_vec[f_ind]
# pm_cat = np.concatenate((pm.real[None, :, f_ind], pm.imag[None,:,  f_ind]))
# pref_cat = np.concatenate((pref.real[None, :, f_ind], pref.imag[None,:,  f_ind]))
# transformer1 = Normalizer().fit(pm_cat)
# transformer2 = Normalizer().fit(pref_cat)
# pm_ = transformer1.transform(pm_cat)
# pref_ = transformer2.transform(pref_cat)
# pm_ = pm_[0] + 1j*pm_[1]
# pref_ = pref_[0] + 1j*pref_[1]
H, k = get_sensing_mat(f, 3000,
                      grid_measured[0],
                      grid_measured[1],
                      grid_measured[2])

Href, _ = get_sensing_mat(f, 3000,
                      grid[0],
                      grid[1],
                      grid[2])

# startlars = time.time()
# coeffs_larsLasso, alpha_larsLasso = LASSOLARS_regression(H, pm_)
# endlars = time.time()
startridge = time.time()
coeffs_ridge, alpha_ridge = Ridge_regression(H, pm[:, f_ind])
endridge = time.time()
startlass = time.time()
coeffs_lasso, alpha_lasso = LASSO_regression_celer(H, pm[:, f_ind])
endlass = time.time()
# startlass = time.time()
# coeffs_lasso, alpha_lasso = LASSO_regression(H, pm_)
# endlass = time.time()
# startortho = time.time()
# coeffs_ortho = OrthoMatchPursuit_regression(H, pm_)
# endortho = time.time()

rmse = lambda x, y: np.sqrt((abs(y - x) ** 2).mean())

# plars = np.squeeze(Href) @ coeffs_larsLasso
plass = np.squeeze(Href) @ coeffs_lasso
pridge = np.squeeze(Href) @ coeffs_ridge
# portho = np.squeeze(Href) @ coeffs_ortho
# print("alpha Lars Lasso: {}, time: {:.4f}, error: {:.5f}".format(alpha_larsLasso, -(startlars - endlars),
#                                                                  rmse(plars, pref_)))
print("alpha Ridge: {}, time: {:.4f}, error: {:.5f}".format(alpha_ridge, -(startridge - endridge),
                                                            rmse(pridge, pref[:, f_ind])))
print("alpha Lasso: {}, time: {:.4f}, error: {:.5f}".format(alpha_lasso, -(startlass - endlass), rmse(plass,  pref[:, f_ind])))
# print("alpha Lasso: {}, time: {:.4f}, error: {:.5f}".format('doesnt apply', -(startlass - endlass), rmse(portho, pref_)))
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax, _ = plot_sf(pref[:, f_ind], grid[0],grid[1], ax=ax)
ax.set_title('truth')
# ax = fig.add_subplot(1, 4, 2)
# ax, _ = plot_sf(plars, grid[0],grid[1], ax=ax)
# ax.set_title('Lars')
# ax = fig.add_subplot(1, 4, 3)
# ax, _ = plot_sf(plass, grid[0],grid[1], ax=ax)
# ax.set_title('Lasso')
ax = fig.add_subplot(1, 3, 2)
ax, _ = plot_sf(pridge, grid[0],grid[1], ax=ax)
ax.set_title('Ridge')
ax = fig.add_subplot(1, 3, 3)
ax, _ = plot_sf(plass, grid[0],grid[1], ax=ax)
ax.set_title('Ridge')
# ax = fig.add_subplot(1, 5, 5)
# ax, _ = plot_sf(portho, grid[0],grid[1], ax=ax)
# ax.set_title('Orthogonal M. P.')
fig.tight_layout()
fig.show()

