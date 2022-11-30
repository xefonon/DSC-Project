import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from celer import LassoCV
import click
import cvxpy as cp
import librosa
import time
import jax
import jax.numpy as jnp
import numpyro
from jax.lib import xla_bridge
from numpyro import distributions as dist
import jax.random as jrandom
from numpyro.infer import MCMC, NUTS
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
import h5py
import time
import os
np.random.seed(42)
numpyro.set_platform("gpu")
numpyro.enable_x64()

""" Temporary / problem specific functions"""

def reference_grid(steps, xmin=-.7, xmax=.7, z=0.):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones_like(X)

    gridnew = np.vstack((X.reshape(1, -1), Y.reshape(1, -1), Z.reshape(1, -1)))
    return gridnew

def subsample_gridpoints(grid, subsample=5):
    r0 = grid.mean(axis=-1)
    tempgrid = grid - r0[:, None]
    xmin, xmax = round(tempgrid[0].min(), 3), round(tempgrid[0].max(), 3)
    # ymin, ymax = round(tempgrid[1].min(), 3), round(tempgrid[1].max(), 3)
    newgrid = reference_grid(subsample, xmin, xmax)
    newgrid += r0[:, None]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(grid.T)
    distances, indices = nbrs.kneighbors(newgrid.T)
    return grid[:, indices.squeeze(-1)], indices.squeeze(-1)

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

def get_measurement_vectors(filename, subsample_points=10, frequency_domain=True):
    data_dict = load_measurement_data(filename)
    refdata = data_dict['RIRs_bottom']
    fs = data_dict['fs']
    grid = data_dict['grid_bottom']
    temperature = data_dict['temperature']
    c = speed_of_sound(temperature)
    # grid_measured = data_dict['grid_bottom']
    # noise = data_dict['Noise_recs_bottom'].mean(axis=1)
    noise = refdata[:, 17000:25000]
    noise = librosa.resample(noise, fs, 8000)
    refdata = librosa.resample(refdata, fs, 8000)
    measureddata = refdata
    fs = 8000
    grid -= grid.mean(axis=-1)[:, None]
    grid_measured, indcs = subsample_gridpoints(grid, subsample=subsample_points)
    measureddata = measureddata[indcs]
    if frequency_domain:
        f_vec = np.fft.rfftfreq(refdata.shape[-1], d=1 / fs)
        refdata = np.fft.rfft(refdata)
        measureddata = np.fft.rfft(measureddata)
    else:
        f_vec = None
    return refdata, fs, grid, measureddata, grid_measured, noise, f_vec, c


""" Plotting """


def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim=None, tex=False, cmap=None, normalise=True,
            colorbar=False, cbar_label='', cbar_loc='bottom',
            interpolated=True):
    """
    Plot spatial soundfield normalised amplitude
    --------------------------------------------
    Args:
        P : Pressure in meshgrid [X,Y]
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)
    """
    # plot_settings()

    N_interp = 1500
    if normalise:
        Pvec = P / np.max(abs(P))
    else:
        Pvec = P
    res = complex(0, N_interp)
    Xc, Yc = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    Pmesh = griddata(points, Pvec, (Xc, Yc), method='cubic', rescale=True)
    if cmap is None:
        cmap = 'coolwarm'
    if f is None:
        f = ''
    # P = P / np.max(abs(P))
    X = Xc.flatten()
    Y = Yc.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * X.ptp() / Pmesh.size
    dy = 0.5 * Y.ptp() / Pmesh.size
    if ax is None:
        _, ax = plt.subplots()  # create figure and axes
    if interpolated:
        im = ax.imshow(Pmesh.real, cmap=cmap, origin='upper',
                       extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy])
        ax.invert_xaxis()
    else:
        if clim is not None:
            lm1, lm2 = clim
        else:
            lm1, lm2 = None, None
        im = ax.scatter(x, y, c=Pvec.real,
                        cmap=cmap, alpha=1., s=10, vmin=lm1, vmax=lm2)
        ax.set_aspect('equal')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    if clim is not None:
        lm1, lm2 = clim
        im.set_clim(lm1, lm2)
    if colorbar:
        if cbar_loc != 'bottom':
            shrink = 1.
            orientation = 'vertical'
        else:
            shrink = 1.
            orientation = 'horizontal'

        cbar = plt.colorbar(im, ax=ax, location=cbar_loc,
                            shrink=shrink)
        # cbar.ax.get_yaxis().labelpad = 15
        titlesize = mpl.rcParams['axes.titlesize']
        # cbar.ax.set_title(cbar_label, fontsize = titlesize)
        cbar.set_label(cbar_label, fontsize=titlesize)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name)
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax, im


""" Preprocessing """

def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)
    return c


def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


def scale_maxabs(data, constant = 1):
    maxabs = abs(data.max())
    datanew = data * constant/maxabs
    return datanew, constant/maxabs

def rescale_maxabs(data, scale):
    return data * scale

def standardise(data, remove_bias = False):
    if remove_bias:
        mu = data.mean(axis = -1)[..., None]
    else:
        mu = np.zeros_like(data.mean(axis = -1)[..., None])
    sigma = data.std(axis = -1)[..., None]
    newdata =  (data - mu)/sigma
    return newdata, mu, sigma
def rescale(data, mu, sigma):
    return data*sigma + mu

def normalise(data):
    norm = np.linalg.norm(data,2)
    newdata = data/norm
    return newdata, norm

def rescale_norm(data, norm):
    return data * norm

def get_lstsq_noise_estimate(H, p):
    lstsq_coeffs = np.linalg.lstsq(H, p, rcond=None)[0]
    return H @ lstsq_coeffs - p


""" Metrics """


def mac_similarity(a, b):
    return abs(a.T.conj() @ b) ** 2 / ((a.T.conj() @ a) * (b.T.conj() @ b))


def nmse(y_true, y_predicted, db=True):
    M = len(y_true)
    nmse_ = 1 / M * np.sum(abs(y_true - y_predicted) ** 2) / np.sum(abs(y_true)) ** 2
    if db:
        nmse_ = np.log10(nmse_)
    return nmse_


""" Scikit-learn functions"""


def MakeGridSearchCV(model, parameters, criterion='nmse', cv=100):
    if criterion == 'nmse':
        score = make_scorer(nmse, greater_is_better=False)
    elif criterion == 'mac':
        score = make_scorer(mac_similarity, greater_is_better=True)
    return GridSearchCV(model, parameters, scoring=score, cv=cv)


def choose_linear_model(choice='larslasso', fit_intercept=False):
    if choice == 'larslasso'.casefold():
        return linear_model.LassoLars(normalize=False, fit_intercept=fit_intercept)
    elif choice == 'lasso'.casefold():
        return linear_model.Lasso(normalize=False, fit_intercept=fit_intercept)
    if choice == 'ridge'.casefold():
        return linear_model.Ridge(normalize=False, fit_intercept=fit_intercept)
    if choice == 'omp'.casefold():
        # takes n_nonzero_coefs as parameter
        # (e.g. parameters = {'n_nonzero_coefs' :np.linspace(10, 1000, 100, dtype = int)},
        return linear_model.OrthogonalMatchingPursuit(normalize=False, fit_intercept=fit_intercept)


def Make_SKlearn_pipeline(normalization='maxabs', linear_model='larslasso'):
    lin_mod = choose_linear_model(choice=linear_model)
    if normalization == 'maxabs':
        pipe = make_pipeline(MaxAbsScaler(), lin_mod)
    elif normalization == 'standardise':
        pipe = make_pipeline(StandardScaler(), lin_mod)
    return pipe


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
        reg = linear_model.RidgeCV(cv=100, alphas=np.geomspace(1e-3, 1e-10, 100),
                                   fit_intercept=False)
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
        reg_las = linear_model.LassoCV(cv=100, alphas=np.geomspace(1e-3, 1e-10, 80),
                                       fit_intercept=True,
                                       normalize=False, verbose=True,
                                       tol=1e-4, n_jobs=10)
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


def LASSOLARS_regression_CV(H, p, n_plwav=None, cv=True):
    """
    Compressive Sensing - Soundfield Reconstruction
    Parameters
    ----------
    cv
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
        reg_las = make_pipeline(MaxAbsScaler(),
                                linear_model.LassoLarsCV(cv=100, fit_intercept=False,
                                                         n_jobs=5, normalize=False,
                                                         verbose=True, max_iter=2000))
        # reg_las =  linear_model.LassoLarsCV(cv=100, fit_intercept=False,
        #                                                  n_jobs=10, normalize=False,
        #                                                  verbose=True, max_iter=5000)

    else:
        alpha_lass = 2.62e-6
        reg_las = linear_model.LassoLars(alpha=alpha_lass, fit_intercept=True, normalize=False, max_iter=2000)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las[1].coef_[:n_plwav] + 1j * reg_las[1].coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        alpha_lass = None
        pass

    return q_las, alpha_lass


def LASSOLARS_regression_IC(H, p, n_plwav=None, Noise_var=None):
    """
    Compressive Sensing - Soundfield Reconstruction
    Parameters
    ----------
    cv
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

    reg_las = make_pipeline(MaxAbsScaler(),
                            linear_model.LassoLarsIC(criterion='aic', fit_intercept=False,
                                                     normalize=False, verbose=True, max_iter=20000,
                                                     noise_variance=Noise_var))

    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)
    np.linalg.lstsq(H, p)
    reg_las.fit(H, p)
    q_las = reg_las[1].coef_[:n_plwav] + 1j * reg_las[1].coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        alpha_lass = None
        pass

    return q_las, alpha_lass


""" CVXPY """


def lasso_cvx(H, p, n_plwav=None, Noise_lvl=None):
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if Noise_lvl is None:
        Noise_lvl = .1
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    # if Noise_lvl.dtype == complex:
    #     Noise_lvl = np.concatenate([Noise_lvl.real * np.ones_like(p.real),
    #                                 Noise_lvl.imag * np.ones_like(p.real)])
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))
    # Create variable.
    x_l1 = cp.Variable(shape=2 * n_plwav)
    epsilon = abs(Noise_lvl)
    # Create constraint.
    # constraints = [cp.norm(H @ x_l1 - p, 2, axis = ) <= epsilon]
    constraints = [cp.norm2(H @ x_l1 - p) <= epsilon]

    # Form objective.
    obj = cp.Minimize(cp.norm(x_l1, 1))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='SCIPY')
    print("status: {}".format(prob.status))

    # Number of nonzero elements in the solution (its cardinality or diversity).
    Noise_lvl = Noise_lvl[0] + 1j * Noise_lvl[0]
    x = x_l1.value[:n_plwav] + 1j * x_l1.value[n_plwav:]
    nnz_l1 = (np.absolute(x) > np.absolute(Noise_lvl)).sum()
    print('Found a feasible x in R^{} that has {} nonzeros.'.format(n_plwav, nnz_l1))
    print("optimal objective value: {}".format(obj.value))

    return x


def lasso_log_cvx(H, p, n_plwav=None, Noise_lvl=None, NUM_RUNS=15):
    nnzs_log = np.array(())
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if Noise_lvl is None:
        Noise_lvl = 1e-6
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if Noise_lvl.dtype == complex:
        Noise_lvl = np.concatenate([Noise_lvl.real * np.ones_like(p.real),
                                    Noise_lvl.imag * np.ones_like(p.real)])
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    # Create variable.
    delta = Noise_lvl.max()
    # Store W as a positive parameter for simple modification of the problem.
    W = cp.Parameter(shape=2 * n_plwav, nonneg=True);
    x_log = cp.Variable(shape=2 * n_plwav)

    # Initial weights.
    W.value = np.ones(2 * n_plwav);

    # Setup the problem.
    obj = cp.Minimize(W.T @ cp.abs(x_log))  # sum of elementwise product
    constraints = [H @ x_log <= p]

    # constraints = [A*x_log <= b]
    prob = cp.Problem(obj, constraints)
    x_all = []
    # Do the iterations of the problem, solving and updating W.
    for k in range(1, NUM_RUNS + 1):
        # Solve problem.
        # The ECOS solver has known numerical issues with this problem
        # so force a different solver.
        prob.solve(solver=cp.SCIPY)

        # Check for error.
        if prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        # Display new number of nonzeros in the solution vector.
        nnz = (np.absolute(x_log.value) > delta).sum()
        nnzs_log = np.append(nnzs_log, nnz)
        print('Iteration {}: Found a feasible x in R^{}'
              ' with {} nonzeros...'.format(k, 2 * n_plwav, nnz))

        # Adjust the weights elementwise and re-iterate
        W.value = np.ones(2 * n_plwav) / (delta * np.ones(2 * n_plwav) + np.absolute(x_log.value))
        x_all.append(x_log.value[:n_plwav] + 1j * x_log.value[n_plwav:])
    # x = x_log.value[:n_plwav] + 1j * x_log.value[n_plwav:]
    return x_all, nnzs_log


def lasso_cvx_cmplx(H, p, n_plwav=None, Noise_var=None, max_iters=5000):
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if Noise_var is None:
        Noise_var = 1e-6
    # Create variable.
    x_l1 = cp.Variable(shape=n_plwav, complex=True)
    epsilon = Noise_var
    # Create constraint.
    # constraints = [cp.norm(H @ x_l1 - p, 2, axis = ) <= epsilon]
    constraints = [cp.norm2(H @ x_l1 - p) <= epsilon]

    # Form objective.
    obj = cp.Minimize(cp.norm(x_l1, 1))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    # works with cp.SCS, max_iters = 10000
    prob.solve(solver=cp.SCS, verbose=True, max_iters=max_iters,
               use_indirect=True, alpha=1.5, warm_start=True)
    print("status: {}".format(prob.status))

    # Number of nonzero elements in the solution (its cardinality or diversity).
    # Noise_lvl = Noise_lvl[0] + 1j * Noise_lvl[1]
    # x = x_l1.value[:n_plwav] + 1j * x_l1.value[n_plwav:]
    nnz_l1 = (np.absolute(x_l1.value) > epsilon).sum()
    print('Found a feasible x in R^{} that has {} nonzeros.'.format(n_plwav, nnz_l1))
    print('Lagrangian multiplier: {}'.format(constraints[0].dual_value))
    print("optimal objective value: {}".format(obj.value))

    return x_l1.value


""" CELER """


# def LASSO_cv_regression_celer(H, p, n_plwav=None, n_jobs=10):
#     """
#     Compressive Sensing - Soundfield Reconstruction
#     Parameters
#     ----------
#     H : Transfer Matrix.
#     p : Measured Pressure.
#     n_plwav : number of plane waves.
#     Returns
#     -------
#     q_las : Plane wave coefficients.
#     alpha_lass : Regularizor.
#     """
#     if n_plwav is None:
#         n_plwav = H.shape[-1]
#     if H.dtype == complex:
#         H = stack_real_imag_H(H)
#     if p.dtype == complex:
#         p = np.concatenate((p.real, p.imag))
#
#     reg_las = LassoCV(cv=100, alphas=np.geomspace(1e-2, 1e-8, 50),
#                       fit_intercept=False, tol=1e-4, n_jobs=n_jobs)
#     # reg_las = Lasso( alpha=1e-2,fit_intercept=True, tol = 1e-6,max_iter=1000)
#
#     reg_las.fit(H, p)
#     q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
#     try:
#         alpha_lass = reg_las.alpha_
#     except:
#         alpha_lass = None
#         pass
#     return q_las, alpha_lass


""" Transfer matrices """


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

def sample_circle(n_samples = 1200, radius = 1., z = 0.):
    angle = np.pi * np.linspace(0, 2., n_samples) - np.pi

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = z*np.ones_like(x)
    return np.stack([x, y, z])


def wavenumber(f, n_PW, c=343, two_dim = True):
    k = 2 * np.pi * f / c
    if two_dim:
        k_grid = sample_circle(n_PW, k)
    else:
        k_grid = fib_sphere(n_PW, k)
    return k_grid


def plane_wave_sensing_matrix(f, sensor_grid, n_pw=1600, k_samp=None, c=343., normalise=False, 
                              two_dim = False):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c=c,
                            two_dim = two_dim)
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
    H = build_sensing_mat(k_out, sensor_grid)
    if normalise:
        column_norm = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / column_norm
    return H, k_out


def build_sensing_mat(k_sampled, sensor_grid):
    kx, ky, kz = k_sampled
    X, Y, Z = sensor_grid
    H = np.exp(-1j * (np.einsum('ij,k -> ijk', kx, X) + np.einsum('ij,k -> ijk', ky, Y) +
                      np.einsum('ij,k -> ijk', kz, Z)))
    return H.squeeze(0).T


""" Hierarchical Bayes """

def run_mcmc(model, data_dict, num_posterior_samples = 1000, num_warmup = None, num_chains = 2,
             thinning= 3 ):
    # numpyro.set_platform(platform)
    print(xla_bridge.get_backend().platform)
    # coefficients = np.zeros((nfreqs, num_posterior_samples * num_chains,
    #                          data_dict['H'].shape[-1]), dtype="complex")
    mcmc_dict = dict(H=data_dict["H"], pm= data_dict["pm"])
    if num_warmup is None:
        num_warmup = int(num_posterior_samples / 2)
    rng_key = jrandom.PRNGKey(42)
    rng_key, rng_key_ = jrandom.split(rng_key)
    my_kernel = NUTS(model, max_tree_depth= 10, target_accept_prob=0.8)
    posterior = MCMC(
        my_kernel,
        thinning= thinning,
        num_samples=num_posterior_samples,
        num_warmup= num_warmup,
        num_chains=num_chains,
        progress_bar=True,
        # chain_method= 'vectorized'
    )
    posterior.run(
        rng_key_,
        data=mcmc_dict,  # This dictionary should be created with measurements, Nw, Nf, Nm, a_w, b_w, device?
    )
    coefficients = posterior.get_samples()["beta_real"] + 1j * posterior.get_samples()["beta_imag"]
    return coefficients, posterior

def hierarchical_model(data):
    """
    Heterscedastic source + noise model (separate model - unpooled)

    """
    A = jnp.asarray(data["H"])
    y = jnp.asarray(data["pm"])
    N = A.shape[1]
    M = y.shape[0]

    # sigma = numpyro.sample("sigma", dist.HalfCauchy(jnp.ones(M)))

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma = 1.0 / jnp.sqrt(prec_obs)


    tau_r = numpyro.sample("tau_r", dist.InverseGamma(concentration=3 * jnp.ones(N), rate=jnp.ones(N)))
    tau_im = numpyro.sample("tau_im", dist.InverseGamma(concentration=3 * jnp.ones(N), rate=jnp.ones(N)))
    b_real = numpyro.sample("beta_real", dist.Normal(jnp.zeros(N), tau_r))
    b_imag = numpyro.sample("beta_imag", dist.Normal(jnp.zeros(N), tau_im))
    mu_real = jnp.einsum("mp, p -> m", A.real, b_real) - jnp.einsum(
        "mp, p -> m", A.imag, b_imag
    )
    mu_imag = jnp.einsum("mp, p -> m", A.imag, b_real) + jnp.einsum(
        "mp, p -> m", A.real, b_imag
    )
    numpyro.sample("y_real", dist.Normal(mu_real, sigma), obs=y.real)
    numpyro.sample("y_imag", dist.Normal(mu_imag, sigma), obs=y.imag)

def sparse_hierarchical_model(data):
    """
    Heterscedastic source + noise model and horseshoe prior

    http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
    """
    H = jnp.asarray(data["H"])
    y = jnp.asarray(data["pm"])
    N = H.shape[1]
    M = y.shape[0]
    m = 1000.
    sigma_0 = 2
    tau_0 = m / (H.shape[1] - m) * sigma_0 / jnp.sqrt(H.shape[0])

    # Each coefficient β_i is modelled as a normal distribution with a variance of λ_i, τ
    # Tau_r = numpyro.sample('tau_r', dist.HalfCauchy(scale=tau_0*jnp.ones(1)))
    Tau = numpyro.sample('tau',  dist.HalfCauchy(scale=tau_0 *jnp.ones(1)))
    Lambda_r = numpyro.sample('lambda_r', dist.HalfCauchy(scale=jnp.ones(N)))
    # Tau_im = numpyro.sample('tau_im', dist.HalfCauchy(scale=tau_0*jnp.ones(1)))
    Lambda_im = numpyro.sample('lambda_im', dist.HalfCauchy(scale=jnp.ones(N)))

    # note that this reparameterization (i.e. coordinate transformation) improves
    # posterior geometry and makes NUTS sampling more efficient
    unscaled_hs_beta_im = numpyro.sample("unscaled_hs_sigma_im", dist.Normal(0.0, jnp.ones(N)))
    unscaled_hs_beta_re = numpyro.sample("unscaled_hs_sigma_re", dist.Normal(0.0, jnp.ones(N)))
    horseshoe_sigma_r = Tau * Lambda_r * unscaled_hs_beta_re
    horseshoe_sigma_im = Tau * Lambda_im * unscaled_hs_beta_im

    # horseshoe_sigma_r = Tau **2 * Lambda_r **2
    # horseshoe_sigma_im = Tau **2 * Lambda_im **2
    # horseshoe_sigma_r = Tau * Lambda_r
    # horseshoe_sigma_im = Tau * Lambda_im
    b_real = numpyro.deterministic("beta_real", horseshoe_sigma_r)
    b_imag = numpyro.deterministic("beta_imag", horseshoe_sigma_im)

    # b_real = numpyro.sample('beta_real', dist.Normal(loc=jnp.zeros(N), scale=scaled_sigma_r))
    # b_imag = numpyro.sample('beta_imag', dist.Normal(loc=jnp.zeros(N), scale=scaled_sigma_im))

    sigma = numpyro.sample("sigma", dist.HalfNormal( sigma_0*jnp.ones(M) ))

    mu_real = jnp.einsum("mp, p -> m", H.real, b_real) - jnp.einsum(
        "mp, p -> m", H.imag, b_imag
    )
    mu_imag = jnp.einsum("mp, p -> m", H.imag, b_real) + jnp.einsum(
        "mp, p -> m", H.real, b_imag
    )
    numpyro.sample("y_real", dist.Normal(mu_real, sigma), obs=y.real)
    numpyro.sample("y_imag", dist.Normal(mu_imag, sigma), obs=y.imag)


def bayesian_lasso(data):
    H = jnp.asarray(data["H"])
    y = jnp.asarray(data["pm"])
    N = H.shape[1]
    M = y.shape[0]

    sigma = numpyro.sample("sigma", dist.HalfNormal(2*jnp.ones(M)))

    tau_r = numpyro.sample('tau_r', dist.InverseGamma(concentration=2 * jnp.ones(N), rate=jnp.ones(N)))
    tau_i = numpyro.sample('tau_i', dist.InverseGamma(concentration=2 * jnp.ones(N), rate=jnp.ones(N)))
    # b_real = numpyro.sample('beta_real', dist.Laplace(loc=jnp.zeros(N), scale=tau_r))
    # b_imag = numpyro.sample('beta_imag', dist.Laplace(loc=jnp.zeros(N), scale=tau_i))
    b_real = numpyro.sample('beta_real', dist.Laplace(loc=jnp.zeros(N), scale=tau_r))
    b_imag = numpyro.sample('beta_imag', dist.Laplace(loc=jnp.zeros(N), scale=tau_i))
    mu_real = jnp.einsum("mp, p -> m", H.real, b_real) - jnp.einsum(
        "mp, p -> m", H.imag, b_imag
    )
    mu_imag = jnp.einsum("mp, p -> m", H.imag, b_real) + jnp.einsum(
        "mp, p -> m", H.real, b_imag
    )
    numpyro.sample("y_real", dist.Normal(mu_real, sigma), obs=y.real)
    numpyro.sample("y_imag", dist.Normal(mu_imag, sigma), obs=y.imag)

# def horshoe_prior(data):
#     m = 500,
#     ss = 3,
#     dof = 25,
#     H = jnp.asarray(data["H"])
#     y = jnp.asarray(data["pm"])
#     N = H.shape[1]
#     M = y.shape[0]
#     sigma_0 = 2
#     sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_0*jnp.ones(M)))
#     tau_0 = m / (H.shape[1] - m) * sigma_0 / jnp.sqrt(H.shape[0])
#     tau = numpyro.sample("tau", dist.HalfCauchy(tau_0))
#
#     c2 = numpyro.sample("c2",dist.InverseGamma('c2', dof / 2, dof / 2 * ss ** 2))
#     lam = pm.HalfCauchy('lam', 1, shape=X.shape[1])
#
#     l1 = lam * tt.sqrt(c2)
#     l2 = tt.sqrt(c2 + tau * tau * lam * lam)
#     lam_d = l1 / l2


# %%
@click.command()
@click.option(
    "--data_dir", default='../Data', type=str, help="Directory of training data"
)
@click.option(
    "--save_dir",
    default="./Reconstructions",
    type=str,
    help="Directory for saving PW model results"
)
@click.option(
    "--reconstruction_method",
    default='ridge',
    type= click.Choice(['sparse bayes', 'ridge', 'lasso', 'bayesian lasso', 'normal bayes'], case_sensitive=False),
    help="Method with which to solve inverse problem"
)
@click.option(
    "--normalisation",
    default='nothing',
    type= click.Choice(['standardise', 'unitnorm', 'nothing'], case_sensitive=False),
    help="standardise measurement data with global mean and std, normalize with inf norm (f \in [-1, 1])"
         " or dont do anything"
)
@click.option(
    "--reconstruction_index",
    default= -1,
    type= int,
    help="To be used for parallelisation. When set to -1, this setting is bypassed"
)

@click.option(
    "--reconstruction_freq",
    default=  456.,
    type= int,
    help="When 'reconstruction_index' is not in use, then reconstruct at this frequency"
)

def run_experiment(data_dir, save_dir = './Reconstructions', reconstruction_method = 'ridge', normalisation = 'unitnorm',
                   reconstruction_index = 0, reconstruction_freq = None,
                   number_of_plane_waves = 1800):
    filename = data_dir + '/SoundFieldControlPlanarDataset.h5'
    pref, fs, grid, pm, grid_measured, noise, f_vec, c = get_measurement_vectors(filename=filename, frequency_domain=False)

    taps = pref.shape[-1]
    noise = noise[:, :taps]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f_vec = np.fft.rfftfreq(pref.shape[-1], d=1 / fs)
    # Pref = np.fft.rfft(pref)
    Pm = np.fft.rfft(pm)
    Noise = np.fft.rfft(noise, n=pm.shape[-1])

    if reconstruction_index == -1:
        if reconstruction_freq is None:
            raise SystemExit('You must set a reconstruction frequency when "reconstruction_index" is set to -1.')
        f_ind = np.argmin(f_vec <= reconstruction_freq)
    else:
        f_ind = reconstruction_index
    f = f_vec[f_ind]

    H, k = plane_wave_sensing_matrix(f, n_pw=number_of_plane_waves,
                                     sensor_grid=grid_measured,
                                     c=c, normalise=False)

    Href, _ = plane_wave_sensing_matrix(f,
                                        k_samp=k,
                                        n_pw=number_of_plane_waves,
                                        sensor_grid=grid,
                                        c=c, normalise=False)

    # Pm_ff, mu_P, scale_P = standardise(Pm[:, f_ind])
    # Pref_ff, mu_Pref, scale_Pref = scale_maxabs(Pm[:, f_ind])
    Noise_estimate = Noise[:, f_ind]

    if normalisation == 'standardise':
        Pm_ff, mu_P, scale_P = standardise(Pm[:, f_ind])
        Noise_estimate = (Noise_estimate - Noise_estimate.mean())/scale_P
    elif normalisation == 'unitnorm':
        Pm_ff, norm = normalise(Pm[:, f_ind])
        Noise_estimate /= norm
    else:
        Pm_ff = Pm[:, f_ind]
    data_dict = {
        "f":f,
        "H" : H,
        "pm" : Pm_ff,
        "epsilon" : Noise_estimate

    }
    print(80*'-')
    print(f"Using {reconstruction_method} to reconstruct pressure at f = {np.round(f,2)} Hz with {Pm_ff.shape} sensors")
    print(80*'-')
    if reconstruction_method == 'ridge':
        start_ridge = time.time()
        coeffs, alpha_ridge = Ridge_regression(H, Pm_ff)
        end_ridge = time.time()
        elapsed_time_ridge = time.strftime("%Mm %Ss", time.gmtime(end_ridge - start_ridge ))
    elif reconstruction_method == 'normal bayes':
        start_bayes = time.time()
        coeffs_post, posterior = run_mcmc(hierarchical_model, data_dict=data_dict
                                     , num_chains=2, num_posterior_samples=1000)
        end_bayes = time.time()
        elapsed_time_normal = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes))
        print("MCMC took ", elapsed_time_normal)
        
        coeffs = coeffs_post.mean(axis = -1)
    elif reconstruction_method == 'bayesian lasso':
        start_bayes = time.time()
        coeffs_post, posterior = run_mcmc(bayesian_lasso, data_dict=data_dict
                                       , num_chains=2, num_posterior_samples=1000)
        end_bayes = time.time()
        elapsed_time_lasso = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes))
        print("MCMC took ", elapsed_time_lasso)
        
        coeffs = coeffs_post.mean(axis = -1)
    elif reconstruction_method == 'sparse bayes':
        start_bayes = time.time()
        coeffs_post, posterior = run_mcmc(sparse_hierarchical_model, data_dict=data_dict
                                       , num_chains=2, num_posterior_samples=1000,
                                       platform='gpu')
        end_bayes = time.time()
        elapsed_time_sparse = time.strftime("%Mm %Ss", time.gmtime(end_bayes - start_bayes))

        print("MCMC took ", elapsed_time_sparse)
        
        coeffs = coeffs_post.mean(axis = -1)
    elif reconstruction_method == 'lasso':
        start_ridge = time.time()
        coeffs =  lasso_cvx_cmplx(H, Pm_ff, Noise_var= abs(data_dict["epsilon"]).mean())
        end_ridge = time.time()
        elapsed_time_ridge = time.strftime("%Mm %Ss", time.gmtime(end_ridge - start_ridge ))
        print("Took ", elapsed_time_ridge)

    Phat = Href.dot(coeffs)
    if normalisation == 'standardise':
        Phat = rescale(Phat, mu_P, scale_P)
    elif normalisation == 'unitnorm':
        Phat *= norm

    np.savez(save_dir + f"/reconstructed_pressure_partion_n_{reconstruction_index}", Phat)

if __name__ == '__main__':
    run_experiment()
