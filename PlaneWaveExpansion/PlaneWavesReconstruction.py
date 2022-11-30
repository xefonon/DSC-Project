import numpy as np
import h5py
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
import time
from utils_soundfields import plot_sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from celer import LassoCV, Lasso
import click
import cvxpy as cp
import librosa
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer


# %%
def reference_grid(steps, xmin=-.7, xmax=.7, z=0.):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones_like(X)

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


def plane_wave_sensing_matrix(f, sensor_grid, n_pw=1600, k_samp=None, c=343., normalise=False):
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


def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


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
def LASSOLARS_regression_IC(H, p, n_plwav=None, Noise_var = None):
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
                                linear_model.LassoLarsIC( criterion='aic', fit_intercept=False,
                                                         normalize=False,verbose=True, max_iter=50000,
                                                          noise_variance= Noise_var))

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

    reg_las = linear_model.OrthogonalMatchingPursuit(cv=5, normalize = False, max_iter=1e4)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    return q_las


def real2complex(q):
    n = q.size // 2
    return q[:n] + 1j * q[n:]


def comlpex2real(p):
    return np.concatenate((p.real, p.imag))

def mac_similarity(a,b):
    return abs(a.T.conj() @ b)**2 / ((a.T.conj()@a) * (b.T.conj()@b))
def MakeGridSearchCV(model, parameters, criterion = 'nmse', cv = 100):

    if criterion == 'nmse':
        score = make_scorer(nmse, greater_is_better=False)
    elif criterion == 'mac':
        score = make_scorer(mac_similarity, greater_is_better=True)
    return GridSearchCV(model, parameters, scoring=score, cv=cv)

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
        reg = linear_model.RidgeCV(cv=100, alphas=np.geomspace(1e-3, 1e-9, 80),
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


def LASSO_cv_regression_celer(H, p, n_plwav=None, n_jobs=10):
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

    reg_las = LassoCV(cv=100, alphas=np.geomspace(1e-2, 1e-8, 50),
                      fit_intercept=False, tol=1e-4, n_jobs=n_jobs)
    # reg_las = Lasso( alpha=1e-2,fit_intercept=True, tol = 1e-6,max_iter=1000)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        alpha_lass = None
        pass
    return q_las, alpha_lass


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
pref, fs, grid, pm, grid_measured, noise, f_vec, c = get_measurement_vectors(filename=filename, frequency_domain=False)

taps = pref.shape[-1]
noise = noise[:, :taps]

plt.magnitude_spectrum(noise.mean(0), Fs=8000, scale='dB')
plt.magnitude_spectrum(pref.mean(0), Fs=8000, scale='dB')
plt.xlim([0, 5000])
plt.show()

f_vec = np.fft.rfftfreq(pref.shape[-1], d=1 / fs)
Pref = np.fft.rfft(pref)
Pm = np.fft.rfft(pm)
Noise = np.fft.rfft(noise, n=pm.shape[-1])

# for i in range(200, 215):
#     plt.hist(noise[i], bins = 400)
#
# plt.show()
# %%
f = 500.
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

H, k = plane_wave_sensing_matrix(f, n_pw=1800,
                                 sensor_grid=grid_measured,
                                 c=c, normalise=False)

Href, _ = plane_wave_sensing_matrix(f,
                                    k_samp=k,
                                    n_pw=1800,
                                    sensor_grid=grid,
                                    c=c, normalise=False)

# startlars = time.time()
# coeffs_larsLasso, alpha_larsLasso = LASSOLARS_regression(H, pm_)
# endlars = time.time()
# Pref_ridge = comlpex2real(Pm[:, f_ind])

# transformer = M().fit(Pref_ridge.reshape(-1, 1))
# Pref_ridge = transformer.transform(Pref_ridge.reshape(-1, 1))
Pref_ridge = Pm[:, f_ind]

#
Pref_lass = Pm[:, f_ind]
# %%
startridge = time.time()
coeffs_ridge, alpha_ridge = Ridge_regression(H, Pref_ridge)
endridge = time.time()

# %%
startlass = time.time()
# coeffs_lasso, alpha_lasso = LASSO_regression_celer(H, pm[:, f_ind])
# coeffs_lasso = lasso_cvx_cmplx(H, Pref_lass, Noise_lvl= Noise[:, f_ind].mean())
coeffs_lasso = lasso_cvx_cmplx(H, Pref_lass, Noise_var=abs(Noise[:, f_ind]).mean())
# coeffs_lasso_all,  nlogs = lasso_log_cvx(H, Pref_lass, Noise_lvl= Noise[:, f_ind].mean())
endlass = time.time()
# %%
startlass = time.time()
# coeffs_lasso2, alpha_lasso = LASSOLARS_regression_CV(H, Pref_lass)
# coeffs_lasso2, alpha_lasso = LASSOLARS_regression_IC(H, Pref_lass, Noise_var = Noise[:, f_ind].var())
coeffs_lasso2, alpha_lasso = LASSO_cv_regression_celer(H, Pref_lass)
# coeffs_lasso, alpha_lasso = LASSO_regression(H, Pref_lass)
# coeffs_lasso_sk = LASSO_regression(H, Pref_lass)
# coeffs_lasso_all,  nlogs = lasso_log_cvx(H, Pref_lass, Noise_lvl= Noise[:, f_ind].mean())
endlass = time.time()
# %%
rmse = lambda x, y: np.sqrt((abs(y - x) ** 2).mean())


def nmse(y_true, y_predicted, db=True):
    M = len(y_true)
    nmse_ = 1 / M * np.sum(abs(y_true - y_predicted) ** 2) / np.sum(abs(y_true)) ** 2
    if db:
        nmse_ = np.log10(nmse_)
    return nmse_


# plars = np.squeeze(Href) @ coeffs_larsLasso
plass = Href @ coeffs_lasso
plass2 = Href @ coeffs_lasso2
# plass = np.squeeze(Href) @ coeffs_lasso_sk[0]
pridge = Href @ coeffs_ridge

# pridge = transformer.inverse_transform(comlpex2real(pridge).reshape(-1, 1))
# pridge = real2complex(pridge).squeeze(-1)
#
# plass = transformer.inverse_transform(comlpex2real(plass).reshape(-1, 1))
# plass = real2complex(plass).squeeze(-1)
# alpha_lasso = None
# portho = np.squeeze(Href) @ coeffs_ortho
# print("alpha Lars Lasso: {}, time: {:.4f}, error: {:.5f}".format(alpha_larsLasso, -(startlars - endlars),
#                                                                  rmse(plars, pref_)))
print("alpha Ridge: {}, time: {:.4f}, error: {:.5f}".format(alpha_ridge, -(startridge - endridge),
                                                            nmse(Pref[:, f_ind], pridge )))
print("alpha Lasso: {}, time: {:.4f}, error: {:.5f}".format(None, -(startlass - endlass), nmse(Pref[:, f_ind], plass)))
print("alpha Lasso: {}, time: {:.4f}, error: {:.5f}".format(None, -(startlass - endlass), nmse(Pref[:, f_ind], plass2)))

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 4, 1)
ax, _ = plot_sf(Pref[:, f_ind], grid[0], grid[1], ax=ax)
ax.set_title('truth')
# ax = fig.add_subplot(1, 4, 2)
# ax, _ = plot_sf(plars, grid[0],grid[1], ax=ax)
# ax.set_title('Lars')
# ax = fig.add_subplot(1, 4, 3)
# ax, _ = plot_sf(plass, grid[0],grid[1], ax=ax)
# ax.set_title('Lasso')
ax = fig.add_subplot(1, 4, 2)
ax, _ = plot_sf(pridge, grid[0], grid[1], ax=ax)
ax.set_title('Ridge')
ax = fig.add_subplot(1, 4, 3)
ax, _ = plot_sf(plass, grid[0], grid[1], ax=ax)
ax.set_title('Sparse')
ax = fig.add_subplot(1, 4, 4)
ax, _ = plot_sf(plass2, grid[0], grid[1], ax=ax)
ax.set_title('Lasso')
# ax = fig.add_subplot(1, 5, 5)
# ax, _ = plot_sf(portho, grid[0],grid[1], ax=ax)
# ax.set_title('Orthogonal M. P.')
fig.tight_layout()
fig.show()
