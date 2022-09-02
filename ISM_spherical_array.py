import numpy as np
import pyroomacoustics as pra
# from plotter import plot_on_sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
# from icecream import ic
# import numba
# from numba import jit
import click
import h5py
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')

def save_paired_responses(dic, filepath, index, sample_len = 16384):
    with open('sound_field_metadata.txt', 'a') as f:
        f.write(80*'='+'\n')
        f.write(f'Sound Field {index}\n')
        f.write(80*'='+'\n')
        for key, value in dic.items():
            if isinstance(value, (list, np.ndarray)):
                if key == 'pm_{}'.format(index):
                    f.write('n_mics:%s\n' % (np.array(value).shape[-1]))
                else:
                    pass
            else:
                f.write('%s:%s\n' % (key, value))
        f.close()
    for key, item in dic.items():
        if key == 'prec_{}'.format(index):
            prec = np.asarray(item)
        if key == 'pref_{}'.format(index):
            pref = np.asarray(item)
    for ii in range(len(pref)):
        np.savez_compressed(filepath + f'/responses_ISM_sf_{index}_{ii}', pref = pref[ii], prec = prec[ii])

# @jit(nopython=True)
def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack

# @jit(nopython=True)
def _build_sensing_mat(kx, ky, kz, X, Y, Z):
    # H = np.exp(-1j * (np.einsum('ij,k -> ijk', kx, X) + \
    #                   np.einsum('ij,k -> ijk', ky, Y) + \
    #                   np.einsum('ij,k -> ijk', kz, Z)))
    # return np.transpose(H, axes=[0, 2, 1])
    H = np.exp(-1j * (np.outer(kx, X) + \
                      np.outer(ky, Y) + \
                      np.outer(kz, Z)))
    H = np.expand_dims(H, axis=0)
    return H

# @jit(nopython=True)
def grid_sphere_fib(n_points):
    """
    This function computes nearly equidistant points on the sphere
    using the fibonacci method
    Parameters
    ----------
    n_points: int
        The number of points to sample
    spherical_points: ndarray, optional
        A 2 x n_points array of spherical coordinates with azimuth in
        the top row and colatitude in the second row. Overrides n_points.
    References
    ----------
    http://lgdv.cs.fau.de/uploads/publications/spherical_fibonacci_mapping.pdf
    http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    # If no list was provided, samples points on the sphere
    # as uniformly as possible

    # Fibonnaci sampling
    offset = 2 / n_points
    increment = np.pi * (3 - np.sqrt(5))

    z = (np.arange(n_points) * offset - 1) + offset / 2
    rho = np.sqrt(1 - z ** 2)

    phi = np.arange(n_points) * increment

    x = np.cos(phi) * rho
    y = np.sin(phi) * rho

    return np.concatenate((np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z))).T

def save_h5_from_dict(dictionary, savepath = '/TD_point_sources.h5'):
    import os
    dirname = './SoundFieldData'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savepath = dirname + savepath
    with h5py.File(savepath, 'w') as f:
        for key in dictionary.keys():
            f[key] = dictionary[key]
        f.close()

def wavenumber( f=1000, n_PW=2000, c=343):
    k = 2 * np.pi * f / c
    k_grid = k*grid_sphere_fib(int(n_PW))
    return k_grid.T

def build_sensing_mat( kx, ky, kz, X, Y, Z):
    H = _build_sensing_mat(kx, ky, kz, X, Y, Z)
    return np.transpose(H, axes=[0, 2, 1])

def get_sensing_mat( f, n_pw, X, Y, Z, k_samp=None, c = None):
    # Basis functions for coefficients
    if c is None:
        c = 343
    if k_samp is None:
        k_samp = wavenumber(f, n_pw, c)
    kx, ky, kz = k_samp
    if kx.ndim < 2:
        kx = kx[np.newaxis, ...]
        ky = ky[np.newaxis, ...]
        kz = kz[np.newaxis, ...]
    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z)
    return H, k_out

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
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-1, 1e-7, 30),
                                   fit_intercept=True, normalize = True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept = True, normalize = True)

    # gcv_mode = 'eigen')
    # reg = linear_model.RidgeCV()
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

def reconstruct_FR(FR, n_pw, freq, grid, grid_ref):
    p_rec_FR = []
    pbar = tqdm(freq)
    for ii, ff in enumerate(pbar):
        pm = FR[:, ii]
        H, k = get_sensing_mat(ff,
                               n_pw,
                               grid[0],
                               grid[1],
                               grid[2])

        Href, _ = get_sensing_mat(ff,
                                  n_pw,
                                  grid_ref[0],
                                  grid_ref[1],
                                  grid_ref[2],
                                  k_samp=k)

        coeffs, alpha_ = Ridge_regression(np.squeeze(H), pm, cv=True)
        p_rec_FR.append(np.squeeze(Href) @ coeffs)
        pbar.set_description('Reconstructing f: {} Hz'.format(ff))
    return np.fft.irfft(np.array(p_rec_FR).T)

# @numba.njit
def get_shp_mesh(azim_res, pol_res, rad, plot=False):
    """
    Get spherical mesh in cartesian coordinates
    ----------------------------------------------------------------
    Args:
        azim_res : azimuth measurement resolution (azim_res = 360/n)
        pol_res  : polar angle resolution (pol_res = 180/n)
        rad      : sphere radius

    Returns:
        X_meas, Y_meas, Z_meas : cartesian coordinates as a grid
        phi, theta             : azimuth, polar angles as a grid
    ----------------------------------------------------------------
    """
    r = rad
    pi = np.pi
    cos = np.cos
    sin = np.sin
    azim_slice = np.complex(0, 360 // azim_res)
    pol_slice = np.complex(0, 360 // pol_res)
    phi, theta = np.mgrid[0.0:2.0 * pi:azim_slice, 0.0:pi:pol_slice]  # azimuth, polar
    X_meas = r * sin(phi) * cos(theta)
    Y_meas = r * sin(phi) * sin(theta)
    Z_meas = r * cos(phi)
    if plot:
        plt.figure()
        ax = plt.gca(projection="3d")
        ax.plot_wireframe(X_meas, Y_meas, Z_meas, color='hotpink', alpha=0.3, rstride=1, cstride=1)
        ax.set_xlim([-2 * rad, 2 * rad])
        ax.set_ylim([-2 * rad, 2 * rad])
        ax.set_zlim([-rad, rad])
        ax.grid()
        ax.scatter(X_meas, Y_meas, Z_meas)
        plt.tight_layout()
        plt.show()
    return X_meas, Y_meas, Z_meas, phi, theta


# @numba.njit
def get_circ_mesh(max_radius, height, N_steps_rad, N_steps_theta, plot=False):
    # res_theta = np.complex(0, N_steps_theta)
    # res_rad = np.complex(0, N_steps_rad)
    # grid_theta, grid_r  = np.mgrid[-np.pi:np.pi:res, 0:0.5:res]
    R = np.linspace(0., max_radius, N_steps_rad)
    theta = np.linspace(-np.pi, np.pi, N_steps_theta)
    x = np.outer(R, np.cos(theta))
    y = np.outer(R, np.sin(theta))
    z = height * np.ones_like(x)
    # if plot:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(x,y,z) # z in case of disk which is parallel to XY plane is constant and you can directly use h
    #     fig.show()
    return x, y, z


def reference_grid(steps, xmin=-.7, xmax=.7, z=0):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones(X.shape)
    return X, Y, Z


def adjustSNR(sig, snrdB=40, td = True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    noise : Vector or Tensor, optional
        Noise Tensor. The default is None.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from wav file
    sig_zero_mean = sig - sig.mean()
    psig = sig_zero_mean.var()

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0**(snrdB/10.0)

    # Find required noise power
    pnoise = psig/snr_lin

    if td:
        # Create noise vector
        # noise = np.sqrt(pnoise)*np.random.randn(sig.shape[0], sig.shape[1] )
        noise = np.sqrt(pnoise)*np.random.normal(0, 1, sig.shape )
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2)/2, size= sig.shape )
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2)/2, size= sig.shape )
        noise = real_noise + 1j*imag_noise
        noise = np.sqrt(pnoise)*abs(noise)*np.exp(1j*np.angle(noise))


    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise

def disk_grid_fibonacci(n, r, c = (0,0), z=None):
    """
    Get circular disk grid points
    Parameters
    ----------
    n : integer N, the number of points desired.
    r : float R, the radius of the disk.
    c : tuple of floats C(2), the coordinates of the center of the disk.
    z : float (optional), height of disk
    Returns
    -------
    cg :  real CG(2,N) or CG(3,N) if z != None, the grid points.
    """
    r0 = r / np.sqrt(float(n) - 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    gr = np.zeros(n)
    gt = np.zeros(n)
    for i in range(0, n):
        gr[i] = r0 * np.sqrt(i + 0.5)
        gt[i] = 2.0 * np.pi * float(i + 1) / phi

    if z is None:
        cg = np.zeros((3, n))
    else:
        cg = np.zeros((2, n))

    for i in range(0, n):
        cg[0, i] = c[0] + gr[i] * np.cos(gt[i])
        cg[1, i] = c[1] + gr[i] * np.sin(gt[i])
        if z != None:
            cg[2, i] = z
    return cg

def get_ISM_RIRs(room_coords,
                 room_height,
                 source_coords,
                 robot_radius,
                 n_mics,
                 fs,
                 rt60,
                 max_order,
                 plot_RIR=False,
                 plot_room=False,
                 plot_array=False,
                 raytrace=True,
                 distributed_measurements=True,
                 array_base=True,
                 snr=45):
    # grid_ref = np.asarray(reference_grid(21))
    n_ref = 1000
    grid_ref = disk_grid_fibonacci(n_ref, r = 1.5)
    grid_ref = np.reshape(grid_ref, (3, n_ref))


    # room dimensions (corners in [x,y] meters)
    room_dim = [room_coords[0, 3], room_coords[1, 2], room_height]

    if distributed_measurements:
        # grid_measured = np.asarray(reference_grid(32))
        # n_mics = np.prod(grid_measured.shape[-2:])
        # grid_measured = np.reshape(grid_measured, (3, n_mics))
        gridx = np.random.uniform(low=-room_dim[0] + 0.01, high=room_dim[0] - 0.01, size=n_mics)
        gridy = np.random.uniform(low=-room_dim[1] + 0.01, high=room_dim[1] - 0.01, size=n_mics)
        gridz = np.random.uniform(low=-room_dim[2] + 0.01, high=room_dim[2] - 0.01, size=n_mics)
        grid_measured = np.c_[robot_radius * np.array([gridx, gridy, gridz])]

    else:
        # get spherical array coords
        grid = pra.doa.GridSphere(n_points=n_mics)

        if array_base:
            mask = np.argwhere(grid.z > -.7)
            grid_measured = np.c_[robot_radius * np.array([grid.x[mask], grid.y[mask], grid.z[mask]])]
            grid_measured = np.squeeze(grid_measured)
            n_mics = grid_measured.shape[-1]
        else:
            grid_measured = np.c_[robot_radius * np.array([grid.x, grid.y, grid.z])]

        interp_ind = np.squeeze(np.argwhere(np.linalg.norm(grid_ref, axis = 0) < robot_radius))

        plus_five_ind = np.random.choice(interp_ind, 5, replace = False)
        grid_fit = np.concatenate((grid_measured, grid_ref[:, plus_five_ind]), axis = -1)
        n_mics += 5
        # rir_fit = np.concatenate((array_rirs, ref_rirs[plus_five_ind]), axis = 0)

        # for disk reference shape:
        # x_ref, y_ref, z_ref = get_circ_mesh(robot_radius + .4 * robot_radius,
        #                                     0,
        #                                     N_steps_rad=10,
        #                                     N_steps_theta=70)
        #
        # grid_ref = np.c_[np.array([x_ref.ravel(), y_ref.ravel(), z_ref.ravel()])]
        #
    # plot 3d array
    if plot_array:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(grid_measured[0], grid_measured[1], grid_measured[2], marker='d', color='b', s=100)
        # ax.scatter(0.5*grid.x, 0.5*grid.y, 0.5*grid.z, marker = 'o', color = 'r', s = 30); plt.show()
        ax.scatter(grid_ref[0], grid_ref[1], grid_ref[2], marker='d', color='r', s=100)
        ax.view_init(0, 90)
        fig.show()

    # set uniform absorption coeffs to walls
    e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
    pra.inverse_sabine(rt60, room_dim)

    # receiver centre coords
    receiver_center = np.asarray(room_dim)[:, np.newaxis] / 2

    # The locations of the microphones can then be computed as
    R = receiver_center + grid_fit
    reference_receiver = receiver_center + grid_ref

    # Create the room
    room = pra.Room.from_corners(
        room_coords,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=raytrace,
        air_absorption=True)
    if raytrace:
        room.set_ray_tracing()

    # make room 3d
    room.extrude(height=room_height, materials=pra.Material(e_absorption))

    # add source to room
    room.add_source(source_coords)
    # add arrays to room
    room.add_microphone_array(R)
    room.add_microphone_array(reference_receiver)

    # compute RIR
    room.compute_rir()
    # assert that arrays are correctly split (e.g. spherical array and reference array)
    # test_valid = R == room.mic_array.R[:, :n_mics]
    # assert (test_valid.all())

    # truncate to same length
    max_len = len(room.rir[0][0])
    for ii in range(len(room.rir)):
        if max_len < len(room.rir[ii][0]):
            max_len = len(room.rir[ii][0])
    trunc = max_len
    # split the arrays
    ref_size = len(room.rir) - n_mics
    RIR_measured = np.zeros((n_mics, trunc))
    RIRs_ref = np.zeros((ref_size, trunc))
    for ii in range(len(room.rir)):
        if ii < n_mics:
            RIR_measured[ii, :] = np.hstack((room.rir[ii][0], np.zeros((trunc - len(room.rir[ii][0], )))))
        else:
            RIRs_ref[ii - n_mics, :] = np.hstack((room.rir[ii][0], np.zeros((trunc - len(room.rir[ii][0], )))))

    RIR_measured = np.pad(RIR_measured, ((0,0), (0, 16384 - trunc)))
    RIRs_ref = np.pad(RIRs_ref, ((0,0), (0, 16384 - trunc)))
    # time samples
    t = np.linspace(0, 16384 / fs, 16384)
    RIR_measured = adjustSNR(RIR_measured, snrdB=snr)
    RIRs_ref = adjustSNR(RIRs_ref, snrdB=snr)

    # plot RIRs
    if plot_RIR:
        fig, ax = plt.subplots(1, 1)
        ax.plot(t[:int(0.2 * fs)], RIR_measured[:3, :int(0.2 * fs)].T)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid('both', ls=':', color='k')
        ax.set_title('Generated ISM RIRs')
        # plt.savefig('Generated ISM RIRs.png', dpi = 150)
        plt.show()
    if plot_room:
        fig = plt.figure()
        fig, ax = room.plot(img_order=1)
        # ax.set_xlim([0, round(room_dim[0]) + 1])
        # ax.set_ylim([0, round(room_dim[1]) + 1])
        # ax.set_zlim([0, round(room_dim[2]) + 1])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        # xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
        # XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
        # ax.set_xlim3d(XYZlim)
        # ax.set_ylim3d(XYZlim)
        # ax.set_zlim3d(XYZlim)
        ax.set_box_aspect((room_dim[0], room_dim[1], room_dim[2]))
        # try:
        #     ax.set_aspect('equal')
        # except NotImplementedError:
        #     pass
        # ax.view_init(azim=90, elev=0)
        plt.show()

    # freq = np.fft.rfftfreq(trunc, d = 1/fs)
    # FR = np.fft.rfft(RIR_measured.T, axis = 0)

    return RIR_measured, RIRs_ref, grid_fit, grid_ref


@click.command()
# options_metavar='<options>'
@click.option('--plot_array', default=False, is_flag=True,
              help='plot the spherical array to visualise in 3d')
@click.option('--plot_room', default=False, is_flag=True,
              help='plot the room with 2nd order image sources \
               to visualise in 3d')
@click.option('--plot_rir', default=False, is_flag=True,
              help='plot the first three RIRs generated with the \
               ISM and view the first 0.2 sec')
@click.option('--raytrace', default=False, is_flag=True,
              help='Use combination of ray tracing and ISM')
@click.option('--RT60', default=0.5, type=float,
              help='Reverberation time from which to calculate \
                     the uniform absorption coefficient with the \
                     Sabine equation')
@click.option('--n_mics', default=100, type=click.IntRange(10, 300),
              help='Number of transducers in array')
@click.option('--array_radius', default=.5, type=float,
              help='Radius of spherical array')
@click.option('--max_XY', nargs=2, default=[7, 5], type=click.Tuple([float, float]),
              help='maximum X Y dimensions of random rooms')
@click.option('--n_rooms', default=1, type=int,
              help='Number of rooms to generate')
# @click.option('--n_shoebox', default=.5, type=click.FloatRange(0., 1.),
#               help='Percentage of shoebox rooms as a fraction of n_rooms')
@click.option('--lsf_number', default=0, type=int,
              help='Given from cluster job number, acts as a seed for random numbers')
@click.option('--max_Z', default=3, type=click.FloatRange(2.4, 5.),
              help='Room height maximum')
@click.option('--max_order', default=11, type=click.IntRange(2, 15),
              help='Maximum order of image sources')
@click.option('--sample_rate', default=16000, type=int,
              help='Sample rate in samples/second')
@click.option('--distributed_measurements', default=False, is_flag=True,
              help='Uniformly distributed arrayr')
@click.option('--data_dir', default='./SoundFieldData', type=str,
              help='Directory to save synthesised sound fields')
def run_ISM(plot_array, plot_room, plot_rir,
            raytrace, rt60, n_mics, array_radius,
            max_xy, n_rooms, max_z,
            max_order, sample_rate, distributed_measurements,
            lsf_number, data_dir,
            ):
    
    np.random.seed(lsf_number)
    rir_sets = {}
    array_data = []
    reference_data = []
    grids_sphere = []
    grid_reference = []
    num_shbox = n_rooms
    maxX, maxY = max_xy
    for n in range(num_shbox):
        first_corner = [0., 0.]
        second_corner = [0., np.random.uniform(2, maxY)]
        third_corner = [np.random.uniform(2., maxX), second_corner[1]]
        fourth_corner = [third_corner[0], 0.]
        roomdim = [first_corner, second_corner, third_corner, fourth_corner]
        # e.g. for room coords in [x, y]:
        # room_coords = np.array([[0, 0], [1.2, 3.3], [2.4, 3.3], [3.6, 0]]).T
        room_coords = np.array(roomdim).T
        room_height = np.random.uniform(2.4, max_z)
        rev_time = np.random.uniform(rt60 - rt60 / 2, rt60 + rt60 / 2)

        # set source, slightly offset from corner and assert that it is within
        # room boundaries
        source_coords = roomdim[np.random.choice([0, 1, 2, 3])]
        source_coords.append(0.)
        if source_coords[0] > 0:
            multx = -1
        else:
            multx = 1
        if source_coords[1] > 0:
            multy = -1
        else:
            multy = 1

        snr = 30
        source_coords = list(np.asarray(source_coords) + np.array([multx * .001, multy * .001, .001]))
        rirs_sphere, rirs_ref, gridsphere, grid_ref = get_ISM_RIRs(room_coords,
                                                                   room_height,
                                                                   source_coords,
                                                                   array_radius,
                                                                   n_mics,
                                                                   sample_rate,
                                                                   rev_time,
                                                                   max_order,
                                                                   plot_RIR=plot_rir,
                                                                   plot_room=plot_room,
                                                                   plot_array=plot_array,
                                                                   raytrace=raytrace,
                                                                   array_base=False,
                                                                   snr=snr,
                                                                   distributed_measurements=distributed_measurements
                                                                   )
        # array_data.append(rirs_sphere)
        # reference_data.append(rirs_ref)
        # grids_sphere.append(gridsphere)
        # grid_reference.append(grid_ref)
        # tdsamples = 16384
        # freq = np.fft.rfftfreq(tdsamples, d = 1/sample_rate)
        # recon_rir = reconstruct_FR(np.fft.rfft(rirs_sphere), 1200, freq, gridsphere, grid_ref)
        # rir_sets["pref_{}".format(lsf_number)] = rirs_ref
        # rir_sets["pm_{}".format(lsf_number)] = rirs_sphere
        # rir_sets["prec_{}".format(lsf_number)] = recon_rir
        # rir_sets["array_loc_{}".format(lsf_number)] = gridsphere
        # rir_sets["ref_loc_{}".format(lsf_number)] = grid_ref
        # save_paired_responses(rir_sets, data_dir, index= lsf_number)
        np.savez_compressed('./ISM_sphere.npz', array_data=rirs_sphere, reference_data=rirs_ref,
                            grids_sphere=gridsphere, grid_reference=grid_ref,
                            snr=snr, rt60=rev_time, room_coords=room_coords,
                            room_height = room_height, source_coords = source_coords,
                            fs = sample_rate)

    # np.savez_compressed('ISM_sphere', array_data=array_data, reference_data=reference_data,
    #                     grids_sphere=grids_sphere, grid_reference=grid_reference,
    #                     snr=snr, rt60=rev_time, room_coords=room_coords,
    #                     room_height = room_height, source_coords = source_coords)
    # 
    # array_data = np.asarray(array_data)
    # reference_data = np.asarray(reference_data)
    # grids_sphere = np.asarray(grids_sphere)
    # grid_reference = np.asarray(grid_reference)
    # f1 = h5py.File("data_ISM.hdf5", "w")
    #
    # dset1 = f1.create_dataset("array_data", array_data.shape, dtype='f',
    #                           data=array_data, chunks=(1, array_data.shape[1], array_data.shape[-1]))
    # dset2 = f1.create_dataset("reference_data", reference_data.shape, dtype='f', data=reference_data)
    # dset1.attrs['grid'] = grids_sphere
    # dset2.attrs['grid'] = grid_reference
    # f1.close()
    #

if __name__ == '__main__':
    print("Synthesising spherical array response set, please wait...")
    run_ISM()
