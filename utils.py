from collections import OrderedDict
import scipy.special as spl
import numpy as np
import mrcfile


def make_volume(mapfile):

    mrc = mrcfile.open(mapfile)
    vol = np.array(mrc.data).astype(np.double)
    
    edge_len = vol.shape[0]
    R = edge_len // 2

    # zero-pad the volume
    vol = np.pad(vol, ((R, R), (R, R), (R, R)), 'constant')
    make_mrc(vol, "vol.mrc")

    return vol


def make_mrc(vol, mapfile):
    
    '''
    Inputs: 
        vol: volume to output
        mapfile: output mrc filename
    '''

    vol = vol.astype(np.float32)
    with mrcfile.new(mapfile, overwrite=True) as m:
        m.set_data(vol)


def make_grid2D(edge_len, kmax=0.5):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        kmax: bandlimit of the 3D Fourier transform

    Outputs:
        (xv, yv): Cartesian coordinates of the 2D grid
        support2D: indices within the circular support
    '''

    tol = 1e-10
    i0 = np.arange(edge_len)
    j0 = np.arange(edge_len)
    i1, j1 = np.meshgrid(i0, j0, indexing='ij')

    R = edge_len // 2
    xv = (j1 - R) * (kmax/R)
    yv = (R - i1) * (kmax/R)
    
    radi = np.sqrt(xv**2 + yv**2)
    radi = radi.flatten()
    support2D = tuple([radi < kmax - tol])

    return (xv, yv, support2D)


def make_grid3D(edge_len, kmax=0.5):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        kmax: bandlimit of the 3D Fourier transform
    
    Outputs:
        (xv, yv, zv): Cartesian coordinates of the 3D grid
        support3D: indices within the spherical support
    '''

    tol = 1e-10
    i0 = np.arange(edge_len)
    j0 = np.arange(edge_len)
    k0 = np.arange(edge_len)
    i1, j1, k1 = np.meshgrid(i0, j0, k0, indexing='ij')
    
    R = edge_len // 2
    xv = (k1 - R) * (kmax/R)
    yv = (R - j1) * (kmax/R)
    zv = (R - i1) * (kmax/R)
    
    radi = np.sqrt(xv**2 + yv**2 + zv**2)
    radi = radi.flatten()
    support3D = tuple([radi < kmax - tol])

    return (xv, yv, zv, support3D)


def make_polar_coor(edge_len, tilt_ang):
    
    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        tilt_ang: tilt angle of the volume

    Outputs:
        polar_coor: polar coordinates in the body frame at different tilt angles
    '''

    tol = 1e-10
    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()[support2D]
    yv = yv.flatten()[support2D]
    radi = np.sqrt(xv**2 + yv**2)
    inv_radi = np.zeros_like(radi)
    inv_radi[radi > tol] = 1./radi[radi > tol]

    num_tilts = tilt_ang.shape[0]
    num_grid2D = radi.shape[0]
    polar_coor = np.zeros((num_tilts, 3, num_grid2D))
    for r in range(num_tilts):
        polar_coor[r, 0, :] = radi
        theta = np.arccos(np.sin(tilt_ang[r])*np.multiply(yv, inv_radi))
        polar_coor[r, 1, :] = theta
        beta = np.angle(-np.cos(tilt_ang[r])*yv - 1j*xv)
        polar_coor[r, 2, :] = beta

    return polar_coor


def enumerate_samples(edge_len, kmax=0.5):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        kmax: bandlimit of the 3D Fourier transform

    Outputs:
        sample_list: list of samples for the moments
    '''
    
    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()[support2D]
    yv = yv.flatten()[support2D]
    num_grid2D = xv.shape[0]
    
    R = edge_len // 2
    xv = np.array(np.round(xv*(R/kmax))).astype(np.int)
    yv = np.array(np.round(yv*(R/kmax))).astype(np.int)
    
    pix_map = np.ones((edge_len, edge_len), dtype=np.int)*(-1)
    for t in range(num_grid2D):
        i = xv[t] + R
        j = yv[t] + R
        pix_map[i][j] = t

    # 1st order
    samples_1st = pix_map[R][R]

    # 2nd order
    samples_2nd = np.zeros((num_grid2D, 2), dtype=np.int)
    for t in range(num_grid2D):
        x1 = xv[t]
        y1 = yv[t]
        x2 = -x1
        y2 = -y1
        samples_2nd[t, 0] = t
        samples_2nd[t, 1] = pix_map[x2 + R][y2 + R]

    # 3rd order
    R2 = R**2
    s3_list = []
    for t1 in range(num_grid2D):
        x1 = xv[t1]
        y1 = yv[t1]
        for t2 in range(num_grid2D):
            x2 = xv[t2]
            y2 = yv[t2]
            x3 = -x1-x2
            y3 = -y1-y2
            if (x3**2 + y3**2 >= R2):
                continue
            t3 = pix_map[x3 + R][y3 + R]
            if (t3 >= 0):
                s3 = tuple(np.sort((t1, t2, t3)))
                s3_list.append(s3)

    s3_list = list(OrderedDict.fromkeys(s3_list))
    samples_3rd = np.zeros((len(s3_list), 3), dtype=np.int)
    for i in range(len(s3_list)):
        for j in range(3):
            samples_3rd[i, j] = s3_list[i][j]
    
    sample_list = [samples_1st, samples_2nd, samples_3rd]
    return sample_list


def sort_sample_list(edge_len, sample_list):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        sample_list: list of samples for the moments

    Outputs:
        sorted_sample_list: sorted by the sample radii in the ascending order
    '''

    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()[support2D]
    yv = yv.flatten()[support2D]
    
    # 1st order
    samples_1st = sample_list[0]

    # 2nd order
    samples_2nd = sample_list[1]
    num_samples_2nd = samples_2nd.shape[0]
    radi = np.zeros(num_samples_2nd)
    for t in range(num_samples_2nd):
        idx = samples_2nd[t, 0]
        radi[t] = np.sqrt(xv[idx]**2 + yv[idx]**2)

    sort_idx = np.argsort(radi)
    sorted_samples_2nd = np.zeros_like(samples_2nd)
    for t in range(num_samples_2nd):
        for i in range(2):
            sorted_samples_2nd[t, i] = samples_2nd[sort_idx[t], i]

    # 3rd order
    samples_3rd = sample_list[2]
    num_samples_3rd = samples_3rd.shape[0]
    radi = np.zeros(num_samples_3rd)
    for t in range(num_samples_3rd):
        rmax = 0.
        for i in range(3):
            idx = samples_3rd[t, i]
            rval = np.sqrt(xv[idx]**2 + yv[idx]**2)
            if (rval > rmax):
                rmax = rval
        radi[t] = rmax

    sort_idx = np.argsort(radi)
    sorted_samples_3rd = np.zeros_like(samples_3rd)
    for t in range(num_samples_3rd):
        for i in range(3):
            sorted_samples_3rd[t, i] = samples_3rd[sort_idx[t], i]

    sorted_sample_list = [samples_1st, sorted_samples_2nd, sorted_samples_3rd]
    return sorted_sample_list
