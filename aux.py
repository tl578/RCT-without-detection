import scipy.optimize as optimize
from scipy import sparse
from utils import *
import finufft


def make_subproblem_config(edge_len, r_limit, kmax=0.5):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        r_limit: cutoff radius in the unit of voxels
        kmax: bandlimit of the 3D Fourier transform
    
    Outputs:
        sub_grid2D: subset of 2D grid points considered in the problem
        sub_grid2D_coor: coordinates of sub_grid2D
        sub_grid3D: subset of 3D grid points considered in the problem
        sub_grid3D_coor: coordinates of sub_grid3D
    '''

    # make sub_grid2D
    R = edge_len // 2
    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()[support2D]*(R/kmax)
    yv = yv.flatten()[support2D]*(R/kmax)
    xv = np.array(np.round(xv)).astype(np.int)
    yv = np.array(np.round(yv)).astype(np.int)
    radi = np.sqrt(xv**2 + yv**2)
    num_grid2D = radi.shape[0]

    sub_grid2D = []
    sub_grid2D_coor = []
    for i in range(num_grid2D):
        if (radi[i] < r_limit):
            sub_grid2D.append(i)
            sub_grid2D_coor.append((xv[i], yv[i]))

    sub_grid2D = np.array(sub_grid2D)
    sub_grid2D_coor = np.array(sub_grid2D_coor)
    num_sub_grid2D = sub_grid2D.shape[0]

    # make sub_grid3D
    (xv, yv, zv, support3D) = make_grid3D(edge_len)
    xv = xv.flatten()[support3D]*(R/kmax)
    yv = yv.flatten()[support3D]*(R/kmax)
    zv = zv.flatten()[support3D]*(R/kmax)
    xv = np.array(np.round(xv)).astype(np.int)
    yv = np.array(np.round(yv)).astype(np.int)
    zv = np.array(np.round(zv)).astype(np.int)
    radi = np.sqrt(xv**2 + yv**2 + zv**2)
    num_grid3D = xv.shape[0]

    sub_grid3D = []
    sub_grid3D_coor = []
    for i in range(num_grid3D):
        if (radi[i] < r_limit):
            sub_grid3D.append(i)
            sub_grid3D_coor.append((xv[i], yv[i], zv[i]))

    sub_grid3D = np.array(sub_grid3D)
    sub_grid3D_coor = np.array(sub_grid3D_coor)
    num_sub_grid3D = sub_grid3D.shape[0]

    return (sub_grid2D, sub_grid2D_coor, sub_grid3D, sub_grid3D_coor)


def make_kgrid(edge_len, tilt_ang, num_alpha, sub_grid2D, kmax=0.5):

    '''
    Inputs:
        edge_len: length of the 3D Fourier transform
        tilt_ang: tilt angle of the volume
        num_alpha: number of samples for the in-plane rotations
        sub_grid2D: subset of 2D grid points considered in the problem
        kmax: bandlimit of the 3D Fourier transform

    Outputs:
        kgrid: non-uniform coordinates sampled by rotated Fourier slices
    '''

    # make coordinates of the rotated pixels in the body frame
    R = edge_len // 2
    polar_coor = make_polar_coor(edge_len, tilt_ang)
    polar_coor = np.transpose(np.squeeze(polar_coor))
    polar_coor[:, 0] *= R/kmax

    num_sub_grid2D = sub_grid2D.shape[0]
    kx = np.zeros(num_alpha*num_sub_grid2D)
    ky = np.zeros(num_alpha*num_sub_grid2D)
    kz = np.zeros(num_alpha*num_sub_grid2D)
    alpha = np.arange(num_alpha)*(2*np.pi/num_alpha)

    for j in range(num_alpha):
        idx_start = j*num_sub_grid2D
        for i in range(num_sub_grid2D):
            idx = idx_start + i
            radi = polar_coor[sub_grid2D[i], 0]
            theta = polar_coor[sub_grid2D[i], 1]
            phi = polar_coor[sub_grid2D[i], 2] - alpha[j]
            kx[idx] = radi*np.sin(theta)*np.cos(phi)
            ky[idx] = radi*np.sin(theta)*np.sin(phi)
            kz[idx] = radi*np.cos(theta)

    rescale = 2*np.pi/edge_len
    kgrid = np.vstack((kx, ky, kz))
    kgrid *= rescale

    return kgrid


def make_sub_sample_list(sorted_sample_list, sub_grid2D):

    '''
    Inputs:
        sorted_sample_list: list of samples for the moments, sorted by 
                            the sample radii in the ascending order
        sub_grid2D: subset of 2D grid points considered in the problem

    Outputs:
        sub_sample_list: list of samples for the moments that lie within sub_grid2D
    '''

    samples_1st = sorted_sample_list[0]
    samples_2nd = sorted_sample_list[1]
    samples_3rd = sorted_sample_list[2]
    
    num_grid2D = 2*samples_1st + 1
    num_sub_grid2D = sub_grid2D.shape[0]
    pix_map = np.ones(num_grid2D, dtype=np.int)*(-1)
    for i in range(num_sub_grid2D):
        pix_map[sub_grid2D[i]] = i

    # 1st order
    sub_samples_1st = num_sub_grid2D // 2

    # 2nd order
    sub_samples_2nd = []
    for i in range(len(samples_2nd)):
        idx1 = samples_2nd[i][0]
        idx2 = samples_2nd[i][1]
        if (pix_map[idx1] < 0 or pix_map[idx2] < 0):
            break
        sub_samples_2nd.append([pix_map[idx1], pix_map[idx2]])
    
    # 3rd order
    sub_samples_3rd = []
    for i in range(len(samples_3rd)):
        idx1 = samples_3rd[i][0]
        idx2 = samples_3rd[i][1]
        idx3 = samples_3rd[i][2]
        if (pix_map[idx1] < 0 or pix_map[idx2] < 0 or pix_map[idx3] < 0):
            break
        sub_samples_3rd.append([pix_map[idx1], pix_map[idx2], pix_map[idx3]])
    
    sub_samples_2nd = np.array(sub_samples_2nd).astype(np.int)
    sub_samples_3rd = np.array(sub_samples_3rd).astype(np.int)
    sub_sample_list = [sub_samples_1st, sub_samples_2nd, sub_samples_3rd]
    
    return sub_sample_list


def make_nonredundant_variables(vol, kmax=0.5):

    edge_len = vol.shape[0]
    R = edge_len // 2
    D = (edge_len + 1) // 2

    (xv, yv, zv, support3D) = make_grid3D(edge_len)
    xv = np.array(np.round(xv.flatten()*(R/kmax))).astype(np.int)
    yv = np.array(np.round(yv.flatten()*(R/kmax))).astype(np.int)
    zv = np.array(np.round(zv.flatten()*(R/kmax))).astype(np.int)
    radi = np.sqrt(xv**2 + yv**2 + zv**2)

    par_support3D = tuple([radi < D // 2])
    indices = np.arange(edge_len**3)
    indices = indices[par_support3D]
    vol = vol.flatten()

    for i in range(indices.shape[0]):
        idx = indices[i]
        if (xv[idx] == 1 and yv[idx] == 0 and zv[idx] == 0):
            i1 = i
        elif (xv[idx] == 0 and yv[idx] == 1 and zv[idx] == 0):
            i2 = i
        elif (xv[idx] == 0 and yv[idx] == 0 and zv[idx] == 1):
            i3 = i

    row_ind = []
    col_ind = []
    data = []    

    num_pmtr = indices.shape[0] - 3
    x0 = np.zeros(num_pmtr)
    pmtr_ct = 0
    
    for i in range(indices.shape[0]):
        if (i == i1 or i == i2 or i == i3):
            continue

        idx = indices[i]
        row_ind.append(idx)
        col_ind.append(pmtr_ct)
        data.append(1.)

        row_ind.append(indices[i1])
        col_ind.append(pmtr_ct)
        data.append(-xv[idx])

        row_ind.append(indices[i2])
        col_ind.append(pmtr_ct)
        data.append(-yv[idx])
        
        row_ind.append(indices[i3])
        col_ind.append(pmtr_ct)
        data.append(-zv[idx])
        
        x0[pmtr_ct] = vol[idx]
        pmtr_ct += 1

    data = np.array(data)
    pmtr2vol_mtx = sparse.csr_matrix((data, (row_ind, col_ind)), \
        shape=(edge_len**3, num_pmtr))

    vox_map = (xv+R)*edge_len**2 + (yv+R)*edge_len + (zv+R)
    M_mtx_sparse = sparse.csr_matrix((data, (vox_map[row_ind], col_ind)), \
        shape=(edge_len**3, num_pmtr))

    return (x0, pmtr2vol_mtx, M_mtx_sparse)


def cal_proj_F(num_alpha, tilt_ang, vol, kmax=0.5):

    edge_len = vol.shape[0]
    R = edge_len // 2
    
    i0 = np.arange(edge_len)
    j0 = np.arange(edge_len)
    k0 = np.arange(edge_len)
    i1, j1, k1 = np.meshgrid(i0, j0, k0, indexing='ij')
    
    xv1 = (k1 - R)
    yv1 = (R - j1)
    zv1 = (R - i1)
    fk = np.zeros_like(vol)
    fk[xv1+R, yv1+R, zv1+R] = vol[i1, j1, k1]
    fk = np.array(fk).astype(np.complex128)
    
    polar_coor = make_polar_coor(edge_len, tilt_ang)
    polar_coor = np.transpose(np.squeeze(polar_coor))
    polar_coor[:, 0] *= R/kmax

    num_grid2D = polar_coor.shape[0]
    kx = np.zeros(num_alpha*num_grid2D)
    ky = np.zeros(num_alpha*num_grid2D)
    kz = np.zeros(num_alpha*num_grid2D)
    alpha = np.arange(num_alpha)*(2*np.pi/num_alpha)
    
    rescale = 2*np.pi/edge_len
    for j in range(num_alpha):
        idx_start = j*num_grid2D
        for i in range(num_grid2D):
            idx = idx_start + i
            radi = polar_coor[i, 0]
            theta = polar_coor[i, 1]
            phi = polar_coor[i, 2] - alpha[j]
            kx[idx] = radi*np.sin(theta)*np.cos(phi)*rescale
            ky[idx] = radi*np.sin(theta)*np.sin(phi)*rescale
            kz[idx] = radi*np.cos(theta)*rescale

    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()[support2D]*(R/kmax)
    yv = yv.flatten()[support2D]*(R/kmax)
    xv = np.array(np.round(xv)).astype(np.int)
    yv = np.array(np.round(yv)).astype(np.int)

    pix_map = np.zeros(num_grid2D, dtype=np.int)
    for i in range(num_grid2D):
        row_idx = R - yv[i]
        col_idx = R + xv[i]
        pix_map[i] = row_idx*edge_len + col_idx

    proj_F = np.zeros((num_alpha, edge_len*edge_len), dtype=np.complex128)
    ck = finufft.nufft3d2(kx, ky, kz, fk, eps=1e-15) / edge_len**1.5 
    for j in range(num_alpha):
        idx_min = j*num_grid2D
        idx_max = idx_min + num_grid2D
        proj_F[j][pix_map] = ck[idx_min:idx_max]

    return proj_F


def make_proj_images(proj_F):

    num_alpha = proj_F.shape[0]
    edge_len = np.int(np.round(np.sqrt(proj_F.shape[1])))
    proj_images = np.zeros((num_alpha, edge_len, edge_len))

    for j in range(num_alpha):
        F = np.reshape(proj_F[j], (edge_len, edge_len))
        F = np.fft.ifftshift(F)
        F = np.fft.ifftn(F)
        proj_images[j] = np.real(np.fft.fftshift(F))

    return proj_images


def check_subproblem_config(num_alpha, tilt_ang, vol, sub_grid2D_coor, F_slices):

    edge_len = vol.shape[0]
    R = edge_len // 2
    num_sub_grid2D = sub_grid2D_coor.shape[0]

    pix_map = np.zeros(num_sub_grid2D, dtype=np.int)
    for i in range(num_sub_grid2D):
        row_idx = R - sub_grid2D_coor[i, 1]
        col_idx = R + sub_grid2D_coor[i, 0]
        pix_map[i] = row_idx*edge_len + col_idx

    proj_F = np.zeros((num_alpha, edge_len*edge_len), dtype=np.complex128)
    for j in range(num_alpha):
        idx_min = j*num_sub_grid2D
        idx_max = idx_min + num_sub_grid2D
        proj_F[j][pix_map] = F_slices[idx_min:idx_max]

    proj_images = make_proj_images(proj_F)
    proj_F = cal_proj_F(num_alpha, tilt_ang, vol)
    cpy_proj_images = make_proj_images(proj_F)

    err_F_slices = 0.
    for j in range(num_alpha):
        diff = cpy_proj_images[j] - proj_images[j]
        err_F_slices += np.linalg.norm(diff)/np.linalg.norm(cpy_proj_images[j])
    
    print("error of F_slices = {0:1.3e}".format(err_F_slices/num_alpha))


def make_aux_mtx(edge_len, num_pmtr, kgrid, num_sub_grid2D, \
    sub_grid3D_coor, M_mtx_sparse, sub_sample_list):

    samples_3rd = sub_sample_list[2]
    num_samples_3rd = samples_3rd.shape[0]
    adj_samp_mtx_info = np.zeros((num_sub_grid2D, 4), dtype=np.int32)
    for i in range(num_samples_3rd):
        s1 = samples_3rd[i, 0]
        s2 = samples_3rd[i, 1]
        s3 = samples_3rd[i, 2]
        adj_samp_mtx_info[s1, 0] += 1
        adj_samp_mtx_info[s2, 1] += 1
        adj_samp_mtx_info[s3, 2] += 1

    count = np.sum(adj_samp_mtx_info[0, 0:3])
    for s in range(1, num_sub_grid2D):
        adj_samp_mtx_info[s, 3] = count
        count += np.sum(adj_samp_mtx_info[s, 0:3])

    adj_samp_mtx_idx = np.zeros(count, dtype=np.int32)
    entry_count = np.zeros((num_sub_grid2D, 3), dtype=np.int32)
    for i in range(num_samples_3rd):
        s1 = samples_3rd[i, 0]
        offset = adj_samp_mtx_info[s1, 3]
        c1 = offset + entry_count[s1, 0]
        entry_count[s1, 0] += 1
        adj_samp_mtx_idx[c1] = i

        s2 = samples_3rd[i, 1]
        offset = adj_samp_mtx_info[s2, 3] + adj_samp_mtx_info[s2, 0]
        c2 = offset + entry_count[s2, 1]
        entry_count[s2, 1] += 1
        adj_samp_mtx_idx[c2] = i

        s3 = samples_3rd[i, 2]
        offset = adj_samp_mtx_info[s3, 3] + np.sum(adj_samp_mtx_info[s3, 0:2])
        c3 = offset + entry_count[s3, 2]
        entry_count[s3, 2] += 1
        adj_samp_mtx_idx[c3] = i

    aux_mtx_list = {}
    aux_mtx_list['adj_samp_mtx_info'] = adj_samp_mtx_info
    aux_mtx_list['adj_samp_mtx_idx'] = adj_samp_mtx_idx
    aux_mtx_list['M_mtx_sparse'] = M_mtx_sparse
    aux_mtx_list['adj_M_mtx_sparse'] = M_mtx_sparse.transpose()        
    aux_mtx_list['kgrid'] = kgrid
    aux_mtx_list['num_alpha'] = kgrid.shape[1] // num_sub_grid2D
    aux_mtx_list['edge_len'] = edge_len
    aux_mtx_list['iter_ct'] = 0
    
    kmax = 0.5
    R = edge_len // 2
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    z = np.arange(-R, R+1)
    (xv, yv, zv) = np.meshgrid(x, y, z)
    radi = np.sqrt(xv**2 + yv**2 + zv**2)*(kmax/R)
    aux_mtx_list['radi'] = radi.flatten()

    shape = (edge_len, edge_len, edge_len)
    aux_mtx_list['F_slices'] = np.zeros(kgrid.shape[1], dtype=np.complex128)
    aux_mtx_list['Jvec_3rd'] = np.zeros(2*kgrid.shape[1], dtype=np.complex128)
    aux_mtx_list['Jv1'] = np.zeros(kgrid.shape[1], dtype=np.complex128)
    aux_mtx_list['Jv2'] = np.zeros(kgrid.shape[1], dtype=np.complex128)
    aux_mtx_list['fk1'] = np.zeros(shape, dtype=np.complex128)
    aux_mtx_list['fk2'] = np.zeros(shape, dtype=np.complex128)
    
    return aux_mtx_list


def convert_auto2moments(autofile, num_alpha, sub_grid2D_coor, sub_sample_list):

    fp = np.load(autofile)
    (a1, a2, a3) = (fp['a1'], fp['a2'], fp['a3'])

    # 1st order
    M1 = a1

    # 2nd order
    edge_len = a2.shape[0]
    R = edge_len // 2
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    radi = np.sqrt(xv**2 + yv**2)
    xv = xv[radi < R]
    yv = yv[radi < R]

    F2 = np.fft.ifftshift(a2)
    F2 = np.fft.fftn(F2)
    F2 = np.real(np.fft.fftshift(F2))
    
    samples_2nd = sub_sample_list[1]
    M2 = np.zeros(samples_2nd.shape[0])
    for i in range(samples_2nd.shape[0]):
        s1 = samples_2nd[i, 0]
        M2[i] = F2[xv[s1]+R, yv[s1]+R]

    # 3rd order
    F3 = np.fft.ifftshift(a3)
    F3 = np.fft.fftn(F3)
    F3 = np.fft.fftshift(F3)

    samples_3rd = sub_sample_list[2]
    M3 = np.zeros(samples_3rd.shape[0], dtype=np.complex128)
    for i in range(samples_3rd.shape[0]):
        s1 = samples_3rd[i, 0]
        s2 = samples_3rd[i, 1]
        M3[i] = F3[xv[s1]+R, yv[s1]+R, xv[s2]+R, yv[s2]+R]
    
    return (M1, M2, M3)
