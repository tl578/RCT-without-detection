import sys
sys.path.append('../')
from utils import *
from costgrad import *
from aux import *


def init_gpu_func():

    fp = open("gpu_code_auto.txt", "r")
    code = ""
    for line in fp:
        code += line
    code_pieces = code.split('"""\n')
    fp.close()

    auto2_code = code_pieces[1]
    mod1 = SourceModule(auto2_code)
    cal_auto2 = mod1.get_function("moments_2nd")

    auto3_code = code_pieces[3]
    mod2 = SourceModule(auto3_code)
    cal_auto3 = mod2.get_function("moments_3rd")

    return (cal_auto2, cal_auto3)


def cal_sigma(proj_images, snr):

    num_alpha = proj_images.shape[0]
    edge_len = proj_images.shape[1]
    length = edge_len*edge_len

    norm2 = 0.
    count = 0
    for j in range(num_alpha):
        vec = proj_images[j].flatten()
        vec = np.flip(np.sort(vec))
        thres = 0.99*np.sum(vec)
        
        vsum = 0.
        for i in range(length):
            vsum += vec[i]
            norm2 += vec[i]**2
            count += 1
            if (vsum > thres):
                break

    norm2 /= count
    sigma = np.sqrt(norm2/snr)
    return sigma


def gen_micrograph(mg_len, num_par, proj_images):

    num_alpha = proj_images.shape[0]
    edge_len = proj_images.shape[1]
    img_sz = (edge_len + 1) // 2
    R = img_sz // 2

    ## mask of size (2R+1) by (2R+1)
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    xv, yv = np.meshgrid(x, y)
    radi = np.sqrt(xv**2 + yv**2)
    xv1 = xv[radi <= R]
    yv1 = yv[radi <= R]

    ## mask of size (6R+1) by (6R+1)
    x = np.arange(-3*R, 3*R+1)
    y = np.arange(-3*R, 3*R+1)
    xv, yv = np.meshgrid(x, y)
    radi = np.sqrt(xv**2 + yv**2)
    xv2 = xv[radi <= 3*R]
    yv2 = yv[radi <= 3*R]

    while (1):
        par_ct = 0
        trial_max = num_par * 20
        cx_list = np.random.randint(R, mg_len-R, trial_max)
        cy_list = np.random.randint(R, mg_len-R, trial_max)
        if_visit = np.zeros((mg_len, mg_len), dtype=np.uint8)
        micrograph = np.zeros((mg_len, mg_len), dtype=np.double)
        
        ## mask the border of the micrograph
        idx_min = img_sz-1
        idx_max = mg_len - (img_sz-1)
        if_visit[0:idx_min, :] = 1
        if_visit[:, 0:idx_min] = 1
        if_visit[idx_max:, :] = 1
        if_visit[:, idx_max:] = 1

        for k in range(trial_max):
            if (par_ct == num_par):
                break
            (cx, cy) = (cx_list[k], cy_list[k])
            (tx, ty) = (cx + xv1, cy + yv1)
            if (np.sum(if_visit[tx, ty]) > 0):
                continue
            
            x_min = cx - img_sz 
            x_max = x_min + edge_len
            y_min = cy - img_sz
            y_max = y_min + edge_len
            
#            alpha_idx = par_ct
            alpha_idx = np.random.randint(0, num_alpha)
            micrograph[x_min:x_max, y_min:y_max] = proj_images[alpha_idx]

            (tx, ty) = (cx + xv2, cy + yv2)
            indices = [(tx >= 0) & (tx < mg_len) & (ty >= 0) & (ty < mg_len)]
            indices = tuple(indices)
            (tx, ty) = (tx[indices], ty[indices])
    
            if_visit[tx, ty] = 1
            par_ct += 1
        
        if (par_ct == num_par):
            break

    return (micrograph, par_ct)


def check_autocorr(A2k, A3k, padded_mg, D, N1, mg_len):

    (cx, cy) = (N1//2, N1//2)

    ## 2nd order
    auto2 = np.fft.irfftn(A2k)
    auto2 = auto2[cx-D+1:cx+D, cy-D+1:cy+D]

    a2 = np.zeros_like(auto2)
    m1 = np.copy(padded_mg[D-1:mg_len+D-1, D-1:mg_len+D-1])
    for i0 in range(2*D-1):
        for i1 in range(2*D-1):
            m2 = np.copy(padded_mg[i0:mg_len+i0, i1:mg_len+i1])
            a2[i0, i1] = np.sum(np.multiply(m1, m2))

    print(np.linalg.norm(a2 - auto2)/np.linalg.norm(auto2))

    ## 3rd order
    A3k = np.reshape(A3k, (N1, N1, N1, N1//2 + 1))
    auto3 = np.fft.irfftn(A3k)
    auto3 = auto3[cx-D+1:cx+D, cy-D+1:cy+D, cx-D+1:cx+D, cy-D+1:cy+D]

    a3 = np.zeros_like(auto3)
    for i0 in range(2*D-1):
        for i1 in range(2*D-1):
            m2 = np.copy(padded_mg[i0:mg_len+i0, i1:mg_len+i1])
            for j0 in range(2*D-1):
                for j1 in range(2*D-1):
                    m3 = np.copy(padded_mg[j0:mg_len+j0, j1:mg_len+j1])
                    a3[i0, i1, j0, j1] = np.sum( \
                        np.multiply(m1, np.multiply(m2, m3)))

    print(np.linalg.norm(a3 - auto3)/np.linalg.norm(auto3))


def make_supported_proj_images(num_alpha, tilt_ang, vol, kmax=0.5):

    ## make polar coordinates of the tilted, rotated detector
    edge_len = vol.shape[0]
    (xv, yv, support2D) = make_grid2D(edge_len)
    xv = xv.flatten()
    yv = yv.flatten()
    radi = np.sqrt(xv**2 + yv**2)
    inv_radi = np.zeros_like(radi)
    inv_radi[radi > 1e-10] = 1./radi[radi > 1e-10]

    num_tilts = tilt_ang.shape[0]
    polar_coor = np.zeros((num_tilts, 3, edge_len**2))
    for r in range(num_tilts):
        polar_coor[r, 0, :] = radi
        theta = np.arccos(np.sin(tilt_ang[r])*np.multiply(yv, inv_radi))
        polar_coor[r, 1, :] = theta
        beta = np.angle(-np.cos(tilt_ang[r])*yv - 1j*xv)
        polar_coor[r, 2, :] = beta

    R = edge_len // 2
    polar_coor = np.transpose(np.squeeze(polar_coor))
    polar_coor[:, 0] *= R/kmax

    kx = np.zeros(num_alpha*edge_len**2)
    ky = np.zeros(num_alpha*edge_len**2)
    kz = np.zeros(num_alpha*edge_len**2)
    alpha = np.arange(num_alpha)*(2*np.pi/num_alpha)
    
    rescale = 2*np.pi/edge_len
    for j in range(num_alpha):
        idx_start = j*edge_len**2
        for i in range(edge_len**2):
            idx = idx_start + i
            radi = polar_coor[i, 0]
            theta = polar_coor[i, 1]
            phi = polar_coor[i, 2] - alpha[j]
            kx[idx] = radi*np.sin(theta)*np.cos(phi)*rescale
            ky[idx] = radi*np.sin(theta)*np.sin(phi)*rescale
            kz[idx] = radi*np.cos(theta)*rescale

    ## change coordinates from ijk to xyz
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

    ## calculate the projection images
    iflag = -1
    eps = 1e-15
    ck = np.zeros(kx.shape[0], dtype=np.complex128)
    ck = finufft.nufft3d2(kx, ky, kz, fk, eps=1e-15) / edge_len**1.5 

    xv = np.array(np.round(xv*(R/kmax))).astype(np.int)
    yv = np.array(np.round(yv*(R/kmax))).astype(np.int)
    pix_map = np.zeros(edge_len**2, dtype=np.int)
    for i in range(edge_len**2):
        row_idx = R - yv[i]
        col_idx = R + xv[i]
        pix_map[i] = row_idx*edge_len + col_idx

    proj_F = np.zeros((num_alpha, edge_len**2), dtype=np.complex128)
    for j in range(num_alpha):
        idx_min = j*edge_len**2
        idx_max = idx_min + edge_len**2
        proj_F[j][pix_map] = ck[idx_min:idx_max]

    proj_images = np.zeros((num_alpha, edge_len, edge_len))
    for j in range(num_alpha):
        F = np.reshape(proj_F[j], (edge_len, edge_len))
        F = np.fft.ifftshift(F)
        F = np.fft.ifftn(F)
        proj_images[j] = np.real(np.fft.fftshift(F))

    return (proj_images, ck)
