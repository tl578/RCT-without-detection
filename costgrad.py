from utils import *
from aux import *
import pycuda.autoinit
import pycuda.cuda as cuda
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import time


def setup_gpu_func(src_code):

    fp = open(src_code, "r")
    code = ""
    for line in fp:
        code += line
    code_pieces = code.split('"""\n')
    fp.close()
    
    gpu_func = {}
    
    res3_code = code_pieces[1]
    mod1 = SourceModule(res3_code)
    cal_res_3rd = mod1.get_function("cal_res_3rd")
    gpu_func['cal_res_3rd'] = cal_res_3rd

    Jvec3_code = code_pieces[3]
    mod2 = SourceModule(Jvec3_code)
    cal_Jvec_3rd = mod2.get_function("cal_Jvec_3rd")
    gpu_func['cal_Jvec_3rd'] = cal_Jvec_3rd

    return gpu_func


def setup_gpu_arr(edge_len, num_alpha, num_pmtr, num_sub_grid2D, \
    num_sub_grid3D, aux_mtx_list, sub_sample_list):

    gpu_arrays = {}
    
    adj_samp_mtx_info = aux_mtx_list['adj_samp_mtx_info'].flatten()
    adj_samp_mtx_info_gpu = gpuarray.to_gpu(adj_samp_mtx_info)
    gpu_arrays['adj_samp_mtx_info'] = adj_samp_mtx_info_gpu

    adj_samp_mtx_idx = aux_mtx_list['adj_samp_mtx_idx']
    adj_samp_mtx_idx_gpu = gpuarray.to_gpu(adj_samp_mtx_idx)
    gpu_arrays['adj_samp_mtx_idx'] = adj_samp_mtx_idx_gpu

    samples_3rd = sub_sample_list[2].flatten()
    samples_3rd = np.array(samples_3rd).astype(np.int32)
    num_samples_3rd = samples_3rd.shape[0] // 3
    samples_3rd_gpu = gpuarray.to_gpu(samples_3rd)
    gpu_arrays['samples_3rd'] = samples_3rd_gpu

    gpu_arrays['F_slices'] = \
        gpuarray.zeros(num_alpha*num_sub_grid2D, dtype=np.complex128)
    gpu_arrays['moment_3rd'] = \
        gpuarray.zeros(num_samples_3rd, dtype=np.complex128)
    gpu_arrays['residual_3rd'] = \
        gpuarray.zeros(num_samples_3rd, dtype=np.complex128)
    gpu_arrays['weight_3rd'] = \
        gpuarray.to_gpu(np.ones(num_samples_3rd, dtype=np.double))
    gpu_arrays['Jvec_3rd'] = \
        gpuarray.zeros(2*num_alpha*num_sub_grid2D, dtype=np.complex128)
    
    info = [num_alpha, num_samples_3rd, \
        num_sub_grid2D, num_sub_grid3D, num_pmtr, edge_len]
    info = np.array(info).astype(np.int32)
    info_gpu = gpuarray.to_gpu(info)
    gpu_arrays['info'] = info_gpu

    return gpu_arrays


def my_F_slices(x_vec, gpu_arrays, aux_mtx_list, vec_lib, plan1, plan2):

    if_F_slices = 0
    if ('F_slices_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['F_slices_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_F_slices = 1

    if (if_F_slices == 1):
        return aux_mtx_list['F_slices']
    
    vec_lib['F_slices_x'] = np.copy(x_vec)
    edge_len = aux_mtx_list['edge_len']
    fk = aux_mtx_list['M_mtx_sparse'].dot(x_vec)
    fk = np.reshape(fk, (edge_len, edge_len, edge_len))
    fk = fk.astype(np.complex128)
    
    ck = aux_mtx_list['F_slices']
    ck = plan2.execute(fk) / edge_len**1.5
    aux_mtx_list['F_slices'] = ck

    return ck


def my_residual(x_vec, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2):

    t1 = time.time()
    if_residual = 0
    if ('residual_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['residual_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_residual = 1

    if (if_residual == 1):
        return
    
    if_F_slices = 0
    if ('F_slices_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['F_slices_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_F_slices = 1

    if (if_F_slices == 0):
        F_slices = my_F_slices(x_vec, gpu_arrays, \
            aux_mtx_list, vec_lib, plan1, plan2)

    t2 = time.time()
    vec_lib['residual_x'] = np.copy(x_vec)
    F_slices_gpu = gpuarray.to_gpu(aux_mtx_list['F_slices'])
    moment_3rd_gpu = gpu_arrays['moment_3rd']
    residual_3rd_gpu = gpu_arrays['residual_3rd']
    weight_3rd_gpu = gpu_arrays['weight_3rd']
    samples_3rd_gpu = gpu_arrays['samples_3rd']
    info_gpu = gpu_arrays['info']
    cal_res_3rd = gpu_func['cal_res_3rd']
    
    (blockdim_x, griddim_x) = (256, 256)
    cal_res_3rd(F_slices_gpu, residual_3rd_gpu, moment_3rd_gpu, \
        weight_3rd_gpu, samples_3rd_gpu, info_gpu, \
        block=(blockdim_x, 1, 1), grid=(griddim_x, 1, 1))

    t3 = time.time()

    return


def my_cost(x_vec, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2):

    t1 = time.time()
    if_residual = 0
    if ('residual_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['residual_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_residual = 1

    if (if_residual == 0):
        my_residual(x_vec, gpu_arrays, gpu_func, \
            aux_mtx_list, vec_lib, plan1, plan2)

    t2 = time.time()
    num_alpha = aux_mtx_list['num_alpha']
    residual_3rd = gpu_arrays['residual_3rd'].get()
    cost0 = 0.5*np.linalg.norm(residual_3rd)**2

    ## regularization
    t3 = time.time()
    reg_weight = aux_mtx_list['reg_weight']
    cost2 = cal_cost2(x_vec, aux_mtx_list) * reg_weight

    cost_3rd = cost0 + cost2
    
    ## optimization statistics
    t4 = time.time()
    aux_mtx_list['iter_ct'] += 1

    t5 = time.time()
    out_str = "iter = {0:05d}: cost = {1:1.5e}, ".\
        format(aux_mtx_list['iter_ct'], cost_3rd)

    out_str = "{0:s}cost0 = {1:1.5e}, cost1 = {2:1.5e}, ".\
        format(out_str, cost0, cost2)

    if ('grad' in vec_lib.keys()):
        grad_3rd = vec_lib['grad']
        grad_norm = np.linalg.norm(grad_3rd)
        out_str = "{0:s}grad_norm = {1:1.5e}".format(out_str, grad_norm)
    
    out_str = "{0:s}\ncost: res = {1:1.2e}s, cost0 = {2:1.2e}s, ".\
        format(out_str, t2-t1, t3-t2)
    out_str = "{0:s}cost2 = {1:1.2e}s, output = {2:1.2e}s, total = {3:1.2e}s".\
        format(out_str, t4-t3, t5-t4, t5-t1)
    print(out_str)

    return cost_3rd


def my_grad(x_vec, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2):

    t1 = time.time()
    if_grad = 0
    if ('grad_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['grad_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_grad = 1

    if (if_grad == 1):
        grad_3rd = vec_lib['grad']
        return grad_3rd

    flag = 0
    if ('if_hessp_grad1' not in vec_lib.keys()):
        flag = 1
    elif (vec_lib['if_hessp_grad1'] == 0):
        flag = 1

    if_residual = 0
    if ('residual_x' in vec_lib.keys()):
        diff = x_vec - vec_lib['residual_x']
        if (np.linalg.norm(diff)/np.linalg.norm(x_vec) < 1e-14):
            if_residual = 1

    if (if_residual == 0):
        my_residual(x_vec, gpu_arrays, gpu_func, \
            aux_mtx_list, vec_lib, plan1, plan2)

    ## calculate Jvec
    t2 = time.time()
    F_slices_gpu = gpuarray.to_gpu(aux_mtx_list['F_slices'])
    residual_3rd_gpu = gpu_arrays['residual_3rd']
    Jvec_3rd_gpu = gpu_arrays['Jvec_3rd']
    weight_3rd_gpu = gpu_arrays['weight_3rd']
    samples_3rd_gpu = gpu_arrays['samples_3rd']
    adj_samp_mtx_info_gpu = gpu_arrays['adj_samp_mtx_info']
    adj_samp_mtx_idx_gpu = gpu_arrays['adj_samp_mtx_idx']
    info_gpu = gpu_arrays['info']
    cal_Jvec_3rd = gpu_func['cal_Jvec_3rd']
    
    (blockdim_x, griddim_x) = (256, 256)
    cal_Jvec_3rd(F_slices_gpu, residual_3rd_gpu, Jvec_3rd_gpu, weight_3rd_gpu, \
        adj_samp_mtx_info_gpu, adj_samp_mtx_idx_gpu, samples_3rd_gpu, info_gpu, \
        block=(blockdim_x, 1, 1), grid=(griddim_x, 1, 1))

    ## calculate pvec
    t3 = time.time()
    edge_len = aux_mtx_list['edge_len']
    num_alpha = aux_mtx_list['num_alpha']
    shape = (edge_len, edge_len, edge_len)
    fk1 = np.zeros(shape, dtype=np.complex128)
    fk2 = np.zeros(shape, dtype=np.complex128)

    Jvec_3rd = Jvec_3rd_gpu.get()
    length = Jvec_3rd.shape[0] // 2
    Jvec_3rd = np.reshape(Jvec_3rd, (length, 2))
    fk1 = plan1.execute(Jvec_3rd[:, 0]) / edge_len**1.5
    fk2 = plan1.execute(Jvec_3rd[:, 1]) / edge_len**1.5
    
    ## calculate grad_3rd
    t4 = time.time()
    adj_M_mtx_sparse = aux_mtx_list['adj_M_mtx_sparse']
    grad0 = np.real(adj_M_mtx_sparse.dot(fk1.flatten())) \
        - np.imag(adj_M_mtx_sparse.dot(fk2.flatten()))

    ## regularization
    t5 = time.time()
    reg_weight = aux_mtx_list['reg_weight']
    grad2 = cal_grad2(x_vec, aux_mtx_list) * reg_weight
    
    grad_3rd = grad0 + grad2

    if (flag == 1):
        vec_lib['grad_x'] = np.copy(x_vec)
        vec_lib['grad'] = grad_3rd

    t6 = time.time()
    out_str = "grad: res = {0:1.2e}s, Jvec = {1:1.2e}s, ".format(t2-t1, t3-t2)
    out_str = "{0:s}pvec = {1:1.2e}s, grad0 = {2:1.2e}s, ".\
        format(out_str, t4-t3, t5-t4)
    out_str = "{0:s}grad2 = {1:1.2e}s, total = {2:1.2e}s".\
        format(out_str, t6-t5, t6-t1)
    print(out_str)

    return grad_3rd


def make_obs_moments(num_alpha, num_sub_grid2D, x0, sub_sample_list, \
    gpu_arrays, gpu_func, aux_mtx_list, plan1, plan2):

    vec_lib = {}
    F_slices = my_F_slices(x0, gpu_arrays, aux_mtx_list, vec_lib, plan1, plan2)

    # 1st order
    M1 = 0j
    samples_1st = sub_sample_list[0]
    for j in range(num_alpha):
        s1 = j*num_sub_grid2D + samples_1st
        M1 += F_slices[s1]
    
    # 2nd order
    samples_2nd = sub_sample_list[1]
    num_samples_2nd = samples_2nd.shape[0]
    M2 = np.zeros(num_samples_2nd, dtype=np.complex128)
    for j in range(num_alpha):
        s0 = j*num_sub_grid2D
        for i in range(num_samples_2nd):
            s1 = s0 + samples_2nd[i, 0]
            s2 = s0 + samples_2nd[i, 1]
            M2[i] += F_slices[s1]*F_slices[s2]

    # 3rd order
    my_residual(x0, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
    
    M1 = np.real(M1)
    M2 = np.real(M2)
    M3 = gpu_arrays['residual_3rd'].get()
    
    return (M1, M2, M3)


def check_costgrad(num_pmtr, gpu_arrays, \
    gpu_func, aux_mtx_list, vec_lib, plan1, plan2):

    t_vals = np.linspace(-10, 0, 41)
    error = np.zeros_like(t_vals)
    y0 = np.random.randn(num_pmtr)
    dy = np.random.randn(num_pmtr)
    
    f0 = my_cost(y0, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
    g_y = my_grad(y0, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
    
    for j in range(len(t_vals)):
        t = 10**t_vals[j]
        y1 = y0 + t*dy
        f1 = my_cost(y1, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
        df = f1 - f0 - t*np.dot(g_y, dy)
        error[j] = np.log10(np.abs(df)/f0)
    
    y_vals = 2*t_vals + (error[-1] - 1)
    return (t_vals, error, y_vals)


def estimate_ave_intens(edge_len, num_alpha, \
    samples_2nd, moment_2nd, sub_grid2D_coor):

    R = edge_len // 2
    ave_intens = np.zeros(R+1)
    inter_weight = np.zeros(R+1)
    num_samples_2nd = samples_2nd.shape[0]

    for i in range(num_samples_2nd):
        val = moment_2nd[i]/num_alpha
        s = samples_2nd[i, 0]
        xv = sub_grid2D_coor[s, 0]
        yv = sub_grid2D_coor[s, 1]
        x = np.sqrt(xv**2 + yv**2)
        tx = np.int(np.round(x))
        ave_intens[tx] += val
        inter_weight[tx] += 1.

    for i in range(R+1):
        if (inter_weight[i] > 0.):
            ave_intens[i] /= inter_weight[i]

    return ave_intens


def make_mat_radial_ave(edge_len):

    R = edge_len // 2
    rmax = np.int(np.round(np.sqrt(3)*R))
    w_arr = np.zeros(rmax+1)

    row_ind = []
    col_ind = []
    data = []

    count = 0
    for i in range(edge_len):
        for j in range(edge_len):
            for k in range(edge_len):
                x = i - R
                y = j - R
                z = k - R
                r = np.sqrt(x**2 + y**2 + z**2)

                tr = np.int(np.round(r))
                row_ind.append(tr)
                col_ind.append(count)
                data.append(1.)

                w_arr[tr] += 1
                count += 1

    data = np.array(data) / edge_len**3
    for i in range(len(row_ind)):
        data[i] /= w_arr[row_ind[i]]

    mat_radial_ave = sparse.csr_matrix((data, (row_ind, col_ind)), \
        shape=(rmax+1, edge_len**3))

    return (mat_radial_ave, w_arr)


def setup_prior_reg(aux_mtx_list):

    ave_intens = aux_mtx_list['ave_intens']
    w_arr = aux_mtx_list['w_arr']
    edge_len = aux_mtx_list['edge_len']
    
    R = edge_len // 2
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    z = np.arange(-R, R+1)
    (xv, yv, zv) = np.meshgrid(x, y, z)
    radi = np.sqrt(xv**2 + yv**2 + zv**2)
    radi = np.array(np.round(radi.flatten())).astype(np.int)

    prior_weight = np.zeros(edge_len**3)
    for r in range(R+1):
        prior_weight[radi == r] = 1/ave_intens[r]

    rescale = 1/(ave_intens[R]*R**2)
    rmax = np.max(radi)
    for r in range(R+1, rmax+1):
         prior_weight[radi == r] = r**2*rescale

    prior_weight /= R**3
    aux_mtx_list['prior_weight'] = prior_weight
    return


def cal_cost2(x, aux_mtx_list):

    edge_len = aux_mtx_list['edge_len']
    M_mtx_sparse = aux_mtx_list['M_mtx_sparse']
    prior_weight = aux_mtx_list['prior_weight']
    
    fk = M_mtx_sparse.dot(x)
    fk = np.reshape(fk, (edge_len, edge_len, edge_len))
    F = np.fft.fftshift(np.fft.fftn(fk))
    F = F.flatten() / edge_len**1.5
    
    u_arr = np.multiply(prior_weight, F)
    cost2 = 0.5*np.dot(np.conj(F), u_arr) / edge_len**3
    cost2 = np.real(cost2)
    
    return cost2


def cal_grad2(x, aux_mtx_list):
    
    edge_len = aux_mtx_list['edge_len']
    M_mtx_sparse = aux_mtx_list['M_mtx_sparse']
    adj_M_mtx_sparse = aux_mtx_list['adj_M_mtx_sparse']
    prior_weight = aux_mtx_list['prior_weight']

    fk = M_mtx_sparse.dot(x)
    fk = np.reshape(fk, (edge_len, edge_len, edge_len))
    F = np.fft.fftshift(np.fft.fftn(fk))
    F = F.flatten() / edge_len**1.5

    u_arr = np.multiply(prior_weight, F)
    U = np.reshape(u_arr, (edge_len, edge_len, edge_len))
    F = np.fft.ifftn(np.fft.ifftshift(U))
    F = F.flatten() * edge_len**1.5
    grad2 = adj_M_mtx_sparse.dot(F) / edge_len**3
    grad2 = np.real(grad2)

    return grad2
