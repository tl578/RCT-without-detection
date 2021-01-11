import matplotlib.pyplot as plt
from utils import *
from aux import *
from costgrad import *
from tf_utils import *

num_alpha = 50
tilt_ang = np.ones(1)*(np.pi/180)*60
mapfile = "Data/BPTI-5A.mrc"
autofile = "estimate-autocorr/noisy-auto.npz"

#### --------------------------------------------------------------------
#### initialization
#### --------------------------------------------------------------------

vol = make_volume(mapfile)
edge_len = vol.shape[0]
R = edge_len // 2

sample_list = enumerate_samples(edge_len)
sorted_sample_list = sort_sample_list(edge_len, sample_list)

src_code = "gpu_code_3rd.txt"
gpu_func = setup_gpu_func(src_code)

#### --------------------------------------------------------------------
#### setup optimization problem
#### --------------------------------------------------------------------

r_limit = R
(sub_grid2D, sub_grid2D_coor, sub_grid3D, sub_grid3D_coor) = \
    make_subproblem_config(edge_len, r_limit)
sub_sample_list = make_sub_sample_list(sorted_sample_list, sub_grid2D)
kgrid = make_kgrid(edge_len, tilt_ang, num_alpha, sub_grid2D)
(x0, pmtr2vol_mtx, M_mtx_sparse) = make_nonredundant_variables(vol)

num_pmtr = x0.shape[0]
num_sub_grid2D = sub_grid2D.shape[0]
num_sub_grid3D = sub_grid3D.shape[0]
aux_mtx_list = make_aux_mtx(edge_len, num_pmtr, kgrid, \
    num_sub_grid2D, sub_grid3D_coor, M_mtx_sparse, sub_sample_list)
gpu_arrays = setup_gpu_arr(edge_len, num_alpha, num_pmtr, \
    num_sub_grid2D, num_sub_grid3D, aux_mtx_list, sub_sample_list)

## setup nufft plans
nufft_tol = 1e-6
shape = (edge_len, edge_len, edge_len)
plan1 = finufft.Plan(1, shape, eps=nufft_tol)
plan1.setpts(kgrid[0, :], kgrid[1, :], kgrid[2, :])
plan2 = finufft.Plan(2, shape, eps=nufft_tol)
plan2.setpts(kgrid[0, :], kgrid[1, :], kgrid[2, :])

## calculate truncation error of moments
(M1, M2, M3) = make_obs_moments(num_alpha, num_sub_grid2D, x0, \
    sub_sample_list, gpu_arrays, gpu_func, aux_mtx_list, plan1, plan2)
(m1, m2, m3) = convert_auto2moments(autofile, \
    num_alpha, sub_grid2D_coor, sub_sample_list)

m1 *= num_alpha
m2 *= num_alpha
m3 *= num_alpha
gpu_arrays['moment_3rd'] = gpuarray.to_gpu(m3)

## calculate radial average of intensity
samples_2nd = sub_sample_list[1]
ave_intens = estimate_ave_intens(edge_len, num_alpha, \
    samples_2nd, m2, sub_grid2D_coor)
(mat_radial_ave, w_arr) = make_mat_radial_ave(edge_len)
aux_mtx_list['ave_intens'] = ave_intens
aux_mtx_list['mat_radial_ave'] = mat_radial_ave
aux_mtx_list['w_arr'] = w_arr

setup_prior_reg(aux_mtx_list)

#### --------------------------------------------------------------------

## check consistency between vol & nonredundant variables
recon_vol = np.reshape(pmtr2vol_mtx.dot(x0), (edge_len, edge_len, edge_len))
error = np.linalg.norm(vol - recon_vol)/np.linalg.norm(vol)
print("error of nonredundant variables = {0:1.3e}".format(error))

## sanity check for F_slices
vec_lib = {}
F_slices = my_F_slices(x0, gpu_arrays, aux_mtx_list, vec_lib, plan1, plan2)
check_subproblem_config(num_alpha, tilt_ang, vol, sub_grid2D_coor, F_slices)

## sanity check of moments
err1 = np.linalg.norm(M2-m2)/np.linalg.norm(m2)
err2 = np.linalg.norm(M3-m3)/np.linalg.norm(m3)
print("error of M2 / M3 = {0:1.3e} / {1:1.3e}".format(err1, err2))

## sanity check for ave_intens
F = np.fft.fftshift(np.fft.fftn(vol))
F = F.flatten()
ave_intens_true = mat_radial_ave.dot(np.abs(F)**2)

## check consistency between cost & grad
vec_lib = {}
vec_lib['x0'] = x0
aux_mtx_list['reg_weight'] = 1.
(t_vals, error, y_vals) = check_costgrad(num_pmtr, \
    gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)

outfile = "check.npz"
np.savez(outfile, n0=ave_intens_true.shape[0], y0=ave_intens_true, \
    n1=ave_intens.shape[0], y1=ave_intens, t_vals=t_vals, \
    error=error, y_vals=y_vals)

## calculate cost & grad at ground truth
vec_lib = {}
vec_lib['x0'] = x0
aux_mtx_list['iter_ct'] = 0
aux_mtx_list['reg_weight'] = 1.
print("\nAt ground truth:")
cost = my_cost(x0, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
grad = my_grad(x0, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
print("grad_norm = {0:1.5e}".format(np.linalg.norm(grad)))
print("edge_len = {0:d}".format(edge_len))
print("length of m2 and m3: ({0:d}, {1:d})\n".format(m2.shape[0], m3.shape[0]))
print("tilt angle = {0:.0f}, num_pmtr = {1:d}".format(tilt_ang[0]*180/np.pi, num_pmtr))

## make inital guess
x = np.arange(-R, R+1)
y = np.arange(-R, R+1)
z = np.arange(-R, R+1)
(xv, yv, zv) = np.meshgrid(x, y, z)
radi = np.sqrt(xv**2 + yv**2 + zv**2)
D = (edge_len + 1) // 2

vol_init = np.exp(-0.5*radi**2/(D//2)**2)
vol_init[radi >= D//2] = 0.
(x_init, pmtr2vol_mtx, M_mtx_sparse) = make_nonredundant_variables(vol_init)
(M1, M2, M3) = make_obs_moments(num_alpha, num_sub_grid2D, x_init, \
    sub_sample_list, gpu_arrays, gpu_func, aux_mtx_list, plan1, plan2)
x_init *= m1/M1

opt_tol = 1e-7
max_iter = 3000
print("opt_tol = {0:1.2e}, max_iter = {1:d}".format(opt_tol, max_iter))
print(autofile)

#### --------------------------------------------------------------------

@tf.function
def tf_costgrad(x):
    return tf_costgrad_(x, gpu_arrays=gpu_arrays, gpu_func=gpu_func, \
        aux_mtx_list=aux_mtx_list, vec_lib=vec_lib, plan1=plan1, plan2=plan2)

@tf.function
def costgrad_with_bfgs():
    return tfp.optimizer.bfgs_minimize( \
        tf_costgrad, initial_position=tf.constant(x1), parallel_iterations=1, \
        f_relative_tolerance=opt_tol, max_iterations=max_iter)

@tf.function
def costgrad_with_lbfgs():
    return tfp.optimizer.lbfgs_minimize( \
        tf_costgrad, initial_position=tf.constant(x1), parallel_iterations=1, \
        f_relative_tolerance=opt_tol, max_iterations=max_iter)

#### --------------------------------------------------------------------

vec_lib = {}
vec_lib['x0'] = x0
x1 = np.copy(x_init)

reg_weight = 1
aux_mtx_list['reg_weight'] = reg_weight
aux_mtx_list['iter_ct'] = 0

trial_ct = 10
for k in range(trial_ct):

    t1 = time.time()
    print('-'*80)

    reg_weight = 10**(-k+7)
    aux_mtx_list['reg_weight'] = reg_weight
    aux_mtx_list['iter_ct'] = 0
    
    result = costgrad_with_bfgs()

    t2 = time.time()
    print("optimization takes {0:1.5e} sec".format(t2-t1))
    print("reg_weight = {0:1.5e}".format(reg_weight))
    
    outfile = "Data/result-{0:d}.npz".format(k)
    x2 = np.array(result.position)
    np.savez(outfile, x0=x0, x1=x1, sol=x2, reg_weight=reg_weight)

#### --------------------------------------------------------------------

(free,total)=drv.mem_get_info()
print("\nFree mem: {0:1.3e} GB, total mem: {1:1.3e} GB\n".\
    format(free/2.**30, total/2.**30))
