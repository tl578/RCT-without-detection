import matplotlib.pyplot as plt
import skcuda.fft as cu_fft
from modules import *
import time

mg_len = 4096
num_mg = 500
num_par = 400
snr = 1

data_dir = "../Data"
blockdim_x = 1024
griddim_x = 1024

num_alpha = 400
tilt_ang = np.ones(1)*(np.pi/180)*60
mapfile = "{0:s}/BPTI-5A.mrc".format(data_dir)

vol = make_volume(mapfile)
proj_images, ck = make_supported_proj_images(num_alpha, tilt_ang, vol)
np.savez("proj-images", proj_images=proj_images)

out_str = "mg_len = {0:d}, num_mg = {1:d}".format(mg_len, num_mg)
out_str = "{0:s}, num_par = {1:d}\n".format(out_str, num_par)
if (len(sys.argv) == 2 and sys.argv[1] == "clean"):
    sigma = 0.
    out_str = "{0:s}sigma = 0.\n".format(out_str)
else:
    sigma = cal_sigma(proj_images, snr)
    out_str = "{0:s}sigma = {1:1.3e}, snr = {2:1.3e}\n".format(out_str, sigma, snr)

out_str = "{0:s}edge_len = {1:d}, tilt_ang = {2:.1f} deg\n".\
    format(out_str, vol.shape[0], tilt_ang[0]*180/np.pi)
out_str = "{0:s}blockdim_x = {1:d}, griddim_x = {2:d}\n".\
    format(out_str, blockdim_x, griddim_x)
print(out_str)

#### --------------------------------------------------------------------

## setup problem configuration
edge_len = proj_images.shape[1]
D = (edge_len + 1) // 2
patch_sz = 4*(D-1)
M = mg_len // patch_sz
if (mg_len % patch_sz != 0):
    M += 1

# zero-pad the micrograph
N0 = M*patch_sz + 2*(D-1)
padded_mg = np.zeros((N0, N0), np.float64)

# overlapping patches excerpted from the micrograph
N1 = patch_sz + 4*(D-1)
x_cpu = np.zeros((2*M**2, N1, N1), np.float64)

# FFT of the patches
xf_gpu = gpuarray.empty((2*M**2, N1, N1//2 + 1), np.complex128)

# arrays that store the autocorrelations
auto1 = 0.
A2k = np.zeros((N1, N1//2 + 1), np.complex128)
A2k_gpu = gpuarray.zeros((N1, N1//2 + 1), np.complex128)
m3_count = N1**3*(N1//2 + 1)
A3k = np.zeros(m3_count, np.complex128)

D3 = N1**2*(N1//2 + 1)
block_sz = 2 * 1024**3 // (16*D3) + 1
pA3k_gpu = gpuarray.zeros(block_sz*D3, np.complex128)
num_outer_iter = N1 // block_sz
if (N1 % block_sz != 0):
    num_outer_iter += 1

print("num_outer_iter = {0:d}".format(num_outer_iter))
print("M = {0:d}".format(M))

# initialize gpu functions that compute the autocorrelations
(cal_auto2, cal_auto3) = init_gpu_func()

t00 = time.time()

## calculate autocorrelations of the micrographs
total_par_ct = 0
for mg_ct in range(num_mg):
     
    t0 = time.time()
    
    idx_min = D-1
    idx_max = idx_min + mg_len
    (padded_mg[idx_min:idx_max, idx_min:idx_max], cur_par_ct) \
        = gen_micrograph(mg_len, num_par, proj_images)
    total_par_ct += cur_par_ct

    idx_min += D-1
    idx_max -= D-1
    padded_mg[idx_min:idx_max, idx_min:idx_max] \
        += sigma*np.random.normal(size=(mg_len-2*(D-1), mg_len-2*(D-1)))

    patch_ct = 0
    for i in range(M):
        for j in range(M):
            (i0, i1) = (i*patch_sz, (i+1)*patch_sz + 2*(D-1))
            (j0, j1) = (j*patch_sz, (j+1)*patch_sz + 2*(D-1))
            (k0, k1) = (D-1, patch_sz + 3*(D-1))
            x_cpu[2*patch_ct, k0:k1, k0:k1] = padded_mg[i0:i1, j0:j1]
            
            (k0, k1) = (2*(D-1), patch_sz + 2*(D-1))
            x_cpu[2*patch_ct+1, k0:k1, k0:k1] = \
                x_cpu[2*patch_ct, k0:k1, k0:k1]
            patch_ct += 1

    t1 = time.time()

    # take FFT of each patch
    x_gpu = gpuarray.to_gpu(x_cpu)
    plan_forward = cu_fft.Plan((N1, N1), \
        np.float64, np.complex128, 2*M**2)
    cu_fft.fft(x_gpu, xf_gpu, plan_forward)
    x_gpu.gpudata.free()

    t2 = time.time()

    # 1st order
    auto1 += np.sum(padded_mg)

    # 2nd order
    info = np.array([patch_ct, N1]).astype(np.int32)
    info_gpu = gpuarray.to_gpu(info)
    A2k_gpu.fill(0j)
    cal_auto2(xf_gpu, A2k_gpu, info_gpu, \
        block=(blockdim_x, 1, 1), grid=(griddim_x, 1, 1))
    A2k += A2k_gpu.get()
    info_gpu.gpudata.free()

    t3 = time.time()
    
    # 3rd order
    dt1 = 0.
    dt2 = 0.
    for k in range(num_outer_iter):
        t4 = time.time()
        k1x_offset = k*block_sz
        info = np.array([patch_ct, N1, k, block_sz]).astype(np.int32)
        info_gpu = gpuarray.to_gpu(info)
        pA3k_gpu.fill(0j)
        cal_auto3(xf_gpu, pA3k_gpu, info_gpu, \
            block=(blockdim_x, 1, 1), grid=(griddim_x, 1, 1))
        info_gpu.gpudata.free()
        
        t5 = time.time()
        dt1 += t5 - t4
    
        idx_min = k*block_sz*D3
        idx_max = (k+1)*block_sz*D3
        if (idx_max > m3_count):
            idx_max = m3_count
        pA3k = pA3k_gpu.get()
        A3k[idx_min:idx_max] += pA3k[0:idx_max-idx_min]
 
        t6 = time.time()
        dt2 += t6 - t5
     
    out_arr = "Micrograph: {0:1.3e} sec".format(t1-t0)
    out_arr = "{0:s}, FFT: {1:1.3e} sec".format(out_arr, t2-t1)
    out_arr = "{0:s}, auto2: {1:1.3e} sec".format(out_arr, t3-t2)
    out_arr = "{0:s}, auto3: {1:1.3e} sec".format(out_arr, dt1)
    out_arr = "{0:s}, data transfer: {1:1.3e} sec".format(out_arr, dt2)
    print(out_arr)

#    check_autocorr(A2k, A3k, padded_mg, D, N1, mg_len)

    if (mg_ct+1 == 1 or mg_ct+1 == 10 or (mg_ct+1) % 100 == 0):

        t01 = time.time()
        (cx, cy) = (N1//2, N1//2)
        auto2 = np.fft.irfftn(A2k)
        auto2 = auto2[cx-D+1:cx+D, cy-D+1:cy+D]
        
        A3k = np.reshape(A3k, (N1, N1, N1, N1//2 + 1))
        auto3 = np.fft.irfftn(A3k)
        auto3 = auto3[cx-D+1:cx+D, cy-D+1:cy+D, cx-D+1:cx+D, cy-D+1:cy+D]
        A3k = A3k.flatten()
        
        outfile = "{0:s}/autocorr_snr-{1:1.3e}_run-{2:05d}-60deg".format(data_dir, snr, mg_ct+1)
        np.savez(outfile, a1=auto1, a2=auto2, a3=auto3, total_par_ct=total_par_ct)

        t02 = time.time()
        print("\ntotal particle count = {0:d}".format(total_par_ct))
        print("output file takes {0:.0f} sec".format(t02-t01))
        (free,total)=drv.mem_get_info()
        print("Free mem: {0:1.3e} GB, total mem: {1:1.3e} GB\n".\
            format(free/2.**30, total/2.**30))
