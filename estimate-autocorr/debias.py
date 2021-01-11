import matplotlib.pyplot as plt
from scipy.optimize import *
import numpy as np


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


def debias_2nd(a2_est, bias):

    edge_len = a2_est.shape[1]
    total_len = edge_len**2
    
    D = (edge_len + 1) // 2
    N = D-1
    N2 = N**2

    out_of_support = []
    in_support = []
    a2_est[N, N] -= bias

    for idx in range(total_len):
        i1 = idx // edge_len
        j1 = idx % edge_len
        if ((i1 - N)**2 + (j1 - N)**2 >= N2):
            out_of_support.append(idx)
        else:
            in_support.append(idx)

    return (a2_est, out_of_support, in_support)


def debias_3rd(a3_est, bias):
    
    edge_len = a3_est.shape[0]
    edge_len2 = edge_len**2
    edge_len3 = edge_len**3
    total_len = edge_len**4
    
    D = (edge_len + 1) // 2
    N = D - 1
    N2 = N**2

    out_of_support = []
    in_support = []
    
    for idx in range(total_len):
        x1 = idx // edge_len3 - N
        y1 = (idx % edge_len3) // edge_len2 - N
        x2 = (idx % edge_len2) // edge_len - N
        y2 = idx % edge_len - N
        if (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
            a3_est[N, N, N, N] -= 3*bias
        elif (x1 == 0 and y1 == 0):
            a3_est[N, N, x2+N, y2+N] -= bias
        elif (x2 == 0 and y2 == 0):
            a3_est[x1+N, y1+N, N, N] -= bias
        elif (x1 == x2 and y1 == y2):
            a3_est[x1+N, y1+N, x2+N, y2+N] -= bias
        
        if (x1**2 + y1**2 >= N2 or x2**2 + y2**2 >= N2 \
            or (x1-x2)**2 + (y1-y2)**2 >= N2):
            out_of_support.append(idx)
        else:
            in_support.append(idx)

    return (a3_est, out_of_support, in_support)


def symmetrize_2nd(a2):

    edge_len = a2.shape[0]
    total_len = edge_len**2
    D = (edge_len + 1) // 2
    N = D - 1

    cpy_a2 = np.zeros_like(a2)
    count = np.zeros_like(a2)
    for idx in range(total_len):
        x1 = idx // edge_len - N
        y1 = idx % edge_len - N
        cpy_a2[x1+N, y1+N] += a2[x1+N, y1+N] + a2[-x1+N, y1+N] \
            + a2[x1+N, -y1+N] + a2[-x1+N, -y1+N]
        count[x1+N, y1+N] += 4

    for idx in range(total_len):
        i1 = idx // edge_len
        j1 = idx % edge_len
        if (count[i1, j1] > 0):
            cpy_a2[i1, j1] /= count[i1, j1]

    return cpy_a2


def bandlimit_3rd(a3):

    edge_len = a3.shape[0]
    D = (edge_len + 1) // 2
    N = D - 1
    N2 = N**2

    F = np.fft.fftshift(np.fft.fftn(a3))
    for k1x in range(-N, N+1):
        for k1y in range(-N, N+1):
            for k2x in range(-N, N+1):
                for k2y in range(-N, N+1):
                    if (k1x**2 + k1y**2 >= N2 or k2x**2 + k2y**2 >= N2 \
                        or (k1x+k2x)**2 + (k1y+k2y)**2 >= N2):
                        F[k1x+N, k1y+N, k2x+N, k2y+N] = 0.

    a3 = np.fft.ifftn(np.fft.ifftshift(F))
    a3 = np.real(a3)
    
    return a3

## -----------------------------------------------------------

data_dir = "../Data"
mg_len = 4096
snr = 1

fp = np.load("proj-images.npz")
proj_images = fp['proj_images']
sigma = cal_sigma(proj_images, snr)
print("sigma = {0:1.3e}".format(sigma))

num_alpha = proj_images.shape[0]
edge_len = proj_images.shape[1]
D = (edge_len + 1) // 2

infile = "{0:s}/BPTI-clean-autocorr-60deg.npz".format(data_dir)
fp = np.load(infile)
(a1_true, a2_true, a3_true) = (fp['a1'], fp['a2'], fp['a3'])

## -----------------------------------------------------------

num_instance = 5
mg_counts = [1, 10]
mg_counts.extend([100*n for n in range(1, num_instance+1)])
print(mg_counts)

a1_est = np.zeros(num_instance + 2)
a2_est = np.zeros((num_instance + 2, 2*D-1, 2*D-1))
a3_est = np.zeros((num_instance + 2, 2*D-1, 2*D-1, 2*D-1, 2*D-1))
par_counts = np.zeros(num_instance + 2)

for i in range(num_instance + 2):
    infile= "{0:s}/autocorr_snr-{1:1.3e}_run-{2:05d}-60deg.npz".\
        format(data_dir, snr, mg_counts[i])
    fp = np.load(infile)
    (a1_est[i], a2_est[i], a3_est[i], par_counts[i]) = \
        (fp['a1'], fp['a2'], fp['a3'], fp['total_par_ct'])

## -----------------------------------------------------------

## 1st order
a1_mean = a1_est/par_counts
error = np.abs(a1_mean - a1_true)/a1_true
out_str = "\n1st order:\n\t"
out_str = "{0:s}a1_true = {1:1.5e}\n\t".format(out_str, a1_true)
out_str = "{0:s}mean = {1:1.5e}, error = {2:1.5e}\n\t".\
    format(out_str, a1_mean[-1], error[-1])
print(out_str)

## -----------------------------------------------------------

## 2nd order
num_pix = (mg_len - 2*(D-1))**2
bias = num_pix*sigma**2
for i in range(len(mg_counts)):
    (a2_est[i], out_of_support, in_support) = \
        debias_2nd(a2_est[i], mg_counts[i]*bias)
    a2_est[i, :, :] /= par_counts[i]

error = np.zeros(len(mg_counts))
for i in range(len(mg_counts)):
    error[i] = np.linalg.norm(a2_est[i] - a2_true)/np.linalg.norm(a2_true)

out_str = "2nd order:\n\t"
out_str = "{0:s}error of mean: {1:1.5e}\n\t".format(out_str, error[-1])
print(out_str)

cpy_a2_mean = symmetrize_2nd(a2_est[-1, :, :])
error = np.linalg.norm(cpy_a2_mean - a2_true)/np.linalg.norm(a2_true)
out_str = "\tafter symmetrization:\n\t"
out_str = "{0:s}error of mean: {1:1.5e}\n\t".format(out_str, error)
print(out_str)

cpy_a2_mean = cpy_a2_mean.flatten()
cpy_a2_mean[out_of_support] = 0.
cpy_a2_mean = np.reshape(cpy_a2_mean, a2_true.shape)
error = np.linalg.norm(cpy_a2_mean - a2_true)/np.linalg.norm(a2_true)
out_str = "\tafter removing out-of-support entries:\n\t"
out_str = "{0:s}error of mean: {1:1.5e}\n\t".format(out_str, error)
print(out_str)

## -----------------------------------------------------------

## 3rd order
bias = a1_mean[-1]*sigma**2
bias = a1_true*sigma**2
for i in range(len(mg_counts)):
    (a3_est[i], out_of_support, in_support) = \
        debias_3rd(a3_est[i], par_counts[i]*bias)
    a3_est[i, :, :, :, :] /= par_counts[i]

error = np.zeros(len(mg_counts))
for i in range(len(mg_counts)):
    error[i] = np.linalg.norm(a3_est[i] - a3_true)/np.linalg.norm(a3_true)

out_str = "3rd order:\n\t"
out_str = "{0:s}error of mean: {1:1.5e}\n\t".format(out_str, error[-1])
print(out_str)

cpy_a3_mean = np.copy(a3_est[-1, :, :, :, :]).flatten()
cpy_a3_mean[out_of_support] = 0.
cpy_a3_mean = np.reshape(cpy_a3_mean, a3_true.shape)
error = np.linalg.norm(cpy_a3_mean - a3_true)/np.linalg.norm(a3_true)
out_str = "\tafter removing out-of-support entries:\n\t"
out_str = "{0:s}error of mean: {1:1.5e}\n\t".format(out_str, error)
print(out_str)

np.savez("noisy-auto.npz", a1=a1_mean[-1], a2=cpy_a2_mean, a3=cpy_a3_mean)
