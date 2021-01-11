import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from costgrad import *
import numpy as np
import time

tf.enable_v2_behavior()
tf.config.run_functions_eagerly(True)

@tf.function
def tf_costgrad_(x, gpu_arrays, gpu_func, aux_mtx_list, vec_lib, plan1, plan2):
    
    x_vec = np.array(x)
    cost_3rd = my_cost(x_vec, gpu_arrays, \
        gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
    grad_3rd = my_grad(x_vec, gpu_arrays, \
        gpu_func, aux_mtx_list, vec_lib, plan1, plan2)
    cost_3rd = tf.constant(cost_3rd)
    grad_3rd = tf.constant(grad_3rd)
    
    return cost_3rd, grad_3rd
