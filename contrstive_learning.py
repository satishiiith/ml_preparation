import numpy as np

p1 = np.array([-0.83483301, -0.16904167, 0.52390721])

p2 = np.array([-0.83455951, -0.16862266, 0.52447767])



neg_exmples = np.array([
 [ 0.70374682, -0.18682394, -0.68544673],
 [ 0.15465702,  0.32303224,  0.93366556],
 [ 0.53043332, -0.83523217, -0.14500935],
 [ 0.68285685, -0.73054075,  0.00409143],
 [ 0.76652431,  0.61500886,  0.18494479]])


pos_dot = p1.dot(p2)

neg_dot_products = np.zeros(len(neg_exmples))


for i, negative in enumerate(neg_exmples):
    neg_dot_products[i] = p1.dot(negative)
    
    
# calculate loss for 

v = np.concatenate(([pos_dot], neg_dot_products))

exp = np.exp(v)

softmax_out = exp/np.sum(exp)

print(softmax_out)
