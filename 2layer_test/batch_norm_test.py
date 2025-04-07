import numpy as np

dummy = np.array(
[[ -2.0,   0.0,   2.0],
 [ -1.0,   0.0,   1.0],
 [  0.0,   0.0,   0.0],
 [  1.0,   0.0,  -1.0],
 [  2.0,   0.0,  -2.0]]
)

batch_size = 5
small_constant = 1e-5
hidden_size = 16
gamma1 = np.ones((hidden_size,))
beta1 = np.zeros((hidden_size,))

def batch_norm(x, gamma, beta):
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0, mean=mean, ddof=0)

    stddev = 1./np.sqrt(variance + small_constant)
    norm = (x-mean)*stddev

    out = gamma * norm + beta
    cache = (x, norm, mean, variance, stddev, gamma,beta)
    return out, cache

test1, cache1 = batch_norm(dummy, gamma1, beta1)
print(test1)
