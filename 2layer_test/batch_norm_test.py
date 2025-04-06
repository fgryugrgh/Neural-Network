import numpy as np

dummy = np.array(
[[ -2.0,   0.0,   2.0],
 [ -1.0,   0.0,   1.0],
 [  0.0,   0.0,   0.0],
 [  1.0,   0.0,  -1.0],
 [  2.0,   0.0,  -2.0]]
)
dummy2 = np.array(
[[5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0]]
)
dummy3 = np.array(
[[ 3.0,  7.0, -1.0],
 [ 5.0,  2.0,  0.0],
 [ 8.0,  6.0,  3.0],
 [ 1.0,  4.0, -2.0],
 [ 4.0,  5.0,  1.0]]
)
batch_size = 5
small_constant = 1e-5

def batch_norm(x):
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0, mean=mean, ddof=0)
    norm = (x-mean)/np.sqrt(variance + small_constant)
    return norm

test1 = batch_norm(dummy)
test2 = batch_norm(dummy2)
test3 = batch_norm(dummy3)
print(f"Test 1: {test1}\nTest 2: {test2}\nTest 3: {test3}")
