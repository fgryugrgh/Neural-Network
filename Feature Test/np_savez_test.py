import numpy as np

load_data = np.load('data.npz')

a = load_data['array_a']
b = load_data['array_b']
#np.savez('data', array_a = a, array_b = b)

print(a)
print(b)
