from tensorflow.keras.datasets import mnist
import numpy as np

def cosine_decay(t, T, eta_max, eta_min):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

x_train = (x_train > 0.5).astype(np.float32)
x_test = (x_test > 0.5).astype(np.float32)

def one_hot(y, num_classes=10): 
    onehot = np.zeros((y.shape[0], num_classes)) 
    onehot[np.arange(y.shape[0]), y] = 1
    return onehot

y_train = one_hot(y_train)
y_test = one_hot(y_test)

def do_nothing(x):
    return x

epochs = 10
batch_size = 32
eta_max = 0.001
eta_min = 0.00001 
learning_rate = 0.001

for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batch_size):
        do_nothing(i)
    print(f"Epoch {epoch}") 
    learning_rate = cosine_decay(epoch, epochs-1, eta_max, eta_min)
    print(learning_rate)
