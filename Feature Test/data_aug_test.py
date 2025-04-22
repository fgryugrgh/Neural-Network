import numpy as np 
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

#def augment_image(image):
#    image_np = image.squeeze().astype(np.float32) 
#    rotated_image = rotate(image_np, 15, reshape = False, mode='nearest')
#    image = tf.convert_to_tensor(rotated_image, dtype=tf.uint8)
#    return image

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1, interpolation='bilinear'),
    tf.keras.layers.RandomZoom(0.1, interpolation='nearest'),
    tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), interpolation='nearest')
])

idx = np.random.randint(0, len(x_train))
augmented = data_augmentation(tf.convert_to_tensor(x_train[idx], dtype=tf.float32))
augmented = augmented.numpy()
original = x_train[idx]

def show_images(original, augmented):
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Augmented")
    plt.imshow(augmented, cmap='gray')
    plt.axis("off")
    
    plt.show()

show_images(original, augmented)
