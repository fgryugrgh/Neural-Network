#spaghetti code my beloved
#i love making hyper complicated code that's barely worth any grade yayyy

# Standard library imports
import time

# Third-party imports
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk
from tensorflow.keras.datasets import mnist

# Tkinter imports
from tkinter import Tk, Frame, Canvas, Text, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure Matplotlib
matplotlib.use("Agg")

#Tkinter window
root = Tk()
root.title("paint for my neural network stuff")
root.geometry("1100x650")
default_bg = root.cget("bg")

#Neural Network Part

#get data from tensorflow mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

x_train = (x_train > 0.5).astype(np.float32)
x_test = (x_test > 0.5).astype(np.float32)

#one hot encoding
def one_hot(y, num_classes=10): 
    onehot = np.zeros((y.shape[0], num_classes)) 
    onehot[np.arange(y.shape[0]), y] = 1
    return onehot

y_train = one_hot(y_train)
y_test = one_hot(y_test)

#Layers      Neuron Amount
input_size = 784 #28*x28
hidden_size = 256
output_size = 10
dropout_rate = 0.2
scaling = 1/(1 - dropout_rate)

#Functino for forward propagation
def relu(x):
    return np.maximum(0,x) 

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Normalize each row

#function for backward propagation
def relu_derivative(x):
    return (x > 0).astype(float)  # Returns 1 for x > 0, else 0

def cosine_decay(t, T, eta_max, eta_min):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

def earlystop(x):
    counter = 0
    for i in range(len(x)):
        if counter == 10:
            print("stopped early")
            return
        else:
            counter = counter + 1 if x[i] == x[i-1] else 0
    return counter

#Weight and Biases initialization (commented because it'a unused after the first time)
#np.random.seed(69) #Set the seed so the random number stays the same
#weight1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
#bias1 = np.zeros((1, hidden_size))
#weight2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
#bias2 = np.zeros((1, output_size))

#import weigt and biases
weight1 = np.load("weight1.npy")
bias1 = np.load("bias1.npy")
weight2 = np.load("weight2.npy")
bias2 = np.load("bias2.npy")

#training variable
epochs = 0
batch_size = 32
eta_max = 0.0001
eta_min = 0.00001
learning_rate = 0.0001

#Training
start_time = time.time()

for epoch in range(epochs):
    val_loss_list = []
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        z1 = np.dot(x_batch, weight1) + bias1
        a1 = relu(z1)
        dropout_mask = np.random.rand(*a1.shape) > dropout_rate
        d1 = a1 * dropout_mask
        scaled_a1 = d1 * scaling
        z2 = np.dot(scaled_a1, weight2) + bias2
        a2 = softmax(z2)

        loss = -np.sum(y_batch * np.log(a2 + 1e-8)) / batch_size

        #Backpropagation
        dL_da2 = a2 - y_batch
        dL_dweight2 = np.dot(a1.T, dL_da2) / batch_size
        dL_dbias2 = np.sum(dL_da2, axis=0, keepdims=True) / batch_size
        
        dL_da1 = np.dot(dL_da2, weight2.T) * relu_derivative(z1)
        dL_dweight1 = np.dot(x_batch.T, dL_da1) / batch_size
        dL_dbias1 = np.sum(dL_da1, axis=0, keepdims=True) / batch_size

        weight2 -= learning_rate * dL_dweight2
        weight1 -= learning_rate * dL_dweight1
        bias2 -= learning_rate * dL_dbias2 
        bias1 -= learning_rate * dL_dbias1 

    learning_rate = cosine_decay(epoch, epochs-1, eta_max, eta_min)

    z1_val = np.dot(x_test, weight1) + bias1
    a1_val = relu(z1_val)
    z2_val = np.dot(a1_val, weight2) + bias2
    a2_val = softmax(z2_val)
    val_loss = -np.sum(y_test * np.log(a2_val + 1e-8)) / x_test.shape[0] 
    val_loss_list.append(round(val_loss, 3))
    earlystop(val_loss_list)

    print(f"Val loss: {val_loss}")
    print(f"Epoch {epoch}, W1 mean: {np.mean(weight1)}, W2 mean: {np.mean(weight2)}")

    np.save("weight1.npy", weight1)
    np.save("bias1.npy", bias1)
    np.save("weight2.npy", weight2)
    np.save("bias2.npy", bias2)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

##Forward pass test
total = 0
count = 0
softmax_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def predict(x):
    global softmax_output
    z1 = np.dot(x, weight1) + bias1
    a1 = relu(z1)
    z2 = np.dot(a1, weight2) + bias2 
    softmax_output = softmax(z2)
#    print(np.sum(softmax(z2)))
#    print(np.floor(np.max(softmax_output)*100))
    return np.argmax(softmax_output, axis=1)

#my sample
def own_sample(event = None):
    global guess
    save_image()
    img = cv2.imread("drawing.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Ensure 28x28 size
    img = (img > 127).astype(np.float32)  # Apply thresholding (like training data)
    img = img.reshape(1, 28*28)  # Flatten to (1, 784)
  
    guess = predict(img)
    plot_softmax()
    changeLabel()

#accuracy test (using mnist)
#for i in range(len(x_test)):
#    prediction = predict(x_test[i].reshape(1, -1))  # Reshape to (1, 784)
#    Y = np.argmax(y_test[i])  # Get the true label  
#
#    if prediction == Y:  # `predict()` already returns class index
#        count += 1
#        print(f"{i+1}. correct")
#    else:
#        print(f"{i+1}. wrong")
#    total += 1  
#
#print(f'Validation accuracy: {count} / {total} = {count / total * 100}%')
#
#for i in range(len(x_train)):
#    prediction = predict(x_train[i].reshape(1, -1))  # Reshape to (1, 784)
#    Y = np.argmax(y_train[i])  # Get the true label  
#
#    if prediction == Y:   
#        count += 1
#    total += 1  
#
#print(f'Training accuracy: {count} / {total} = {count / total * 100}%')

#Graph time :DDD
digits = np.array([0,1,2,3,4,5,6,7,8,9])
probability_ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

fig1, ax1 = plt.subplots()
fig1.patch.set_facecolor(default_bg)
ax1.set_facecolor(default_bg)
fig1.set_size_inches(6, 4)

bar = FigureCanvasTkAgg(fig1, root)
bar.draw_idle()
bar.get_tk_widget().grid(row = 0, column = 1, sticky="w", pady = 20)

def plot_softmax():
    global softmax_output
    softmax_output = softmax_output.flatten()
    ax1.cla()
    ax1.set_facecolor(default_bg)
    ax1.bar(digits, softmax_output, color="#4a61a4")
    ax1.set_xticks(digits)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(probability_ticks)
    ax1.set_title("NN Prediction")
    ax1.set_xlabel("Number")
    ax1.set_ylabel("Probability")

    bar.draw_idle()

plot_softmax()

def changeLabel():
    guessLabel.config(text=f"The NN think it's the number {guess.item()}\nand its {np.floor(np.max(softmax_output)*10000)/100}% sure")

guessLabel = Label(root,font=("Arial", 13))
guessLabel.grid(row=0, column=2, sticky="w")

info = Label(root, text = "This is a simple neural network to predict a handwritten digit. it's trained using the MNIST dataset \nand it achieved a 97.54% accuracy on 10000 MNIST test data")
info.grid(row = 1, column = 1)

#paint

# Constants
CANVAS_SIZE = 280  # Big visible canvas
GRID_SIZE = 28  # Logical resolution (28x28)
PIXEL_SIZE = CANVAS_SIZE // GRID_SIZE  # Scale factor
STROKE_SIZE = 2

#make canvas 
image = Image.new("L", (GRID_SIZE, GRID_SIZE), "black")  # 28x28 grayscale image
draw = ImageDraw.Draw(image)

frame = Frame(root)
frame.grid(row=0, column=0, pady=15, padx=15, sticky="w")

canvaslabel = Label(frame, text="Canvas")
canvaslabel.pack()

canvas = Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

#paint functions
def clear(event):
    canvas.delete("all")  # Clear the Tkinter canvas
    draw.rectangle([0, 0, 28, 28], fill="black") 
    
def erase(event): 
    x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE  # Scale down
    
    for dx in range(STROKE_SIZE):
        for dy in range(STROKE_SIZE):
            if 0 <= x + dx < 28 and 0 <= y + dy < 28:  # Stay within bounds
                draw.rectangle([x + dx, y + dy, x + dx, y + dy], fill="black")  
    
    canvas.create_rectangle(
        x * PIXEL_SIZE, y * PIXEL_SIZE, 
        (x + STROKE_SIZE) * PIXEL_SIZE, (y + STROKE_SIZE) * PIXEL_SIZE, 
        fill="black", outline="")

def paint(event):
    x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE  # Scale down
    
    for dx in range(STROKE_SIZE):
        for dy in range(STROKE_SIZE):
            if 0 <= x + dx < 28 and 0 <= y + dy < 28:  # Stay within bounds
                draw.rectangle([x + dx, y + dy, x + dx, y + dy], fill="white")  
    
    canvas.create_rectangle(
        x * PIXEL_SIZE, y * PIXEL_SIZE, 
        (x + STROKE_SIZE) * PIXEL_SIZE, (y + STROKE_SIZE) * PIXEL_SIZE, 
        fill="white", outline="")

guide = Label(root, anchor="w", justify="left", text="Left Click Drag: Draw\nRight Click Drag: Erase\nMiddle Click: Clear Canvas\nS or Enter: Save and make prediction\nDraw the number on the center and not too big for the best result")
guide.grid(row=1, column=0, sticky="w", padx=15)

def save_image(event=None):
    image.save("drawing.png")  # Save as PNG

#Keybind
root.bind("<Return>", own_sample)
root.bind("<s>", own_sample)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<B3-Motion>", erase)
canvas.bind("<Button-2>", clear)

root.mainloop()

