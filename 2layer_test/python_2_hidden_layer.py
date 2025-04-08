#spaghetti code my hhhheloved
#i love making hyper complicated code that's barely worth any grade yayyy
import cv2
import time
import numpy as np 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tkinter import Tk, Frame, Canvas, Text, Label
from PIL import Image, ImageDraw, ImageTk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
matplotlib.use("Agg")  # Use a non-GUI backend
# Constants
CANVAS_SIZE = 280  # Big visible canvas
GRID_SIZE = 28  # Logical resolution (28x28)
PIXEL_SIZE = CANVAS_SIZE // GRID_SIZE  # Scale factor

image = Image.new("L", (GRID_SIZE, GRID_SIZE), "black")  # 28x28 grayscale image
draw = ImageDraw.Draw(image)

#Tkinter window
root = Tk()
root.title("paint for my neural network stuff")
root.geometry("1100x650")
default_bg = root.cget("bg")

frame = Frame(root)
frame.grid(row=0, column=0, pady=15, padx=15, sticky="w")

canvaslabel = Label(frame, text="Canvas")
canvaslabel.pack()

canvas = Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

#Function for the paint program
def clear(event):
    canvas.delete("all")  # Clear the Tkinter canvas
    draw.rectangle([0, 0, 28, 28], fill="black") 
    
STROKE_SIZE = 2
    
def erase(event): 
    x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE  # Scale down
    
    for dx in range(STROKE_SIZE):
        for dy in range(STROKE_SIZE):
            if 0 <= x + dx < 28 and 0 <= y + dy < 28:  # Stay within bounds
                draw.rectangle([x + dx, y + dy, x + dx, y + dy], fill="black")  
    
    canvas.create_rectangle(
        x * PIXEL_SIZE, y * PIXEL_SIZE, 
        (x + STROKE_SIZE) * PIXEL_SIZE, (y + STROKE_SIZE) * PIXEL_SIZE, 
        fill="black", outline=""
    )

def paint(event):
    x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE  # Scale down
    
    for dx in range(STROKE_SIZE):
        for dy in range(STROKE_SIZE):
            if 0 <= x + dx < 28 and 0 <= y + dy < 28:  # Stay within bounds
                draw.rectangle([x + dx, y + dy, x + dx, y + dy], fill="white")  
    
    canvas.create_rectangle(
        x * PIXEL_SIZE, y * PIXEL_SIZE, 
        (x + STROKE_SIZE) * PIXEL_SIZE, (y + STROKE_SIZE) * PIXEL_SIZE, 
        fill="white", outline=""
    )

guide = Label(root, anchor="w", justify="left", text="Left Click Drag: Draw\nRight Click Drag: Erase\nMiddle Click: Clear Canvas\nS or Enter: Save and make prediction\nDraw the number on the center and not too big for the best result")
guide.grid(row=1, column=0, sticky="w", padx=15)

def save_image(event=None):
    image.save("drawing.png")  # Save as PNG

#Neural Network Part
rng = np.random.default_rng()

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

#Layers      Neuron Amount
input_size = 784 #28*x28
hidden_size = 512
hidden_2_size = 256
output_size = 10
dropout_rate = 0.3
scaling = 1/(1 - dropout_rate)
small_constant = 1e-5
momentum = 0.9
#running_mean1 = np.zeros((hidden_size,))
#running_var1 = np.ones((hidden_size,))
#running_mean2 = np.zeros((hidden_2_size,))
#running_var2 = np.ones((hidden_2_size,))

#Functino for forward propagation
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Normalize each row

def batch_norm(x, gamma, beta, running_mean, running_var, training=True):
    if training:
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
    
        stddev = 1./np.sqrt(variance + small_constant)
        norm = (x-mean)*stddev
    
        running_mean[:] = momentum * running_mean + (1 - momentum) * mean
        running_var[:] = momentum * running_var + (1 - momentum) * variance 

        out = gamma * norm + beta
        cache = (x, norm, mean, variance, stddev, gamma,beta)
        return out, cache
    else:
        norm = (x-running_mean)/np.sqrt(running_var + small_constant)        
        out = gamma * norm + beta
        return out, None

#function for backward propagation
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def cosine_decay(t, T, eta_max, eta_min):
    if t <= warmup_epochs:
        return eta_start + (eta_max - eta_start) * t / warmup_epochs
    else:
        t_adjusted = t - warmup_epochs
        t_cosine = T - warmup_epochs
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t_adjusted / t_cosine))

def earlystop(x):
    counter = 0
    for i in range(len(x)):
        if counter == 10:
            print("stopped early")
            return True
        else:
            counter = counter + 1 if x[i] == x[i-1] else 0
    return False

def backward_batch_norm(dout, cache):
    x, norm, mean, variance, stddev, gamma, beta = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * norm, axis=0)
    dxhat = dout * gamma
    
    divar = np.sum(dxhat * (x-mean), axis=0)
    dsqrtvar = -divar / (np.sqrt(variance+small_constant)**2)
    dvar = 0.5 * dsqrtvar / np.sqrt(variance + small_constant)
    
    dxmu = dxhat * (1./np.sqrt(variance + small_constant)) + 2 * (x-mean) * dvar / N
    dmu = -np.sum(dxmu, axis=0)
    dx = dxmu + dmu / N

    return dx, dgamma, dbeta

#Weight and Biases
#np.random.seed(42)
#weight1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
#bias1 = np.zeros((1, hidden_size))
#gamma1 = np.ones((hidden_size,))
#beta1 = np.zeros((hidden_size,))
#weight2 = np.random.randn(hidden_size, hidden_2_size) * np.sqrt(2.0 / hidden_size)
#bias2 = np.zeros((1, hidden_2_size))
#gamma2 = np.ones((hidden_2_size,))
#beta2 = np.zeros((hidden_2_size,))
#weight3 = np.random.randn(hidden_2_size, output_size) * np.sqrt(2.0 / hidden_2_size)
#bias3 = np.zeros((1, output_size))

epochs = 0
warmup_epochs = 10
batch_size = 128
eta_start = 0.0005
eta_max = 0.005
eta_min = 0.00001
learning_rate = 0.0005

weight1 = np.load("weight1.npy")
bias1 = np.load("bias1.npy")
gamma1 = np.load("gamma1.npy")
beta1 = np.load("beta1.npy")
weight2 = np.load("weight2.npy")
bias2 = np.load("bias2.npy")
gamma2 = np.load("gamma2.npy")
beta2 = np.load("beta2.npy") 
weight3 = np.load("weight3.npy") 
bias3 = np.load("bias3.npy")
running_mean1 = np.load("running_mean1.npy")
running_var1 = np.load("running_var1.npy")
running_mean2 = np.load("running_mean2.npy")
running_var2 = np.load("running_var2.npy")

#Training
#best_val_loss = float('inf')
best_val_loss = np.load("best_val_loss.npy")
val_loss_list = []
val_acc_list = []
start_time = time.time()

for epoch in range(epochs):
    epoch_train_loss = 0
    batch_num = 0
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        z1 = np.dot(x_batch, weight1) + bias1
        z1_norm, cache1 = batch_norm(z1, gamma1, beta1, running_mean1, running_var1, training=True)
        a1 = leaky_relu(z1_norm)
        dropout_mask1 = np.random.rand(*a1.shape) > dropout_rate
        d1 = a1 * dropout_mask1
        scaled_a1 = d1 * scaling
        z2 = np.dot(scaled_a1, weight2) + bias2
        z2_norm, cache2 = batch_norm(z2, gamma2, beta2, running_mean2, running_var2, training=True)
        a2 = leaky_relu(z2_norm)
        dropout_mask2 = np.random.rand(*a2.shape) > dropout_rate
        d2 = a2 * dropout_mask2 
        scaled_a2 = d2 * scaling
        z3 = np.dot(scaled_a2, weight3) + bias3
        a3 = softmax(z3)
        loss = -np.sum(y_batch * np.log(a3 + 1e-8)) / batch_size
        epoch_train_loss += loss
        batch_num += 1
        l2_lambda = 1e-5 # Tune this hyperparameter
        l2_penalty = l2_lambda * (np.sum(weight1**2) + np.sum(weight2**2) + np.sum(weight3**2))
        loss += l2_penalty

        #Backpropagation
        dL_da3 = a3 - y_batch
        dL_dweight3 = np.dot(scaled_a2.T, dL_da3) / batch_size
        dL_dbias3 = np.sum(dL_da3, axis=0, keepdims=True) / batch_size

        dL_dz2_norm = np.dot(dL_da3, weight3.T)
        dL_dz2, dgamma2, dbeta2 = backward_batch_norm(dL_dz2_norm * leaky_relu_derivative(z2_norm), cache2)# * dropout_mask2 * scaling, cache2)
        dL_dweight2 = np.dot(scaled_a1.T, dL_dz2) / batch_size
        dL_dbias2 = np.sum(dL_dz2, axis=0, keepdims=True) / batch_size
        
        dL_dz1_norm = np.dot(dL_dz2, weight2.T)
        dL_dz1, dgamma1, dbeta1 = backward_batch_norm(dL_dz1_norm * leaky_relu_derivative(z1_norm) * dropout_mask1 * scaling, cache1)
        dL_dweight1 = np.dot(x_batch.T, dL_dz1) / batch_size
        dL_dbias1 = np.sum(dL_dz1, axis=0, keepdims=True) / batch_size

        weight3 -= learning_rate * dL_dweight3
        weight2 -= learning_rate * dL_dweight2
        weight1 -= learning_rate * dL_dweight1
        bias3 -= learning_rate * dL_dbias3
        bias2 -= learning_rate * dL_dbias2 
        bias1 -= learning_rate * dL_dbias1 
        gamma1 -= learning_rate * dgamma1
        gamma2 -= learning_rate * dgamma2
        beta1 -= learning_rate * dbeta1
        beta2 -= learning_rate * dbeta2

    learning_rate = cosine_decay(epoch, epochs-1, eta_max, eta_min)

    z1_val = np.dot(x_test, weight1) + bias1
    z1_val_norm, cache1 = batch_norm(z1_val, gamma1, beta1, running_mean1, running_var1, training=False)
    a1_val = leaky_relu(z1_val_norm)
    z2_val = np.dot(a1_val, weight2) + bias2
    z2_val_norm, cache2 = batch_norm(z2_val, gamma2, beta2, running_mean2, running_var2, training=False)
    a2_val = leaky_relu(z2_val_norm)
    z3_val = np.dot(a2_val, weight3) + bias3
    a3_val = softmax(z3_val)
    val_pred = np.argmax(a3_val, axis=1)
    val_true = np.argmax(y_test, axis=1)
    val_acc = np.mean(val_pred == val_true)
    val_loss = -np.sum(y_test * np.log(a3_val + 1e-8)) / x_test.shape[0] 
    val_loss_list.append(round(val_loss, 4))
    val_acc_list.append(val_acc)
    if earlystop(val_loss_list):
        break
    avg_train_loss = epoch_train_loss/batch_num

    print(f"Avg Train Loss: {avg_train_loss}")
    print(f"Val loss: {val_loss}, Best Val Loss: {best_val_loss}")
    print(f"Current val acc: {val_acc}")
    print(f"Epoch {epoch}, W1 mean: {np.mean(weight1)}, W2 mean: {np.mean(weight2)}, W3 Mean: {np.mean(weight3)}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"New best model at epoch {epoch}, val loss: {val_loss}")
        np.save("weight1.npy", weight1)
        np.save("bias1.npy", bias1)
        np.save("gamma1.npy", gamma1)
        np.save("beta1.npy", beta1)
        np.save("weight2.npy", weight2)
        np.save("bias2.npy", bias2)
        np.save("gamma2.npy", gamma2)
        np.save("beta2.npy", beta2)
        np.save("weight3.npy", weight3)
        np.save("bias3.npy", bias3)
        np.save("running_mean1.npy", running_mean1)
        np.save("running_var1.npy", running_var1)
        np.save("running_mean2.npy", running_mean2)
        np.save("running_var2.npy", running_var2)
        np.save("best_val_loss.npy", best_val_loss)
    
plt.plot(range(1, epochs + 1), val_loss_list, linestyle='-', color='green', label='validation loss')
plt.plot(range(1, epochs + 1), val_acc_list, linestyle='-', color='orange', label='validation accuracy')
plt.title('Validation per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("val_graph.png")
plt.close()

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

##Forward pass test
total = 0
count = 0
softmax_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def predict(x):
    global softmax_output
    z1 = np.dot(x, weight1) + bias1
    z1_norm, cache1 = batch_norm(z1, gamma1, beta1, running_mean1, running_var1, training=False)
    a1 = leaky_relu(z1_norm)
    z2 = np.dot(a1, weight2) + bias2 
    z2_norm, cache2 = batch_norm(z2, gamma2, beta2, running_mean2, running_var2, training=False)
    a2 = leaky_relu(z2_norm)
    z3 = np.dot(a2, weight3) + bias3
    softmax_output = softmax(z3)
#    print(np.sum(softmax(z3)))
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
for i in range(len(x_test)):
    prediction = predict(x_test[i].reshape(1, -1))  # Reshape to (1, 784)
    Y = np.argmax(y_test[i])  # Get the true label  

    if prediction == Y:  # `predict()` already returns class index
        count += 1
#        print(f"{i+1}. correct")
#    else:
#        print(f"{i+1}. wrong")
    total += 1  

print(f'Validation accuracy: {count} / {total} = {count / total * 100}%')

#for i in range(len(x_train)):
#    prediction = predict(x_train[i].reshape(1, -1))  # Reshape to (1, 784)
#    Y = np.argmax(y_train[i])  # Get the true label  
#
#    if prediction == Y:   
#        count += 1
#    total += 1  

print(f'Training accuracy: {count} / {total} = {count / total * 100}%')

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

guessLabel = Label(root,font=("Helvetica", 13), text = f"The NN Think it's the number\nand it's 0% sure")
guessLabel.grid(row=0, column=2, sticky="w")

info = Label(root, text = "This is a simple neural network to predict a handwritten digit. it's trained using the MNIST dataset \nand it achieved a 98.08% accuracy on 10000 MNIST test data")
info.grid(row = 1, column = 1)

#Keybind
root.bind("<Return>", own_sample)
root.bind("<s>", own_sample)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<B3-Motion>", erase)
canvas.bind("<Button-2>", clear)

root.mainloop()

