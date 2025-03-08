import numpy as np
from activations_functions import *
from loss import *
from optimizers import *
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


# Loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the pixel values
x_train = x_train/ 255
x_test = x_test/ 255

print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")
print()
# One Hot Encoding of the Labels
def oneHot(num_class, y):
    Y = np.eye(num_class)[y]
    return Y

y_test = oneHot(10, y_test)
y_train = oneHot(10, y_train)

print(f"Training Label data shape after one hot encoding: {y_train.shape}")
print(f"Testing Label data shape after one hot encoding: {y_test.shape}")
print()
# Flattening the Images: (28 x 28) - > (784,)
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)
print(f"Training data shape after flattening: {x_train.shape}")
print(f"Test data shape after flattening: {x_test.shape}")
print()
# Splitting the training data
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.1)

print(f"Training data shape after splitting: {X_train.shape}")
print(f"Validation data shape after splitting: {X_val.shape}")
print()
X_train = X_train.T
X_test = x_test.T
X_val = X_val.T
print("Applied Transpose:")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")
print()
Y_train = Y_train.T  # Shape: (10, 54000)
Y_val = Y_val.T      # Shape: (10, 6000)
Y_test = y_test.T    # Shape: (10, 10000)


def initialize_weights(layers, init_method="xavier"):
    """
    Initializing the weights and biases for the feedforward neural network.
    
    Parameters:
    layers : list
        A list containing the number of neurons in each layer, including the input and output layers.
        Example: [784, 256, 64, 10] 
        
    init_method : str
        The method used to initialize weights:
        - "random" : Initializes weights with random values sampled from normal distribution ~ N(0,1)(scaled by 0.01).
        - "xavier" : Uses Xavier initialization to improve training stability.
                    Formula: W ~ N(0, sqrt(2 / (n_input + n_output)))
                    
    Returns:
    Weights : dict
        A dictionary where keys are layer indices (1 to L) and values are weight matrices.
        - Weights[i] has shape (neurons in layer i, neurons in layer i-1)
        - Weights[i] connects neurons from layer i-1 to layer i
        
    bias : dict
        A dictionary where keys are layer indices (1 to L) and values are bias vectors.
        - bias[i] has shape (neurons in layer i, 1), initialized to zeros.
    """
    
    Weights = {} 
    bias = {} 
    
    for i in range(1, len(layers)):
        if init_method == "random":
            Weights[i] = np.random.randn(layers[i], layers[i-1]) * 0.01
            
        elif init_method == "xavier":
            Weights[i] = np.random.randn(layers[i], layers[i-1]) * (np.sqrt(2/ (layers[i]+ layers[i-1])))
        
        else:
            print("Wrong Weights Initialization Method")
            return
            
        # Bias initialized as a zero vector of shape (neurons in current layer, 1)
        bias[i] = np.zeros((layers[i],1))
        
    return Weights, bias

def forward_propagation(X, W, b, activations):
    """
    Forward Propogation of data once through the Neural Network
    
    Parameters:
    
    X: numpy array
        shape: [784, batch_size]
        
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    activations: string
        Activation function of the hidden layers; Except the output layer
    
    """
    A = X # Inputs
    # To store Intermediate values which would be used during BackPropogation
    pre_Activation = {}
    Activation = {}
    Activation["A0"] = A # Initial Activation is the input itself
    
    for i in range(1, len(W)+1):
        # Pre activation at layer i
        Z = np.add(np.matmul(W[i], A), b[i])
        
        # Activation at layer i
        if i == len(W): # Output layer (softmax)
            A = softmax(Z)
        else:
            if activations == "relu":
                A = relu(Z)

            elif activations == "tanh":
                A = tanh(Z)

            elif activations == "sigmoid":
                A = sigmoid(Z)
                
            else:
                print("Wrong Activation function")
                return
                
        pre_Activation[f"Z{i}"] = Z
        Activation[f"A{i}"] = A
        
    return A, pre_Activation, Activation

    
def backward_propagation(Y, pre_Activation, Activation, W, b, activation_func):
    """
    To Compute the gradients with respect to weights and biases
    
    Parameters:
    
    Y: numpy array
        shape: [num_classes, batch_size]
        
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    pre_Activation: dict
        Dictionary containing Z values of each Layer
    
    Activation: dict
        Dictionary containing H values of each layer
        
    activation_func: string
        Activation function utilized in the hidden layer
    """
    
    gradients = {}
    L = len(W) # Number of Layers
    
    # Assuming cross entropy loss
    dA = Activation[f"A{L}"] - Y # output layer
    
    
    for i in range(L, 0, -1):
        dZ = dA
        
        if i < L:
            Z = pre_Activation[f"Z{i}"]
            if activation_func == "relu":
                dZ = dA * d_relu(Z)
                
            elif activation_func == "sigmoid":
                dZ = dA * d_sigmoid(Z)
                
            elif activation_func == "tanh":
                dZ = dA * d_tanh(Z)
                
        dW = np.dot(dZ, Activation[f"A{i-1}"].T) / Y.shape[1]  # Normalizing by batch size
        db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        
        gradients[f"dW{i}"] = dW
        gradients[f"db{i}"] = db
        
        dA = np.dot(W[i].T, dZ)
        
    return gradients  

# Defining network architecture
layers = [784, 128, 128, 128, 128, 10]  # Input -> Hidden Layers -> Output
activation_func = "relu"  # Hidden layer activation

# Initialize weights and biases
W, b = initialize_weights(layers, init_method="xavier")

# Hyperparameters
epochs = 20
batch_size = 64
learning_rate = 0.005
beta = 0.9  # Momentum coefficient
eps = 1e-4 
W_decay = 0

uW = {i: np.zeros_like(W[i]) for i in W}
ub = {i: np.zeros_like(b[i]) for i in b}

mW = {i: np.zeros_like(W[i]) for i in W}
mb = {i: np.zeros_like(b[i]) for i in b}

num_batches = X_train.shape[1] // batch_size  
print("Training the Feed forward Network:")
for epoch in range(epochs):
    
    epoch_loss = 0  

    for i in range(num_batches):

        start = i * batch_size
        end = start + batch_size
        X_batch = X_train[:, start:end]
        Y_batch = Y_train[:, start:end]
        
        y_pred, pre_Activation, Activation = forward_propagation(X_batch, W, b, activation_func)
        
        loss = cross_entropy(Y_batch, y_pred)
        epoch_loss += loss  
        
        # Computing the gradients
        gradients = backward_propagation(Y_batch, pre_Activation, Activation, W, b, activation_func)

        # Updating weights using optimizer
        W, b, uW, ub = RMSProp(W, b, uW, ub, gradients, learning_rate, beta, eps, W_decay)

    # Print loss averaged over all batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")

# Testing

y_test_pred, _, _ = forward_propagation(X_test, W, b, activation_func)
y_test_pred = np.argmax(y_test_pred, axis=0)
y_actual = np.argmax(Y_test, axis=0)

accuracy = np.mean(y_test_pred == y_actual)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

