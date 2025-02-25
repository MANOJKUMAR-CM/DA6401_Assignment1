import numpy as np
from activations_functions import *
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split


# Loading the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

print("The Input layer is length of the flattened vector, output Layer is set to number of classes")
print("!!Input Layers and Output layer is not part of the hidden layers!!")
print()
n = int(input("Enter the Number of Hidden layers:"))
layers = []
layers.append(X_train.shape[0])
for i in range(1, n+1):
    k = int(input(f"Enter the Number of neurons in Hidden layer{i}: "))
    layers.append(k)
layers.append(Y_test.shape[0])

activation_func = "tanh"

W, b = initialize_weights(layers, "random")

X = X_train[:, :256] # First 256 samples
Y = Y_train[:, :256] 

output, _,_ = forward_propagation(X, W, b, activation_func)
print()
print("Shape of the output from forward propagation:", output[:, ].shape)  # (10, batch size)
print("Class probabilities for the second sample in the batch:", output[:, 2])
print(f"Sum of probabilities for the second sample (~1): {np.sum(output[:, 2]):,.4f}")