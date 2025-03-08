import os
import wandb
from sklearn.model_selection import train_test_split
from optimizers import *
from loss import *
from activations_functions import *
from keras.datasets import fashion_mnist, mnist
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

wandb.login()
parser = argparse.ArgumentParser()

# Default Values
project_name = "Testing Feed Forward Network"
entity_name = "manoj_da24s018-iit-madras" # Change it to your wandb entity name
dataset = "fashion_mnist"

# Default Hyper Parameters: (Best Parameters obtained through Sweeps for Cross Entropy Loss Function)

epochs = 20
batch_size = 64
loss = "cross_entropy"
optim = "adam"
learning_rate = 0.001
momentum = 0.9
beta = 0.9
beta1 = 0.9
beta2 = 0.999
eps = 1e-4
w_decay = 0
w_Init = "xavier"
no_Hidden_layer = 4
hidden_size = 128
activation_func = "tanh"


parser.add_argument("-wp", "--wandb_project", type=str, default=project_name, required=True,
                    help="WandB project name (Required).")

parser.add_argument("-we", "--wandb_entity", type=str, default=entity_name, required=True,
                    help="WandB entity name (Required).")

parser.add_argument("-d", "--dataset", type=str, default=dataset,
                    help="Dataset to use. Options: 'fashion_mnist', 'mnist'.")

parser.add_argument("-e", "--epochs", type=int, default=epochs,
                    help="Number of training epochs.")

parser.add_argument("-b", "--batch_size", type=int, default=batch_size,
                    help="Batch size for training.")

parser.add_argument("-l", "--loss", type=str, default=loss,
                    help="Loss function to use. Options: 'cross_entropy', 'mean_squared_error'.")

parser.add_argument("-o", "--optimizer", type=str, default=optim,
                    help="Optimizer to use. Options: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'.")

parser.add_argument("-lr", "--learning_rate", type=float, default=learning_rate,
                    help="Learning rate for optimization.")

parser.add_argument("-m", "--momentum", type=float, default=momentum,
                    help="Momentum factor (used for 'momentum' and 'NAG' optimizers).")

parser.add_argument("-beta", "--beta", type=float, default=beta,
                    help="Beta parameter for RMSProp optimizer.")

parser.add_argument("-beta1", "--beta1", type=float, default=beta1,
                    help="Beta1 parameter for Adam and Nadam optimizers.")

parser.add_argument("-beta2", "--beta2", type=float, default=beta2,
                    help="Beta2 parameter for Adam and Nadam optimizers.")

parser.add_argument("-eps", "--epsilon", type=float, default=eps,
                    help="Epsilon value for Adam and Nadam optimizers.")

parser.add_argument("-w_d", "--weight_decay", type=float, default=w_decay,
                    help="Weight decay (L2 regularization factor).")

parser.add_argument("-w_i", "--weight_init", type=str, default=w_Init,
                    help="Weight initialization method. Options: 'random', 'xavier'.")

parser.add_argument("-nhl", "--num_layers", type=int, default=no_Hidden_layer,
                    help="Number of hidden layers in the neural network.")

parser.add_argument("-sz", "--hidden_size", type=int, default=hidden_size,
                    help="Number of neurons per hidden layer.")

parser.add_argument("-a", "--activation", type=str, default=activation_func,
                    help="Activation function to use. Options: 'sigmoid', 'tanh', 'relu'.")

args = parser.parse_args()

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

""" Data PreProcessing"""
if(args.dataset == "fashion_mnist"):
    # Loading Fashion MNIST data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif(args.dataset == "mnist"):
    # Loading MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    raise ValueError(f"Invalid dataset '{args.dataset}'! Please choose from: ['fashion_mnist', 'mnist'].")

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

layers = [X_train.shape[0]] + [args.hidden_size] * args.num_layers + [Y_train.shape[0]]

# Initialize weights and biases
W, b = initialize_weights(layers, init_method=args.weight_init)

uW = {i: np.zeros_like(W[i]) for i in W}
ub = {i: np.zeros_like(b[i]) for i in b}
    
mW = {i: np.zeros_like(W[i]) for i in W}
mb = {i: np.zeros_like(b[i]) for i in b}

num_batches = X_train.shape[1] // args.batch_size

# Choosing the Loss Function
if (args.loss == "cross_entropy"):
    L_func = cross_entropy
elif(args.loss == "mean_squared_error"):
    L_func = Mean_Squared_Error
else:
    raise ValueError(f"Invalid Loss '{args.loss}'! Please choose from: ['cross_entropy', 'mean_squared_error'].")

# Initializing Wandb
wandb.init(project=args.wandb_project, name=args.wandb_entity) 

print("Training the Feed forward Network:")
for epoch in range(args.epochs):
    epoch_loss = 0 
    for i in range(num_batches):
        
        start = i * args.batch_size
        end = start + args.batch_size
        X_batch = X_train[:, start:end]
        Y_batch = Y_train[:, start:end]
        
        y_pred, pre_Activation, Activation = forward_propagation(X_batch, W, b, args.activation)
        
        loss = L_func(Y_batch, y_pred)
        epoch_loss += loss
        
        # Computing the gradients
        gradients = backward_propagation(Y_batch, pre_Activation, Activation, W, b, args.activation)
        
        # Updating weights using optimizer
        
        if args.optimizer == "sgd":
            W, b = SGD(W, b, gradients, args.learning_rate, args.weight_decay)
        elif args.optimizer == "momentum":
            W, b, uW, ub = Momentum_GD(W, b, uW, ub, gradients, args.learning_rate, args.momentum, args.weight_decay)
        elif args.optimizer == "nesterov":
            W, b, uW, ub = Nesterov_Accelerated_GD(W, b, uW, ub, gradients, args.learning_rate, args.momentum, args.weight_decay)
        elif args.optimizer == "rmsprop":
            W, b, uW, ub = RMSProp(W, b, uW, ub, gradients, args.learning_rate, args.beta, args.epsilon, args.weight_decay)
        elif args.optimizer == "adam":
            W, b, uW, ub = Adam(W, b, uW, ub, mW, mb, gradients, args.learning_rate, args.epsilon, i+1, args.weight_decay, args.beta1, args.beta2)
        elif args.optimizer == "nadam":
            W, b, uW, ub = NAdam(W, b, uW, ub, mW, mb, gradients, args.learning_rate, args.epsilon, i+1, args.weight_decay, args.beta1, args.beta2)
    
    # Validation Set Performace
    val_pred,_,_ = forward_propagation(X_val, W, b, args.activation)
    val_loss = L_func(Y_val, val_pred)
    
    val_pred = np.argmax(val_pred, axis=0)
    y_val = np.argmax(Y_val, axis=0)
    val_accuracy = accuracy_score(val_pred, y_val)
    
    # Training set performance
    train_pred,_,_ = forward_propagation(X_train, W, b, args.activation)
    train_pred = np.argmax(train_pred, axis = 0)
    y_train = np.argmax(Y_train, axis=0)
    train_loss = epoch_loss
    train_acc = accuracy_score(y_train, train_pred)
    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_accuracy
    })
    
    # Print loss averaged over all batches
    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss / num_batches:.4f}, Train Accuracy: {round(train_acc, 4)}, Val Loss: {round(val_loss, 4)}, Val Accuracy: {round(val_accuracy, 4)}")

y_test_pred, _, _ = forward_propagation(X_test, W, b, args.activation)
test_loss = L_func(Y_test, y_test_pred)
y_test_pred = np.argmax(y_test_pred, axis=0)
y_actual = np.argmax(Y_test, axis=0)

test_accuracy = np.mean(y_test_pred == y_actual)

wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
})
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
wandb.finish()

