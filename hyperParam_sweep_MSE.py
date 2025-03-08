import numpy as np
from activations_functions import *
from loss import *
from optimizers import *
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import wandb
wandb.login()

# One Hot Encoding of the Labels
def oneHot(num_class, y):
    Y = np.eye(num_class)[y]
    return Y


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
    
    # Assuming MSE loss
    dA = (Activation[f"A{L}"] - Y)*d_softmax(pre_Activation[f"Z{L}"]) # output layer
    
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

sweep_config = {
    "name": "Feedforward Network - Hyper parameter search",
    "metric": {
        "name": "Validation Loss",
        "goal": "MINIMIZE"
    },
    "method": "random",
    "parameters": {
        "num_epochs": {
            "values": [5, 10, 20]
            },
        "num_hiddenLayers": {
            "values": [3, 4, 5]
            },
        "hiddenLayer_Size": {
            "values": [32, 64, 128]
            },
        "weightDecay": {
            "values": [0, 0.0005, 0.5]
            },
        "learningRate": {
            "values": [1e-3, 1e-4]
            },
        "optimizer": {
            "values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
            },
        "batchSize": {
            "values": [16, 32, 64]
            },
        "weightInit": {
            "values": ["random", "xavier"]
            },
        "activationFunc": {
            "values": ["tanh", "relu", "sigmoid"]
            }
    }
    
}

def sweep_hyperParameters():
    default_config = {
        'num_epochs': 10,
        'num_hiddenLayers': 3,
        'hiddenLayer_Size': 32,
        'weightDecay': 0,
        'learningRate': 1e-3,
        'optimizer': 'Nesterov_Accelerated_GD',
        'batchSize': 32,
        'weightInit': 'xavier',
        'activationFunc': 'sigmoid'
    }
    
    wandb.init(config = default_config)
    config = wandb.config # To get the Hyper Parameters from sweep_config
    
    # Loading the data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalizing the pixel values
    x_train = x_train/ 255
    x_test = x_test/ 255
    
    y_test = oneHot(10, y_test)
    y_train = oneHot(10, y_train)
    
    # Flattening the Images: (28 x 28) - > (784,)
    x_train = x_train.reshape(60000, -1)
    x_test = x_test.reshape(10000, -1)
    
    # Splitting the training data
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, random_state=42, test_size=0.1)
    
    X_train = X_train.T
    X_test = x_test.T
    X_val = X_val.T
    
    Y_train = Y_train.T  # Shape: (10, 54000)
    Y_val = Y_val.T      # Shape: (10, 6000)
    Y_test = y_test.T    # Shape: (10, 10000)
    
    # Parameters from wandb.config
    num_epochs = config.num_epochs
    num_hiddenLayers = config.num_hiddenLayers
    hiddenLayer_size = config.hiddenLayer_Size
    weightDecay = config.weightDecay
    learningRate = config.learningRate
    optimizer = config.optimizer
    batchSize = config.batchSize
    weightInit = config.weightInit
    activationFunc = config.activationFunc
    
    # Name
    run_name = f"LR: {learningRate}, AC: {activationFunc}, BS: {batchSize}, Optim: {optimizer}, WI: {weightInit}, WD: {weightDecay}, No_HL: {num_hiddenLayers}, HS: {hiddenLayer_size}"
    print("Sweep Name: ",run_name)
    
    # To Initialize Weights
    layers = [X_train.shape[0]] + [hiddenLayer_size] * num_hiddenLayers + [Y_train.shape[0]]
    W, b = initialize_weights(layers, weightInit)
    
    beta = 0.9
    eps = 1e-4 
    
    uW = {i: np.zeros_like(W[i]) for i in W}
    ub = {i: np.zeros_like(b[i]) for i in b}
    
    mW = {i: np.zeros_like(W[i]) for i in W}
    mb = {i: np.zeros_like(b[i]) for i in b}
    
    num_batches = X_train.shape[1] // batchSize 
    
    for epoch in range(num_epochs):
        epoch_loss = 0 
        for i in range(num_batches):
            
            start = i * batchSize
            end = start + batchSize
            X_batch = X_train[:, start:end]
            Y_batch = Y_train[:, start:end]
            
            y_pred, pre_Activation, Activation = forward_propagation(X_batch, W, b, activationFunc)
            loss = Mean_Squared_Error(Y_batch, y_pred)
            epoch_loss += loss
            
            # Computing the gradients
            gradients = backward_propagation(Y_batch, pre_Activation, Activation, W, b, activationFunc)
            
            # Updating weights using optimizer
            
            if optimizer == "sgd":
                W, b = SGD(W, b, gradients, learningRate, weightDecay)
            elif optimizer == "momentum":
                W, b, uW, ub = Momentum_GD(W, b, uW, ub, gradients, learningRate, beta, weightDecay)
            elif optimizer == "nesterov":
                W, b, uW, ub = Nesterov_Accelerated_GD(W, b, uW, ub, gradients, learningRate, beta, weightDecay)
            elif optimizer == "rmsprop":
                W, b, uW, ub = RMSProp(W, b, uW, ub, gradients, learningRate, beta, eps, weightDecay)
            elif optimizer == "adam":
                W, b, uW, ub = Adam(W, b, uW, ub, mW, mb, gradients, learningRate, eps, i+1, weightDecay)
            elif optimizer == "nadam":
                W, b, uW, ub = NAdam(W, b, uW, ub, mW, mb, gradients, learningRate, eps, i+1, weightDecay)
            
            #W, b, uW, ub = optim(W, b, uW, ub, gradients, learningRate, beta)
            
        # Validation Set Performace
        val_pred,_,_ = forward_propagation(X_val, W, b, activationFunc)
        val_loss = Mean_Squared_Error(Y_val, val_pred)
        
        val_pred = np.argmax(val_pred, axis=0)
        y_val = np.argmax(Y_val, axis=0)
        val_accuracy = accuracy_score(val_pred, y_val)
        
        # Training set performance
        train_pred,_,_ = forward_propagation(X_train, W, b, activationFunc)
        train_pred = np.argmax(train_pred, axis = 0)
        y_train = np.argmax(Y_train, axis=0)
        train_loss = epoch_loss
        train_acc = accuracy_score(y_train, train_pred)
        
        # Print loss averaged over all batches
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / num_batches:.4f}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        # logging Results
        loss_value = round(epoch_loss / num_batches, 4)
        wandb.log({
            "epoch": epoch, 
            "val_loss": val_loss, 
            "val_acc": val_accuracy, 
            "train_acc": train_acc,
            "train_loss": loss_value
        })
    wandb.run.name = run_name
    wandb.run.save()
    
    return W, b

sweep_ID = wandb.sweep(sweep_config, entity="manoj_da24s018-iit-madras", project="DNN-HyperParameter-Sweep-MSE")
wandb.agent(sweep_ID, sweep_hyperParameters, count=200)

wandb.finish()