# Contains Activations functions and their derivatives 
import numpy as np

def sigmoid(x):
    f_x = 1/ (1 + np.exp(-x))
    return f_x

def tanh(x):
    f_x = np.tanh(x)
    return f_x

def relu(x):
    f_x = np.maximum(0, x)
    return f_x

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Stability fix
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)  # Normalize per column (sample)



# Derivatives 

def d_sigmoid(x):
    d_fx = sigmoid(x)*(1 - sigmoid(x))
    return d_fx

def d_relu(x):
    d_fx = np.where(x > 0, 1, 0)
    return d_fx

def d_tanh(x):
    d_fx = 1 - np.tanh(x) ** 2
    return d_fx

def d_softmax(x):
    d_fx = softmax(x) * (1-softmax(x))
    return d_fx

