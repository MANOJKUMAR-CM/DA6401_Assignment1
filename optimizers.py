import numpy as np

def SGD(W, b, gradients, learning_rate):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    """
    
    L = len(W)
    
    for i in range(1, L+1):
        W[i] = W[i] - learning_rate * gradients[f"dW{i}"]
        b[i] = b[i] - learning_rate * gradients[f"db{i}"]
        
    return W, b

def Momentum_GD(W, b, uW, ub, gradients, learning_rate, beta):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
        uW: dict
        Dictionary storing exponentially weighted moving averages of past weight gradients (momentum terms)
    
    ub: dict
        Dictionary storing exponentially weighted moving averages of past bias gradients (momentum terms)
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    
    beta: float
        momentum coefficient - [0, 1]
    """
    
    L = len(W)
    
    for i in range(1, L+1):
        uW[i] = beta * uW[i] +  gradients[f"dW{i}"]
        ub[i] = beta * ub[i] +  gradients[f"db{i}"] 
        
        W[i] = W[i] - learning_rate * uW[i]
        b[i] = b[i] - learning_rate * ub[i]
        
    return W, b, uW, ub

def Nesterov_Accelerated_GD(W, b, uW, ub, gradients, learning_rate, beta):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
        uW: dict
        Dictionary storing exponentially weighted moving averages of past weight gradients (momentum terms)
    
    ub: dict
        Dictionary storing exponentially weighted moving averages of past bias gradients (momentum terms)
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    
    beta: float
        momentum coefficient - [0, 1]
    """
    L= len(W)
    
    for i in range(1, L+1):
        uW[i] = beta * uW[i] - learning_rate * gradients[f"dW{i}"]
        ub[i] = beta * ub[i] - learning_rate * gradients[f"db{i}"]
        
        W[i] = W[i] + beta * uW[i] - learning_rate * gradients[f"dW{i}"]
        b[i] = b[i] + beta * ub[i] - learning_rate * gradients[f"db{i}"]
        
    return W, b, uW, ub