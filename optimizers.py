import numpy as np

def SGD(W, b, gradients, learning_rate, weight_decay):
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
    
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.

    """
    
    L = len(W)
    
    for i in range(1, L+1):
        W[i] = W[i] - learning_rate * (gradients[f"dW{i}"] + weight_decay * W[i])
        b[i] = b[i] - learning_rate * gradients[f"db{i}"]
        
    return W, b

def Momentum_GD(W, b, uW, ub, gradients, learning_rate, beta, weight_decay):
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
    
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.
    """
    
    L = len(W)
    
    for i in range(1, L+1):
        uW[i] = beta * uW[i] +  gradients[f"dW{i}"]
        ub[i] = beta * ub[i] +  gradients[f"db{i}"] 
        
        W[i] = W[i] - learning_rate * (uW[i] + weight_decay * W[i])
        b[i] = b[i] - learning_rate * ub[i]
        
    return W, b, uW, ub

def Nesterov_Accelerated_GD(W, b, uW, ub, gradients, learning_rate, beta, weight_decay):
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
    
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.
    """
    L= len(W)
    
    for i in range(1, L+1):
        uW[i] = beta * uW[i] - learning_rate * gradients[f"dW{i}"]
        ub[i] = beta * ub[i] - learning_rate * gradients[f"db{i}"]
        
        W[i] = W[i] + beta * uW[i] - learning_rate * (gradients[f"dW{i}"] + weight_decay * W[i])
        b[i] = b[i] + beta * ub[i] - learning_rate * gradients[f"db{i}"]
        
    return W, b, uW, ub

def RMSProp(W, b, uW, ub, gradients, learning_rate, beta, eps, weight_decay):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    uW: dict
        Dictionary storing exponentially weighted moving averages of squared past weight gradients 
    
    ub: dict
        Dictionary storing exponentially weighted moving averages of squared past bias gradients 
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    
    beta: float
        momentum coefficient - [0, 1]
    
    eps: float
        Added to ensure Numerical Stability; prevents divison by zero is uW/ ub - > 0
    
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.
    """
    
    L = len(W)
    for i in range(1, L+1):
        uW[i] = beta * uW[i] + (1 - beta) * np.square(gradients[f"dW{i}"])
        ub[i] = beta * ub[i] + (1 - beta) * np.square(gradients[f"db{i}"])
        
        W[i] = W[i] - (learning_rate/ np.sqrt(uW[i] + eps)) * gradients[f"dW{i}"] - (learning_rate * weight_decay *W[i])
        b[i] = b[i] - (learning_rate/ np.sqrt(ub[i] + eps)) * gradients[f"db{i}"]
        
    return W, b, uW, ub

def Adam(W, b, uW, ub, mW, mb, gradients, learning_rate, eps, t, weight_decay, beta1 = 0.9, beta2 = 0.999):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    uW: dict
        Dictionary storing exponentially weighted moving averages of squared past weight gradients 
    
    ub: dict
        Dictionary storing exponentially weighted moving averages of squared past bias gradients
        
    mW: dict
        Dictionary storing exponentially weighted moving averages of past weight gradients (momentum terms)
    
    mb: dict
        Dictionary storing exponentially weighted moving averages of past bias gradients (momentum terms)
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    
    eps: float
        Added to ensure Numerical Stability; prevents divison by zero is uW/ ub - > 0
    
    t: int
        Current iteration (for bias correction)
        
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.
    
    beta 1: float
        coefficient - [0, 1], typically = 0.9
    
    beta 2: float
        coefficient - [0, 1], typically = 0.999
    """
    L = len(W)
    for i in range(1, L+1):
        mW[i] = beta1 * mW[i] + (1 - beta1) * gradients[f"dW{i}"]
        mb[i] = beta1 * mb[i] + (1 - beta1) * gradients[f"db{i}"]
        
        # Bias Correction
        mWt_hat = mW[i]/ (1 - (beta1 ** t))
        mbt_hat = mb[i]/ (1 - (beta1 ** t))
        
        uW[i] = beta2 * uW[i] + (1 - beta2) * np.square(gradients[f"dW{i}"])
        ub[i] = beta2 * ub[i] + (1 - beta2) * np.square(gradients[f"db{i}"])
        
        # Bias Correction
        uWt_hat = uW[i]/ (1 - (beta2 ** t))
        ubt_hat = ub[i]/ (1 - (beta2 ** t))
        
        W[i] = W[i] - (learning_rate / (np.sqrt(uWt_hat) + eps))* mWt_hat - (learning_rate * weight_decay* W[i])
        b[i] = b[i] - (learning_rate / (np.sqrt(ubt_hat) + eps)) * mbt_hat
        
    return W, b, uW, ub

def NAdam(W, b, uW, ub, mW, mb, gradients, learning_rate, eps, t, weight_decay, beta1 = 0.9, beta2 = 0.999):
    """
    To Update the Weights and bias
    
    Parameters:
    W: dict
        Dictionary containing weights
    
    b: dict
        Dictionary containing bias vectors
    
    uW: dict
        Dictionary storing exponentially weighted moving averages of squared past weight gradients 
    
    ub: dict
        Dictionary storing exponentially weighted moving averages of squared past bias gradients
        
    mW: dict
        Dictionary storing exponentially weighted moving averages of past weight gradients (momentum terms)
    
    mb: dict
        Dictionary storing exponentially weighted moving averages of past bias gradients (momentum terms)
    
    gradients: dict
        Dictionary containing gradients with respect to both weights and biases of each layer
    
    learning_rate: float
        The rate at which the weights and bias is to be updated
    
    eps: float
        Added to ensure Numerical Stability; prevents divison by zero is uW/ ub - > 0
    
    t: int
        Current iteration (for bias correction)
    
    weight_decay: float
        L2 regularization term (weight decay) to prevent overfitting by penalizing large weights.
    
    beta 1: float
        coefficient - [0, 1], typically = 0.9
    
    beta 2: float
        coefficient - [0, 1], typically = 0.999
    """
    L = len(W)
    for i in range(1, L+1):
        mW[i] = beta1 * mW[i] + (1 - beta1) * gradients[f"dW{i}"]
        mb[i] = beta1 * mb[i] + (1 - beta1) * gradients[f"db{i}"]
        
        # Bias Correction
        mWt_hat = mW[i]/ (1 - (beta1 ** t))
        mbt_hat = mb[i]/ (1 - (beta1 ** t))
        
        uW[i] = beta2 * uW[i] + (1 - beta2) * np.square(gradients[f"dW{i}"])
        ub[i] = beta2 * ub[i] + (1 - beta2) * np.square(gradients[f"db{i}"])
        
        # Bias Correction
        uWt_hat = uW[i]/ (1 - (beta2 ** t))
        ubt_hat = ub[i]/ (1 - (beta2 ** t))
        
        # Nesterov Correction
        N_mW = (beta1 * mWt_hat) + (((1 - beta1) * gradients[f"dW{i}"])/ (1 - (beta1 ** t)))
        N_mb = (beta1 * mbt_hat) + (((1 - beta1) * gradients[f"db{i}"])/ (1 - (beta1 ** t)))

        W[i] = W[i] - (learning_rate/ (np.sqrt(uWt_hat) + eps)) * N_mW - (learning_rate * weight_decay * W[i])
        b[i] = b[i] - (learning_rate/ (np.sqrt(ubt_hat)+ eps)) * N_mb
    
    return W, b, uW, ub