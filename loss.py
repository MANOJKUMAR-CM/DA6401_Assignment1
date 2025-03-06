import numpy as np

def cross_entropy(Y, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    loss = - np.sum(Y * np.log(y_pred))/ Y.shape[1] 
    return loss 

def Mean_Squared_Error(Y, y_pred):
    loss = 0.5 * np.mean((Y - y_pred)**2)
    return loss 

