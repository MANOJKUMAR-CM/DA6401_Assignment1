import numpy as np

def cross_entropy(Y, y_pred):
    loss = - np.sum(np.multiply(Y, np.log(y_pred)))/ Y.shape[1] 
    return loss

def Mean_Squared_Error(Y, y_pred):
    loss = 0.5 * np.mean((Y - y_pred)**2)
    return loss