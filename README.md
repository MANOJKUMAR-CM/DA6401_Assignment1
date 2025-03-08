# DA6401_Assignment1  

This repository contains files for **Assignment 1** of the course **DA6401 - Introduction to Deep Learning** at **IIT Madras**.  

## Assignment Overview  
The objective of this Assignment is to **implement a Feedforward Neural Network with Backpropagation from scratch**. The implementation includes:  

- **Forward and Backward Propagation**  
- **Optimizers**  
- **Loss Functions**   
- **Experiment tracking using [wandb.ai](https://wandb.ai/)** 

## Optimizers Implemented  

- **SGD** - Stochastic Gradient Descent  
- **Momentum** - Momentum-based SGD  
- **NAG** - Nesterov Accelerated Gradient (optimized version)  
- **RMSProp** - Root Mean Square Propagation  
- **Adam** - Adaptive Moment Estimation  
- **Nadam** - Nesterov Adaptive Moment Estimation  

## Loss Functions Implemented  

- **Cross Entropy Loss**  
- **Mean Squared Error (MSE)**

### Question 1
Execute `Question1.ipynb`, make sure to update `wandb.init()` with your own wandb credentials before running.

### Question 2
Run `Question2.py`, it has been implemented to accept the number of hidden layers and the number of neurons for each of the hidden layers from user and outputs the probability distribution over the 10 classes.

The rest of the codes accepts the number of hidden layers and number of neurons in hidden layers as arguments / parameters. 

### Question 3 - Question 6
#### Back Propagation and Optimizers
  The backpropagation algorithm has been implemented entirely from scratch, with all operations performed using NumPy for efficient numerical computations.
  
  The Optimizers have been implemented in `optimizers.py`. New Optimizers can be added as a function to this file.

#### Hyper Parameters sweep using Wandb
  `train_test_split` from `sklearn` is used to create random validation sets (10%) from the train data for the hyper parameter search.
  `random` search strategy provided by `wandb.sweep` is used to find the near optimal hyper parameters.
   
- **Hyperparameter Sweeps Configuration:**  

  ##### Fashion MNIST dataset 

     ```python
     sweep_config = {
         "name": "Feedforward Network - Hyperparameter Search",
         "metric": {
             "name": "Validation Loss",
             "goal": "MINIMIZE"
         },
         "method": "random",
         "parameters": {
             "num_epochs": {"values": [5, 10, 20]},
             "num_hiddenLayers": {"values": [3, 4, 5]},
             "hiddenLayer_Size": {"values": [32, 64, 128]},
             "weightDecay": {"values": [0, 0.0005, 0.5]},
             "learningRate": {"values": [1e-3, 1e-4]},
             "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
             "batchSize": {"values": [16, 32, 64]},
             "weightInit": {"values": ["random", "xavier"]},
             "activationFunc": {"values": ["tanh", "relu", "sigmoid"]}
         }
     }
     ```
     

##### MNIST dataset

  ```python
    sweep_config = {
          "name": "Feedforward Network - Hyper parameter search",
          "metric": {
              "name": "Validation Loss",
              "goal": "MINIMIZE"
          },
          "method": "random",
          "parameters": {
              "learningRate": {
                  "values": [1e-3,5e-3, 1e-4]
                  },
              "optimizer": {
                  "values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
                  },
              "activationFunc": {
                  "values": ["tanh", "relu", "sigmoid"]
                  }
               }
            }
  ```
#### Update the wandb.sweep(sweep_config, entity = <name>, project = <project_name>) with your own wandb credentials before running.

- To Perform Hyper Parameter search on Fashion MNIST dataset, with Cross Entropy as Loss function run : `hyperParam_sweep.py`
- To Perform Hyper Parameter search on Fashion MNIST dataset, with MSE as Loss function run : `hyperParam_sweep_MSE.py`
- To Perform Hyper Parameter search on MNIST dataset run : `hyperParam_sweep_MNIST.py`




  
