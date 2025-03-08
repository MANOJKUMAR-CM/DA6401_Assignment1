# DA6401_Assignment1  

This repository contains files for **Assignment 1** of the course **DA6401 - Introduction to Deep Learning** at **IIT Madras**.  

## Assignment Overview  
The objective of this Assignment is to **implement a Feedforward Neural Network with Backpropagation from scratch**. The implementation includes:  

- **Forward and Backward Propagation**  
- **Optimizers**  
- **Loss Functions**
- **Activation Functions**  
- **Experiment tracking using [wandb.ai](https://wandb.ai/)**

## [Wandb Report](https://api.wandb.ai/links/manoj_da24s018-iit-madras/ha0mw34v)

## Optimizers Implemented  

- **SGD** - Stochastic Gradient Descent  
- **Momentum** - Momentum-based SGD  
- **NAG** - Nesterov Accelerated Gradient   
- **RMSProp** - Root Mean Square Propagation  
- **Adam** - Adaptive Moment Estimation  
- **Nadam** - Nesterov Adaptive Moment Estimation  

## Loss Functions Implemented  

- **Cross Entropy Loss**  
- **Mean Squared Error (MSE)**
  
The functions are implemented in `loss.py`.

## Activation Functions Implemented  

- **Sigmoid**  
- **Tanh**
- **Relu**
- **Softmax**
  
The functions are implemented in `activations_functions.py`.


### Question 1
Execute `Question1.ipynb`, make sure to update `wandb.init()` with your own wandb credentials before running.

### Question 2
``` python
python Question2.py
``` 
The code been implemented to accept the number of hidden layers and the number of neurons for each of the hidden layers from user and outputs the probability distribution over the 10 classes.

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
#### Update WandB Credentials  
Before running the sweep, update the following command with your own WandB credentials:  

```python
wandb.sweep(sweep_config, entity="<your_name>", project="<your_project_name>")
```

- To Perform Hyper Parameter search on Fashion MNIST dataset, with Cross Entropy as Loss function run :
  ``` python
  python hyperParam_sweep.py
  ```
  
### Question 7
To train the feed forward network with the best hyper parameters obtained from the sweeps run:
```python
python main.py
```
The code reports the `Test Accuracy` on Fashion_MNIST dataset and plots the `Confusion Matrix`. 

### Question 8
To Perform Hyper Parameter search on Fashion MNIST dataset, with MSE as Loss function run :
  ``` python
  python hyperParam_sweep_MSE.py
  ```

To train the feed forward network with the best hyper parameters obtained from the sweeps run:
  ```python
  python main_MSE.py
  ```
The code reports the `Test Accuracy` on Fashion_MNIST dataset and plots the `Confusion Matrix`. 

### Question 10
To Perform Hyper Parameter search on MNIST dataset run : 
   ``` python
    python hyperParam_sweep_MNIST.py
   ```
To train the feed forward network with the best hyper parameters obtained from the sweeps run:
  ```python
  python main_MNIST.py
  ```
The code reports the `Test Accuracy` on MNIST dataset. 

  
