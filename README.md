# DA6401_Assignment1  

This repository contains files for **Assignment 1** of the course **DA6401 - Introduction to Deep Learning** at **IIT Madras**.  

## Assignment Overview  
The objective of this Assignment is to **implement a Feedforward Neural Network with Backpropagation from scratch**. The implementation includes:  

- **Forward and Backward Propagation**  
- **Optimizers**  
- **Loss Functions**
- **Activation Functions**  
- **Experiment tracking using [wandb.ai](https://wandb.ai/)**

## Links:

## [Wandb Report](https://api.wandb.ai/links/manoj_da24s018-iit-madras/ha0mw34v)

## [Github: Assignment1 - DA6401](https://github.com/MANOJKUMAR-CM/DA6401_Assignment1)

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

### To train the network with custom parameters, check:
```python
python train.py -h
```
It would list all the parameters that can be configured:
```python
usage: train.py [-h] -wp WANDB_PROJECT -we WANDB_ENTITY [-d DATASET] [-e EPOCHS] [-b BATCH_SIZE] [-l LOSS] [-o OPTIMIZER]
                [-lr LEARNING_RATE] [-m MOMENTUM] [-beta BETA] [-beta1 BETA1] [-beta2 BETA2] [-eps EPSILON] [-w_d WEIGHT_DECAY]
                [-w_i WEIGHT_INIT] [-nhl NUM_LAYERS] [-sz HIDDEN_SIZE] [-a ACTIVATION]

optional arguments:
  -h, --help            show this help message and exit
  -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT
                        WandB project name (Required).
  -we WANDB_ENTITY, --wandb_entity WANDB_ENTITY
                        WandB entity name (Required).
  -d DATASET, --dataset DATASET
                        Dataset to use. Options: 'fashion_mnist', 'mnist'.
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -l LOSS, --loss LOSS  Loss function to use. Options: 'cross_entropy', 'mean_squared_error'.
  -o OPTIMIZER, --optimizer OPTIMIZER
                        Optimizer to use. Options: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for optimization.
  -m MOMENTUM, --momentum MOMENTUM
                        Momentum factor (used for 'momentum' and 'NAG' optimizers).
  -beta BETA, --beta BETA
                        Beta parameter for RMSProp optimizer.
  -beta1 BETA1, --beta1 BETA1
                        Beta1 parameter for Adam and Nadam optimizers.
  -beta2 BETA2, --beta2 BETA2
                        Beta2 parameter for Adam and Nadam optimizers.
  -eps EPSILON, --epsilon EPSILON
                        Epsilon value for Adam and Nadam optimizers.
  -w_d WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Weight decay (L2 regularization factor).
  -w_i WEIGHT_INIT, --weight_init WEIGHT_INIT
                        Momentum factor (used for 'momentum' and 'NAG' optimizers).
  -nhl NUM_LAYERS, --num_layers NUM_LAYERS
                        Number of hidden layers in the neural network.
  -sz HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Number of neurons per hidden layer.
  -a ACTIVATION, --activation ACTIVATION
                        Activation function to use. Options: 'sigmoid', 'tanh', 'relu'.
```
Choose the appropriate parameters and train the network, for example:
```python
python train.py -wp Test4 -we manoj_da24s018-iit-madras -d fashion_mnist -lr 0.001 -eps 0.0001 -beta1 0.9 -beta2 0.999 -e 20 -l mean_squared_error -b 32 -o nadam -w_d 0.0005 -nhl 5 -sz 128 -a relu
```

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

  
