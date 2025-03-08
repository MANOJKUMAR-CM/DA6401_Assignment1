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



