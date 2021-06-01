# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Acknowledgment - Arpandeep Khatua

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()

        self.layer_one = nn.Linear(in_features=in_size, out_features=128, bias=True)
        self.layer_two = nn.Linear(in_features=128, out_features=out_size, bias=True)
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.SGD(self.get_parameters(), lr=lrate)

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        self.parameters = [self.layer_one.weight, self.layer_one.bias, self.layer_two.weight, self.layer_two.bias]
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        return [self.layer_one.weight, self.layer_one.bias, self.layer_two.weight, self.layer_two.bias]

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        hidden = nn.functional.relu(self.layer_one(x))
        output = self.layer_two(hidden)
        return output

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        self.optimizer.zero_grad()

        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        x1 = (x - means) / stds

        y_hat = self.forward(x1)
        loss = self.loss_fn(y_hat, y)
        loss.backward()

        self.optimizer.step()

        return float(loss.item())


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    
    model = NeuralNet(0.01, torch.nn.CrossEntropyLoss(), train_set.shape[1], 2)

    losses = []

    for n in range(100):

        for i in range(0, len(train_labels) - batch_size + 1, batch_size):

            inputs = train_set[i:i + batch_size]
            target = train_labels[i:i + batch_size]

            losses.append(model.step(inputs, target))

        print(sum(losses)/len(losses))

    outputs = model.forward(dev_set)

    _, y_hats = torch.max(outputs, 1)

    y_hats = y_hats.tolist()

    return losses, y_hats, model
