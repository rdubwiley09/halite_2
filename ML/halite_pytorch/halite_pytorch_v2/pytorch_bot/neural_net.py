import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_bot.common import PLANET_MAX_NUM, PER_PLANET_FEATURES

def normalize_input(input_data):

    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)

class TwoLayerNetwork(nn.Module):
    FIRST_LAYER_SIZE = 12
    SECOND_LAYER_SIZE = 6

    #TO DO: CHANGE OVER TO PYTORCH
    def __init__(self):
        super(TwoLayerNetwork, self).__init__()
        self.first_layer = torch.nn.Linear(PLANET_MAX_NUM*PER_PLANET_FEATURES,self.FIRST_LAYER_SIZE)
        self.second_layer = torch.nn.Linear(self.FIRST_LAYER_SIZE,self.SECOND_LAYER_SIZE)
        self.predictions = torch.nn.Linear(self.SECOND_LAYER_SIZE,PLANET_MAX_NUM)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        first_relu = self.first_layer(x).clamp(min=0)
        second_relu = self.second_layer(first_relu).clamp(min=0)
        predictions = self.predictions(second_relu)
        return self.softmax(predictions)

class NeuralNet:
    def __init__(self):
        self.model = TwoLayerNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)

    def reshape(self,input_data):
        """
        Takes in the per planet tensor and reshapes it to a long tensor
        """
        return input_data.view(-1,PLANET_MAX_NUM*PER_PLANET_FEATURES)

    def compute_loss(self, input_data, expected_output_data):
        """
        Takes in input data and returns the cv_loss
        """
        criterion = torch.nn.BCELoss()
        predictions = self.model(self.reshape(input_data))
        loss = criterion(predictions, expected_output_data)
        return loss, loss.data[0]

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, loss_value = self.compute_loss(input_data, expected_output_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_value

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self.model(self.reshape(input_data))


    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        torch.save(self.model,path)
