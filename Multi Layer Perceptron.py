import numpy as np
from random import uniform
from math import exp
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class NeuralNetwork:

    class hiddenLayer:
        def __init__(self, rows, columns, activation):
            self.numNeurons = rows
            self.activation = activation
            self.values = 0
            self.matrix = np.zeros((rows, columns))

            for row in self.matrix:
                for i in range(len(row)):
                    row[i] = uniform(-1,1)

    def __init__(self, numInputs, learningRate=1, bias=1):
        self.numInputs = numInputs
        self.lr = learningRate
        self.bias = bias
        self.hiddenLayers = []
        self.activations = []
        self.previousLayer = numInputs

    def Sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def activation(self, layer):
        for i in range(len(layer.values)):
            if layer.activation == 'Sigmoid':
                layer.values[i] = self.Sigmoid(-layer.values[i])
            elif layer.activation == 'ReLu':
                layer.values[i] = max([0,layer.values[i]])

    def insertLayer(self, numNeurons, activation):
        self.hiddenLayers.append(self.hiddenLayer(numNeurons, self.previousLayer, activation))
        self.previousLayer = numNeurons

    def createNetwork(self, numOutputs, activation):
        self.numOutputs = numOutputs
        self.hiddenLayers.append(self.hiddenLayer(self.numOutputs, self.previousLayer, activation))

    def backPropagation(self, error, inputs):
        for i in range(len(self.hiddenLayers) - 1, -1, -1):
            gradient = np.multiply(self.hiddenLayers[i].values, np.subtract(1, self.hiddenLayers[i].values))
            gradient = np.multiply(error, gradient)
            gradient *= self.lr
            if i > 0:
                layerError = np.matmul(gradient, np.transpose(self.hiddenLayers[i-1].values))
                error = np.matmul(np.transpose(self.hiddenLayers[i].matrix), np.divide(error, np.transpose([np.sum(self.hiddenLayers[i].matrix, axis=1)])))
            else:
                layerError = np.matmul(gradient, np.transpose(inputs))
            
            self.hiddenLayers[i].matrix = np.add(self.hiddenLayers[i].matrix, layerError) 
            
    def feedForward(self, inputs):
        array = inputs
        for layer in self.hiddenLayers:
            layer.values = np.matmul(layer.matrix, array)
            self.activation(layer)
            array = layer.values
        guess = array
        
        return guess
    
    def train(self, inputs, targets):
        for i in range(len(inputs)):
            inputValue = np.transpose([inputs[i]])
            target = np.transpose([targets[i]])
            guess = self.feedForward(inputValue)
            error = np.subtract(target, guess)
            cost = error**2
            self.backPropagation(error, inputValue)

    def test(self, inputs, targets):
        for i in range(len(inputs)):
            inputValue = np.transpose([inputs[i]])
            target = np.transpose([targets[i]])
            guess = self.feedForward(inputValue)
            error = np.subtract(target, guess)
            cost = error**2

    def guess(self, inputs):
        for i in range(len(inputs)):
            inputValue = np.transpose([inputs[i]])
            guess = self.feedForward(inputValue)

        return guess

def f(row):
    for i in range(len(uniqueClass)):
        if row['CLASS'] == uniqueClass[i]:
            return i

data = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'iris.csv'), delimiter=';')
uniqueClass = data['CLASS'].unique()
data['NUM_CLASS'] = data.apply(f, axis=1)

train, test = train_test_split(data, test_size = 0.3)

'''nn = NeuralNetwork(4, 0.001)
nn.insertLayer(3, 'Sigmoid')
nn.insertLayer(3, 'Sigmoid')
nn.createNetwork(3, 'Sigmoid')
nn.train([[5,8,7,9]], [[0,1,0]])'''