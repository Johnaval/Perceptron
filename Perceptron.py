from tkinter import *
from random import uniform
import time
import matplotlib.pyplot as plt

ballSize = 10

class Draw:

    def __init__(self, width, height, y0, y1):
        self.width = width
        self.height = height

        self.x0 = 0
        self.y0 = y0
        self.x1 = self.width
        self.y1 = y1

        self.root = Tk()
        self.canvas = Canvas(self.root, width = self.width, height = self.height, background='white')
        self.canvas.grid()

    def returnPixel(self, x, y):
        return self.width/2 + x * self.width/2, self.height/2 + y * self.height/2

    def draw(self, points, perceptronPoints, linePoints):
        try:
            self.canvas.delete(ALL)
            self.canvas.create_line(self.x0, self.height/2 + self.y0 * self.height/2, self.x1, self.height/2 + self.y1 * self.height/2)
            self.canvas.create_line(0, self.height/2 + linePoints[0] * self.height/2, self.width, self.height/2 + linePoints[1] * self.height/2, fill='red')
            for point in points:
                x, y = self.returnPixel(point.x, point.y)
                self.canvas.create_oval(x - ballSize/2, y - ballSize/2, x + ballSize/2, y + ballSize/2, fill = 'black' if point.label == 1 else 'white')

            for point in perceptronPoints:
                x, y = self.returnPixel(point.x, point.y)
                self.canvas.create_oval(x - ballSize/4, y - ballSize/4, x + ballSize/4, y + ballSize/4, fill = 'green' if point.correct == True else 'red')
            
            self.canvas.update()
        except: pass

class Point:

    def random(self):
        self.x = uniform(-1,1)
        self.y = uniform(-1,1)
        self.label = 0

    def create(self, x, y, label, correct):
        self.x = x
        self.y = y
        self.label = label
        self.correct = correct

class Perceptron:

    def __init__(self, weights):
        self.iterations = 0
        self.weights = []
        self.meanError = []
        for i in range(weights):
            self.weights.append(uniform(-1,1))
        self.bias = 1
        self.lr = 0.0001
    
    def line(self):
        y0 = -(-1 * self.weights[0] + self.bias * self.weights[-1])/self.weights[1]
        y1 = -(1 * self.weights[0] + self.bias * self.weights[-1])/self.weights[1]

        return [y0, y1]
    
    def f(self, x):
        return -0.2 * x + 0.1

    def neuron(self, inputs):
        result = 0
        for i in range(len(inputs)):
            result += inputs[i] * self.weights[i]
        result += self.bias * self.weights[-1]
        return result
    
    def activation(self, inputs):
        result = self.neuron(inputs)
        if result > 0:
            return 1
        return -1

    def train(self, points):
        self.linePoints = self.line()
        self.perceptronPoints = []
        totalError = 0
        for point in points:
            inputs = [point.x, point.y, self.bias]
            label = self.activation([point.x, point.y])
            error = (point.label - label)
            totalError += error
            for i in range(len(self.weights)):
                self.weights[i] += error * self.lr * inputs[i]
            p = Point()
            p.create(point.x, point.y, label, True if error == 0 else False)
            self.perceptronPoints.append(p)
        self.meanError.append(totalError/len(points))
        self.iterations += 1
        return self.perceptronPoints, self.linePoints

    def plotError(self):
        plt.plot([i for i in range(self.iterations)], self.meanError)
        plt.show()

class Main:

    def __init__(self):
        self.width = 2000
        self.height = 800

        n = 500 #population       
        self.points = []
        iterations = 500

        for i in range(n):
            point = Point()
            point.random()
            self.points.append(point)
        self.p = Perceptron(3)
        self.defineLabels()
        gui = Draw(self.width, self.height, self.p.f(-1), self.p.f(1))
        currentIter = 0
        while currentIter <= iterations:
            perceptronPoints, linePoints = self.p.train(self.points)
            gui.draw(self.points, perceptronPoints, linePoints)
            time.sleep(0.01)
            currentIter += 1
        self.p.plotError()

    def defineLabels(self):
        for point in self.points:
            if point.y > self.p.f(point.x):
                point.label = 1
            else:
                point.label = -1
    
main = Main()

        