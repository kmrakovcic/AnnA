from matplotlib import pyplot as plt
from math import cos, sin, atan


class Neuron ():
    def __init__(self, x, y, radius, bias):
        self.x = x
        self.y = y
        self.neuron_radius = radius
        self.b = bias
        self.color=(self.b,self.b,self.b)

    def draw(self):
        circle = plt.Circle((self.x, self.y), radius=self.neuron_radius, color=self.color, fill=True)
        plt.gca().add_patch(circle)

class Weight ():
    def __init__ (self, neuron1, neuron2, weight):
        self.neuron_radius=neuron1.neuron_radius
        self.sx=neuron1.x
        self.sy=neuron1.y
        self.ex=neuron2.x
        self.ey=neuorn2.y
        self.w=weight
        self.color=(self.w,self.w,self.w)

    def draw ():
        angle = atan((self.ex - self.sx) / float(self.ey - self.sy))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((self.sx - x_adjustment, self.ex + x_adjustment), (self.sy - y_adjustment, self.ey + y_adjustment), color=self.color)
        pyplot.gca().add_line(line)

class Layer ():
    def __init__ (self, arh, bias=0, weight=0):
        self.neurons=[]
        self.weights=[]

    def birth (self, arh):
        self.neurons=[]
        self.weights=[]
        for i,j in enumerate(arh):
            n=[]
            for k in range (j):
                neu=Neuron(0,0,0,0)
                self.neurons.append(neu)
if __name__ == '__main__':
    a=Layer([1,1])
    a.birth([2,5])
    print (a.neurons)
