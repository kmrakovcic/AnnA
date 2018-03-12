import math
import numpy as np

class Activationfunction:
	def nofunction (x):
		y=x
		return y
	def binarystep(x):
		if x<0:
			y=0
		else: y=1
		return y
	def sigmoid(x, d=False):
		y= 1 / (1 + np.exp(-x))
		return y if d==False else y*(1-y)
	def tanh(x):
		y=(2/(1+np.exp(-2*x)))-1
		return y
	def ReLU(x):
		x.clip (0, out=x)
		return y
	def leakyReLU(x):
		y=np.where (x<0, x*0.01, x)
		return 