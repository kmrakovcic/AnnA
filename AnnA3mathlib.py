import math
import numpy as np

class Activationfunction:
	def nofunction (x, d=False):
		y=x
		if d:
			return 1
		return y

	identity = nofunction

	def binarystep(x, d=False):
		if d:
			return 0
		y = x.clip(0)
		y[y>0]=1
		return y

	def sigmoid(x, d=False):
		y= 1 / (1 + np.exp(-x))
		return y if d==False else y*(1-y)

	def tanh(x, d=False):
		#y=(2/(1+np.exp(-2*x)))-1
		if d: return np.tanh(x)
		else: return 1-(np.tanh(x)*np.tanh(x))

	def ReLU(x, d=False):
		y = x.clip (0, out=x)
		if d:
			y[y>0]=1
		return y

	def leakyReLU(x, d=False):
		y=np.where (x<0, x*0.01, x)
		return y

class Errorfunction:
	def meanSquaredError (neurons,result):
		return np.sum(np.square(neurons-result))
	
	def cost (neurons, result):
		neurons=neurons.T
		result=result.T
		if 1 in neurons:
			neurons=neurons-np.isin(neurons,1)*0.1
		if 0 in neurons:
			neurons=neurons+np.isin(neurons,0)*0.1
		return -np.sum(result*np.log(neurons)+(1-result)*np.log(1-neurons))

def getaccuracy (n,y):
	n=n.T
	y=y.T
	return 100*((abs(n-y)<0.5).dot(np.ones(n.shape[1]))==n.shape[1]).sum (axis=0)/n.shape[0]


if __name__ == '__main__':
	pass
