import math
import numpy as np

class Activationfunction:
	def nofunction (x):
		return x
	def binarystep(x):
		if x<0:
			y=0
		else: y=1
		return y
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))
	def tanh(x):
		return (2/(1+np.exp(-2*x)))-1
	def ReLU(x):
		return x.clip (0, out=x)
	def leakyReLU(x):
		return np.where (x<0, x*0.01, x)

class Errorfunction:
	def meanSquaredError (neurons,result):
		out=0
		for mjerenje,i in enumerate(neurons):
			out+=(0.5*((result[mjerenje]-neurons[mjerenje])**2).sum(axis=0))
		return out/len(neurons)
	
	def cost (neurons, result):
		mjerenja_u_redu=len(neurons.shape)-2
		out=((result*np.log(neurons)+(1-result)*np.log(1-neurons)).sum (axis=mjerenja_u_redu)).sum (axis=mjerenja_u_redu)/neurons.shape[mjerenja_u_redu]
		return out

def tensortovector (input):
	max=0
	count=0
	for i in input:
		max+=i.size
	output=np.zeros (max)
	for i in input:
		output [count:(count+i.size)]=i.flatten ()
		count+=i.size
	return output

def derivate (style,fx,fdx,result):
	der=Errorfunction ()
	out=exec("der."+style+"(style,fdx,result)-der."+style+"(style,f,result))/derivationstep")

if __name__ == '__main__':
	a=np.array (([6,7,8]))
	b=np.array (([0,1,2],[3,4,5],[6,7,8]))
	c=a+b
	#c=np.rot90 (c,1,(0,2))
	print (c)
	print ("_________")
	print (b)
	print("_________")
	print (c*b)
	#print((a+b).sum (axis=2))


