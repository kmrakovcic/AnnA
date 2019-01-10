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

def getstats1 (n,y):
	n=n.T
	y=y.T
	#print (n)
	return 100*((abs(n-y)<0.5).dot(np.ones(n.shape[1]))==n.shape[1]).sum (axis=0)/n.shape[0]

def getstats (n,y, threshold=0.5):
	n=n.T
	y=y.T
	n=(n>threshold)*1
	tp=np.sum(np.logical_and(n==1,y==1)*1)  #true positives
	tn=np.sum(np.logical_and(n==0,y==0)*1)	#true negatives
	fp=np.sum(y)-tp                         #false positives
	fn=np.sum (y==0)-tn                     #false negatives
	accuracy=(tp+tn)/(tp+tn+fp+fn)    #accuracy
	tpr=tp/(tp+fn)                    #sensitivity or true positive rate
	fpr=fp/(fp+tp)                    #fallout or false positive rate
	ppv=tp/(tp+fp)                    #precision or positive predicitve value
	f1=2*ppv*tpr/(ppv+tpr)            #f1 score
	return accuracy, f1, tpr, fpr, ppv
if __name__ == '__main__':	
	a=np.array ([[1],[0],[0],[1],[1],[0],[0],[1]])
	b=np.array ([[0.7],[0.2],[0.6],[0.2],[0.2],[0.6],[0.2],[0.1]])
	k= getstats1 (b,a)
	print (k)
	pass
