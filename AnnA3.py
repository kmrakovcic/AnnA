import numpy as np
from AnnA3mathlib import *
class Brain: 
	def __init__ (self, arhitecture, mjerenja=[], alfa=0.1, activationFunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost):
		self.n=[]
		self.b=[]
		self.w=[]
		self.z=[]
		self.delta=[]
		self.alpha=alfa
		self.mjerenja=mjerenja
		self.activationfunction=activationFunction
		self.errorFunction=errorFunction
		arhitecture[0]=mjerenja[0].shape [0]
		arhitecture[len(arhitecture)-1]=mjerenja[1].shape[0]
		self.arhitecture=np.array(arhitecture)

	def birth (self):
		M=self.mjerenja[1].shape[1]
		self.b=[0]
		self.w=[0]
		self.delta=[0]
		for i,j in enumerate (self.arhitecture):
			self.n.append (np.ones((j,M)) )
			self.z.append (np.ones ((j,M))) 
			if not (i==0):
				self.b.append (np.random.random (j) )
				self.w.append (np.random.random ((j, self.arhitecture [i-1])))
				self.delta.append ([])

	def fowardpropagation (self):
		self.n[0]=self.mjerenja[0]
		for i in range(len(self.n)-1):
			self.z[i+1]=self.w[i+1].dot(self.n[i])+self.b[i+1].reshape(-1,1)
			self.n[i+1]=self.activationfunction (self.z[i+1])

	def backpropagation (self):
		y=self.mjerenja[1]
		for l in reversed(range(1,len(self.n))):
			if l==(len(self.n)-1):
				self.delta[l]=self.n[l]-y
			else:
				self.delta[l]=self.w[l+1].T.dot(self.delta[l+1])*self.activationfunction (self.z[l], d=True)

	def evolve (self):
		M=self.mjerenja[1].shape[1]
		for l in range (1,len(self.n)):
			self.w[l]=self.w[l]-(self.alpha/M)*self.delta[l].dot(self.n[l-1].T)
			self.b[l]=self.b[l]-(self.alpha/M)*np.sum (self.delta[l],axis=1)

	def learn (self, mjerenja=[]):
		if not mjerenja==[]: self.mjerenja=mjerenja
		self.fowardpropagation ()
		self.backpropagation ()
		self.evolve ()
		acc=getaccuracy (self.n[len(self.n)-1],self.mjerenja[1])
		err=self.errorFunction (self.n[len(self.n)-1],self.mjerenja[1])
		return err,acc

if __name__ == '__main__':
	pass