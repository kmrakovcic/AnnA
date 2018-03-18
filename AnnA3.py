import numpy as np
from numpy.linalg import norm
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

			# print(str(l)+". Delta: ",np.sum(self.delta[l]), norm(self.delta[l]))

	def evolve (self):
		M=self.mjerenja[1].shape[1]
		for l in range (1,len(self.n)):
			self.w[l]=self.w[l]-(self.alpha/M)*self.delta[l].dot(self.n[l-1].T)
			self.b[l]=self.b[l]-(self.alpha/M)*np.sum (self.delta[l],axis=1)
			self.delta[l]=0

	def learn (self, batchNum=1, mjerenja=[]):
		if not mjerenja==[]: self.mjerenja=mjerenja
		M = self.mjerenja[1].shape[1]
		batchSize=M//batchNum
		if M%batchNum > 0:
			batchNum+=1

		tmpMjerenja = self.mjerenja

		self.mjerenja= [None,None]
		for i in range(batchNum):
			for j in range(2):
				self.mjerenja[j]=tmpMjerenja[j][:,i*batchSize:(i+1)*batchSize]
			self.fowardpropagation ()
			self.backpropagation ()
			self.evolve ()

		self.mjerenja=tmpMjerenja
		if batchNum>0:
			self.fowardpropagation ()

		acc=getaccuracy (self.n[len(self.n)-1],self.mjerenja[1])
		err=self.errorFunction (self.n[len(self.n)-1],self.mjerenja[1])
		return err,acc

	def savebrain (self,name):
		output=np.append(np.array(len(self.arhitecture)),self.arhitecture )
		for i in range(1,len(self.arhitecture)):
			output=np.append(output, np.append (self.w[i].flatten(),self.b[i].flatten()) )
		with open (name, "wb") as file:
			output.tofile (file)

	def loadbrain (self, name):
		b=[0]
		w=[0]
		with open (name, "rb") as file:
			input=np.fromfile(file)
		l,arh,o=np.split(input,[1, int(input[0])+1 ])
		arh=arh.astype (int)
		for i in range (1,len(arh)):
			w1,b1,o=np.split(o, [arh[i-1]*arh[i], (arh[i-1]*arh[i])+arh[i] ] )
			w1=w1.reshape ([arh[i],arh[i-1]])
			b.append(b1)
			w.append(w1)
		self.arhitecture=arh
		self.w=w
		self.b=b

if __name__ == '__main__':
	pass