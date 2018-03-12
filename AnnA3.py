import numpy as np
from AnnA3mathlib import *
class Brain: 
	def __init__ (self, arhitecture, mjerenja):
		self.n=[]
		self.b=[]
		self.w=[]
		self.z=[]
		self.delta=[]
		self.alpha=0.1
		self.mjerenja=mjerenja
		self.arhitecture=np.array(arhitecture)
		self.activationfunction=Activationfunction.sigmoid

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

	def korak1 (self):
		self.n[0]=self.mjerenja[0]
		for i in range(len(self.n)-1):
			self.z[i+1]=self.w[i+1].dot(self.n[i])+self.b[i+1].reshape(-1,1)
			self.n[i+1]=self.activationfunction (self.z[i+1])

	def korak2 (self):
		y=self.mjerenja[1]
		for l in reversed(range(1,len(self.n))):
			if l==(len(self.n)-1):
				self.delta[l]=self.n[l]-y
			else:
				self.delta[l]=self.w[l+1].T.dot(self.delta[l+1])*self.activationfunction (self.z[l],True)

	def korak3 (self):
		M=self.mjerenja[1].shape[1]
		for l in range (1,len(self.n)-1):
			self.w[l]=self.w[l]-(self.alpha/M)*self.delta[l].dot(self.n[l-1].T)
			self.b[l]=self.b[l]-(self.alpha/M)*np.sum (self.delta[l],axis=1)


if __name__ == '__main__':
	brm=5
	arh=[4,3,2]
	mjerenja=[np.random.random ( (arh[0],brm) ),np.random.random ((arh[len(arh)-1],brm))]
	a=Brain (arh, mjerenja)
	a.birth ()
	a.korak1 ()
	a.korak2 ()
	a.korak3 ()