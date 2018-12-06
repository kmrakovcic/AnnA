from AnnA3mathlib import *

def resetParameters (arh):
	w=[0]
	b=[0]
	for i,j in enumerate (arh):
		if not (i==0):
			w.append([])
			b.append([])
	return w,b

class Brain: 
	def __init__ (self, arhitecture=[1,1], mjerenja=[], alpha=0.1, activationFunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost, fixedAlpha=True):
		self.n=[]
		self.b=[]
		self.w=[]
		self.newb=[]
		self.neww=[]
		self.z=[]
		self.delta=[]
		self.alpha=alpha
		self.mjerenja=mjerenja
		self.activationfunction=activationFunction
		self.errorFunction=errorFunction
		self.score=np.random.uniform()
		if not mjerenja==[]:
			arhitecture[0]=mjerenja[0].shape [0]
			arhitecture[len(arhitecture)-1]=mjerenja[1].shape[0]
		self.arhitecture=np.array(arhitecture)
		self.fixedAlpha = fixedAlpha

	def birth (self):
		if not self.mjerenja==[]:
			M=self.mjerenja[1].shape[1]
		else: M=1
		self.b=[0]
		self.w=[0]
		self.newb=[0]
		self.neww=[0]
		self.delta=[0]
		for i,j in enumerate (self.arhitecture):
			self.n.append (np.ones((j,M)) )
			self.z.append (np.ones ((j,M))) 
			if not (i==0):
				self.b.append (np.random.random (j) )
				self.w.append (np.random.random ((j, self.arhitecture [i-1])))
				self.delta.append ([])
				self.neww.append ([])
				self.newb.append ([])

	def fowardpropagation (self, new=False):
		self.n[0]=self.mjerenja[0]
		if new:
			for i in range(len(self.n)-1):
				self.z[i+1]=self.neww[i+1].dot(self.n[i])+self.newb[i+1].reshape(-1,1)
				self.n[i+1]=self.activationfunction (self.z[i+1])
		else:
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
		self.neww, self.newb = resetParameters(self.arhitecture)
		M=self.mjerenja[1].shape[1]
		for l in range (1,len(self.n)):
			self.neww[l]=self.w[l]-(self.alpha/M)*self.delta[l].dot(self.n[l-1].T)
			self.newb[l]=self.b[l]-(self.alpha/M)*np.sum (self.delta[l],axis=1)

	def checkConvergence (self):
		error1 = self.errorFunction (self.n[len(self.n)-1], self.mjerenja[1])
		self.fowardpropagation(new=True)
		error2 = self.errorFunction (self.n[len(self.n)-1], self.mjerenja[1])

		if error1 == error2:
			return True

		if error2 < error1:
			self.alpha += self.alpha*0.1
			for i in range(len(self.w)):
				self.w[i] = self.neww[i]
				self.b[i] = self.newb[i]
			return True

		if error1 < error2:
			self.alpha -= self.alpha*0.1
			return False

	def learn (self, mjerenja=[]):
		if not mjerenja==[]: self.mjerenja=mjerenja
		self.fowardpropagation ()
		self.backpropagation ()
		self.evolve ()
		if self.fixedAlpha:
			self.w = self.neww
			self.b = self.newb
			conv = True
		else:
			conv = self.checkConvergence ()

		acc=getaccuracy (self.n[len(self.n)-1],self.mjerenja[1])
		err=self.errorFunction (self.n[len(self.n)-1],self.mjerenja[1])
		return err,acc,conv

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