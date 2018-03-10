from AnnA2mathlib import *
import numpy as np
import random


class Brain: 
	def __init__ (self,arhitecture):
		self.n=[]
		self.b=[]
		self.w=[]
		self.arhitecture=np.array(arhitecture)#np.array([3,3,2])#[784,16,16,10]) [3,3,2])
		self.activationfunction=Activationfunction.sigmoid       #"nofunction","binarystep","sigmoid","tanh","ReLU","leakyReLU"
		self.errorfunction=Errorfunction.cost        #"meanSquaredError", "cost"
		self.derivationstep=0.1
		self.learningcoefficientw=0.1
		self.learningcoefficientb=0.1
		self.outputneuronsfile="outputneurons.txt"
		self.changedneuronsfile="changedneurons.txt"
		self.resultfile="results.txt"

	def birth (self):
		for j,i in enumerate (self.arhitecture):
			self.n.append (np.ones (i))
			self.b.append (np.random.random (i))
			if not (j==(len(self.arhitecture)-1)):
				self.w.append (np.random.random ((self.arhitecture [j+1],i)))


	def connect (self,input_layer):
		self.n[0]=input_layer
		for i in range(len(self.n)-1):
			self.n[i+1]=self.activationfunction (self.w[i].dot(self.n[i])-self.b[i+1])
		return self.n [len(self.n)-1]

	def getfdx (self, variable):	
		if variable=="w":
			weight=tensortovector (self.w)
			dvect=weight+np.eye (weight.size)*self.derivationstep
			output=[]
			cutpos=0
			for i in range(len(self.w)):    #vector to tensor for w
				dimx,dimy=self.w[i].shape
				b=dvect[:,cutpos:cutpos+dimx*dimy]
				b=b.reshape (-1,dimx,dimy)
				output.append (b)
				cutpos+=dimx*dimy
			neuroni =self.n[0]				#neurons from changed w
			for i in range (len(output)):
				neuroni=self.activationfunction((output[i]*neuroni).sum (axis=2)-self.b[i+1])
				if not (i==len(output)-1):
					neuroni=np.expand_dims(neuroni, axis=1)
		elif variable=="b":
			bias =tensortovector (self.b)
			dvect=bias+np.eye (bias.size)*self.derivationstep
			output=[]
			cutpos=0
			for i in range(len(self.b)):	  #vector to tensor for b
				dimx=self.b[i].shape [0]
				b=dvect[:,cutpos:cutpos+dimx]
				output.append (b)
				cutpos+=dimx
			neuroni= (np.zeros(output[0].shape) + self.n[0]).T	#neurons from changed b
			for i in range (len(self.n)-1):
				neuroni=self.activationfunction(self.w[i].dot(neuroni) - output[i+1].T )
			neuroni=neuroni.T [self.arhitecture[0]:,:]
		return neuroni

	def mjeri (self,input,result,style=1):
		if style==0:
			with open(self.outputneuronsfile, "a") as file:
				np.savetxt (file,self.n[len(self.n)-1])
			with open(self.resultfile, "a") as file:
				np.savetxt (file,result)
			with open(self.changedneuronsfile, "a") as file:
				np.savetxt(file,self.getfdx ("w"), delimiter="\n")
				np.savetxt(file,self.getfdx ("b"), delimiter="\n")
		elif style==1:
			with open(self.outputneuronsfile, "ab") as file:
				np.save (file,self.n[len(self.n)-1])
			with open(self.resultfile, "ab") as file:
				np.save (file,result)
			with open(self.changedneuronsfile, "ab") as file:
				a=self.getfdx ("w")
				a=np.append (a, self.getfdx ("b"))
				np.save (file,a)

	def getmjerenja (self,style=1):
		if style==0:
			with open(self.outputneuronsfile, "r") as file:
				outputneurons=np.reshape(np.loadtxt(file),(-1,self.n[len(self.n)-1].size))
			with open(self.resultfile, "r") as file:
				result=np.reshape(np.loadtxt(file),(-1,self.n[len(self.n)-1].size))
			with open(self.changedneuronsfile, "r") as file:
				changedneurons=np.reshape(np.loadtxt(file),(sum(self.arhitecture)+sum([i.size for i in self.w])-self.arhitecture[0],-1,self.n[len(self.n)-1].size))
		elif style==1:
			with open(self.outputneuronsfile, "rb") as file:
				outputneurons=np.reshape(np.load(file),(-1,self.n[len(self.n)-1].size))
			with open(self.resultfile, "rb") as file:
				result=np.reshape(np.load(file),(-1,self.n[len(self.n)-1].size))
			with open(self.changedneuronsfile, "rb") as file:
				changedneurons=np.reshape(np.load(file),(sum(self.arhitecture)+sum([i.size for i in self.w])-self.arhitecture[0],-1,self.n[len(self.n)-1].size))
		open(self.outputneuronsfile, 'w').close()
		open(self.resultfile, 'w').close()
		open(self.changedneuronsfile, 'w').close()
		return outputneurons, result, changedneurons

	def uci (self, outputneurons, result, changedneurons):
		derivation=(self.errorfunction(changedneurons,result)-self.errorfunction(outputneurons,result))/self.derivationstep
		pomw=derivation [:derivation.size-sum(self.arhitecture)+self.arhitecture[0]]
		pomb=np.concatenate((np.zeros(self.arhitecture[0]), derivation [derivation.size-sum(self.arhitecture)+self.arhitecture[0]:]))
		a=0
		for i in range (len (self.arhitecture)):		#change b
			self.b[i]-=pomb[a:a+self.arhitecture[i]]*self.learningcoefficientb
			a+=self.arhitecture[i]
		a=0
		for i in range (len (self.w)):					#change w
			self.w[i]-=pomw[a:a+self.w[i].size].reshape(self.w[i].shape)*self.learningcoefficientw*(-1)
			a+=self.w[i].size
		return 

if __name__ == '__main__':
	()