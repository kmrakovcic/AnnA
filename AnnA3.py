from AnnA3mathlib import *

def resetParameters (arh):  #postavlja sve weightove i biase na 0
	w=[0]
	b=[0]
	for i,j in enumerate (arh):
		if not (i==0):
			w.append([])
			b.append([])
	return w,b

class Brain: 
	def __init__ (self, arhitecture=[1,1], mjerenja=[], alpha=0.1, activationFunction=[Activationfunction.sigmoid,Activationfunction.sigmoid], errorFunction=Errorfunction.cost):
		self.n=[]           	#neuroni
		self.b=[]				#biasi
		self.w=[]				#weightovi
		self.newb=[]			#pomoćni biasi
		self.neww=[]			#pomoćni weightovi
		self.z=[]				#neuroni bez aktivacijske funkcije
		self.delta=[]			#delte
		self.alpha=alpha 		#learning rate
		self.mjerenja=mjerenja  #izmjereni podatci
		self.activationfunction=activationFunction  #aktivacijska funkcija
		self.errorFunction=errorFunction 			#error funkcija
		if not mjerenja==[]:    #ako su mjerenja ubačena prilikom stvaranja mozga onda mozak automatski radi arhitekturu tako da na input layer stavlja neurona koliko postoji mjerenja
			arhitecture[0]=mjerenja[0].shape [0]
			arhitecture[len(arhitecture)-1]=mjerenja[1].shape[0]
		self.arhitecture=np.array(arhitecture)      #arhitektura mozga
		
		#dio koji provjerava točnost aktivacijskih funkcija u odnosu na arhitekturu
		if not len(self.arhitecture)==len(self.activationfunction):
			self.activationfunction=[]  #ako nije dobre duljine vektor, stavlja sve sigmoide u vektor
			for i in self.arhitecture:
				self.activationfunction.append(Activationfunction.sigmoid)
		self.activationfunction [len(self.activationfunction)-1]=Activationfunction.sigmoid #zadnji layer treba bit sigmoida

	def birth (self):    #stvara tenzore alfi, neurona, weightova i biasa, te stavlja u njih random vrijednosti
		if not self.mjerenja==[]:
			M=self.mjerenja[1].shape[1]  #gleda tenzor mjerenja i iz njega gleda koliko mjerenja ima
		else: M=1
		self.b=[0]
		self.w=[0]
		self.newb=[0]
		self.neww=[0]
		self.delta=[0]
		for i,j in enumerate (self.arhitecture):
			self.n.append (np.ones((j,M)) )     #radi jediničnu matricu neurona (arhitektura mozga x broj mjerenja) za svaku liniju neurona
			self.z.append (np.ones ((j,M))) 	#isto to za z
			if not (i==0):
				self.b.append (np.random.random (j) )  #radi vektor biasa (duljine koliko kaže arhitektura mozga) za svaki red neurona
				self.w.append (np.random.random ((j, self.arhitecture [i-1]))) #radi tenzor weightova za svaki red neurona, s tim da je 0. red neurona prazan i weigtovi su od pozicije 2
				self.delta.append ([]) #prazni vektori za svaki red neurona za delte i pomoćne neurone i weightove
				self.neww.append (np.random.random (j)) 
				self.newb.append (np.random.random ((j, self.arhitecture [i-1])))

	def fowardpropagation (self, new=False):
		self.n[0]=self.mjerenja[0] #na input red neurona stavlja izmjerene vrijednosti
		for i in range(len(self.n)-1):
			self.z[i+1]=self.w[i+1].dot(self.n[i])+self.b[i+1].reshape(-1,1)   #sumiranje prijašnijh neurona u z (neuron bez aktivacijske funkcije)
			self.n[i+1]=self.activationfunction [i+1](self.z[i+1])                  #djelovanje aktivacijske funkcije u smjeru z-a, i to je onda neuron :D

	def backpropagation (self):
		y=self.mjerenja[1]
		for l in reversed(range(1,len(self.n))):  #ide od zadnjeg reda neurona prema prvom
			if l==(len(self.n)-1):                #algoritam za backpropagaciju
				self.delta[l]=self.n[l]-y
			else:
				self.delta[l]=self.w[l+1].T.dot(self.delta[l+1])*self.activationfunction [l+1] (self.z[l], d=True)

	def evolve (self):
		self.neww, self.newb = resetParameters(self.arhitecture)
		M=self.mjerenja[1].shape[1]
		for l in range (1,len(self.n)):
			self.neww[l]=self.w[l]-(self.alpha/M)*self.delta[l].dot(self.n[l-1].T)
			self.newb[l]=self.b[l]-(self.alpha/M)*np.sum (self.delta[l],axis=1)

	def learn (self, mjerenja=[]):    #algoritam za učenje
		if not mjerenja==[]: self.mjerenja=mjerenja #stavlja na input neurone mjerenja, ako još nisu stavljena
		self.fowardpropagation ()   
		self.backpropagation ()          #algoritam za učenje
		self.evolve ()
		self.w = self.neww               #stavlja pomoćne neurone u prave neurone (može se to elegantnije, to je tu radi legacy razloga)
		self.b = self.newb
		return self.n[len(self.n)-1], self.mjerenja[1]

	def test (self, mjerenja=[]):
		if not mjerenja==[]: self.mjerenja=mjerenja
		self.fowardpropagation ()
		return self.n[len(self.n)-1], self.mjerenja[1]

	def savebrain (self,name):   #sprema weightove i biase u datoteku
		output=np.append(np.array(len(self.arhitecture)),self.arhitecture )
		for i in range(1,len(self.arhitecture)):
			output=np.append(output, np.append (self.w[i].flatten(),self.b[i].flatten()) )
		with open (name, "wb") as file:
			output.tofile (file)

	def loadbrain (self, name):  #preuzima weightove i biase iz datoteke
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