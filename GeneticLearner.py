from AnnA3 import *
class Population():
		def __init__ (self, arhitecture=[2,1], numPerGen=10, alpha=0.1, activationFunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost, fixedAlpha=True):
			self.numPerGen=numPerGen
			self.arhitecture=arhitecture
			self.alpha=alpha
			self.activationFunction=activationFunction
			self.errorFunction=errorFunction
			self.fixedAlpha=fixedAlpha
			self.generation=self.genesis(num=self.numPerGen)

		def genesis (self,num=1): #napravi prvu populaciju
			out=np.array([],dtype=object)
			for i in range (num):
				mozak=Brain(arhitecture=self.arhitecture, alpha=self.alpha, activationFunction=self.activationFunction, errorFunction=self.errorFunction, fixedAlpha=self.fixedAlpha)
				mozak.birth()
				out=np.append (out,mozak)
			return out

		def sex (self,brain1,brain2,rand=0.1):
			rnd=[0]
			child=[]
			for i,j in enumerate (self.arhitecture):
				if not (i==0):
					rnd.append (np.random.random ((j)) ) #napravi matricu random brojeva (baca kocku)
			for i,j in enumerate(rnd):
				j=np.array(j)
				mater =(np.where(j<=(1-rand)/2, 1, np.where( (j>(1-rand)/2) & (j<=1-rand), 0, 0))*np.array(brain1.w[i]).T ).T #zemi samo od mame gene ostalo 0
				otac  =(np.where(j<=(1-rand)/2, 0, np.where( (j>(1-rand)/2) & (j<=1-rand), 1, 0))*np.array(brain2.w[i]).T ).T #zemi samo od oca  gene ostalo 0
				random=(np.where(j<=(1-rand)/2, 0, np.where( (j>(1-rand)/2) & (j<=1-rand), 0, 1))*np.random.random ((np.array(brain1.w[i]).shape)).T ).T #random geni ostalo 0
				if i==0:
					mater=int(mater)
					otac=int (otac) #čisto konzistencije radi da nije 0.0 nego 0 na nultom mjestu
					random=int (random)
				child.append(mater+otac+random)#zbroji sve 
			return child
		
		def najbolji (self, population, fitnesses, numberofsurvived):
			breed=[]
			fitnesses=-np.sort(-fitnesses)				#sortira vektor scoreova
			for i in range (numberofsurvived):
				for j in range (self.numPerGen):		 #traži koji mozak ima score jednak prih deset scorove u sortiranom vektoru
					if (fitnesses[i]==population[j].score):
						breed.append (population[j])
			return breed.copy()
		
		def roulette(self,population, fitnesses, num):

			total_fitness = float(sum(fitnesses))
			rel_fitness = [f/total_fitness for f in fitnesses]
		    # Generate probability intervals for each individual
			probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
		    # Draw new population
			new_population = []
			for n in range(num):
				r = np.random.uniform()
				for (i, individual) in enumerate(population):
					if r <= probs[i]:
						new_population.append(individual)
						print (individual.w)
						break
			return new_population

		def darwin (self, numberofsurvived=4, rulet=2): #rulet: 1(najbolji) 2(rulet za najviše šanse ima najbolji) ostalo (svi)
			oldgeneration=[]
			sc=np.array([]) 	#radi vektor u kojeg stavlja scorove
			for i in range (self.numPerGen):
				sc=np.append(sc,self.generation[i].score)
			if rulet==1:
				oldgeneration=self.najbolji(self.generation,sc,numberofsurvived)   #stavlja trenutnu generaciju u staru generaciju i onda radi novu generaciju praznih mozgova
			elif rulet==2:
				oldgeneration=self.roulette(self.generation,sc,numberofsurvived)
			else: oldgeneration=self.generation.copy()
			self.generation=self.genesis(num=int(numberofsurvived*(numberofsurvived-1)/2)) #num=broj mogućih kombinacija
			u=0
			for i in range(numberofsurvived):			 #sexa mozak svaki sa svakim, bez ponavljanja {3,1}={1,3} i bez {3,3}
				for j in range(i+1,numberofsurvived):
					self.generation[u].w=self.sex(oldgeneration[i],oldgeneration[j])
					#print (self.generation[u].w)		 #stavlja nove mozgove u trenutnu generaciju
					u+=1

if __name__ == '__main__':
	pop=Population()
	pop.darwin()