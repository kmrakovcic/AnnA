from AnnA2 import *
import cv2
import os

def fileinit (files):
	filescreated=""
	thisfile=os.listdir(".")
	for i in files [:len(files)-1]:
		if not (i in thisfile):
			open(i, 'w').close()
			filescreated+=i+" "
	if not (files[len(files)-1] in thisfile):
		os.makedirs(".\\"+files[len(files)-1])
		filescreated+=files[len(files)-1]+" directory "
	return filescreated

def getinput (folder):
	result=np.empty (0)
	input=np.empty (0)
	for i in os.listdir(".\\"+folder):
		input=np.append(input, np.ravel(cv2.imread(".\\"+folder+"\\"+i,0)/255),axis=0)
		result=np.append(result,np.array(os.path.splitext(i)[0].split ("(")[0].split ("-")).astype(np.float))
	input=input.reshape (len(os.listdir(".\\"+folder)),-1)
	result=result.reshape (len(os.listdir(".\\"+folder)),-1)
	return input, result

def printstate (brain):
	outn="NEURONS:\n"
	outw="WEIGHTS:\n"
	outb="BIAS:\n"
	for i in brain.n:
		for j in i:
			outn+="|"+str ("{:5.2f}".format(j))+"|   "
		outn+="\n"
	for i in brain.w:
		for j in i:
			outw+="|"
			for k in j:
				outw+="|"+str ("{:5.2f}".format(k))
			outw+="||   "
		outw+="\n"
	for i in brain.b:
		for j in i:
			outb+="|"+str ("{:5.2f}".format(j))+"|   "
		outb+="\n"
	return outw,outb,outn

def automatic_arh (input,output,alpha=0): #alpha>2 za 1 hidden layer, 0 za 0, inače 2 hidden layera
	N=output.shape[0] #sample size
	m=output.shape[1] #output neurons
	n=input.shape [1]
	if (alpha==0):
		arh=[n,m]
	elif (alpha<2):
		hidden1=int(round(math.sqrt((m+2)*N)+2*math.sqrt(N/(m+2))))
		hidden2=int (round (m*math.sqrt(N/(m+2))))
		arh=[n,hidden1,hidden2,m]
	else:
		hidden1=int(round(N/(alpha*(n+m))))
		arh=[n,hidden1,m]
	return arh



if __name__ == '__main__':
	f=["weights.txt","bias.txt","tests2x2"]
	x1=""
	fileinit (f)
	input1, result=getinput (f[len(f)-1])
	arh=automatic_arh (input1,result)
	a=Brain (arh)
	a.birth ()
	while not (x1=="x"):
		log=""
		os.system('cls')
		for j in range (len(input1)):
			a.connect (input1[j])
			a.mjeri (input1[j],result[j])
			log+="TEST: "+str (j+1)+"\n"
			for i in printstate (a):
				log+=i
			log+="RESULT: "+str(result[j])+"\n------------------------\n"
		k=a.getmjerenja ()
		a.uci (k[0],k[1],k[2])
		print (log)
		x1=input()

