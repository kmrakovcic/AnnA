from AnnA3 import *
import cv2
import os

def inputfromjpg (folder):
	input=np.empty (0)
	result=np.empty (0)
	a=0
	for i in os.listdir(folder):
		a+=1
		print (str ("{:5.2f}".format (100*a/len(os.listdir(folder))))+"% slika ucitano")
		input=np.append(input, np.ravel(cv2.imread(folder+os.sep+i,0)/255),axis=0)
		result=np.append(result,np.array(os.path.splitext(i)[0].split ("(")[0].split ("-")).astype(np.float))
	input=input.reshape (len(os.listdir(folder)),-1).T
	result=result.reshape (len(os.listdir(folder)),-1).T
	with open (folder+"(x).npy", "wb") as file:
		np.append(input.shape,input.flatten()).tofile (file)
	with open (folder+"(y).npy", "wb") as file:
		np.append(result.shape,result.flatten()).tofile (file)
	os.system('cls')
	return input, result

def inputfromnpy (ime):
	mjerenja=[0,0]
	with open (ime+"(x).npy", "rb") as file:
		got=np.fromfile(file)
		shape=[int(got[0]),int(got[1])]
		mjerenja[0]=np.delete(got,[0,1]).reshape(shape)
	with open (ime+"(y).npy", "rb") as file:
		got=np.fromfile(file)
		shape=[int(got[0]),int(got[1])]
		mjerenja[1]=np.delete(got,[0,1]).reshape(shape)
	return mjerenja

def imput (folder):
	thisf=os.listdir(".")
	if (folder+"(x).npy") in thisf and (folder+"(x).npy") in thisf:
		return inputfromnpy (folder)
	elif folder in thisf:
		return inputfromjpg (folder)
	else:
		return np.array( [[1,0], [0,1], [0,0], [1,1]]).T, np.array([[1,1,0,0]]) 

def printstate (brain, mjerenja):
	mj=mjerenja[1].T
	k1=""
	k2=""
	k=""
	for i1,j1 in enumerate(brain.n[len(brain.n)-1].T):
		for i2,j2 in enumerate (j1):
			k1+="|"+str ("{:5.2f}".format(j2))+"| " 
			k2+="|"+str ("{:5.2f}".format(mj[i1][i2]))+"| "
		k+=k1+"----> "+k2+"\n"
		k1=""
		k2=""
	return k


def automatic_arh (mjerenja,alpha=0): # 0 hiddden layera alpha=0, 1 hidden layer alpha>2, 2 hiddden layer alpha=2
	N=mjerenja[1].shape[1] #sample size
	m=mjerenja[1].shape[0] #output neurons
	n=mjerenja[0].shape [0] #input neurons
	if (alpha==0):
		arh=[n,m]
	elif (alpha<2):
		hidden1=int(round(math.sqrt((m+2)*N)+2*math.sqrt(N/(m+2))))
		hidden2=int (round (m*math.sqrt(N/(m+2))))
		arh=[n,hidden1,hidden2,m]
	else:
		hidden1=int(round(N/(alpha*(n+m))))
		if hidden1==0: 
			hidden1=1
		arh=[n,hidden1,m]
	return arh

def mainloop (mjerfolder="",arh=[0], briteracija=1, alpha=0.1):
	brojprinteva=1
	mjerenja=imput(mjerfolder)
	if arh==[0]:
		arh= automatic_arh(mjerenja,2)
	a=Brain (arh,mjerenja,alpha)
	a.birth ()
	print (a.arhitecture)
	scr=""
	for j in range (1,briteracija+1):
		a.korak1 ()
		a.korak2 ()
		a.korak3 ()
		error=a.errorFunction (a.n[len(a.n)-1], mjerenja[1])
		progressbar="ITERATION: "+str(j)+"/"+str(briteracija)+"  "+str ("{:3.0f}".format(j/briteracija*100))+"% ----- ERROR: "+str ("{:7.5f}".format(error))
		print (progressbar)
		if  (j%(briteracija//brojprinteva)==0):
			scr+=str(j)+". iteration\n"+printstate (a,mjerenja)
	input ("Press Any Key To See Log")
	os.system('cls')
	print (scr)

mjerfolder=input ("Folder mjerenja?\n")
briteracija=int(input ("Broj iteracija?\n"))
alpha=float(input ("Learning rate?\n"))
i=int (input("Broj hidden layera?\n"))
if i==0: arh=[0]
else:
	arh=[1]
	for j in range (1,i+1):
		arh.append ( int( input("Broj neurona u "+str(j)+". hidden layeru?\n" ) ) )
	arh.append (1)
mainloop (mjerfolder, arh, briteracija, alpha)
input()
