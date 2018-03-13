from AnnA3 import *
import cv2
import os

def getinput (folder):
	input=np.empty (0)
	result=np.empty (0)
	for i in os.listdir(folder):
		input=np.append(input, np.ravel(cv2.imread(folder+"/"+i,0)/255),axis=0)
		result=np.append(result,np.array(os.path.splitext(i)[0].split ("(")[0].split ("-")).astype(np.float))
	input=input.reshape (len(os.listdir(folder)),-1).T
	result=result.reshape (len(os.listdir(folder)),-1).T
	return input, result

def printstate (brain, mjerenja):
	mj=mjerenja[1].T
	k1=""
	k2=""
	k=""
	for i1,j1 in enumerate(brain.n[len(brain.n)-1].T):
		for i2,j2 in enumerate (j1):
			k1+="|"+str ("{:5.2f}".format(j2))+"|   " 
			k2+="|"+str ("{:5.2f}".format(mj[i1][i2]))+"|   "
		k+=k1+"---------------------------------->   "+k2+"\n"
		k1=""
		k2=""
	return k

mjerenja=getinput ("tests2x2")
a=Brain ([4,20,2],mjerenja,alfa=5)
a.birth ()
x1=""
for j in range (1,1001):
	a.korak1 ()
	a.korak2 ()
	a.korak3 ()
	if  (j%100==0):
		print (str(j)+". mjerenje")
		print (printstate (a,mjerenja))

