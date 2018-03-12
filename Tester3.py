from AnnA3 import *
import cv2
import os

def getinput (folder):
	input=np.empty (0)
	result=np.empty (0)
	for i in os.listdir(folder):
		input=np.append(input, np.ravel(cv2.imread(folder+"\\"+i,0)/255),axis=0)
		result=np.append(result,np.array(os.path.splitext(i)[0].split ("(")[0].split ("-")).astype(np.float))
	input=input.reshape (len(os.listdir(folder)),-1).T
	result=result.reshape (len(os.listdir(folder)),-1).T
	return input, result

mjerenja=getinput ("tests2x2")
a=Brain ([4,3,2],mjerenja)
a.birth ()
x1=""
#while not (x1=="x"):
for j in range (10000):
	print (j)
	a.korak1 ()
	a.korak2 ()
	a.korak3 ()
os.system('cls')
print (a.n[0])
print ("|||||")
print (a.n[2])
print("________")
print (mjerenja[1])
#	x1=input()