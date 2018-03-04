from AnnA2 import *
import cv2
import os

folder="test"
a=Brain ()
a.arhitecture = np.array([2,1])
a.birth ()
for j in range (100):	
	print (j)
	for i in os.listdir(".\\"+folder):
		input=np.ravel(cv2.imread(".\\"+folder+"\\"+i,0)/255)
		result=np.array([os.path.splitext(i)[0].split ("(")[0]]).astype(np.float)
		a.connect (input)
		a.mjeri (input,result)
	k=a.getmjerenja ()
	a.uci (k[0],k[1],k[2])
for i in os.listdir(".\\"+folder):
	input=np.ravel(cv2.imread(".\\"+folder+"\\"+i,0)/255)
	result=np.array([os.path.splitext(i)[0].split ("(")[0]]).astype(np.float)
	a.connect (input)
	print("result:"+str(result)+"     output:"+str(a.n[1]))

