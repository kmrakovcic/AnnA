from AnnA3 import *
from SQL   import *
import sqlite3
import numpy as np

def getdata (TableName="Iris_dataset",db_file='data.db'): #get data from SQL
	try:
		db = sqlite3.connect(db_file)
	except Exception as e:
		error_print+=str(e)
	c = db.cursor()
	c.execute("SELECT * FROM "+TableName)
	col_name_list=""
	for i in [tuple[0] for tuple in c.description] [:-1]:
		col_name_list+=i+","
	col_name_list_input=col_name_list.split("OutputNeuron")[0]
	col_name_list_output=col_name_list.split("InputNeuron")[-1][3:]
	
	c.execute("SELECT "+col_name_list_input[:-1]+" FROM "+TableName)
	input=np.array(c.fetchall())
	c.execute("SELECT "+col_name_list_output[:-1]+" FROM "+TableName)
	output=np.array(c.fetchall())
	c.execute("SELECT Result FROM "+TableName)
	rezultati=np.array(c.fetchall())
	return np.array(input.T), np.array(output.T)

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

def learner (lista, arh=[0], briteracija=1, alpha=0.1):
	if arh==[0]:
		arh= automatic_arh(lista)
	mozak=Brain (arh, lista, alpha=alpha, fixedAlpha=False)
	mozak.birth()
	for j in range (1,briteracija+1):
		convergance= False
		while not convergance:
			error,accuracy,convergance= mozak.learn ()

		progressbar="EPOH: "+str(j)+"/"+str(briteracija)+" ----- ERROR: "+str ("{:7.5f}".format(error))+" ACCURACY: "+str ("{:7.5f}".format(accuracy))
		if not mozak.fixedAlpha:
			progressbar+= " LEARNING RATE: "+str(mozak.alpha)
		print (progressbar)
	mjerfolder="test"
	mozak.savebrain (mjerfolder+"_save.npy")

learner (getdata())