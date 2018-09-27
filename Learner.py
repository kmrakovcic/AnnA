from AnnA3 import *
import sqlite3
import os
import pickle

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
	col_name_list_input  = ",".join([s for s in col_name_list.split(",")[:-1] if "Input"   in s])
	col_name_list_output = ",".join([s for s in col_name_list.split(",")[:-1] if "Output"  in s])
	
	c.execute("SELECT "+col_name_list_input+" FROM "+TableName)
	input=np.array(c.fetchall())
	c.execute("SELECT "+col_name_list_output+" FROM "+TableName)
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

def learner (lista, alpha=0.1, activationfunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost, arh=[0], fixedAlpha=False, briteracija=1, name_brain="UnnamedBrain"):
	output=""
	start=1
	if arh==[0]:
		arh= automatic_arh(lista)

	if (name_brain+"_save.npy") in os.listdir("."):
		with open (name_brain+"_info.npy", "rb") as file:
			[alpha, activationfunction, errorFunction, arh, fixedAlpha, start, error, accuracy] = pickle.load(file)

	mozak=Brain (arhitecture=arh, mjerenja=lista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction, fixedAlpha=fixedAlpha)
	mozak.birth()
	
	if (name_brain+"_save.npy") in os.listdir("."):
		mozak.loadbrain (name_brain+"_save.npy")

	for j in range (start, briteracija+start):
		convergance= False
		while not convergance:
			error,accuracy,convergance= mozak.learn ()

		progressbar="EPOH: "+str("{:7.0f}".format(j))+"/"+str(briteracija+start-1)+" ----- ERROR:"+str ("{:9.4f}".format(error))+"   ACCURACY: "+str ("{:7.4f}".format(accuracy))
		if not mozak.fixedAlpha:
			progressbar+= "   LEARNING RATE: "+str("{:7.4f}".format(mozak.alpha))
		output+=progressbar+"\n"
		print(progressbar)
	mozak.savebrain (name_brain+"_save.npy")
	with open (name_brain+"_info.npy", "wb") as file:
		pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, mozak.fixedAlpha, briteracija+start, error, accuracy], file)
	return output


if __name__ == '__main__':
	learner (getdata(),briteracija=1000, arh=[4,10,3],activationfunction=Activationfunction.sigmoid)
	input ()