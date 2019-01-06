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

def learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_dev", db_file='Data.db', briteracija=100, alpha=0.1, activationfunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost, arh=[0], name_brain="UnnamedBrain", save_brain=True, print_stats=True):
	start=1
	trainlista=getdata(TableName=TableNameTrain,db_file=db_file)
	devlista = getdata(TableName=TableNameDev,db_file=db_file)
	if (name_brain+"_save.npy") in os.listdir("."):
		with open (name_brain+"_info.npy", "rb") as file:
			[alpha, activationfunction, errorFunction, arh, start, error, accuracy] = pickle.load(file)

	mozak=Brain (arhitecture=arh, mjerenja=trainlista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction)
	mozak.birth()
	if (name_brain+"_save.npy") in os.listdir("."):
		mozak.loadbrain (name_brain+"_save.npy")

	for j in range (start, briteracija+start):
		error,accuracy= mozak.learn (mjerenja=trainlista)
		accuracy1= mozak.test (mjerenja=devlista)
		if print_stats:
			print("EPOH: "+str("{:7.0f}".format(j))+"/"+str(briteracija+start-1)+"  -----"+"   ACCURACY(test): "+str ("{:7.4f}".format(accuracy))+"   ACCURACY(dev): "+str ("{:7.4f}".format(accuracy1)) + "   ERROR:"+str ("{:9.4f}".format(error)))

	if save_brain:
		mozak.savebrain (name_brain+"_save.npy")
		with open (name_brain+"_info.npy", "wb") as file:
			pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, briteracija+start, error, accuracy], file)
	return briteracija+start,accuracy, accuracy1, error



if __name__ == '__main__':
	learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_train", db_file='Data.db', briteracija=1, alpha=0.0001, arh=[1,10,10,1], activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid])
#precision, recall, f1 score, ROC curve