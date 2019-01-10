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

def learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_dev", db_file='Data.db', briteracija=0, nauceno=100, alpha=0.1, threshold=0.5, activationfunction=Activationfunction.sigmoid, errorFunction=Errorfunction.cost, arh=[0], name_brain="UnnamedBrain", save_brain=True, print_stats=True):
	start=1
	trainlista=getdata(TableName=TableNameTrain,db_file=db_file)
	devlista = getdata(TableName=TableNameDev,db_file=db_file)
	if (name_brain+"_save.npy") in os.listdir("."):
		with open (name_brain+"_info.npy", "rb") as file:
			[alpha2, activationfunction, errorFunction, arh, start, error, accuracy] = pickle.load(file)

	mozak=Brain (arhitecture=arh, mjerenja=trainlista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction)
	mozak.birth()
	if (name_brain+"_save.npy") in os.listdir("."):
		mozak.loadbrain (name_brain+"_save.npy")
	j=start
	ctrl = True
	if not (briteracija==0):
		nuceno=100
	try:	
		while ctrl:
			n,y= mozak.learn (mjerenja=trainlista)
			accuracytrain=getstats(n,y,threshold=threshold)[0]
			errortrain = errorFunction (n,y)
			n,y=mozak.test (mjerenja=devlista)
			accuracydev,f1score=getstats(n,y,threshold=threshold)[0:2]
			if print_stats:
				print("EPOH:"+str("{:7.0f}".format(j))+"/"+str(briteracija+start-1)+" -----"+ "   ERROR:"+str ("{:9.4f}".format(errortrain))+"   ACCURACY(train):"+str ("{:7.4f}".format(accuracytrain))+"   ACCURACY(dev):"+str ("{:7.4f}".format(accuracydev))+"   F1_SCORE:"+str ("{:7.4f}".format(f1score))  )
			j+=1
			if (j==start+briteracija) or (accuracydev>=nauceno): #prekida učenje ako je naučio sa dovoljnim accuracy ili je dovoljno puta učio
				ctrl=False
	
	except KeyboardInterrupt: #sprema mozak u slučaju ctrl+c
		if save_brain:
				mozak.savebrain (name_brain+"_save.npy")
				with open (name_brain+"_info.npy", "wb") as file:
					pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, j, error, accuracy], file)

	tpr=[]
	fpr=[]
	ppv=[]
	accuracy=[]
	f1score=[]
	thrashold=[]
	for i in np.arange (0,1,0.1):
		accuracy1,f1score1, tpr1, fpr1, ppv1=getstats(n,y,threshold=i)
		accuracy.append(accuracy1)
		f1score.append(f1score1)
		tpr.append(tpr1)
		fpr.append(fpr1)
		ppv.append(ppv1)
		thrashold.append (i)

	if save_brain: #sprema mozak na kraju
		mozak.savebrain (name_brain+"_save.npy")
		with open (name_brain+"_info.npy", "wb") as file:
			pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, j, error, accuracy], file)
	return briteracija+start, errortrain, thrashold, accuracy, f1score, tpr, fpr, ppv



if __name__ == '__main__':
	a=learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_train", db_file='Data.db', briteracija=0, nauceno=90, alpha=0.001, threshold=0.5, arh=[1,10,10,1], activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid])
	print (a[4])
#ROC curve