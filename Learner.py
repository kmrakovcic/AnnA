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

def learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_dev", db_file='Data.db', briteracija=0, nauceno=90, alpha=1, tau=0.001, threshold=0.5, activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid], errorFunction=Errorfunction.cost, LRschedule=AdaptiveLR.timebased, arh=[1,10,10,10,1], name_brain="UnnamedBrain", save_brain=True, print_stats=True):
	start=1
	if not (os.path.isdir(name_brain)):
		os.mkdir (name_brain)
	if not(os.path.isdir(name_brain+"/save")):
		os.mkdir (name_brain+"/save")
	trainlista=getdata(TableName=TableNameTrain, db_file=db_file)
	devlista = getdata(TableName=TableNameDev, db_file=db_file)
	if ("info.npy") in os.listdir(name_brain+"/save"):
		with open (name_brain+"/save"+"/info.npy", "rb") as file:
			[alpha2, activationfunction, errorFunction, arh, start, error, accuracydev, tau] = pickle.load(file)

	mozak=Brain (arhitecture=arh, mjerenja=trainlista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction)
	mozak.birth()
	if ("wsave.npy") in os.listdir(name_brain+"/save"):
		mozak.loadbrain (name_brain+"/save"+"/wsave.npy")
	epoh=start
	ctrl = True
	if not (briteracija==0):
		nuceno=100
	try:	
		while ctrl:
			n1,y1= mozak.learn (mjerenja=trainlista)
			accuracytrain=getstats(n1,y1,threshold=threshold)[0]
			errortrain = errorFunction (n1,y1)
			n,y=mozak.test (mjerenja=devlista)
			accuracydev,f1score=getstats(n,y,threshold=threshold)[0:2]
			if print_stats:
				print("EPOH: "+str(epoh)+"/"+str(briteracija+start-1)+" (LR="+str("{:.2e}".format(mozak.alpha))+")"+" -/-/- "+ "ERROR:"+str ("{:.4f}".format(errortrain))+" ||ACCURACY train:"+str ("{:.4f}".format(accuracytrain*100))+", dev:"+str ("{:.4f}".format(accuracydev*100))+"|| F1 SCORE:"+str ("{:.4f}".format(f1score))  )
			epoh+=1 
			mozak.alpha=LRschedule(alpha,epoh,tau)
			if (epoh==start+briteracija) or (accuracydev>=nauceno): #prekida učenje ako je naučio sa dovoljnim accuracy ili je dovoljno puta učio
				ctrl=False
	
	except KeyboardInterrupt: #sprema mozak u slučaju ctrl+c
		if save_brain:
				mozak.savebrain (name_brain+"/save"+"/wsave.npy")
				with open (name_brain+"/save"+"/info.npy", "wb") as file:
					pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, epoh, errortrain, accuracydev, tau], file)



	if save_brain: #sprema mozak na kraju
		mozak.savebrain (name_brain+"/save"+"/wsave.npy")
		with open (name_brain+"/save"+"/info.npy", "wb") as file:
			pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, epoh, errortrain, accuracydev, tau], file)
	return briteracija+start

def tester (TableNameTest="MAGIC_dev", db_file='Data.db', name_brain="UnnamedBrain", resolution=100, graf=False):
	
	testlista=getdata(TableName=TableNameTest, db_file=db_file)
	
	#checking folders and loading info
	if not (os.path.isdir(name_brain)):
		print ("Error: Must learn before test")
		quit()
	if not (os.path.isdir(name_brain+"/save")):
		print ("Error: Must learn before test")
		quit()
	if not(os.path.isdir(name_brain+"/plot")):
		os.mkdir (name_brain+"/plot")
	if ("wsave.npy") in os.listdir(name_brain+"/save"):
		with open (name_brain+"/save"+"/info.npy", "rb") as file:
			[alpha, activationfunction, errorFunction, arh, fixedAlpha, start, error, accuracy] = pickle.load(file)
	else:
		print ("Error: Must learn before test")
		quit()
	

	mozak=Brain (arhitecture=arh, mjerenja=testlista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction)
	mozak.birth()
	mozak.loadbrain (name_brain+"/save/wsave.npy")
	n,y=mozak.test (mjerenja=testlista)
	
	#geting statistic
	tpr=[]
	tprlog=[]
	fpr=[]
	fprlog=[]
	ppv=[]
	accuracy=[]
	f1score=[]
	thrashold=[]
	thrasholdlog=[]
	for i in np.arange (0,1+1/resolution,1/resolution):
		accuracy1,f1score1, tpr1, fpr1, ppv1=getstats(n,y,threshold=i)
		accuracy.append(accuracy1)
		f1score.append(f1score1)
		tpr.append(tpr1)
		fpr.append(fpr1)
		ppv.append(ppv1)
		thrashold.append (i)
	for j in np.logspace(0.1, 1, num=resolution)/10:
		accuracy1,f1score1, tpr1, fpr1, ppv1=getstats(n,y,threshold=j)
		thrasholdlog.append (j)
		tprlog.append(tpr1)
		fprlog.append(fpr1)

	ROC=np.column_stack((fprlog,tprlog))
	accu=np.column_stack((thrashold,accuracy))
	f1=np.column_stack((thrashold,f1score))
	np.savetxt(name_brain+"/plot"+"/ROC.txt",ROC, delimiter=' ')
	np.savetxt(name_brain+"/plot"+"/accuracy.txt",accu, delimiter=' ')
	np.savetxt(name_brain+"/plot"+"/f1score.txt",f1, delimiter=' ')

	
	#plotting
	try: 
		from matplotlib import pyplot as plt
		fig = plt.figure("Accuracy and F1 Score")
		plt.ylim(0,1)
		plt.xlim(0,1)
		plt.xticks(np.arange(0,1.1,0.1))
		plt.yticks(np.arange(0,1.1,0.1))
		plt.xlabel('Threshold')
		plt.plot (thrashold, accuracy, "b")
		plt.plot (thrashold, f1score , "r")
		plt.legend(["Accuracy","F1 Score"])
		plt.savefig (name_brain+"/plot"+"/AccuracyAndF1.png")
		fig= plt.figure ("ROC curve")
		plt.ylim(0,1)
		plt.xlim(0,1)
		plt.xticks(np.arange(0,1.1,0.1))
		plt.yticks(np.arange(0,1.1,0.1))
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.plot (fprlog,tprlog, "-o",markersize=2)
		plt.plot (thrashold,thrashold,"--")
		plt.legend(["ROC curve","Random Chance"])
		plt.savefig (name_brain+"/plot"+"/ROCcurve.png")
		if graf:
			plt.show()
	except: print ("No plotting 4U D:")

if __name__ == '__main__':

	learner (briteracija=0, save_brain=True, alpha=0.0001, tau=0, arh=[1,1])#TableNameTrain="MAGIC_train", TableNameDev="MAGIC_dev", db_file='Data.db', briteracija=1, nauceno=90, alpha=1, tau=0.01, threshold=0.5, arh=[1,10,10,1], activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid], LRschedule=AdaptiveLR.timebased)
	tester (graf=False)