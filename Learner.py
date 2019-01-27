from AnnA3 import *
import sqlite3
import os
import pickle

if __name__ == '__main__':
	def unos (notsublime=True):
		TableNameTrain="MAGIC_train"
		TableNameDev="MAGIC_dev"
		db_file='Data.db'
		briteracija=0
		nauceno=90
		alpha=1
		tau=0.1
		threshold=0.5
		activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid]
		errorFunction=Errorfunction.cost
		LRschedule=AdaptiveLR.activeSeljacki
		arh=[1,16,1]
		name_brain="UnnamedBrain"
		save_brain=True
		print_stats=True
		if notsublime:
			name_brain=input ("Unesi ime NEURALNE MREZE: ")
			if name_brain=="": name_brain="UnnamedBrain"
		else: name_brain="UnnamedBrain"
		if not (name_brain=="UnnamedBrain" or os.path.isdir(name_brain) ):
			pot=""
			while (pot=="N" or pot=="n" or pot==""):
				os.system('cls')
				print ("||| Kreiraj novu neuralnu mrezu "+name_brain+" |||")
				arh= [int(x) for x in input("Unesi arhitekturu neuralne mreze: ").split()]
				alpha=float(input ("Unesi learning rate: "))
				tau= float(input ("Unesi koeficijent promjene learning rate-a: "))
				briteracija= int(input ("Unesi broj epoha ucenja: "))
				while not (pot=="y" or pot=="Y" or pot=="N" or pot == "n"):
					os.system('cls')
					print ("Neuralna mreza "+name_brain+":","\nArhitektura:", arh, "\nLearning rate:",alpha,"\nKoeficjent promjene learning rate-a:",tau,"\nBroj epoha:", briteracija,"\nUNESI y ZA POTVRDU, n ZA PONOVNI UPIS")
					pot=input ()
				os.system ('cls')
		
		return TableNameTrain, TableNameDev, db_file, briteracija, nauceno, alpha, tau, threshold, activationfunction, errorFunction, LRschedule, arh, name_brain, save_brain, print_stats


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
	return Normalization.rescaling(np.array(input.T)), np.array(output.T)

def learner (TableNameTrain="MAGIC_train", TableNameDev="MAGIC_dev", db_file='Data.db', briteracija=0, nauceno=90, alpha=1, tau=0.001, threshold=0.5, activationfunction=[Activationfunction.sigmoid,Activationfunction.sigmoid], errorFunction=Errorfunction.cost, LRschedule=AdaptiveLR.activeSeljacki, arh=[1,1], name_brain="UnnamedBrain", save_brain=True, print_stats=True):
	errorb4=9223372036854775807
	start=1
	#making folders
	if not (os.path.isdir(name_brain)):
		os.mkdir (name_brain)
	if not(os.path.isdir(name_brain+"/save")):
		os.mkdir (name_brain+"/save")
	#geting data
	trainlista=getdata(TableName=TableNameTrain, db_file=db_file)
	devlista = getdata(TableName=TableNameDev, db_file=db_file)
	if ("info.npy") in os.listdir(name_brain+"/save"):
		with open (name_brain+"/save"+"/info.npy", "rb") as file:
			[alpha, activationfunction, errorFunction, arh, start, error, accuracydev, tau] = pickle.load(file)

	mozak=Brain (arhitecture=arh, mjerenja=trainlista, alpha=alpha, activationFunction=activationfunction, errorFunction=errorFunction)
	mozak.birth()
	if ("wsave.npy") in os.listdir(name_brain+"/save"):
		mozak.loadbrain (name_brain+"/save"+"/wsave.npy")
	epoh=start
	ctrl = True
	if not (briteracija==0):
		nauceno=100
	try:	
		while ctrl:
			n1,y1= mozak.learn (mjerenja=trainlista)
			accuracytrain=getstats(n1,y1,threshold=threshold)[0]
			errortrain = errorFunction (n1,y1)/np.shape(trainlista[0])[1]
			n,y=mozak.test (mjerenja=devlista)
			accuracydev,f1score=getstats(n,y,threshold=threshold)[0:2]
			if print_stats:
				print ("EPOH: "+str(epoh)+"/"+str(briteracija+start-1)+" (LR="+str("{:.2e}".format(mozak.alpha))+")"+" -/-/- "+ "LOSS:"+str ("{:.4f}".format(errortrain))+" ||ACCURACY train:"+str ("{:.4f}".format(accuracytrain*100))+", dev:"+str ("{:.4f}".format(accuracydev*100))+"|| F1 SCORE:"+str ("{:.4f}".format(f1score)) )
			epoh+=1 
			if LRschedule==AdaptiveLR.activeSeljacki:
				if (errortrain>errorb4):
					mozak.alpha=LRschedule(mozak.alpha,epoh,tau)
			else: mozak.alpha=LRschedule(alpha,epoh,tau)
			if (epoh==start+briteracija) or (accuracydev>=nauceno): #prekida učenje ako je naučio sa dovoljnim accuracy ili je dovoljno puta učio
				ctrl=False
			errorb4=errortrain
			with open (name_brain+"/save"+"/epoch_stats.txt", "a") as file:
				np.savetxt(file, np.array([epoh-1, mozak.alpha, errortrain,accuracytrain, accuracydev, np.nan_to_num(f1score)])[None] )
	except: pass

	if save_brain: #sprema mozak na kraju
		mozak.savebrain (name_brain+"/save"+"/wsave.npy")
		with open (name_brain+"/save"+"/info.npy", "wb") as file:
			pickle.dump([mozak.alpha, mozak.activationfunction, mozak.errorFunction, mozak.arhitecture, epoh, errortrain, accuracydev, tau], file)
	return TableNameDev, db_file, name_brain

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
	for i in np.arange (0,1+1/resolution, 1/resolution):
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

	try: plotting (name_brain, thrashold,accuracy,f1score, fpr, tpr, ppv, fprlog, tprlog, thrasholdlog, mozak.arhitecture)

	except Exception as a:
		print (a)
		print ("No plotting 4U D:")	
		ROC=np.column_stack((fprlog,tprlog))
		accu=np.column_stack((thrashold,accuracy))
		f1=np.column_stack((thrashold,f1score))
		np.savetxt(name_brain+"/plot"+"/ROC.txt",ROC, delimiter=' ')
		np.savetxt(name_brain+"/plot"+"/accuracy.txt",accu, delimiter=' ')
		np.savetxt(name_brain+"/plot"+"/f1score.txt",f1, delimiter=' ')

def plotting (name_brain, thrashold,accuracy,f1score, fpr, tpr, ppv, fprlog, tprlog, thrasholdlog, arh):
	import matplotlib as mpl
	mpl.use('Agg' ) 
	from matplotlib import pyplot as plt
	
	#Plotting accuracy and f1 score
	fig = plt.figure("Accuracy and F1 Score")
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.grid(True)
	plt.xticks(np.arange(0,1.1,0.1))
	plt.yticks(np.arange(0,1.1,0.1))
	plt.xlabel('Threshold')
	plt.plot (thrashold, accuracy, "b")
	plt.plot (thrashold, f1score , "r--")
	plt.legend(["Accuracy","F1 Score"])
	plt.savefig (name_brain+"/plot"+"/AccuracyAndF1.png")

	#plotting FPR, TPR, PPV
	fig = plt.figure("FPR_TRP_PPV")
	plt.ylim(0,1.25)
	plt.xlim(0,1)
	plt.grid(True)
	plt.xticks(np.arange(0,1.1,0.1))
	plt.yticks(np.arange(0,1.25,0.1))
	plt.xlabel('Threshold')
	plt.plot (thrashold, tpr, "b")
	plt.plot (thrashold, fpr , "r--")
	plt.plot (thrashold, ppv , "k:")
	plt.legend(["true positive rate (recall)","false positive rate (fallout)","positive predicitve value (precision)"],loc=9)
	plt.savefig (name_brain+"/plot"+"/FPR_TRP_PPV.png")
	
	#plotting ROC curve
	fig= plt.figure ("ROC curve")
	plt.ylim(0,1)
	plt.xlim(0,1)
	plt.grid(True)
	plt.xticks(np.arange(0,1.1,0.1))
	plt.yticks(np.arange(0,1.1,0.1))
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.plot (fprlog,tprlog, "-o",markersize=2)
	plt.plot (thrashold,thrashold,"--")
	for i,j in enumerate (thrasholdlog):
		if (i%4==0):
			plt.text(fprlog[i]+.04, tprlog[i]-.04, str("{:.2f}".format(j)), fontsize=5)
		elif (i%2==0):
			plt.text(fprlog[i]-.03, tprlog[i]+.03, str("{:.2f}".format(j)), fontsize=5)
	plt.legend(["ROC curve","Random Chance"])
	plt.savefig (name_brain+"/plot"+"/ROCcurve.png")
	
	#plotting brain arhitecture
	import Draw
	Draw.DrawNN(arh).draw(name_brain+"/plot"+"/Arhitecture.jpeg")

	
	epoh, alpha, errortrain, accuracytrain, accuracydev, f1score1 =np.delete(np.loadtxt(name_brain+"/save"+"/epoch_stats.txt").T, 0, 1)

	#plotting Train stats
	fig= plt.figure ("Train set")
	plt.title ("Train set")
	plt.ylim (0,1)
	plt.xlabel('Epoh')
	plt.ylabel('%')
	plt.plot (epoh,errortrain, "b--")
	plt.plot (epoh, accuracytrain, "r")
	plt.legend(["Loss","Accuracy"])
	plt.savefig (name_brain+"/plot"+"/Train_error_and_acc.png")

	#plotting Dev stats
	fig=plt.figure ("Dev set")
	plt.title ("Dev set")
	plt.ylim (0,1)
	plt.xlabel('Epoh')
	plt.ylabel('%')
	plt.plot (epoh,f1score1,"b")
	plt.plot (epoh, accuracydev,"r--")
	plt.legend(["F1 score","Accuracy"])
	plt.savefig (name_brain+"/plot"+"/Dev_error_and_acc.png")

	#plotting Alpha curve
	fig=plt.figure ("Learning rate curve")
	plt.title ("Learning rate curve")
	plt.ylim (0,1)
	plt.xlabel('Epoh')
	plt.ylabel('Learning rate')
	plt.plot (epoh,alpha)
	plt.savefig (name_brain+"/plot"+"/LRcurve.png")


if __name__ == '__main__':
	try: tester(*learner (*unos(True)), resolution=100 )
	except Exception as a:
		print (a)
		input ()
	#learner ()
	#tester (graf=False)