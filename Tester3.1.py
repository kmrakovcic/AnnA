import sqlite3
from AnnA3 import *
import numpy as np

def getdata (TableName="Iris_dataset",db_file='data.db'):
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

def gogo (lista, arh=[0], briteracija=1, alpha=0.1):
	arh=[4,3]
	mozak=Brain (arh, lista, alpha)
	mozak.birth()
	for j in range (1,briteracija+1):
		error,accuracy= mozak.learn ()
		progressbar="EPOH: "+str(j)+"/"+str(briteracija)+" ----- ERROR: "+str ("{:7.5f}".format(error))+" ACCURACY: "+str ("{:7.5f}".format(accuracy))
		print (progressbar)
	mjerfolder="test"
	mozak.savebrain (mjerfolder+"_save.npy")


gogo(getdata())

