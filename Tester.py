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
	print (col_name_list_input)
	c.execute("SELECT "+col_name_list_input+" FROM "+TableName)
	input=np.array(c.fetchall())
	return np.array(input.T)

def tester (lista, arh=[1,1], name_brain="UnnamedBrain"):
	if (name_brain+"_save.npy") in os.listdir("."):
		with open (name_brain+"_info.npy", "rb") as file:
			[alpha, activationfunction, errorFunction, arh, fixedAlpha, start, error, accuracy] = pickle.load(file)

	mozak=Brain (arh,[lista,lista])
	mozak.birth()
	mozak.loadbrain (name_brain+"_save.npy")
	mozak.fowardpropagation()
	return mozak.n[len(mozak.n)-1]

if __name__ == '__main__':	
	print(np.round(tester(getdata (),arh=[4,3]).T,0))
	input()