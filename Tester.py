from AnnA3 import *
import sqlite3

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

	c.execute("SELECT "+col_name_list_input[:-1]+" FROM "+TableName)
	input=np.array(c.fetchall())
	return np.array(input.T)

def tester (lista, arh=[1,1], name_brain="UnnamedBrain"):
	mozak=Brain (arh,[lista,lista])
	mozak.birth()
	mozak.loadbrain (name_brain+"_save.npy")
	mozak.fowardpropagation()
	return mozak.n[len(mozak.n)-1]

if __name__ == '__main__':
	print(np.round(tester(getdata (),arh=[4,3]).T,0))
	input()