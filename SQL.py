import urllib.request
import sqlite3
import numpy as np

def url_to_list(URL='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'):
	try:
		response = urllib.request.urlopen(URL)
	except Exception as e:
		print (e)
		input ()
	html = []
	l=0
	for i in str(response.read() ).replace("b'","").replace("\\r\\n",'\\n').split('\\n'):
		k=i.split(',')
		for t in range(len(k)):
			try:
				k[t]=float(k[t])
			except Exception as e:
				pass
		if not (html==[]):
			l=len(html[0])
		if l==len(k) or (l==0):
			html.append(tuple(k)) #MUST BE TUPLE
	return html

def sql_to_list (TableName="Iris_dataset",db_file=':memory:'):
	try:
		db = sqlite3.connect(db_file)
	except Exception as e:
		error_print+=str(e)
	c = db.cursor()
	c.execute("SELECT * FROM "+TableName)
	mjerenja=c.fetchall()
	return mjerenja

def list_to_sql(lista, size, name="Iris_dataset",db_file=':memory:'):
	#building SQL command for creating table
	create_table_sql="CREATE TABLE IF NOT EXISTS "+name+"\n ("
	for i in range (size[0]):
		create_table_sql+="InputNeuron_"+str(i+1)+" REAL, "
	for i in range (size[1]):
		create_table_sql+="OutputNeuron_"+str(i+1)+" REAL, "
	create_table_sql+= "Result TEXT);"
	
	#building SQL command for inserting values
	insert_value_sql="INSERT INTO "+name+" VALUES ("
	for i in range (len(lista[0])-1):
		insert_value_sql+="?,"
	insert_value_sql+="?)"
	
	try:
		db = sqlite3.connect(db_file)
	except Exception as e:
		print (e)
		input ()
	c = db.cursor()
	c.execute(create_table_sql)
	c.executemany(insert_value_sql,lista)
	db.commit()
	db.close()

def edit_list(lista):
	a=np.array(list(set(np.array(lista)[:,len(lista[0])-1])))
	lista=np.array (lista)
	b=np.array([a,]*len(lista))
	result=np.array([1*np.isin(b[i],lista [:,len(lista[0])-1][i]) for i in range(len(lista))])
	out=np.hstack((lista[:,:-1], result,lista[:,-1][:, np.newaxis]))
	return out,[lista.shape[1]-1, result.shape[1]]

def divideList (lista):
	np.random.shuffle(lista) #TRAIN pa DEV pa TEST
	return lista[:int(round(lista.shape[0]*0.6,0)),:], lista[int(round(lista.shape[0]*0.6,0)):int(round(lista.shape[0]*0.8,0)),:], lista[int(round(lista.shape[0]*0.8,0)):,:]

def mainSQL(notdivided=True):   #notdivided=True djeli listu na train, dev i test
	URL=input ("URL?: ")
	if URL == "": URL="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	name=input ("Table name?: ")
	if name == "": name="Iris"
	db=input ("Database name?: ")
	if db == "": db="Datasets"
	a= edit_list(url_to_list(URL))
	if notdivided:
		b=divideList (a[0])
		list_to_sql (b[0],a[1],name+"_train",db+".db")
		list_to_sql (b[1],a[1],name+"_dev",db+".db")
		list_to_sql (b[2],a[1],name+"_test",db+".db")
		print ("Table "+name+"_train, "+name+"_dev and "+name+"_test created in database "+db)
	else:
		list_to_sql (a[0],a[1],name,db+".db")
		print ("Table "+name+" created in database "+db)
	input ("PRESS ENTER TO EXIT")

if __name__ == '__main__':
	mainSQL(True)