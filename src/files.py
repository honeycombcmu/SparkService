import sys

def download(localFilePath, hdfsFilePath):
	#readLocalFile("/Users/jacobliu/SparkService/data/sample_libsvm_data.txt")
	subprocess.call(["HADOOP_USER_NAME=hdfs","hdfs","dfs","-get",localFilePath, hdfsFilePath])

def upLoad(localFilePath, hdfsFilePath):
	subprocess.call(["HADOOP_USER_NAME=hdfs","hdfs","dfs","-put",localFilePath,hdfsFilePath])

def deleteFile(hdfsFilePath):
	#if the directory already exists, delete it
	ifExisted = subprocess.call(["hdfs","dfs","-test","-d",hdfsFilePath])
	if ifExisted == 0:
		subprocess.call(["hdfs","dfs","-rm","-r", hdfsFilePath])