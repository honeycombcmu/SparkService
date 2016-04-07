import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import shutil
import re

#ML modules
import random_forest
import isotonic_regression
import gradient_boostedtrees
import naive_bayes
import alternating_least_squares
import logistic_regression

import Json
import k_means
import linear_regression

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.sql import SQLContext
from pyspark.sql import DataFrame
from numpy import array
import math
from pyspark.mllib.regression import IsotonicRegression, IsotonicRegressionModel
import subprocess



#"/Users/jacobliu/HoneyPySpark/spark-1.5.2-bin-hadoop2.6/README.md"  # Should be some file on your system
sc = SparkContext("local", "PySpark")
SQLContext = SQLContext(sc)

def main():
	loadTrainingFilePath = sys.argv[1]		#tainning file path
	loadTestingFilePath = sys.argv[2]		#testing file path
	taskid = sys.argv[3]
	#dumpFilePath = sys.argv[3]				#output file path
	# hdfsFilePath = "/user/honeycomb/sparkteam/output"
	model_name = sys.argv[4]   #"Regression"				#model_name

	# print sys.argv[1]

	##test##
	#readLocalFile("/Users/jacobliu/SparkService/data/sample_libsvm_data.txt")

	#if the directory already exists, delete it
	#ifExisted = subprocess.call(["hdfs","dfs","-test","-d",hdfsFilePath])
	#if ifExisted == 0:
	#	subprocess.call(["hdfs","dfs","-rm","-r", hdfsFilePath])
	#if os.path.exists(dumpFilePath):
		#shutil.rmtree(dumpFilePath)
		#hdfs.delete_file_dir(dumpFilePath)

	if model_name == "LinearRegression":
		linear_regression.LinearRegression(loadTrainingFilePath, sc)

	elif model_name == "IsotonicRegression":
		isotonic_regression.Isotonic_Regression(loadTrainingFilePath,sc)

	elif model_name == "GradientBoostedTrees":
		gradient_boostedtrees.Gradient_BoostedTrees(loadTrainingFilePath,sc)

	elif model_name == "ALS":
		alternating_least_squares.Alternating_Least_Squares(loadTrainingFilePath, sc)

	elif model_name == "NaiveBayes":
		naive_bayes.Naive_Bayes(loadTrainingFilePath, sc)

	elif model_name == "RandomForest":
		random_forest.Random_Forest(loadTrainingFilePath, loadTestingFilePath, taskid, sc)

	elif model_name == "LogisticRegression":
		logistic_regression.logisticRegression(loadTrainingFilePath, loadTestingFilePath, taskid, sc)


	elif model_name == "KMeans":
		k_dssmeans.k_means(loadTrainingFilePath, sc)
		# # Load and parse the data
		# data = sc.textFile(loadTrainingFilePath)
		# parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
		# # Build the model (cluster the data)
		# clusters = KMeans.train(parsedData, 3, maxIterations=10, runs=30, initializationMode="random")

		# WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)

		# print("Within Set Sum of Squared Error = " + str(WSSSE))

		# #write to file as JSON
		# #res = [('k_means',dumpFilePath, WSSSE)]
		# #rdd = sc.parallelize(res)
		# #SQLContext.createDataFrame(rdd).collect()
		# #df = SQLContext.createDataFrame(rdd,['model_name','res_path', 'WSSSE'])
		# #df.toJSON().saveAsTextFile(dumpFilePath)



#Read from local file, sample test read a txt file and output the columns
def readLocalFile(filename):
	with open(filename, 'r') as f:
		for line in f.readlines():
			for words in line.strip().split(" "):
				print words


# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

# def LinearRegression(filename):
# 	data = sc.textFile(filename)
# 	parsedData = data.map(parsePoint)

# 	# train the model
# 	model = LinearRegressionWithSGD.train(parsedData)

# 	# Evaluate the model on training data
# 	valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# 	MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
# 	print("\n\n\n\n\n\nMean Squared Error = " + str(MSE) + "\n\n\n\n\n")

# 	# Save and load model
# 	#model.save(sc, "myModelPath")
# 	#sameModel = LinearRegressionModel.load(sc, "myModelPath")

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

main()
