import sys
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

def k_means(loadTrainingFilePath, sc):
	# Load and parse the data
	loadTrainingFilePath = "../data/kmeans_data.txt"
	data = sc.textFile(loadTrainingFilePath)
	parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
	# Build the model (cluster the data)
	clusters = KMeans.train(parsedData, 3, maxIterations=10, runs=30, initializationMode="random")

	WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	
	print("Within Set Sum of Squared Error = " + str(WSSSE))
	
	#write to file as JSON
	#res = [('k_means',dumpFilePath, WSSSE)]
	#rdd = sc.parallelize(res)
	#SQLContext.createDataFrame(rdd).collect()
	#df = SQLContext.createDataFrame(rdd,['model_name','res_path', 'WSSSE'])
	#df.toJSON().saveAsTextFile(dumpFilePath)

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))