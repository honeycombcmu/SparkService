from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.util import MLUtils

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

def LinearRegression(trainFile, testFile, taskid,sc):
	# filename = "/Users/Jacob/repository/SparkService/data/lpsa.data"
	# data = sc.textFile(filename)
	# parsedData = data.map(parsePoint)

	trainData = MLUtils.loadLibSVMFile(sc, trainFile)
	testData = MLUtils.loadLibSVMFile(sc, testFile)

	# train the model
	model = LinearRegressionWithSGD.train(trainData)

	# Evaluate the model on training data
	# predictionAndLabels = parsedData.map(lambda p: (p.label, model.predict(p.features)))
	predictionAndLabels = testData.map(lambda p: (p.label, model.predict(p.features)))
	MSE = predictionAndLabels.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / predictionAndLabels.count()
	print("\n\n\n\n\n\nMean Squared Error = " + str(MSE) + "\n\n\n\n\n")

	# Save and load model
	#model.save(sc, "myModelPath")
	#sameModel = LinearRegressionModel.load(sc, "myModelPath")