from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

def LinearRegression(filename, sc):
	filename = "/Users/Jacob/repository/SparkService/data/lpsa.data"
	data = sc.textFile(filename)
	parsedData = data.map(parsePoint)

	# train the model
	model = LinearRegressionWithSGD.train(parsedData)

	# Evaluate the model on training data
	valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
	MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
	print("\n\n\n\n\n\nMean Squared Error = " + str(MSE) + "\n\n\n\n\n")

	# Save and load model
	#model.save(sc, "myModelPath")
	#sameModel = LinearRegressionModel.load(sc, "myModelPath")