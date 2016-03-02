import math
from pyspark.mllib.regression import IsotonicRegression, IsotonicRegressionModel

def Isotonic_Regression(filename, sc):
	filename = "/Users/Jacob/SparkService/data/sample_isotonic_regression_data.txt"

	data = sc.textFile(filename)

	# Create label, feature, weight tuples from input data with weight set to default value 1.0.
	parsedData = data.map(lambda line: tuple([float(x) for x in line.split(',')]) + (1.0,))

	# Split data into training (60%) and test (40%) sets.
	training, test = parsedData.randomSplit([0.6, 0.4], 11)

	# Create isotonic regression model from training data.
	# Isotonic parameter defaults to true so it is only shown for demonstration
	model = IsotonicRegression.train(training)

	# Create tuples of predicted and real labels.
	predictionAndLabel = test.map(lambda p: (model.predict(p[1]), p[0]))

	# Calculate mean squared error between predicted and real labels.
	meanSquaredError = predictionAndLabel.map(lambda pl: math.pow((pl[0] - pl[1]), 2)).mean()
	print("Mean Squared Error = " + str(meanSquaredError))

	# Save and load model
	#model.save(sc, "target/tmp/myIsotonicRegressionModel")
	#sameModel = IsotonicRegressionModel.load(sc, "target/tmp/myIsotonicRegressionModel")