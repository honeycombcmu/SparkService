from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

def Random_Forest(filename, sc):

	filename = "/Users/Jacob/SparkService/data/sample_libsvm_data.txt"
	# Load and parse the data file into an RDD of LabeledPoint.
	data = MLUtils.loadLibSVMFile(sc, filename)
	# Split the data into training and test sets (30% held out for testing)
	(trainingData, testData) = data.randomSplit([0.7, 0.3])

	# Train a RandomForest model.
	#  Empty categoricalFeaturesInfo indicates all features are continuous.
	#  Note: Use larger numTrees in practice.
	#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
	model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
	                                     numTrees=3, featureSubsetStrategy="auto",
	                                     impurity='gini', maxDepth=4, maxBins=32)

	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
	print('Test Error = ' + str(testErr))
	print('Learned classification forest model:')
	print(model.toDebugString())

	# Save and load model
	#model.save(sc, "target/tmp/myRandomForestClassificationModel")
	#sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")