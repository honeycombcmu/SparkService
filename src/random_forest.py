from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
import Json

def Random_Forest(trainFile, testFile, taskid, sc):

	# filename = "/Users/Jacob/SparkService/data/sample_libsvm_data.txt"
	# Load and parse the data file into an RDD of LabeledPoint.
	trainData = MLUtils.loadLibSVMFile(sc, trainFile)
	testData = MLUtils.loadLibSVMFile(sc, testFile)

	labelNum = trainData.map(lambda lp: lp.label).distinct().count()
	# Split the data into training and test sets (30% held out for testing)
	# (trainingData, testData) = data.randomSplit([0.7, 0.3])

	# Train a RandomForest model.
	#  Empty categoricalFeaturesInfo indicates all features are continuous.
	#  Note: Use larger numTrees in practice.
	#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
	model = RandomForest.trainClassifier(trainData, numClasses=3, categoricalFeaturesInfo={},
	                                     numTrees=labelNum, featureSubsetStrategy="auto",
	                                     impurity='gini', maxDepth=4, maxBins=32)

	# Evaluate model on test instances and compute test error
	predictions = model.predict(testData.map(lambda x: x.features))
	labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

	Json.generateJson("LogisticRegression", taskid, trainData, labelsAndPredictions);


	# predictionAndLabels = testData.map(lambda lp: (float(model.predict(lp.features)), lp.label))


	# Instantiate metrics object
	# metrics = MulticlassMetrics(predictionAndLabels)
	# metrics = MulticlassMetrics(labelsAndPredictions)

	# # Overall statistics
	# precision = metrics.precision()
	# recall = metrics.recall()
	# f1Score = metrics.fMeasure()
	# #confusion_matrix = metrics.confusionMatrix().toArray()

	# print("Summary Stats")
	# print("Precision = %s" % precision)
	# print("Recall = %s" % recall)
	# print("F1 Score = %s" % f1Score)


	# # Statistics by class
	# labels = trainData.map(lambda lp: lp.label).distinct().collect()
	# for label in sorted(labels):
	#     print("Class %s precision = %s" % (label, metrics.precision(label)))
	#     print("Class %s recall = %s" % (label, metrics.recall(label)))
	#     print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

	# # Weighted stats
	# print("Weighted recall = %s" % metrics.weightedRecall)
	# print("Weighted precision = %s" % metrics.weightedPrecision)
	# print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
	# print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
	# print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

	# testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
	# print('Test Error = ' + str(testErr))
	# print('Learned classification forest model:')
	# print(model.toDebugString())

	# Save and load model
	#model.save(sc, "target/tmp/myRandomForestClassificationModel")
	#sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")