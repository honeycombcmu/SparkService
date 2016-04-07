from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics

import Json

def logisticRegression(trainFile, testFile, taskid, sc):
	# Load training data in LIBSVM format
	trainData = MLUtils.loadLibSVMFile(sc, trainFile)
	testData = MLUtils.loadLibSVMFile(sc, testFile)

	# Split data into training (60%) and test (40%)
	# traindata, testdata = data.randomSplit([0.6, 0.4], seed = 11L)
	# traindata.cache()

	# Load testing data in LIBSVM format
	#testdata = MLUtils.loadLibSVMFile(sc, loadTestingFilePath)

	labelNum = trainData.map(lambda lp: lp.label).distinct().count()

	# Run training algorithm to build the model
	model = LogisticRegressionWithLBFGS.train(trainData, numClasses=labelNum)

	# Compute raw scores on the test set
	predictionAndLabels = testData.map(lambda lp: (float(model.predict(lp.features)), lp.label))

	Json.generateJson("LogisticRegression", taskid, trainData, predictionAndLabels);
	# Instantiate metrics object
	metrics = MulticlassMetrics(predictionAndLabels)

	# Overall statistics
	precision = metrics.precision()
	recall = metrics.recall()
	f1Score = metrics.fMeasure()
	#confusion_matrix = metrics.confusionMatrix().toArray()

	print("Summary Stats")
	print("Precision = %s" % precision)
	print("Recall = %s" % recall)
	print("F1 Score = %s" % f1Score)


	# Statistics by class
	labels = trainData.map(lambda lp: lp.label).distinct().collect()
	for label in sorted(labels):
	    print("Class %s precision = %s" % (label, metrics.precision(label)))
	    print("Class %s recall = %s" % (label, metrics.recall(label)))
	    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

	# Weighted stats
	print("Weighted recall = %s" % metrics.weightedRecall)
	print("Weighted precision = %s" % metrics.weightedPrecision)
	print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
	print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
	print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

	# #return model parameters
	# res = [('1','Yes','TP Rate', metrics.truePositiveRate(0.0)),
	# 	   ('2','Yes','FP Rate', metrics.falsePositiveRate(0.0)),
	# 	   ('3','Yes','Precision', metrics.precision(0.0)),
	# 	   ('4','Yes','Recall', metrics.recall(0.0)),
	#        ('5','Yes','F-Measure', metrics.fMeasure(0.0, beta=1.0)),
	#        ('1','Yes','TP Rate', metrics.truePositiveRate(1.0)),
	# 	   ('2','Yes','FP Rate', metrics.falsePositiveRate(1.0)),
	#        ('3','Yes','Precision', metrics.precision(1.0)),
	# 	   ('4','Yes','Recall', metrics.recall(1.0)),
	#        ('5','Yes','F-Measure', metrics.fMeasure(1.0, beta=1.0)),
	#        ('1','Yes','TP Rate', metrics.truePositiveRate(2.0)),
	# 	   ('2','Yes','FP Rate', metrics.falsePositiveRate(2.0)),
	#        ('3','Yes','Precision', metrics.precision(2.0)),
	#        ('4','Yes','Recall', metrics.recall(2.0)),
	#        ('5','Yes','F-Measure', metrics.fMeasure(2.0, beta=1.0))]

	# #save output file path as JSON and dump into dumpFilePath
	# rdd = sc.parallelize(res)
	# SQLContext.createDataFrame(rdd).collect()
	# df = SQLContext.createDataFrame(rdd,['Order','CLass','Name', 'Value'])

	#tempDumpFilePath = dumpFilePath + "/part-00000"
	#if os.path.exists(tempDumpFilePath):
	#	os.remove(tempDumpFilePath)

	#df.toJSON().saveAsTextFile(hdfsFilePath)
	#tmpHdfsFilePath = hdfsFilePath + "/part-00000"
	#subprocess.call(["hadoop","fs","-copyToLocal", tmpHdfsFilePath, dumpFilePath])

	# Save and load model
	#clusters.save(sc, "myModel")
	#sameModel = KMeansModel.load(sc, "myModel")