from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics

TRAINING_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

# Load and parse the data file into an RDD of LabeledPoint.
trainingData = MLUtils.loadLibSVMFile(sc, TRAINING_FILE)
testData = MLUtils.loadLibSVMFile(sc, TEST_FILE)

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
metrics = BinaryClassificationMetrics(labelsAndPredictions)
print metrics

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
print('Test Error = ' + str(testErr))
# print('Learned classification tree model:')
# print(model.toDebugString())