from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

TRAINING_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

# Load and parse the data file into an RDD of LabeledPoint.
trainingData = MLUtils.loadLibSVMFile(sc, TRAINING_FILE)
testData = MLUtils.loadLibSVMFile(sc, TEST_FILE)

# Build the model
model = SVMWithSGD.train(trainingData, iterations=100)

# Evaluating the model on training data
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda p: (p.label).zip(predictions)
testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testData.count())
print("Training Error = " + str(trainErr))

# Save and load model
# model.save(sc, "myModelPath")
# sameModel = SVMModel.load(sc, "myModelPath")
