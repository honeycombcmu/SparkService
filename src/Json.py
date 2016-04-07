from pyspark.mllib.evaluation import MulticlassMetrics
import json

def generateJson(AlgorithmName, taskid, traindata, predictionAndLabels):
	jsonContent = dict()
	jsonContent['AlgorithmName'] = AlgorithmName
	jsonContent['TaskId'] = taskid

	labels = traindata.map(lambda lp: lp.label).distinct().collect()
	jsonContent['LabelNum'] = len(labels)

	metrics = MulticlassMetrics(predictionAndLabels)
	precision = metrics.precision()
	recall = metrics.recall()
	f1Score = metrics.fMeasure()
	confusion_matrix = metrics.confusionMatrix().toArray()

	jsonContent['Precision'] = precision
	jsonContent['Recall'] = recall
	jsonContent['F1Score'] = f1Score
	jsonContent['ConfusionMatrix'] = confusion_matrix.tolist()

	jsonContent['Labels'] = dict()
	for label in sorted(labels):
		jsonContent['Labels'][label] = dict()
		jsonContent['Labels'][label]['Precession'] = metrics.precision(label)
		jsonContent['Labels'][label]['Recall'] = metrics.recall(label)
		jsonContent['Labels'][label]['F1Measure'] = metrics.fMeasure(label, beta=1.0)

	jsonContent['WeightedStats'] = dict()
	jsonContent['WeightedStats']['Precision'] = metrics.weightedRecall
	jsonContent['WeightedStats']['F1Score'] = metrics.weightedFMeasure()
	jsonContent['WeightedStats']['FalsePositiveRate'] = metrics.weightedFalsePositiveRate

	with open(taskid + '.json', 'w') as jsonFile:
		json.dump(jsonContent, jsonFile, indent=4, separators=(',', ': '))
		jsonFile.flush()


