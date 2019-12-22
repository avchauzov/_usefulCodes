"""
custom NN written in numPy for target value optimization (=maximization)
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from copy import deepcopy
from random import randint, uniform

mpl.use('TkAgg')

import matplotlib.pyplot as plt


def getTransportCosts(value, W1, b1):
	if value < 30:
		return b1
	
	else:
		return W1 * value


def initializeParameters(weights, volumes):
	W1 = weights[0]
	b1 = weights[1]
	
	return {'W1': W1, 'b1': b1}


def forwardPropagation(model, xSet, xSetAdditional):
	W1, b1 = model['W1'], model['b1']
	
	z1 = []
	for index in range(len(xSet)):
		z1.append(getTransportCosts(xSetAdditional[index][0], W1[index], b1[index]) * xSetAdditional[index][1] + \
		          getTransportCosts(xSetAdditional[index][2], W1[index], b1[index]) + xSet[index])
	z1 = np.array(z1, dtype=np.float)
	
	a1 = np.array([[np.sum(z1)]], dtype=np.float).reshape((1, 1))
	
	return {'a0': xSet, 'z1': z1, 'a1': a1}


def forwardPropagationFull(model, xSet, xSetAdditional, W, b):
	z1 = []
	
	for index in range(len(xSet)):
		z1.append(getTransportCosts(xSetAdditional[index][0], W[index], b[index]) * xSetAdditional[index][1] + \
		          getTransportCosts(xSetAdditional[index][2], W[index], b[index]) + xSet[index])
	z1 = np.array(z1, dtype=np.float)
	
	return z1


def lossDerivative(yTarget, yPredicted):
	return yPredicted - yTarget


def backwardPropagation(model, cache, yTarget):
	W1, b1 = model['W1'], model['b1']
	a0, a1 = cache['a0'], cache['a1']
	
	m = yTarget.shape[0]
	dz1 = lossDerivative(yTarget=yTarget, yPredicted=a1)
	
	dW1 = 1 / m * np.dot(dz1[0][0], a0.T)
	db1 = 1 / m * np.sum(dz1, axis=0)
	# db1 = np.array([[0.0]], dtype=np.float).reshape((1, 1))
	
	return {'dW1': dW1, 'db1': db1}


def updateParameters(model, gradients, learningRate):
	W1, b1 = model['W1'], model['b1']
	
	#
	W1New = W1 - learningRate * gradients['dW1']
	b1New = b1 - learningRate * gradients['db1']
	
	tempArray1 = forwardPropagationFull(model, xTrain, xTrainAdditional, W1, b1)
	tempArray2 = forwardPropagationFull(model, xTrain, xTrainAdditional, W1New, b1New)
	
	for index in range(len(W1)):
		if tempArray2[index] < tempArray1[index] and tempArray2[index] > 0:
			# print(W1New[index], b1New[index], averagePriceLowerBoundW1[index], averagePriceLowerBoundb1[index])
			# W1New[index] = W1[index] + learningRate * np.abs(averagePriceLowerBoundW1[index] - W1[index])
			# b1New[index] = b1[index] + learningRate * np.abs(averagePriceLowerBoundb1[index] - b1[index])
			# print(W1New[index], b1New[index])
			
			W1New[index] = W1[index] + learningRate * np.abs(averagePriceUpperBoundW1[index] - W1[index])
			b1New[index] = b1[index] + learningRate * np.abs(averagePriceUpperBoundb1[index] - b1[index])
		
		elif tempArray2[index] < tempArray1[index] and tempArray2[index] < 0:
			# print(W1New[index], b1New[index])
			W1New[index] = W1[index] + learningRate * np.abs(averagePriceUpperBoundW1[index] - W1[index])
			b1New[index] = b1[index] + learningRate * np.abs(averagePriceUpperBoundb1[index] - b1[index])
	# print(W1New[index], b1New[index])
	
	#
	
	# W1New = W1 - learningRate * gradients['dW1']
	
	W1New = np.maximum(averagePriceUpperBoundW1, W1New)
	W1New = np.minimum(averagePriceLowerBoundW1, W1New)
	
	# b1New = b1 - learningRate * gradients['db1']
	
	b1New = np.maximum(averagePriceUpperBoundb1, b1New)
	b1New = np.minimum(averagePriceLowerBoundb1, b1New)
	
	return {'W1': W1New, 'b1': b1New}


def predict(model, xSet):
	return forwardPropagation(model, xSet, xTrainAdditional)


globalLosses = []
globalLossesLast = []
globalMetric = []
globalMetricLast = []
globalPredictions = []
globalPredictionsLast = []

globalPricesUnder30 = []
globalPricesLastUnder30 = []
globalPricesOver30 = []
globalPricesLastOver30 = []

globalPredictionsByEachRoute = []
globalPredictionsByEachRouteLast = []


def trainModel(model, xSet, ySet, learningRate, epochs=1000):
	learningRateBase = deepcopy(learningRate)
	
	losses = []
	metric = []
	predictions = []
	
	pricesUnder30 = []
	pricesOver30 = []
	
	predictionsFull = []
	
	for i in range(epochs):
		W1Old = deepcopy(model['W1'])
		cache = forwardPropagation(model, xSet, xTrainAdditional)
		
		gradients = backwardPropagation(model, cache, ySet)
		a1 = cache['a1']
		
		#
		if len(losses) > 0:
			
			for _ in range(7):
				tempModel = updateParameters(model=model, gradients=gradients, learningRate=learningRate)
				yHat = predict(tempModel, xSet)['a1']
				tempLoss = ((ySet - a1) ** 2)[0][0]
				
				if tempLoss >= losses[-1]:
					learningRate /= 2
				
				else:
					break
		#
		
		model = updateParameters(model=model, gradients=gradients, learningRate=learningRate)
		
		yHat = predict(model, xSet)['a1']
		
		print 'Loss [' + str(i) + ']: ' + str(((ySet - a1) ** 2)[0][0]), 'MAE: ' + str(np.abs(yHat - ySet)[0][0])
		
		losses.append(((ySet - a1) ** 2)[0][0])
		metric.append(np.absolute(ySet - a1)[0][0])
		predictions.append(yHat[0][0])
		
		pricesUnder30.append(np.round(model['b1'], 5))
		pricesOver30.append(np.round(model['W1'], 5))
		
		predictionsFull.append(forwardPropagationFull(model, xSet, xTrainAdditional, model['W1'], model['b1']))
		
		if losses[-1] == 0.0 or str(losses[-1]) == 'nan':
			break
		
		W1New = model['W1']
		if max(np.absolute((np.subtract(W1Old, W1New)))) < 0.01:
			learningRate *= 2
			
			continue
		
		if learningRate > learningRateBase:
			break
		
		else:
			learningRate /= 2
	
	if len(losses) > 0:
		globalLosses.append(losses)
		globalLossesLast.append(losses[-1])
		
		globalMetric.append(metric)
		globalMetricLast.append(metric[-1])
		
		globalPredictions.append(predictions)
		globalPredictionsLast.append(predictions[-1])
		
		globalPricesUnder30.append(pricesUnder30)
		globalPricesLastUnder30.append(pricesUnder30[-1])
		globalPricesOver30.append(pricesOver30)
		globalPricesLastOver30.append(pricesOver30[-1])
		
		globalPredictionsByEachRoute.append(predictionsFull)
		globalPredictionsByEachRouteLast.append(predictionsFull[-1])
	
	return model, losses


###

#

predictions = pd.read_csv('1.1.predictionsUpdated.csv', index_col=0)
predictions['routeID'] = [int(float(value)) for value in predictions['routeID']]

pricesData = pd.read_csv('1.5.prices.csv', index_col=0)
weightsData = pd.read_csv('1.5.weights.csv', index_col=0)

pricesData['shipping_date'] = weightsData['shipping_date']
pricesData = pricesData[['shipping_date', 'routeID', 'pricePerKilo']]

pricesData = pricesData.loc[pricesData['shipping_date'] < '2018-05-17']

dictionaryPrices = {}
for routeID in sorted(list(set(pricesData['routeID'].values))):
	
	if routeID == 'group' or int(float(routeID)) not in predictions['routeID'].values:
		continue
	
	tempDF = pricesData.loc[pricesData['routeID'] == routeID]
	
	dictionaryPrices[int(float(routeID))] = np.mean(tempDF['pricePerKilo'].values)

# print(sorted(list(set(predictions['routeID'].values))))
# print(sorted(list(set(dictionaryPrices.keys()))))

predictions = predictions.loc[predictions['routeID'].isin(dictionaryPrices.keys())]
predictions.sort_values('routeID', inplace=True)

dictionaryPrices = [(key, value) for key, value in dictionaryPrices.items()]
dictionaryPrices = sorted(dictionaryPrices, key=lambda tup: tup[0])

# print(len(predictions['routeID'].values))
# print(len(dictionaryPrices))

# print(predictions['routeID'].values)
# print([value[0] for value in dictionaryPrices])

xTrain = []
for index in range(len(predictions['nanmean'].values)):
	xTrain.append((predictions['nanmean'].values[index] * predictions['amount'].values[index] + predictions['rest'].values[index]) * dictionaryPrices[index][1])
xTrain = np.array(xTrain, dtype=np.float)

W1 = []
for index in range(len(dictionaryPrices)):
	W1.append(-0.67921 * uniform(0.5, 1.5))
W1 = np.array(W1, dtype=np.float)

b1 = []
for index in range(len(dictionaryPrices)):
	b1.append(-20.37 * uniform(0.5, 1.5))
b1 = np.array(b1, dtype=np.float)

upperBoundCoefficient = 1.25
lowerBoundCoefficient = 0.75

averagePriceUpperBoundW1 = np.round(W1.dot(upperBoundCoefficient), 2)
averagePriceLowerBoundW1 = np.round(W1.dot(lowerBoundCoefficient), 2)

averagePriceUpperBoundb1 = np.round(b1.dot(upperBoundCoefficient), 2)
averagePriceLowerBoundb1 = np.round(b1.dot(lowerBoundCoefficient), 2)

xTrainAdditional = []
for index in range(len(predictions['nanmean'].values)):
	xTrainAdditional.append((predictions['nanmean'].values[index], predictions['amount'].values[index], predictions['rest'].values[index]))
xTrainAdditional = np.array(xTrainAdditional, dtype=np.float)

routeNames = [value[0] for value in dictionaryPrices]

model = initializeParameters(weights=[W1, b1], volumes=xTrain)
ySet = forwardPropagation(model, xTrain, xTrainAdditional)

basicModel = deepcopy(model)

if ySet['a1'] > 0:
	targetValue = ySet['a1']

else:
	targetValue = np.array([[np.sum(xTrain)]], dtype=np.float).reshape((1, 1))

for mainIteration in range(1000):
	print 'mainIteration: ', mainIteration
	
	if mainIteration == 0:
		pass
	
	else:
		targetValue = targetValue.dot(1.01)
	
	learningRate = 0.0001 / (1 + mainIteration)
	
	W1Old = deepcopy(model['W1'])
	
	model, losses = trainModel(model, xTrain, targetValue, learningRate=learningRate, epochs=100)
	
	if len(losses) > 0:
		
		if str(losses[-1]) == 'nan':
			print '***Stop because of nan!'
			break
	
	if len(globalLossesLast) > 1:
		
		if globalLossesLast[-1] > globalLossesLast[-2]:
			print '***Stop because of loss increasing!'
			model['W1'] = W1Old
			break

# print(model)

minIndexLastLoss = globalLossesLast.index(min(value for value in globalLossesLast if value > 0))
minIndexLoss = globalLosses[minIndexLastLoss].index(min(value for value in globalLosses[minIndexLastLoss] if value > 0))

print 'minLoss: ', round(globalLosses[minIndexLastLoss][minIndexLoss], 2)

print 'optimalPricesUnder30: ', np.round(globalPricesUnder30[minIndexLastLoss][minIndexLoss], 5)
print 'optimalPricesOver30: ', np.round(globalPricesOver30[minIndexLastLoss][minIndexLoss], 5)

model['b1'] = globalPricesUnder30[minIndexLastLoss][minIndexLoss]
model['W1'] = globalPricesOver30[minIndexLastLoss][minIndexLoss]

print 'basicRevenue: ', round(forwardPropagation(basicModel, xTrain, xTrainAdditional)['a1'][0][0], 2)
print 'expectedRevenue: ', round(forwardPropagation(model, xTrain, xTrainAdditional)['a1'][0][0], 2)

globalLosses = globalLosses[1: minIndexLastLoss + 1]
globalLossesLast = globalLossesLast[1: minIndexLastLoss + 1]
globalMetric = globalMetric[1: minIndexLastLoss + 1]
globalMetricLast = globalMetricLast[1: minIndexLastLoss + 1]
globalPredictions = globalPredictions[1: minIndexLastLoss + 1]
globalPredictionsLast = globalPredictionsLast[1: minIndexLastLoss + 1]

globalPricesUnder30 = globalPricesUnder30[1: minIndexLastLoss + 1]
globalPricesLastUnder30 = globalPricesLastUnder30[1: minIndexLastLoss + 1]
globalPricesOver30 = globalPricesOver30[1: minIndexLastLoss + 1]
globalPricesLastOver30 = globalPricesLastOver30[1: minIndexLastLoss + 1]

globalPredictionsByEachRoute = globalPredictionsByEachRoute[1: minIndexLastLoss + 1]
globalPredictionsByEachRouteLast = globalPredictionsByEachRouteLast[1: minIndexLastLoss + 1]

print(globalLosses)
print(globalLossesLast)
print(globalMetric)
print(globalMetricLast)
print(globalPredictions)
print(globalPredictionsLast)

globalLosses = [value for subValue in globalLosses for value in subValue]
globalMetric = [value for subValue in globalMetric for value in subValue]
globalPredictions = [value for subValue in globalPredictions for value in subValue]

globalPricesUnder30 = [value for subValue in globalPricesUnder30 for value in subValue]
globalPricesOver30 = [value for subValue in globalPricesOver30 for value in subValue]

globalPredictionsByEachRoute = [value for subValue in globalPredictionsByEachRoute for value in subValue]

plt.plot(globalLosses)
plt.title('loss (mse)')
plt.show()

plt.clf()

plt.plot(globalMetric)
plt.title('metric (mae)')
plt.show()

plt.clf()

plt.plot(globalPredictions)
plt.title('summarized revenue')
plt.show()

plt.clf()

###

'''tempIndices = []
tempArrayX = [float(value) for value in globalPredictions]
for index in range(len(globalPricesUnder30[0])):
    tempArrayY = [float(value[index]) for value in globalPricesUnder30]

    points = [(value1, value2) for value1, value2 in zip(tempArrayX, tempArrayY)]
    points = sorted(points, key=lambda tup: tup[0])

    if tempArrayY[-1] < tempArrayY[0]:
        tempIndices.append(index)

    tempArrayX = [value[0] for value in points]
    tempArrayY = [value[1] for value in points]

    plt.plot(tempArrayX, tempArrayY)
    plt.title('price Under30 for route index: ' + str(index))
    plt.show()

    plt.clf()

print(tempIndices)

###

tempIndices = []
tempArrayX = [float(value) for value in globalPredictions]
for index in range(len(globalPricesOver30[0])):
    tempArrayY = [float(value[index]) for value in globalPricesOver30]

    points = [(value1, value2) for value1, value2 in zip(tempArrayX, tempArrayY)]
    points = sorted(points, key=lambda tup: tup[0])

    if tempArrayY[-1] < tempArrayY[0]:
        tempIndices.append(index)

    tempArrayX = [value[0] for value in points]
    tempArrayY = [value[1] for value in points]

    plt.plot(tempArrayX, tempArrayY)
    plt.title('price Over30 for route index: ' + str(index))
    plt.show()

    plt.clf()

print(tempIndices)'''

###

tempIndices = []
tempArrayX = [float(value) for value in globalPredictions]
for index in range(len(globalPredictionsByEachRoute[0])):
	tempArrayY = [float(value[index]) for value in globalPredictionsByEachRoute]
	
	points = [(value1, value2) for value1, value2 in zip(tempArrayX, tempArrayY)]
	points = sorted(points, key=lambda tup: tup[0])
	
	if tempArrayY[-1] < tempArrayY[0]:
		tempIndices.append(index)
	
	tempArrayX = [value[0] for value in points]
	tempArrayY = [value[1] for value in points]
	
	plt.plot(tempArrayX, tempArrayY)
	plt.title('revenue for route index: ' + str(index))
	plt.show()
	
	plt.clf()

print(tempIndices)
