"""
rnn predictions + confusion matrix
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, concatenate, Dense, Input, LSTM, Masking
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight


#

def plotConfusionMatrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap='coolwarm'):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print('Normalized confusion matrix')
	
	else:
		print('Confusion matrix, without normalization')
	
	print(cm)
	
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	# plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
		         horizontalalignment='center',
		         color='white' if cm[i, j] > thresh else 'black')
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()


#

xDataPRS = np.load('data/prsColumn.npy')
xDataTeam = np.load('data/teamColumn.npy')
yData = np.load('data/targetColumn.npy')

stratifyParameter = []

xDataPRSFlatten = [value for subValue in xDataPRS for value in subValue]
scalerPRS = MinMaxScaler((0, 1)).fit(np.array(xDataPRSFlatten).reshape(-1, 1))

xDataTeamFlatten = [value for subValue in xDataTeam for value in subValue]
scalerTeam = MinMaxScaler((0, 1)).fit(np.array(xDataTeamFlatten).reshape(-1, 1))

xDataPRSConverted = []
xDataTeamConverted = []
for index in range(len(xDataPRS)):
	
	stratifyParameter.append(str(index // 6) + '/' + str(yData[index]))
	
	tempArray = scalerPRS.transform(np.array(xDataPRS[index]).reshape(1, -1))
	
	if len(xDataPRS[index]) < 6:
		tempArray = np.append(tempArray, [-1.0] * (6 - len(xDataPRS[index])))
		xDataPRSConverted.append(tempArray)
	
	else:
		xDataPRSConverted.append(tempArray[0])
	
	tempArray = scalerTeam.transform(np.array(xDataTeam[index]).reshape(1, -1))
	
	if len(xDataTeam[index]) < 6:
		tempArray = np.append(tempArray, [-1.0] * (6 - len(xDataTeam[index])))
		xDataTeamConverted.append(tempArray)
	
	else:
		xDataTeamConverted.append(tempArray[0])

xDataPRSConverted = np.array(xDataPRSConverted)
xDataPRS = xDataPRSConverted.reshape((xDataPRSConverted.shape[0], xDataPRSConverted.shape[1], 1))
print(xDataPRS.shape)

xDataTeamConverted = np.array(xDataTeamConverted)
xDataTeam = xDataTeamConverted.reshape((xDataTeamConverted.shape[0], xDataTeamConverted.shape[1], 1))
print(xDataTeam.shape)

yDataCategorical = to_categorical(yData)

weights = class_weight.compute_class_weight('balanced',
                                            np.unique(yData),
                                            yData)


def createModel(numOfCells):
	input1 = Input(shape=(len(xDataPRS[0]), 1))
	mask1 = Masking(mask_value=-1.0, input_shape=(len(xDataPRS[0]), 1))(input1)
	lstmLayer1 = LSTM(numOfCells)(mask1)
	bn1 = BatchNormalization()(lstmLayer1)
	
	input2 = Input(shape=(len(xDataTeam[0]), 1))
	mask2 = Masking(mask_value=-1.0, input_shape=(len(xDataTeam[0]), 1))(input2)
	lstmLayer2 = LSTM(numOfCells)(mask2)
	bn2 = BatchNormalization()(lstmLayer2)
	
	mergeLayer = concatenate([bn1, bn2], axis=1)
	
	output = Dense(len(yDataCategorical[0]), activation='softmax')(mergeLayer)
	
	model = Model(inputs=[input1, input2], outputs=output)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	model.summary()
	
	return model


#

'''xTrain, xTest, yTrain, yTest = train_test_split(xData, yDataCategorical, test_size=0.01, stratify=yData, shuffle=True)

history = model.fit(xTest, yTest,
                    epochs=1000, batch_size=8, class_weight=weights)'''

earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto')

from sklearn.model_selection import StratifiedKFold

outputReal = []
outputPredicted = []
outputProbability = []
testIndices = []
skf = StratifiedKFold(n_splits=40, shuffle=True)
for trainIndex, testIndex in skf.split(xDataPRS, yData):
	xTrainPRS, xTestPRS = xDataPRS[trainIndex], xDataPRS[testIndex]
	xTrainTeam, xTestTeam = xDataTeam[trainIndex], xDataTeam[testIndex]
	yTrain, yTest = yDataCategorical[trainIndex], yDataCategorical[testIndex]
	
	model = createModel(2)
	
	history = model.fit([xTrainPRS, xTrainTeam], yTrain,
	                    epochs=300, batch_size=1024, class_weight=weights)
	
	print(history.history.keys())
	
	'''plt.plot(history.history['categorical_accuracy'])
	# plt.plot(history.history['val_categorical_accuracy'])
	plt.title('model categorical_accuracy')
	plt.ylabel('categorical_accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()'''
	
	predictions = model.predict([xTestPRS, xTestTeam])
	outputReal.extend([value[1] for value in yTest])
	outputPredicted.extend([round(value[1]) for value in predictions])
	outputProbability.extend([value[1] for value in predictions])
	testIndices.extend(testIndex)
	
	cm = confusion_matrix(outputReal, outputPredicted)
	
	np.set_printoptions(precision=2)
	
	figure = plt.figure(figsize=(10, 5))
	
	plt.subplot(1, 1, 1)
	plotConfusionMatrix(cm, classes=['class_0', 'class_1'],
	                    title='startPrediction')
	
	# plt.show()
	plt.clf()

metaData = pd.read_csv('data/metaData.csv', index_col=0)

dataPredictions = pd.DataFrame()
dataPredictions['testIndex'] = testIndices
dataPredictions['realValue'] = outputReal
dataPredictions['predictedValue'] = outputPredicted
dataPredictions['predictedProbability'] = outputProbability

dataPredictions.sort_values('testIndex', inplace=True)
dataPredictions = pd.merge(dataPredictions, metaData, left_on='testIndex', right_index=True)
dataPredictions['split'] = [int(str(value).split('[')[1].replace(']', '')) for value in dataPredictions['name']]

for index in range(1, 7):
	tempDataPredictions = dataPredictions.loc[dataPredictions['split'] == index]
	
	tempMetricAccuracy = np.round(balanced_accuracy_score(tempDataPredictions['realValue'].values,
	                                                      tempDataPredictions['predictedValue'].values), 3)
	tempMetricLogLoss = np.round(log_loss(tempDataPredictions['realValue'].values,
	                                      tempDataPredictions['predictedProbability'].values), 3)
	
	cm = confusion_matrix(tempDataPredictions['realValue'].values,
	                      tempDataPredictions['predictedValue'].values)
	
	np.set_printoptions(precision=2)
	
	figure = plt.figure(figsize=(10, 5))
	
	plt.subplot(1, 1, 1)
	plotConfusionMatrix(cm, classes=['bad', 'good'],
	                    title='split[' + str(index) + '][' + str(tempMetricAccuracy) + '][' + str(tempMetricLogLoss) + ']')
	
	plt.savefig('split[' + str(index) + '].png')
	
	# plt.show()
	plt.clf()

dataPredictions.to_csv('predictions.csv')
