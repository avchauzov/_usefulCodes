"""
autoencoder for TS
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Input, MaxPooling1D, UpSampling1D
from keras.models import Model
from scipy.interpolate import interp1d
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

#

data = pd.read_csv('../_dataProcessing/data/PRS[agg_1].csv', index_col=0)

trajectories = []
for filter in sorted(list(set(data['to_member_id']))):
	tempDataPRS = data.loc[data['to_member_id'] == filter]
	
	if len(tempDataPRS.index) < 6:
		print(filter)
		
		continue
	
	trajectories.append(tempDataPRS['prsScaled'].values)

maxLength = max([len(value) for value in trajectories])
for index, value in enumerate(trajectories):
	
	if len(value) == maxLength:
		continue
	
	timeSteps = list(range(len(value)))
	
	interpolationFunction = interp1d(timeSteps, value)
	timeStepsNew = np.arange(min(timeSteps), max(timeSteps) - 0.1, max(timeSteps) / (maxLength - 1))
	
	reshapedValue = interpolationFunction(timeStepsNew)
	trajectories[index] = reshapedValue

maxLength = 50


def createModel(_convSize, _maxPoolingSize, _activation):
	inputWindow = Input(shape=(maxLength, 1))
	
	layer = Conv1D(_convSize, 2, activation='relu', padding='same')(inputWindow)
	layer = MaxPooling1D(_maxPoolingSize, padding='same')(layer)
	layer = Conv1D(1, 2, activation='relu', padding='same')(layer)
	encoded = MaxPooling1D(1, padding='same')(layer)
	
	encoder = Model(inputWindow, encoded)
	
	layer = Conv1D(1, 2, activation='relu', padding='same')(encoded)
	layer = UpSampling1D(1)(layer)
	layer = Conv1D(_convSize, 1, activation='relu')(layer)
	layer = UpSampling1D(_maxPoolingSize)(layer)
	decoded = Conv1D(1, 2, activation=_activation, padding='same')(layer)
	
	autoencoder = Model(inputWindow, decoded)
	autoencoder.summary()
	
	autoencoder.compile(optimizer='adam', loss='mean_squared_error',
	                    metrics=['mean_squared_error', 'mean_absolute_error'])
	
	return encoder, autoencoder, encoded


earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=25,
                              verbose=0, mode='auto')

modelResults = {}
for convSize in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
	
	for maxPoolingSize in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
		
		activation = 'linear'
		
		xTrain = [value for value in np.array(trajectories)]
		xTest = [value for value in np.array(trajectories)]
		
		for index, value in enumerate(xTrain):
			xTrain[index] = np.array(value).reshape(-1, 1)[: 50]
		
		for index, value in enumerate(xTest):
			xTest[index] = np.array(value).reshape(-1, 1)[: 50]
		
		xTrain = np.array(xTrain).reshape(len(xTrain), len(xTrain[0]), 1)
		xTest = np.array(xTest).reshape(len(xTest), len(xTest[0]), 1)
		
		try:
			encoder, autoencoder, encoded = createModel(convSize, maxPoolingSize, activation)
			
			if encoded.shape.dims[1].value >= 50 or encoded.shape.dims[1].value <= 10:
				continue
			
			history = autoencoder.fit(xTrain, xTrain,
			                          epochs=1,
			                          batch_size=1024,
			                          verbose=0)
		
		except:
			print(convSize, maxPoolingSize, activation)
			
			continue
		
		for activation in ['linear', 'sigmoid', 'softmax']:
			
			for trainIndices, testIndices in RepeatedKFold(10, 1).split(trajectories):
				print(convSize, maxPoolingSize, activation)
				
				xTrain = [value for value in np.array(trajectories)[trainIndices]]
				xTest = [value for value in np.array(trajectories)[testIndices]]
				
				scaler = MinMaxScaler((0, 1))
				flattenXTrain = [value for subValue in xTrain for value in subValue]
				scaler.fit(np.array(flattenXTrain).reshape(-1, 1))
				
				for index, value in enumerate(xTrain):
					xTrain[index] = scaler.transform(np.array(value).reshape(-1, 1))[: 50]
				
				for index, value in enumerate(xTest):
					xTest[index] = scaler.transform(np.array(value).reshape(-1, 1))[: 50]
				
				xTrain = np.array(xTrain).reshape(len(xTrain), len(xTrain[0]), 1)
				xTest = np.array(xTest).reshape(len(xTest), len(xTest[0]), 1)
				
				encoder, autoencoder, encoded = createModel(convSize, maxPoolingSize, activation)
				
				history = autoencoder.fit(xTrain, xTrain,
				                          epochs=1000,
				                          batch_size=1024,
				                          shuffle=True,
				                          validation_data=[xTest, xTest],
				                          callbacks=[earlyStopping])
				
				if '; '.join([str(convSize), str(maxPoolingSize), str(activation)]) not in modelResults.keys():
					modelResults['; '.join([str(convSize), str(maxPoolingSize), str(activation)])] = [[encoded.shape.dims[1].value,
					                                                                                   history.history['loss'][-1],
					                                                                                   history.history['val_loss'][-1],
					                                                                                   history.history['mean_squared_error'][-1],
					                                                                                   history.history['val_mean_squared_error'][-1],
					                                                                                   history.history['mean_absolute_error'][-1],
					                                                                                   history.history['val_mean_absolute_error'][-1],
					                                                                                   len(history.history['loss'])]]
				
				else:
					modelResults['; '.join([str(convSize), str(maxPoolingSize), str(activation)])].append([encoded.shape.dims[1].value,
					                                                                                       history.history['loss'][-1],
					                                                                                       history.history['val_loss'][-1],
					                                                                                       history.history['mean_squared_error'][-1],
					                                                                                       history.history['val_mean_squared_error'][-1],
					                                                                                       history.history['mean_absolute_error'][-1],
					                                                                                       history.history['val_mean_absolute_error'][-1],
					                                                                                       len(history.history['loss'])])
			
			modelResultsUpdated = {}
			for key, value in modelResults.items():
				modelResultsUpdated[key] = list(np.mean(np.array(value), axis=0))
			
			modelResultsUpdated = pd.DataFrame.from_dict(modelResultsUpdated, orient='index', columns=['shape', 'loss', 'val_loss', 'mse', 'val_mse', 'mae', 'val_mae', 'iterations'])
			modelResultsUpdated.to_csv('data/1.0.0.modelResults.csv')
