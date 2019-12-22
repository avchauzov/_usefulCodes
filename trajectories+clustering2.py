"""
trajectories clustering via distance matrices + synthetic cases generation
"""

import csv
import itertools
import os
import shutil
import traces
import warnings
from dateutil.relativedelta import relativedelta
from dtreeviz.trees import *
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def warn(*args, **kwargs):
	pass


warnings.filterwarnings('ignore')
warnings.warn = warn

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

with warnings.catch_warnings():
	warnings.filterwarnings('ignore', category=FutureWarning)
	warnings.filterwarnings('ignore', category=DeprecationWarning)


def spearman(array1, array2):
	return 2 - (spearmanr(array1, array2)[0] + 1)


def cid(array1, array2):
	distance = np.sum([(value1 - value2) ** 2 for value1, value2 in zip(array1, array2)])
	
	CEp = 0
	for index in range(1, len(array1)):
		CEp += (array1[index] - array1[index - 1]) ** 2
	
	CEq = 0
	for index in range(1, len(array2)):
		CEq += (array2[index] - array2[index - 1]) ** 2
	
	CEp = np.sqrt(CEp)
	CEq = np.sqrt(CEq)
	
	return distance * max(CEp, CEq) / min(CEp, CEq)


def freshet(array1, array2):
	distance = [np.abs(value1 - value2) for value1, value2 in zip(array1, array2)]
	
	return np.max(distance)


def euclidean(array1, array2):
	distance = np.sum([(value1 - value2) ** 2 for value1, value2 in zip(array1, array2)])
	
	return np.sqrt(distance)


def mjc(array1, array2):
	distance = 0
	
	for index in range(len(array1)):
		distance += np.abs(array1[index] - array2[index + 1])
	
	return distance


def loadData(df, aggregation):
	mainDataConverted = pd.DataFrame()
	scalingTS = {}
	
	df = df.sort_values(['to_member_id', 'cycle'], ascending=[True, True])
	df.drop_duplicates(inplace=True)
	
	for id, filter in enumerate(sorted(list(set(df['to_member_id'])))):
		
		if id % 100 == 0:
			print(id)
		
		try:
			data = df.loc[df['to_member_id'] == filter]
			data['cycle'] = pd.to_datetime(data['cycle'])
			minDate = min(data['cycle'])
			
			for cycle, aggregate in zip(data['cycle'], data['aggregate']):
				
				if cycle not in scalingTS.keys():
					scalingTS[cycle] = [aggregate]
				
				else:
					scalingTS[cycle].append(aggregate)
			
			data = data.loc[data['cycle'] >= minDate + relativedelta(months=3)]
			
			data = data.groupby('cycle').agg({'aggregate': np.mean}).reset_index()
			
			data = pd.DataFrame(data.set_index('cycle')['aggregate'])
			data.columns = ['aggregate']
			
			data = data.reindex(pd.date_range(start=data.index.min(),
			                                  end=data.index.max(),
			                                  freq='d')).interpolate(method='linear')
			
			data = traces.TimeSeries(data['aggregate'])
			data = data.moving_average(60 * 60 * 24 * aggregation * 3 * (365 / 12), pandas=True)
			data = pd.DataFrame(data)
			
			data.reset_index(inplace=True)
			data.columns = ['cycle', 'aggregate']
			
			data['to_member_id'] = [filter] * len(data.index)
			
			if len(data.index) >= 3 and len(sorted(list(set(data['aggregate'].values)))) > 1:
				mainDataConverted = pd.concat((mainDataConverted, data))
		
		except Exception as message:
			print(message)
			
			pass
	
	try:
		shutil.rmtree('data/aggregation[' + str(aggregation) + ']')
	
	except:
		pass
	
	os.makedirs('data/aggregation[' + str(aggregation) + ']', 0o0755)
	mainDataConverted.to_csv('data/aggregation[' + str(aggregation) + ']/mainDataConverted.csv')
	
	if not Path('data/scalingTS.csv').exists():
		
		for key, value in scalingTS.items():
			scalingTS[key] = [min(-25, np.percentile(value, 10)), max(np.percentile(value, 90), 25)]
		
		scalingTS = pd.DataFrame.from_dict(scalingTS, orient='index')
		scalingTS = scalingTS.reindex(pd.date_range(start=min(scalingTS.index) - relativedelta(months=3),
		                                            end=max(scalingTS.index) + relativedelta(months=3),
		                                            freq='d')).interpolate(method='linear')
		
		scalingTS.columns = ['minScore', 'maxScore']
		scalingTS = scalingTS.rolling(window=90).mean()
		
		scalingTS['minScore'].fillna(-25, inplace=True)
		scalingTS['maxScore'].fillna(25, inplace=True)
		
		scalingTS.to_csv('data/scalingTS.csv')
	
	else:
		scalingTS = pd.read_csv('data/scalingTS.csv', index_col=0)
	
	return mainDataConverted, scalingTS


def scalingData(df, dfScaled):
	df['cycle'] = pd.to_datetime(df['cycle'])
	df['cycle'] = df['cycle'].dt.normalize()
	
	dfScaled.index = pd.to_datetime(dfScaled.index)
	df = pd.merge(df, dfScaled, how='left', left_on='cycle', right_index=True)
	
	scaledMin = -1
	scaledMax = 1
	
	tempColumn = []
	for aggregate, minValue, maxValue in zip(df['aggregate'],
	                                         df['minScore'],
	                                         df['maxScore']):
		newAggregate = (((aggregate - minValue) * (scaledMax - scaledMin)) / (maxValue - minValue)) + scaledMin
		
		tempColumn.append(newAggregate)
	
	df['aggregateScaled'] = tempColumn
	
	return df


def dmCalculation(df, aggregation):
	scaledValues = {}
	cycles = []
	
	for id, filter in enumerate(sorted(list(set(df['to_member_id'])))):
		data = df.loc[df['to_member_id'] == filter]
		cycles.append([(value - min(data['cycle'])) / np.timedelta64(1, 's') for value in data['cycle']])
	
	maxSteps = max([len(value) for value in cycles])
	maxLength = max([max(value) for value in cycles])
	
	maxNumber = len(list(set(df['to_member_id'])))
	
	distanceMatrixSpearmanOriginal = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	distanceMatrixSpearmanScaled = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	
	distanceMatrixCIDScaled = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	distanceMatrixFreshetScaled = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	distanceMatrixEuclideanScaled = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	distanceMatrixMJCScaled = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	
	distanceMatrixLengths = [[0 for _ in range(maxNumber)] for _ in range(maxNumber)]
	
	for id1, filter1 in enumerate(sorted(list(set(df['to_member_id'])))):
		
		if filter1 in scaledValues.keys():
			continue
		
		if id1 % 2 == 0:
			print(id1)
		
		if id1 > maxNumber - 1:
			continue
		
		data1 = df.loc[df['to_member_id'] == filter1]
		
		for id2, filter2 in enumerate(sorted(list(set(df['to_member_id'])))):
			
			if filter2 in scaledValues.keys():
				continue
			
			if id2 > maxNumber - 1 or id2 > id1:
				continue
			
			data2 = df.loc[df['to_member_id'] == filter2]
			
			cycles1 = data1['cycle'].values
			cycles1 = [(value - min(cycles1)) / np.timedelta64(1, 's') for value in cycles1]
			
			cycles2 = data2['cycle'].values
			cycles2 = [(value - min(cycles2)) / np.timedelta64(1, 's') for value in cycles2]
			
			'''distanceMatrixLengths[id1][id2] = euclidean([len(cycles1)], [len(cycles2)])
			distanceMatrixLengths[id2][id1] = distanceMatrixLengths[id1][id2]'''
			
			if max(cycles1) <= maxLength:
				cycles1 = [value / max(cycles1) * maxLength for value in cycles1]
			
			if max(cycles2) <= maxLength:
				cycles2 = [value / max(cycles2) * maxLength for value in cycles2]
			
			# function1 = interp1d(cycles1, data1['aggregate'].values, kind='linear')
			# function2 = interp1d(cycles2, data2['aggregate'].values, kind='linear')
			
			xNew = np.arange(0, maxLength, int(maxLength) / maxSteps)
			
			# aggregate1 = function1(xNew)
			# aggregate2 = function2(xNew)
			
			'''distanceMatrixSpearmanOriginal[id1][id2] = spearman(aggregate1, aggregate2)
			distanceMatrixSpearmanOriginal[id2][id1] = distanceMatrixSpearmanOriginal[id1][id2]'''
			
			function1 = interp1d(cycles1, data1['aggregateScaled'].values, kind='linear')
			function2 = interp1d(cycles2, data2['aggregateScaled'].values, kind='linear')
			
			aggregateScaled1 = function1(xNew)
			aggregateScaled2 = function2(xNew)
			
			'''distanceMatrixSpearmanScaled[id1][id2] = spearman(aggregateScaled1, aggregateScaled2)
			distanceMatrixSpearmanScaled[id2][id1] = distanceMatrixSpearmanScaled[id1][id2]

			distanceMatrixCIDScaled[id1][id2] = cid(aggregateScaled1, aggregateScaled2)
			distanceMatrixCIDScaled[id2][id1] = distanceMatrixCIDScaled[id1][id2]

			distanceMatrixFreshetScaled[id1][id2] = freshet(aggregateScaled1, aggregateScaled2)
			distanceMatrixFreshetScaled[id2][id1] = distanceMatrixFreshetScaled[id1][id2]

			distanceMatrixEuclideanScaled[id1][id2] = euclidean(aggregateScaled1, aggregateScaled2)
			distanceMatrixEuclideanScaled[id2][id1] = distanceMatrixEuclideanScaled[id1][id2]

			distanceMatrixMJCScaled[id1][id2] = mjc(aggregateScaled1, aggregateScaled2)
			distanceMatrixMJCScaled[id2][id1] = distanceMatrixMJCScaled[id1][id2]'''
			
			if filter1 not in scaledValues.keys():
				scaledValues[filter1] = aggregateScaled1
			
			if filter2 not in scaledValues.keys():
				scaledValues[filter2] = aggregateScaled2
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixSpearmanOriginal.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixSpearmanOriginal)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixSpearmanScaled.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixSpearmanScaled)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixCIDScaled.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixCIDScaled)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixFreshetScaled.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixFreshetScaled)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixEuclideanScaled.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixEuclideanScaled)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixMJCScaled.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixMJCScaled)
	
	with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixLengths.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(distanceMatrixLengths)
	
	scaledValues = pd.DataFrame.from_dict(scaledValues)
	scaledValues.to_csv('data/scaledValues.csv')
	
	distanceMatrixSpearmanOriginal = np.array(distanceMatrixSpearmanOriginal)
	distanceMatrixSpearmanScaled = np.array(distanceMatrixSpearmanScaled)
	distanceMatrixCIDScaled = np.array(distanceMatrixCIDScaled)
	distanceMatrixFreshetScaled = np.array(distanceMatrixFreshetScaled)
	distanceMatrixEuclideanScaled = np.array(distanceMatrixEuclideanScaled)
	distanceMatrixMJCScaled = np.array(distanceMatrixMJCScaled)
	distanceMatrixLengths = np.array(distanceMatrixLengths)
	
	return scaledValues, \
	       distanceMatrixSpearmanOriginal, distanceMatrixSpearmanScaled, distanceMatrixCIDScaled, \
	       distanceMatrixFreshetScaled, distanceMatrixEuclideanScaled, distanceMatrixMJCScaled, \
	       distanceMatrixLengths


def statDataCalculation(df, column, aggregation):
	extractedFeatures = extract_features(df,
	                                     column_id='to_member_id',
	                                     column_value=column,
	                                     show_warnings=False)
	extractedFeatures.replace([-np.inf, np.inf], np.nan, inplace=True)
	extractedFeatures.dropna(axis=1, inplace=True)
	extractedFeatures = impute(extractedFeatures)
	
	extractedFeatures.to_csv('data/aggregation[' + str(aggregation) + ']/extractedFeatures[' + str(column) + '].csv')
	
	return extractedFeatures


def runTestClustering(distanceMatrixSpearmanOriginal, distanceMatrixSpearmanScaled,
                      distanceMatrixCIDScaled, distanceMatrixFreshetScaled,
                      distanceMatrixEuclideanScaled, distanceMatrixMJCScaled,
                      distanceMatrixLengths, distanceMatrixOriginalFloat,
                      distanceMatrixOriginalBoolean, distanceMatrixScaledFloat,
                      distanceMatrixScaledBoolean,

                      spearmanOriginalWeightArray, spearmanScaledWeightArray,
                      cidScaledWeightArray, freshetScaledWeightArray,
                      euclideanScaledWeightArray, mjcScaledWeightArray,
                      lengthsWeightArray, originalFloatWeightArray,
                      originalBooleanWeightArray, scaledFloatWeightArray,
                      scaledBooleanWeightArray,

                      scalingArray, linkageArray,

                      scaledValues,
                      searchStep, maxNumOfClusters,
                      aggregation):
	results = []
	
	totalCount = 0
	for spearmanOriginalWeight, spearmanScaledWeight, cidScaledWeight, \
	    freshetScaledWeight, euclideanScaledWeight, mjcScaledWeight, \
	    lengthsWeight, originalFloatWeight, originalBooleanWeight, \
	    scaledFloatWeight, scaledBooleanWeight in itertools.product(spearmanOriginalWeightArray,
	                                                                spearmanScaledWeightArray, cidScaledWeightArray,
	                                                                freshetScaledWeightArray,
	                                                                euclideanScaledWeightArray, mjcScaledWeightArray,
	                                                                lengthsWeightArray, originalFloatWeightArray,
	                                                                originalBooleanWeightArray,
	                                                                scaledFloatWeightArray, scaledBooleanWeightArray):
		
		if sum([spearmanOriginalWeight, spearmanScaledWeight, cidScaledWeight,
		        freshetScaledWeight, euclideanScaledWeight, mjcScaledWeight,
		        lengthsWeight, originalFloatWeight, originalBooleanWeight,
		        scaledFloatWeight, scaledBooleanWeight
		        ]) == 1.0:
			totalCount += 1
		
		else:
			continue
	
	totalCount = totalCount * len(scalingArray) * len(range(2, maxNumOfClusters, searchStep)) * len(linkageArray)
	
	tempCount = 1
	for spearmanOriginalWeight, spearmanScaledWeight, cidScaledWeight, \
	    freshetScaledWeight, euclideanScaledWeight, mjcScaledWeight, \
	    lengthsWeight, originalFloatWeight, originalBooleanWeight, \
	    scaledFloatWeight, scaledBooleanWeight in itertools.product(spearmanOriginalWeightArray,
	                                                                spearmanScaledWeightArray, cidScaledWeightArray,
	                                                                freshetScaledWeightArray,
	                                                                euclideanScaledWeightArray, mjcScaledWeightArray,
	                                                                lengthsWeightArray, originalFloatWeightArray,
	                                                                originalBooleanWeightArray,
	                                                                scaledFloatWeightArray, scaledBooleanWeightArray):
		
		if sum([spearmanOriginalWeight, spearmanScaledWeight, cidScaledWeight,
		        freshetScaledWeight, euclideanScaledWeight, mjcScaledWeight,
		        lengthsWeight, originalFloatWeight, originalBooleanWeight,
		        scaledFloatWeight, scaledBooleanWeight
		        ]) != 1.0:
			continue
		
		dm = distanceMatrixSpearmanOriginal * spearmanOriginalWeight + \
		     distanceMatrixSpearmanScaled * spearmanScaledWeight + \
		     distanceMatrixCIDScaled * cidScaledWeight + \
		     distanceMatrixFreshetScaled * freshetScaledWeight + \
		     distanceMatrixEuclideanScaled * euclideanScaledWeight + \
		     distanceMatrixMJCScaled * mjcScaledWeight + \
		     distanceMatrixLengths * lengthsWeight + \
		     distanceMatrixOriginalFloat * originalFloatWeight + \
		     distanceMatrixOriginalBoolean * originalBooleanWeight + \
		     distanceMatrixScaledFloat * scaledFloatWeight + \
		     distanceMatrixScaledBoolean * scaledBooleanWeight
		
		dmScaled = pd.DataFrame(dm).values
		
		for scaling in scalingArray:
			
			if scaling == 1:
				dmScaled = MinMaxScaler((0, 1)).fit_transform(dmScaled)
			
			for linkage in linkageArray:
				
				for nClusters in range(11, maxNumOfClusters, searchStep):
					clusterer = AgglomerativeClustering(n_clusters=nClusters,
					                                    affinity='precomputed',
					                                    linkage=linkage).fit(dmScaled)
					clusters = clusterer.labels_
					
					clusterDataFrame = pd.DataFrame()
					clusterDataFrame['cluster'] = clusters
					
					clusterDataFrame['name'] = sorted(list(set(mainDataConverted['to_member_id'])))
					
					tempColumnTS = []
					for filter in sorted(list(set(mainDataConverted['to_member_id']))):
						tempColumnTS.append(scaledValues[filter].values)
					
					clusterDataFrame['ts'] = tempColumnTS
					
					#
					absMetric = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						if len(clusterDataFrameFilter) == 1:
							absMetric = [np.nan]
							
							break
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						
						tempArray = []
						for item in clusterDataFrameFilter['ts'].values:
							tempArray.append(
									np.mean([np.abs(value1 - value2) for value1, value2 in zip(item, smoothSetMean)]))
						
						absMetric.append(np.median(tempArray))
					#
					
					#
					euclideanMetric = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						if len(clusterDataFrameFilter) == 1:
							euclideanMetric = [np.nan]
							
							break
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						
						tempArray = []
						for item in clusterDataFrameFilter['ts'].values:
							tempArray.append(
									np.mean([(value1 - value2) ** 2 for value1, value2 in zip(item, smoothSetMean)]))
						
						euclideanMetric.append(np.median(tempArray))
					#
					
					#
					absMaxMetric = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						if len(clusterDataFrameFilter) == 1:
							absMaxMetric = [np.nan]
							
							break
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						
						tempArray = []
						for item in clusterDataFrameFilter['ts'].values:
							tempArray.append(
									np.max([np.abs(value1 - value2) for value1, value2 in zip(item, smoothSetMean)]))
						
						absMaxMetric.append(np.median(tempArray))
					#
					
					#
					euclideanMaxMetric = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						if len(clusterDataFrameFilter) == 1:
							euclideanMaxMetric = [np.nan]
							
							break
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						
						tempArray = []
						for item in clusterDataFrameFilter['ts'].values:
							tempArray.append(
									np.max([(value1 - value2) ** 2 for value1, value2 in zip(item, smoothSetMean)]))
						
						euclideanMaxMetric.append(np.median(tempArray))
					#
					
					#
					spearmanMetric = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						if len(clusterDataFrameFilter) == 1:
							spearmanMetric = [np.nan]
							
							break
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						
						tempArray = []
						for item in clusterDataFrameFilter['ts'].values:
							tempArray.append(spearmanr(item, smoothSetMean))
						
						spearmanMetric.append(np.median(tempArray))
					#
					
					# ***
					smoothSetArrays = []
					for cluster in sorted(list(set(clusterDataFrame['cluster']))):
						clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
						
						smoothSetMean = np.mean(clusterDataFrameFilter['ts'].values, axis=0)
						smoothSetArrays.append(smoothSetMean)
					
					absMetricNegative = []
					tempArray = []
					for index1, item1 in enumerate(smoothSetArrays):
						
						for index2, item2 in enumerate(smoothSetArrays):
							
							if index1 <= index2:
								continue
							
							tempArray.append(
									np.mean([np.abs(value1 - value2) for value1, value2 in
									         zip(item1, item2)]))
					
					absMetricNegative.append(np.median(tempArray))
					#
					
					#
					euclideanMetricNegative = []
					tempArray = []
					for index1, item1 in enumerate(smoothSetArrays):
						
						for index2, item2 in enumerate(smoothSetArrays):
							
							if index1 <= index2:
								continue
							
							tempArray.append(
									np.mean([(value1 - value2) ** 2 for value1, value2 in
									         zip(item1, item2)]))
					
					euclideanMetricNegative.append(np.median(tempArray))
					#
					
					#
					absMaxMetricNegative = []
					tempArray = []
					for index1, item1 in enumerate(smoothSetArrays):
						
						for index2, item2 in enumerate(smoothSetArrays):
							
							if index1 <= index2:
								continue
							
							tempArray.append(
									np.min([np.abs(value1 - value2) for value1, value2 in
									        zip(item1, item2)]))
					
					absMaxMetricNegative.append(np.median(tempArray))
					#
					
					#
					euclideanMaxMetricNegative = []
					tempArray = []
					for index1, item1 in enumerate(smoothSetArrays):
						
						for index2, item2 in enumerate(smoothSetArrays):
							
							if index1 <= index2:
								continue
							
							tempArray.append(
									np.min([(value1 - value2) ** 2 for value1, value2 in
									        zip(item1, item2)]))
					
					euclideanMaxMetricNegative.append(np.median(tempArray))
					#
					
					#
					spearmanMetricNegative = []
					tempArray = []
					for index1, item1 in enumerate(smoothSetArrays):
						
						for index2, item2 in enumerate(smoothSetArrays):
							
							if index1 <= index2:
								continue
							
							tempArray.append(spearmanr(item1, item2))
					
					spearmanMetricNegative.append(np.median(tempArray))
					#
					
					results.append([
							spearmanOriginalWeight, spearmanScaledWeight, cidScaledWeight,
							freshetScaledWeight, euclideanScaledWeight, mjcScaledWeight,
							lengthsWeight, originalFloatWeight, originalBooleanWeight,
							scaledFloatWeight, scaledBooleanWeight,
							scaling, linkage,
							nClusters,
							np.median(absMetric),
							np.median(euclideanMetric),
							np.median(absMaxMetric),
							np.median(euclideanMaxMetric),
							np.median(spearmanMetric),
							
							np.median(absMetricNegative),
							np.median(euclideanMetricNegative),
							np.median(absMaxMetricNegative),
							np.median(euclideanMaxMetricNegative),
							np.median(spearmanMetricNegative),
					])
					
					print(tempCount, totalCount)
					tempCount += 1
	
	results = pd.DataFrame(results, columns=['spearmanOriginalWeight', 'spearmanScaledWeight', 'cidScaledWeight',
	                                         'freshetScaledWeight', 'euclideanScaledWeight', 'mjcScaledWeight',
	                                         'lengthsWeight', 'originalFloatWeight', 'originalBooleanWeight',
	                                         'scaledFloatWeight', 'scaledBooleanWeight',
	                                         'scaling', 'linkage',
	                                         'nClusters',
	                                         'np.median(absMetric)',
	                                         'np.median(euclideanMetric)',
	                                         'np.median(absMaxMetric)',
	                                         'np.median(euclideanMaxMetric)',
	                                         'np.median(spearmanMetric)',
	
	                                         'np.median(absMetricNegative)',
	                                         'np.median(euclideanMetricNegative)',
	                                         'np.median(absMaxMetricNegative)',
	                                         'np.median(euclideanMaxMetricNegative)',
	                                         'np.median(spearmanMetricNegative)',
	                                         ])
	
	results.to_csv('data/aggregation[' + str(aggregation) + ']/results.csv')
	
	return results


loadDataOption = True
countDMOption = True
statDataOption = True
runTestOption = True

if __name__ == '__main__':
	mainData = pd.read_csv('../../cuberoot/Peer Rank Deltas - 2019_02_13.csv', index_col=0)
	
	# mainData = mainData.loc[mainData['to_member_id'].isin(sorted(list(set(mainData['to_member_id'])))[: 100])]
	
	firedData = pd.read_csv('../../../data/firedData.csv', index_col=None, sep=';')
	firedData = firedData[~firedData['endDate'].isnull()]
	firedData = firedData['email'].values
	
	salesData = pd.read_csv('../../../data/salesData.csv', index_col=None)
	salesData = salesData['email'].values
	
	mainData = mainData.loc[~mainData['to_member_id'].isin(salesData)]
	
	mainData = mainData[['date', 'to_member_id', 'aggregate_prs']]
	mainData.columns = ['cycle', 'to_member_id', 'aggregate']
	
	tempColumn = []
	for item in mainData['to_member_id']:
		
		if item in firedData:
			tempColumn.append(item + ' (F)')
			
			continue
		
		tempColumn.append(item)
	
	mainData['to_member_id'] = tempColumn
	
	for aggregation in range(1, 2):
		
		if loadDataOption:
			mainDataConverted, scalingTS = loadData(mainData, aggregation)
		
		else:
			mainDataConverted = pd.read_csv('data/aggregation[' + str(aggregation) + ']/mainDataConverted.csv',
			                                index_col=0)
			scalingTS = pd.read_csv('data/scalingTS.csv', index_col=0)
		
		mainDataConverted = scalingData(mainDataConverted, scalingTS)
		
		mainDataConverted.to_csv('mainDataConverted.csv')
		
		if countDMOption:
			scaledValues, \
			distanceMatrixSpearmanOriginal, distanceMatrixSpearmanScaled, distanceMatrixCIDScaled, \
			distanceMatrixFreshetScaled, distanceMatrixEuclideanScaled, distanceMatrixMJCScaled, \
			distanceMatrixLengths = dmCalculation(mainDataConverted, aggregation)
		
		else:
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixSpearmanOriginal.csv') as file:
				distanceMatrixSpearmanOriginal = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixSpearmanOriginal = [[float(float(subValue)) for subValue in value] for value in
			                                  distanceMatrixSpearmanOriginal]
			distanceMatrixSpearmanOriginal = np.array(distanceMatrixSpearmanOriginal)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixSpearmanScaled.csv') as file:
				distanceMatrixSpearmanScaled = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixSpearmanScaled = [[float(float(subValue)) for subValue in value] for value in
			                                distanceMatrixSpearmanScaled]
			distanceMatrixSpearmanScaled = np.array(distanceMatrixSpearmanScaled)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixCIDScaled.csv') as file:
				distanceMatrixCIDScaled = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixCIDScaled = [[float(float(subValue)) for subValue in value] for value in
			                           distanceMatrixCIDScaled]
			distanceMatrixCIDScaled = np.array(distanceMatrixCIDScaled)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixFreshetScaled.csv') as file:
				distanceMatrixFreshetScaled = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixFreshetScaled = [[float(float(subValue)) for subValue in value] for value in
			                               distanceMatrixFreshetScaled]
			distanceMatrixFreshetScaled = np.array(distanceMatrixFreshetScaled)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixEuclideanScaled.csv') as file:
				distanceMatrixEuclideanScaled = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixEuclideanScaled = [[float(float(subValue)) for subValue in value] for value in
			                                 distanceMatrixEuclideanScaled]
			distanceMatrixEuclideanScaled = np.array(distanceMatrixEuclideanScaled)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixMJCScaled.csv') as file:
				distanceMatrixMJCScaled = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixMJCScaled = [[float(float(subValue)) for subValue in value] for value in
			                           distanceMatrixMJCScaled]
			distanceMatrixMJCScaled = np.array(distanceMatrixMJCScaled)
			
			with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixLengths.csv') as file:
				distanceMatrixLengths = [value.split(',') for value in file if str(value) != '\n']
			
			distanceMatrixLengths = [[float(float(subValue)) for subValue in value] for value in distanceMatrixLengths]
			distanceMatrixLengths = np.array(distanceMatrixLengths)
			
			scaledValues = pd.read_csv('data/scaledValues.csv', index_col=0)
		
		import sys
		
		sys.exit(0)
		
		if statDataOption:
			extractedFeaturesOriginal = statDataCalculation(mainDataConverted, 'aggregate', aggregation)
			extractedFeaturesScaled = statDataCalculation(mainDataConverted, 'aggregateScaled', aggregation)
		
		else:
			extractedFeaturesOriginal = pd.read_csv(
					'data/aggregation[' + str(aggregation) + ']/extractedFeatures[aggregate].csv', index_col=0)
			extractedFeaturesScaled = pd.read_csv(
					'data/aggregation[' + str(aggregation) + ']/extractedFeatures[aggregateScaled].csv', index_col=0)
		
		columnsOriginalFloat = []
		columnsOriginalBoolean = []
		for column in list(extractedFeaturesOriginal):
			
			if sorted(list(set(extractedFeaturesOriginal[column].values))) == [0, 1]:
				columnsOriginalBoolean.append(column)
			
			elif len(list(set(extractedFeaturesOriginal[column].values))) > 1:
				columnsOriginalFloat.append(column)
		
		extractedFeaturesOriginalBoolean = extractedFeaturesOriginal[columnsOriginalBoolean]
		extractedFeaturesOriginalFloat = extractedFeaturesOriginal[columnsOriginalFloat]
		
		distanceMatrixOriginalBoolean = squareform(pdist(extractedFeaturesOriginalBoolean.values, 'hamming'))
		with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixOriginalBoolean.csv', 'w',
		          newline='') as file:
			writer = csv.writer(file)
			writer.writerows(distanceMatrixOriginalBoolean)
		
		distanceMatrixOriginalFloat = squareform(pdist(extractedFeaturesOriginalFloat.values, 'euclidean'))
		with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixOriginalFloat.csv', 'w',
		          newline='') as file:
			writer = csv.writer(file)
			writer.writerows(distanceMatrixOriginalFloat)
		
		columnsScaledFloat = []
		columnsScaledBoolean = []
		for column in list(extractedFeaturesScaled):
			
			if sorted(list(set(extractedFeaturesScaled[column].values))) == [0, 1]:
				columnsScaledBoolean.append(column)
			
			elif len(list(set(extractedFeaturesScaled[column].values))) > 1:
				columnsScaledFloat.append(column)
		
		extractedFeaturesScaledBoolean = extractedFeaturesScaled[columnsScaledBoolean]
		extractedFeaturesScaledFloat = extractedFeaturesScaled[columnsScaledFloat]
		
		distanceMatrixScaledBoolean = squareform(pdist(extractedFeaturesScaledBoolean.values, 'hamming'))
		with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixScaledBoolean.csv', 'w',
		          newline='') as file:
			writer = csv.writer(file)
			writer.writerows(distanceMatrixScaledBoolean)
		
		distanceMatrixScaledFloat = squareform(pdist(extractedFeaturesScaledFloat.values, 'euclidean'))
		with open('data/aggregation[' + str(aggregation) + ']/distanceMatrixScaledFloat.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(distanceMatrixScaledFloat)
		
		# 0.02238806	0.02238806	0.134328358	0.097014925	0.141791045	0.201492537	0.02238806	0.179104478	0.097014925	0.014925373	0.067164179
		
		spearmanOriginalWeightArray = [0.0]  # np.linspace(0.0, 0.25, num=3)
		spearmanScaledWeightArray = [0.0]  # np.linspace(0.0, 0.25, num=3)
		cidScaledWeightArray = np.linspace(0.0, 0.25, num=3)
		freshetScaledWeightArray = np.linspace(0.0, 0.25, num=3)
		euclideanScaledWeightArray = [0.0]  # np.linspace(0.0, 0.375, num=3)
		mjcScaledWeightArray = np.linspace(0.0, 0.5, num=3)
		lengthsWeightArray = [0.0]  # np.linspace(0.0, 0.5, num=3)
		originalFloatWeightArray = [0.0]  # np.linspace(0.0, 1.0, num=3)
		originalBooleanWeightArray = np.linspace(0.0, 0.5, num=3)
		scaledFloatWeightArray = [0.0]  # np.linspace(0.0, 0.5, num=3)
		scaledBooleanWeightArray = np.linspace(0.0, 0.5, num=3)
		
		linkageArray = ['complete']  # , 'average', 'single']
		
		scalingArray = [0]  # , 1]
		
		if runTestOption:
			results = runTestClustering(distanceMatrixSpearmanOriginal, distanceMatrixSpearmanScaled,
			                            distanceMatrixCIDScaled, distanceMatrixFreshetScaled,
			                            distanceMatrixEuclideanScaled, distanceMatrixMJCScaled,
			                            distanceMatrixLengths, distanceMatrixOriginalFloat,
			                            distanceMatrixOriginalBoolean, distanceMatrixScaledFloat,
			                            distanceMatrixScaledBoolean,
			
			                            spearmanOriginalWeightArray, spearmanScaledWeightArray,
			                            cidScaledWeightArray, freshetScaledWeightArray,
			                            euclideanScaledWeightArray, mjcScaledWeightArray,
			                            lengthsWeightArray, originalFloatWeightArray,
			                            originalBooleanWeightArray, scaledFloatWeightArray,
			                            scaledBooleanWeightArray,
			
			                            scalingArray, linkageArray,
			
			                            scaledValues,
			                            1, min(20, len(distanceMatrixSpearmanOriginal)),
			                            aggregation)
		
		else:
			results = [[0.089939024, 0.033536585, 0.117378049, 0.091463415, 0.106707317, 0.092987805,
			            0.106707317, 0.047256098, 0.31097561, 0.00304878, 1, 29]]
	
	import sys
	
	sys.exit(0)
	
	cidWeight = results[0][0]
	dtwWeight = results[0][1]
	firedWeight = results[0][2]
	scaledBooleanWeight = results[0][3]
	scaledFloatWeight = results[0][4]
	scaledZScoreBooleanWeight = results[0][5]
	scaledZScoreFloatWeight = results[0][6]
	spearmanWeight = results[0][7]
	spearmanZScoreWeight = results[0][8]
	lengthsWeight = results[0][9]
	scaling = results[0][10]
	nClusters = results[0][11]
	
	dm = distanceMatrixCID * cidWeight + \
	     distanceMatrixDTW * dtwWeight + \
	     distanceMatrixFired * firedWeight + \
	     distanceMatrixScaledBoolean * scaledBooleanWeight + \
	     distanceMatrixScaledFloat * scaledFloatWeight + \
	     distanceMatrixScaledZScoreBoolean * scaledZScoreBooleanWeight + \
	     distanceMatrixScaledZScoreFloat * scaledZScoreFloatWeight + \
	     distanceMatrixSpearman * spearmanWeight + \
	     distanceMatrixSpearmanZScore * spearmanZScoreWeight + \
	     distanceMatrixLengths * lengthsWeight
	
	dmScaled = pd.DataFrame(dm).values
	
	if scaling == 1:
		dmScaled = MinMaxScaler((0, 1)).fit_transform(dmScaled)
	
	clusterer = AgglomerativeClustering(n_clusters=nClusters,
	                                    affinity='precomputed',
	                                    linkage='complete').fit(dmScaled)
	clusters = clusterer.labels_
	
	clusterDataFrame = pd.DataFrame()
	clusterDataFrame['cluster'] = clusters
	
	clusterDataFrame['name'] = sorted(list(set(mainDataConverted['to_member_id'])))
	
	clusterDataFrame.to_csv('data/clusters.csv')
	
	mainDataConverted = mainDataConverted.sort_values(['to_member_id', 'cycle'], ascending=[True, True])
	mainDataConverted.drop_duplicates(inplace=True)
	
	mainData = mainData.sort_values(['to_member_id', 'cycle'], ascending=[True, True])
	mainData.drop_duplicates(inplace=True)
	
	mainData['tempCycle'] = pd.to_datetime(mainData['cycle'])
	maxDataPeriod = max(mainData['tempCycle'])
	
	tempColumnTS = []
	tempColumnDate = []
	tempColumnTSScaled = []
	tempColumnTSOld = []
	tempColumnDateOld = []
	for filter in sorted(list(set(mainDataConverted['to_member_id']))):
		mainDataConvertedFilter = mainDataConverted.loc[mainDataConverted['to_member_id'] == filter]
		
		tempColumnTS.append(mainDataConvertedFilter['aggregate'].values)
		tempColumnDate.append(mainDataConvertedFilter['cycle'].values)
		
		tempColumnTSScaled.append(scaledValues[filter].values)
		
		mainDataFilter = mainData.loc[mainData['to_member_id'] == filter]
		mainDataFilter = mainDataFilter.sort_values(['to_member_id', 'cycle'], ascending=[True, True])
		
		tempColumnTSOld.append(mainDataFilter['aggregate'].values)
		tempColumnDateOld.append(mainDataFilter['cycle'].values)
	
	clusterDataFrame['ts'] = tempColumnTS
	clusterDataFrame['cycle'] = tempColumnDate
	clusterDataFrame['tsScaled'] = tempColumnTSScaled
	clusterDataFrame['tsOld'] = tempColumnTSOld
	clusterDataFrame['cycleOld'] = tempColumnDateOld
	
	clusterDataFrame.to_csv('data/clusterDataFrame.csv')
	
	tempColumnTSOldFlatten = [item for sublist in tempColumnTSOld for item in sublist]
	
	scalingTS.index = [str(value) for value in scalingTS.index]
	scalingTS = scalingTS.to_dict()
	
	syntheticPeople = pd.DataFrame()
	
	stdSet = []
	for cluster in sorted(list(set(clusterDataFrame['cluster']))):
		clusterDataFrameFilter = clusterDataFrame.loc[clusterDataFrame['cluster'] == cluster]
		
		clusterDataFrameFilterConverted = pd.DataFrame()
		for filter in sorted(list(set(clusterDataFrameFilter['name']))):
			clusterDataFrameFilterName = clusterDataFrameFilter.loc[clusterDataFrameFilter['name'] == filter]
			
			dateValues = clusterDataFrameFilterName['cycle'].values[0]
			dateValues = [datetime.strptime(str(value).split('T')[0], '%Y-%m-%d') for value in dateValues]
			
			minDate = min(dateValues)
			clusterDataFrameFilterName['cycle'] = [[(value - minDate).total_seconds() for value in dateValues]]
			
			clusterDataFrameFilterConverted = pd.concat((clusterDataFrameFilterConverted, clusterDataFrameFilterName))
		
		NUM_COLORS = len(list(set(clusterDataFrameFilterConverted['name'])))
		
		cm = plt.get_cmap('gist_rainbow')
		itemColors = [cm(1. * index / NUM_COLORS) for index in
		              range(len(list(set(clusterDataFrameFilterConverted['name']))))]
		
		figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
		axesFlatten = axes.flatten()
		
		for item, timeStamps, color in zip(clusterDataFrameFilterConverted['ts'],
		                                   clusterDataFrameFilterConverted['cycle'],
		                                   itemColors):
			axesFlatten[0].plot(timeStamps, item, color=color)
			
			axesFlatten[0].set_xlabel('timestamp')
			axesFlatten[0].set_ylabel('interpolated aggregated score (shifted to the left)')
		
		globalTimeStamps = [datetime.strptime(value, '%Y-%m-%d')
		                    for subValue in clusterDataFrameFilterConverted['cycleOld'] for value in subValue]
		globalTimeStamps = [datetime(value.year, value.month, 1) for value in globalTimeStamps]
		
		newMin = -1
		newMax = 1
		for item, timeStamps, color in zip(clusterDataFrameFilterConverted['tsOld'],
		                                   clusterDataFrameFilterConverted['cycleOld'],
		                                   itemColors):
			newTimeStamps = [datetime.strptime(value, '%Y-%m-%d') for value in timeStamps]
			timeStampsConverted = [str(value) for value in newTimeStamps]
			
			minValue = [scalingTS.get('minScore').get(value) for value in timeStampsConverted]
			maxValue = [scalingTS.get('maxScore').get(value) for value in timeStampsConverted]
			
			newItem = [(((value - minSubValue) * (newMax - newMin)) / (maxSubValue - minSubValue)) + newMin
			           for value, minSubValue, maxSubValue in zip(item, minValue, maxValue)]
			
			tempArray = zip(newTimeStamps, newItem)
			minTS = min(newTimeStamps)
			
			tempArray = [value for value in tempArray if value[0] >= minTS + relativedelta(months=3)]
			
			newTimeStampsCut = [value[0] for value in tempArray]
			newItem = [value[1] for value in tempArray]
			
			axesFlatten[1].plot(newTimeStampsCut, newItem, color=color)
			
			axesFlatten[1].set_xlim((min(globalTimeStamps), max(globalTimeStamps)))
			
			axesFlatten[1].set_xlabel('real time')
			axesFlatten[1].set_ylabel('scaled aggregated score')
		
		xNewSmooth = []
		smoothSet = []
		for item, color in zip(clusterDataFrameFilterConverted['tsScaled'], itemColors):
			item = stats.zscore(item)
			axesFlatten[2].plot(item, color=color)
			
			axesFlatten[2].set_xlabel('timestamp')
			axesFlatten[2].set_ylabel('interpolated scaled + zscore aggregated score (expanded)')
			
			smoothSet.append(item)
		
		smoothSetMean = np.mean(smoothSet, axis=0)
		smoothSetStd = np.std(smoothSet, axis=0)
		
		stdSet.append((np.mean(smoothSetStd), cluster))
		
		axesFlatten[2].plot(smoothSetMean, linewidth=3, linestyle='-', color='black')
		axesFlatten[2].fill_between(range(0, len(smoothSetMean)),
		                            (smoothSetMean - 2 * smoothSetStd),
		                            (smoothSetMean + 2 * smoothSetStd),
		                            color='black', alpha=0.25, linestyle='-')
		
		for item, timeStamps, filter, color in zip(clusterDataFrameFilterConverted['tsOld'],
		                                           clusterDataFrameFilterConverted['cycleOld'],
		                                           clusterDataFrameFilterConverted['name'],
		                                           itemColors):
			timeStamps = [datetime.strptime(value, '%Y-%m-%d') for value in timeStamps]
			axesFlatten[3].plot(timeStamps, item, label=filter, color=color)
			
			axesFlatten[3].set_xlim((min(globalTimeStamps), max(globalTimeStamps)))
			
			axesFlatten[3].set_xlabel('real time')
			axesFlatten[3].set_ylabel('aggregated score')
		
		tempWeight = sum([1 if '(F)' in value else 0 for value in clusterDataFrameFilterConverted['name'].values]) / \
		             len(clusterDataFrameFilterConverted['name'].values)
		
		'''firedItems = list(zip(clusterDataFrameFilterConverted['tsOld'],
							  clusterDataFrameFilterConverted['cycleOld'],
							  clusterDataFrameFilterConverted['name']))

		totalItems = len(firedItems)
		firedItems = [(value[0], value[1]) for value in firedItems if '(F)' in value[2]]
		fItems = len(firedItems)

		firedTS = [value[0] for value in firedItems]
		firedCycle = [[datetime.strptime(str(subValue), '%Y-%m-%d').timestamp() for subValue in value[1]] for value in
					  firedItems]'''
		
		'''personIndex = 0
		for tempCount in range(2, len(firedItems) + 1):

			for index, item in enumerate(itertools.combinations(list(zip(firedTS, firedCycle)), tempCount)):
				print(cluster, index, len(list(itertools.combinations(list(zip(firedTS, firedCycle)), tempCount))))

				if len(item) == 1:
					continue

				itemTS = np.asarray([value[0] for value in item])
				itemCycle = [value[1] for value in item]

				itemCycleConverted = []
				for item in itemCycle:
					itemCycleConverted.append([value - min(item) for value in item])

				maxLength = max([max(value) for value in itemCycleConverted])
				maxPeriod = max([len(value) for value in itemCycle])

				for index, item in enumerate(itemCycleConverted):

					if max(item) <= maxLength:
						itemCycleConverted[index] = [value / max(item) * maxLength for value in item]

				function = []
				for cycle, item in zip(itemCycleConverted, itemTS):
					function.append(interp1d(cycle, item, kind='linear'))

				xNew = np.arange(0, maxLength, int(maxLength) / maxPeriod)

				aggregate = []
				for subFunction in function:
					aggregate.append(subFunction(xNew))

				startPoint = np.median([value[0] for value in itemCycle])
				xNew = [value + startPoint for value in xNew]

				if datetime.fromtimestamp(xNew[-1]).replace(hour=0, minute=0, second=0, microsecond=0) > maxDataPeriod:
					continue

				aggregateMean = np.mean(aggregate, axis=0)

				data = pd.DataFrame()
				data['cycle'] = [datetime.fromtimestamp(value).replace(hour=0, minute=0, second=0, microsecond=0) for
								 value in xNew]
				data['aggregate'] = aggregateMean

				data = pd.DataFrame(data.set_index('cycle')['aggregate'])
				data.columns = ['aggregate']

				data = data.reindex(pd.date_range(start=data.index.min(),
												  end=data.index.max(),
												  freq='d')).interpolate(method='linear')

				data = traces.TimeSeries(data['aggregate'])
				data = data.moving_average(60 * 60 * 24 * 7 * (365 / 12), pandas=True)
				data = pd.DataFrame(data)

				data.reset_index(inplace=True)
				data.columns = ['cycle', 'aggregate']

				data['to_member_id'] = ['synthetic[' + str(cluster) + '-' + str(personIndex) + '-' + str(
					tempCount) + ']'] * len(data.index)
				data['weight'] = [tempWeight / tempCount] * len(data.index)

				if len(data.index) >= 9 and len(sorted(list(set(data['aggregate'].values)))) > 1:
					syntheticPeople = pd.concat((syntheticPeople, data))

				personIndex += 1

				if personIndex >= 50:
					break'''
		#
		
		handles, _ = axesFlatten[3].get_legend_handles_labels()
		figure.legend(handles, clusterDataFrameFilterConverted['name'],
		              loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)
		
		plt.tight_layout()
		
		tempWeight = np.round(tempWeight * 100, 2)
		
		figure.suptitle(
				'Cluster: ' + str(cluster) + '; ' + str(
						len(clusterDataFrameFilterConverted['name'])) + ' items; ' + str(tempWeight) + '%')
		figure.subplots_adjust(top=0.88)
		
		plt.rcParams['figure.figsize'] = 120, 36
		figure.savefig('plots/' + str(tempWeight) + '%_cluster[' + str(cluster) + '].png', dpi=300, bbox_inches='tight')
		
		plt.clf()
	
	syntheticPeople.to_csv('data/syntheticPeople.csv')
