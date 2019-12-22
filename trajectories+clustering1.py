"""
trajectories clustering - DTW
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import warnings
from copy import deepcopy
from dtaidistance import dtw
from scipy.interpolate import interp1d
from scipy.stats import boxcox, zscore
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)

with warnings.catch_warnings():
	warnings.filterwarnings('ignore', category = FutureWarning)
	warnings.filterwarnings('ignore', category = DeprecationWarning)

#

GETSCALEDVALUES = False
GETCLUSTERS = True

METRICBOUND = [1, 2, 3]

report = pd.DataFrame()
metricColumn = []
notIncludedColumn = []
averageCountColumn = []
numOfClustersColumn = []

mainData = pd.read_csv('../_dataProcessing/data/PRS[agg_2].csv', index_col = 0)

mainData = mainData[['cycle', 'to_member_id', 'prsScaled']]

firedData = pd.read_csv('../../../data/firedDataUpdated.csv', index_col = None, sep = ';')
firedData.dropna(inplace = True)
firedData = firedData.set_index('user').T.to_dict('list')

tempColumn = []
for item in mainData['to_member_id']:
	
	if item in firedData.keys():
		tempColumn.append(item + ' ' + str(firedData.get(item)))
		
		continue
	
	tempColumn.append(item)

mainData['to_member_id'] = tempColumn

#

trajectories = {}
trajectoriesZScores = {}

for filter in sorted(list(set(list(mainData['to_member_id'])))):
	tempArray = mainData.loc[mainData['to_member_id'] == filter]['prsScaled'].values
	
	if len(tempArray) == 51:
		trajectories[filter] = tempArray[1:]
	
	else:
		trajectories[filter] = tempArray

maxLength = max([len(value) for value in trajectories.values()])
for key, value in trajectories.items():
	
	if len(value) == maxLength:
		continue
	
	timeSteps = list(range(len(value)))
	
	interpolationFunction = interp1d(timeSteps, value)
	timeStepsNew = np.arange(0, len(value) - 1, (len(value) - 1) / maxLength)
	
	reshapedValue = interpolationFunction(timeStepsNew)
	
	reshapedValue = reshapedValue[: maxLength]
	
	if len(reshapedValue) != maxLength:
		print(123)
	
	trajectories[key] = reshapedValue
	
	minValue = min(reshapedValue)
	if minValue <= 0:
		reshapedValue = [value - minValue + 0.0001 for value in reshapedValue]
	
	reshapedValue = boxcox(reshapedValue)[0]
	trajectoriesZScores[key] = zscore(reshapedValue)

for threshold in METRICBOUND:
	trajectoriesToWork = deepcopy(trajectoriesZScores)
	iteration = 1
	
	dmDictionary = {}
	
	check = True
	while check:
		check = False
		print('threshold: ', threshold, 'iteration: ', iteration)
		
		dm = [[0 for _ in range(len(trajectoriesToWork))] for _ in range(len(trajectoriesToWork))]
		
		for index1, (filter1, item1) in enumerate(trajectoriesToWork.items()):
			tempArray = []
			
			for index2, (filter2, item2) in enumerate(trajectoriesToWork.items()):
				
				if index1 > index2:
					continue
				
				elif index1 == index2:
					dm[index1][index2] = 0
					dm[index2][index1] = 0
					
					continue
				
				tempList = [filter1, filter2]
				sorted(tempList)
				tempList = ';'.join(tempList)
				
				if tempList in dmDictionary.keys():
					dm[index1][index2] = dmDictionary.get(tempList)
					dm[index2][index1] = dm[index1][index2]
					
					continue
				
				if isinstance(item1, list) or isinstance(item2, list):
					
					if not isinstance(item1, list):
						item1 = [item1]
					
					if not isinstance(item2, list):
						item2 = [item2]
					
					metric = []
					for subIndex1, subItem1 in enumerate(item1):
						
						for subIndex2, subItem2 in enumerate(item2):
							metric.append(dtw.distance_fast(subItem1, subItem2, psi = 1))
					
					metric = max(metric)
				
				else:
					metric = dtw.distance_fast(item1, item2, psi = 1)
				
				dm[index1][index2] = metric
				dm[index2][index1] = dm[index1][index2]
				
				dmDictionary[tempList] = metric
		
		minValue = min(dmDictionary.values())
		
		if minValue > threshold:
			continue
		
		minIndex = [[index1, index2]
		            for index1, row in enumerate(dm)
		            for index2, value in enumerate(row)
		            if index1 < index2 and value == minValue]
		
		if len(minIndex) == 1:
			minIndex = minIndex[0]
			
			names = [key for key, value in trajectoriesToWork.items()]
			name1 = names[minIndex[0]]
			name2 = names[minIndex[1]]
			
			item1 = trajectoriesToWork.get(name1)
			item2 = trajectoriesToWork.get(name2)
			
			if not isinstance(item1, list):
				item1 = [item1]
			
			itemMean = item1
			
			if isinstance(item2, list):
				itemMean.extend(item2)
			
			else:
				itemMean.append(item2)
			
			trajectoriesToWork = {key: value for key, value in trajectoriesToWork.items() if
			                      key not in [name1, name2]}
			
			tempList = [name1, name2]
			sorted(tempList)
			tempList = ';'.join(tempList)
			
			trajectoriesToWork[tempList] = itemMean
			
			dmDictionary = {key: value for key, value in dmDictionary.items()
			                if name1 not in key and name2 not in key}
			
			check = True
			iteration += 1
		
		else:
			print(minIndex)
			indicesFlatten = [item for subList in minIndex for item in subList]
			
			if len(list(set(indicesFlatten))) == len(indicesFlatten):
				
				for subMinIndex in minIndex:
					names = [key for key, value in trajectoriesToWork.items()]
					name1 = names[subMinIndex[0]]
					name2 = names[subMinIndex[1]]
					
					item1 = trajectoriesToWork.get(name1)
					item2 = trajectoriesToWork.get(name2)
					
					if not isinstance(item1, list):
						item1 = [item1]
					
					itemMean = item1
					
					if isinstance(item2, list):
						itemMean.extend(item2)
					
					else:
						itemMean.append(item2)
					
					trajectoriesToWork = {key: value for key, value in trajectoriesToWork.items() if
					                      key not in [name1, name2]}
					
					tempList = [name1, name2]
					sorted(tempList)
					tempList = ';'.join(tempList)
					
					trajectoriesToWork[tempList] = itemMean
					
					dmDictionary = {key: value for key, value in dmDictionary.items()
					                if name1 not in key and name2 not in key}
					
					check = True
					iteration += 1
	
	#
	
	try:
		shutil.rmtree('output/case[' + str(threshold) + ']')
	
	except Exception as _:
		pass
	
	os.makedirs('output/case[' + str(threshold) + ']', 0o0755)
	
	#
	mainData = pd.read_csv('../_dataProcessing/data/PRS.csv', index_col = 0)
	
	tempColumn = []
	for item in mainData['to_member_id']:
		
		if item in firedData.keys():
			tempColumn.append(item + ' ' + str(firedData.get(item)))
			
			continue
		
		tempColumn.append(item)
	
	mainData['to_member_id'] = tempColumn
	
	realScores = {}
	scaledScores = {}
	timeStamps = {}
	for to_member_id in sorted(list(set(mainData['to_member_id']))):
		
		if to_member_id not in sorted(list(set(list(mainData['to_member_id'])))):
			continue
		
		tempDF = mainData.loc[mainData['to_member_id'] == to_member_id]
		tempDF['cycle'] = pd.to_datetime(tempDF['cycle'])
		
		realScores[to_member_id] = tempDF['prs'].values
		scaledScores[to_member_id] = tempDF['prsScaled'].values
		timeStamps[to_member_id] = tempDF['cycle'].values
	#
	
	cm = plt.get_cmap('gnuplot')
	
	countNotClustered = 0
	tempCountKeys = []
	numOfClusters = 0
	
	tempNames = []
	tempClusters = []
	tempClusterNumber = []
	
	tempCount = 1
	clusters = []
	clusterVolumes = []
	meanTrajectories = []
	for key, _ in trajectoriesToWork.items():
		key = str(key).split(';')
		key = sorted(list(set(key)))
		
		if len(key) == 1:
			countNotClustered += 1
			
			continue
		
		tempCountKeys.append(len(key))
		numOfClusters += 1
		
		#
		NUMCOLORS = len(key)
		itemColors = [cm(1. * index / NUMCOLORS) for index in range(NUMCOLORS)]
		
		figure, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 10))
		axesFlatten = axes.flatten()
		
		# original values
		for subKey, color in zip(key, itemColors):
			axesFlatten[0].plot(timeStamps.get(subKey), realScores.get(subKey), color = color)
		
		axesFlatten[0].set_xlabel('original timestamps')
		axesFlatten[0].set_ylabel('real values')
		
		# scaled values
		for subKey, color in zip(key, itemColors):
			axesFlatten[1].plot(timeStamps.get(subKey), scaledScores.get(subKey), color = color)
		
		axesFlatten[1].set_xlabel('original timestamps')
		axesFlatten[1].set_ylabel('scaled values')
		
		yRanges1 = axesFlatten[1].get_ylim()
		
		# aggregated values
		for subKey, color in zip(key, itemColors):
			axesFlatten[2].plot(range(len(trajectories.get(subKey))), trajectories.get(subKey), color = color,
			                    label = subKey)
		
		axesFlatten[2].set_xlabel('aggregated steps')
		axesFlatten[2].set_ylabel('aggregated scaled values')
		
		yRanges2 = axesFlatten[2].get_ylim()
		
		minY = min(yRanges1[0], yRanges2[0])
		maxY = max(yRanges1[1], yRanges2[1])
		
		axesFlatten[1].set_ylim([minY, maxY])
		axesFlatten[2].set_ylim([minY, maxY])
		
		# trajectories
		aggregation = 2
		meanTrajectory = []
		for subKey, color in zip(key, itemColors):
			tempItem = trajectoriesZScores.get(subKey)
			tempItem = MinMaxScaler((0, 1)).fit_transform(np.array(tempItem).reshape(-1, 1))
			tempItem = [value[0] for value in tempItem]
			
			axesFlatten[3].plot(range(len(trajectoriesZScores.get(subKey))), tempItem,
			                    color = color, label = subKey)
			
			meanTrajectory.append(tempItem)
		
		axesFlatten[3].get_xaxis().set_visible(False)
		axesFlatten[3].set_ylim([-0.25, 1.25])
		
		try:
			meanTrajectory = np.mean(meanTrajectory, axis = 0)
		# continue
		
		except:
			for item in meanTrajectory:
				print(item, len(item))
			
			# print(meanTrajectory)
			
			continue
		
		meanTrajectory = np.convolve(meanTrajectory, np.ones((aggregation * 2,)) / (aggregation * 2),
		                             mode = 'same')
		meanTrajectory = meanTrajectory[aggregation: len(meanTrajectory) - aggregation]
		
		axesFlatten[3].plot(range(0 + aggregation, len(trajectoriesZScores.get(subKey)) - aggregation),
		                    meanTrajectory, color = 'red', linewidth = 3, linestyle = '--')
		
		handles, _ = axesFlatten[3].get_legend_handles_labels()
		
		figure.legend(handles, key, loc = 'center left', bbox_to_anchor = (1, 0.5),
		              ncol = int(len(key) / 25) + 1)
		
		plt.tight_layout()
		
		regrettableCount = 0
		notRegrettableCount = 0
		notRegrettableStrengthsCount = 0
		unknownCount = 0
		notFiredCount = 0
		for subKey in key:
			
			if "'regrettable'" in subKey:
				regrettableCount += 1
			
			elif "'not_regrettable'" in subKey:
				notRegrettableCount += 1
			
			elif "'not_regrettable + strengths'" in subKey:
				notRegrettableStrengthsCount += 1
			
			elif "'unknown'" in subKey:
				unknownCount += 1
			
			else:
				notFiredCount += 1
		
		titleString = 'Cluster #' + str(tempCount) + '\n'
		
		if notFiredCount != 0:
			titleString += 'Not_Fired: ' + str(notFiredCount) + '\n'
		
		if regrettableCount != 0:
			titleString += 'Regrettable: ' + str(regrettableCount) + '\n'
		
		if notRegrettableCount != 0:
			titleString += 'Not_Regrettable: ' + str(notRegrettableCount) + '\n'
		
		if notRegrettableStrengthsCount != 0:
			titleString += 'Not_Regrettable + Strengths: ' + str(notRegrettableStrengthsCount)
		
		if unknownCount != 0:
			titleString += 'Unknown: ' + str(unknownCount)
		
		figure.suptitle(titleString)
		figure.subplots_adjust(top = 0.88)
		
		plt.rcParams['figure.figsize'] = 120, 36
		
		fileName = [('[regrettable]', regrettableCount),
		            ('[not_regrettable]', notRegrettableCount),
		            ('[not_regrettable + strengths]', notRegrettableStrengthsCount),
		            ('[unknown]', unknownCount),
		            ('[not_fired]', notFiredCount)]
		fileName = [value[0] for value in fileName if value[1] > 0]
		
		if len(fileName) == 1:
			figure.savefig('output/case[' + str(threshold) +
			               ']/' + str(fileName[0]) + ']_cluster[' + str(tempCount) +
			               '].png', dpi = 300, bbox_inches = 'tight')
		
		else:
			figure.savefig('output/case[' + str(threshold) +
			               ']/cluster[' + str(tempCount) +
			               '].png', dpi = 300, bbox_inches = 'tight')
		
		plt.clf()
		plt.close(figure)
		
		tempNames.extend(sorted(list(set(key))))
		
		shortClusterName = str(tempCount)
		
		if notFiredCount != 0:
			shortClusterName += '; nf: ' + str(notFiredCount)
		
		if regrettableCount != 0:
			shortClusterName += '; r: ' + str(regrettableCount)
		
		if notRegrettableCount != 0:
			shortClusterName += '; nr: ' + str(notRegrettableCount)
		
		if notRegrettableStrengthsCount != 0:
			shortClusterName += '; nrs: ' + str(notRegrettableStrengthsCount)
		
		if unknownCount != 0:
			shortClusterName += '; u: ' + str(unknownCount)
		
		tempClusters.extend([shortClusterName] * len(sorted(list(set(key)))))
		tempClusterNumber.extend([tempCount] * len(sorted(list(set(key)))))
		
		tempCount += 1
	
	metricColumn.append(threshold)
	notIncludedColumn.append(countNotClustered)
	averageCountColumn.append(np.mean(tempCountKeys))
	numOfClustersColumn.append(np.mean(numOfClusters))
	
	clusteringResults = pd.DataFrame()
	clusteringResults['name'] = tempNames
	clusteringResults['cluster'] = tempClusters
	clusteringResults['clusterNumber'] = tempClusterNumber
	
	clusteringResults.to_csv('output/case[' + str(threshold) + ']/clusteringResults.csv', index = None)

report['metric'] = metricColumn
report['notIncluded'] = notIncludedColumn
report['averageCount'] = averageCountColumn
report['numOfClusters'] = numOfClustersColumn

report.to_csv('output/report.csv', index = None)
