"""
token similaritis via gensim & gloVe - v2
"""

import pandas as pd
import spacy
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from numpy import percentile

GENSIM_MODEL = 'outputData//modelUpdated.txt'
GLOVE_MODEL = 'outputData//gloveModel'
INPUT_FILE = 'outputData//2.1.wordLemma.csv'
FILE_TO_UPDATE = 'outputData//2.3.1.tokenToken.csv'

OUTPUT_FILE = 'outputData//2.3.2.tokenToken.csv'

THRESHOLD_GENSIM = 0.50
THRESHOLD_GLOVE = 0.75

MAX_CONNECTIONS = 3
PERCENTILE_VALUE = 0.75

nlp = spacy.load('en')


###

class similarityProcess:
	
	def __init__(self):
		# self.model = Word2Vec.load(GENSIM_MODEL)
		self.model = KeyedVectors.load_word2vec_format(GENSIM_MODEL, binary=False)
		self.gloveModel = KeyedVectors.load(GLOVE_MODEL)
		self.wordLemmas = pd.read_csv(INPUT_FILE, index_col=0)
		
		self.data = pd.read_csv(FILE_TO_UPDATE, index_col=0)
	
	def run(self):
		keysToUse = self.getPopularTokens(self.data)
		
		###
		
		self.tokensUsed, self.dataTuples = self.getTuples(self.data)
		
		for index1, key1 in enumerate(keysToUse):
			
			for index2, key2 in enumerate(keysToUse):
				
				if index1 >= index2:
					continue
				
				index1Gensim = self.getIndices(self.data, 'positiveGensim', key1)
				index2Gensim = self.getIndices(self.data, 'positiveGensim', key2)
				
				index1Glove = self.getIndices(self.data, 'positiveGlove', key1)
				index2Glove = self.getIndices(self.data, 'positiveGlove', key2)
				
				if len(index1Gensim) != 0 and len(index2Gensim) != 0:
					
					for subItem1 in index1Gensim:
						
						if self.filter(subItem1):
							continue
						
						for subItem2 in index2Gensim:
							
							if self.filter(subItem2) or subItem1 == subItem2:
								continue
							
							subItem1Clean = subItem1.split('|')[0]
							subItem2Clean = subItem2.split('|')[0]
							
							self.similarityCalculation(self.model, THRESHOLD_GENSIM, subItem1Clean, subItem2Clean,
							                           subItem1, subItem2, 'positiveGensim')
				
				if len(index1Glove) != 0 and len(index2Glove) != 0:
					
					for subItem1 in index1Glove:
						
						if self.filter(str(subItem1) + '|' + nlp(unicode(subItem1))[0].pos_):
							continue
						
						for subItem2 in index2Glove:
							
							if self.filter(str(subItem2) + '|' + nlp(unicode(subItem2))[0].pos_) or subItem1 == subItem2:
								continue
							
							self.similarityCalculation(self.gloveModel, THRESHOLD_GLOVE,
							                           subItem1, subItem2,
							                           None, None, 'positiveGlove')
		
		# print(self.tokensUsed)
		self.exportFile(self.dataTuples)
	
	@classmethod
	def filter(cls, value):
		
		if not any(tag in value for tag in ['|NOUN', '|VERB', '|ADJ']):
			return True
		
		if nlp.vocab[unicode(value.split('|')[0])].is_stop:
			return True
		
		return False
	
	@classmethod
	def exportFile(cls, tempArray):
		tempArray.sort(key=lambda tup: tup[2], reverse=True)
		
		result = pd.DataFrame()
		result['originalToken'] = [value[0] for value in tempArray]
		result['connectedToken'] = [value[1] for value in tempArray]
		result['similarity'] = [value[2] for value in tempArray]
		result['type'] = [value[3] for value in tempArray]
		result['index1'] = [value[4] for value in tempArray]
		result['index2'] = [value[5] for value in tempArray]
		
		result.to_csv(OUTPUT_FILE)
	
	@classmethod
	def getPopularTokens(cls, data):
		counter = Counter(data['originalToken'])
		
		tempArray = []
		for key, value in counter.items():
			
			if value >= MAX_CONNECTIONS:
				tempArray.append(value)
		
		tempValue = percentile(tempArray, PERCENTILE_VALUE)
		
		return [key for key, value in counter.items() if value >= tempValue]
	
	@classmethod
	def getTuples(cls, data):
		
		tempArray = []
		for item1, item2, item3 in zip(data['originalToken'],
		                               data['connectedToken'],
		                               data['type']):
			tempArray.append(str(item1) + '|' + str(item2))
		
		# print(tempArray)
		return tempArray, [tuple(value) for value in data.values]
	
	@classmethod
	def getIndices(cls, data, type, key):
		dataSmall = data.loc[(data['originalToken'] == key) & \
		                     data['type'].isin([type, type + 'Backwards'])]
		
		index = sorted(list(set(dataSmall['index1'].values)))
		
		return index
	
	def similarityCalculation(self, similarityModel, thresholdType, item1, item2, item1Clean, item2Clean, type):
		
		if item1Clean is None:
			item1Clean = item1
			item2Clean = item2
		
		similarity = similarityModel.similarity(item1Clean, item2Clean)
		if similarity >= thresholdType \
				and str(item1Clean) + '|' + str(item2Clean) not in self.tokensUsed:
			self.dataTuples.append((item1, item2, similarity, type + 'Additional',
			                        item1Clean, item2Clean))
			self.tokensUsed.append(str(item1) + '|' + str(item2))
		
		similarity = similarityModel.similarity(item2Clean, item1Clean)
		if similarity >= thresholdType \
				and str(item2Clean) + '|' + str(item1Clean) not in self.tokensUsed:
			self.dataTuples.append((item2, item1, similarity, type + 'BackwardsAdditional',
			                        item2Clean, item1Clean))
			self.tokensUsed.append(str(item2) + '|' + str(item1))


sp = similarityProcess()
similarityProcess.run(sp)
