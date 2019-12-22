"""
token similaritis via gensim & gloVe
"""

import pandas as pd
import spacy
from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors

GENSIM_MODEL = 'outputData//modelUpdated.txt'
GLOVE_MODEL = 'outputData//gloveModel'
INPUT_FILE = 'outputData//2.1.wordLemma.csv'

OUTPUT_FILE = 'outputData//2.3.1.tokenToken.csv'

THRESHOLD_GENSIM = 0.75
THRESHOLD_GLOVE = 0.95

nlp = spacy.load('en')


class similarityProcess:
	
	def __init__(self):
		# self.model = Word2Vec.load(GENSIM_MODEL)
		self.model = KeyedVectors.load_word2vec_format(GENSIM_MODEL, binary=False)
		self.gloveModel = KeyedVectors.load(GLOVE_MODEL)
		self.wordLemmas = pd.read_csv(INPUT_FILE, index_col=0)
	
	def run(self):
		tokensUsed = []
		resultArray = []
		
		for key, _ in self.model.wv.vocab.items():
			
			if self.filter(key):
				continue
			
			keyClean = key.split('|')[0]
			
			wordToCheck = [keyClean]
			wordLemmasSmall = self.wordLemmas.loc[self.wordLemmas['word'] == keyClean]
			
			for item in wordLemmasSmall['lemma']:
				wordToCheck.append(item)
			
			wordToCheck = sorted(list(set(wordToCheck)))
			
			for word in wordToCheck:
				
				try:
					
					for keySimilar, valueSimilar in self.model.most_similar(positive=[key], topn=10):
						
						if self.filter(keySimilar):
							continue
						
						if valueSimilar >= THRESHOLD_GENSIM \
								and keyClean != keySimilar.split('|')[0] \
								and str(keyClean) + '|' + str(keySimilar.split('|')[0]) not in tokensUsed:
							resultArray.append(
									(keyClean, keySimilar.split('|')[0], valueSimilar, 'positiveGensim', key, keySimilar))
							tokensUsed.append(str(keyClean) + '|' + str(keySimilar.split('|')[0]))
						
						similarity = self.model.similarity(keySimilar, key)
						if similarity >= THRESHOLD_GENSIM \
								and keyClean != keySimilar.split('|')[0] \
								and str(keySimilar.split('|')[0]) + '|' + str(keyClean) not in tokensUsed:
							resultArray.append(
									(keySimilar.split('|')[0], keyClean, similarity, 'positiveGensimBackwards', keySimilar, key))
							tokensUsed.append(str(keySimilar.split('|')[0]) + '|' + str(keyClean))
				
				except:
					pass
				
				###
				
				try:
					
					for keySimilar, valueSimilar in self.gloveModel.most_similar(positive=[word], topn=10):
						
						if self.filter(keySimilar + '|' + nlp(keySimilar)[0].pos_):
							continue
						
						if valueSimilar >= THRESHOLD_GLOVE \
								and keyClean != keySimilar \
								and str(keyClean) + '|' + str(keySimilar) not in tokensUsed:
							resultArray.append(
									(keyClean, keySimilar, valueSimilar, 'positiveGlove', keyClean, keySimilar))
							tokensUsed.append(str(keyClean) + '|' + str(keySimilar))
						
						similarity = self.gloveModel.similarity(keySimilar, keyClean)
						if similarity >= THRESHOLD_GLOVE \
								and keyClean != keySimilar \
								and str(keySimilar) + '|' + str(keyClean) not in tokensUsed:
							resultArray.append(
									(keySimilar, keyClean, similarity, 'positiveGloveBackwards', keySimilar, keyClean))
							tokensUsed.append(str(keySimilar) + '|' + str(keyClean))
				
				except:
					pass
		
		self.exportFile(resultArray)
	
	@classmethod
	def filter(cls, value):
		
		if not any(tag in value for tag in ['|NOUN', '|VERB', '|ADJ']):
			return True
		
		if nlp.vocab[value.split('|')[0]].is_stop:
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


sp = similarityProcess()
similarityProcess.run(sp)
